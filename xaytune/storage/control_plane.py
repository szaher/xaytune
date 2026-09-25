"""The public write surface (ADR-005 §3–§5).

PR-004 deliberately shipped no public write method, because a transition and its
event have to commit together and there were no events yet. This module is the
other half: every operation here is one transaction, and the units are the ones
the ADR names.

```text
transition_experiment / _node / _run / _attempt
                                            state + event + outbox        §3
create_attempt_with_submit_intent(...)      attempt + INTENDED operation  §4
confirm_operation / mark_operation_sent / fail_operation
request_cancellation(...)                   Action + cancel operation     §5
reconcile_cancellation(...)                 Action settled from observed state

begin_evaluation_cycle(...)                 node EVALUATING + its runs    ADR-015
create_evaluation_attempt_with_submit_intent(...)
hold_evaluation_completion(...)             pending completion + cursor
record_evaluation_result(...)               result + attempt + run SUCCEEDED
reconcile_evaluating_node(...)              wait / DECIDING / EvaluationStalled
```

Evaluation shares the journal, the cancellation path and the telemetry cursor
with training, and none of the aggregates: an evaluation attempt is cancelled
by the same ``cancel-attempt`` Action through the same operation journal, and
lives in its own table.

There is still no ``save_experiment()``. The row writers stay private on
:class:`~xaytune.storage.repository.AggregateStore` and
:mod:`xaytune.storage.journal`, and are composed here rather than exposed, so a
caller cannot write state without its event or request an effect without durable
intent -- the API simply does not offer those operations separately.

The asymmetry this buys, from ADR-005 §9: it is always safe to hold durable
intent with no effect, and never safe to have an effect with no durable intent.
Every boundary below is arranged so only the first can happen.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any, Literal, Protocol, TypeVar

from xaytune.core.clock import utc_now
from xaytune.core.domain.action import (
    Action,
    ActionOutcome,
    ActionStatus,
    ActionTarget,
)
from xaytune.core.domain.decision import Decision, DecisionOutcome, DecisionProposal
from xaytune.core.domain.evaluation import (
    EvaluationAttempt,
    EvaluationResult,
    EvaluationRun,
    result_provenance_problems,
)
from xaytune.core.domain.event import DomainEvent, OutboxRecord
from xaytune.core.domain.experiment import Experiment, ExperimentNode
from xaytune.core.domain.operation import (
    RuntimeOperation,
    RuntimeOperationTarget,
)
from xaytune.core.domain.run import Run, RunAttempt
from xaytune.core.errors import ConcurrentModificationError
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import (
    ActionId,
    EvaluationAttemptId,
    EvaluationRunId,
    EventId,
    ExperimentId,
    ExperimentNodeId,
    OperationId,
    RunAttemptId,
    RunId,
)
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor, ArtifactRef, RuntimeRef
from xaytune.core.state.status import (
    EvaluationAttemptStatus,
    EvaluationRunStatus,
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)
from xaytune.core.telemetry import EvaluationCompletedPayload
from xaytune.storage.actions import ActionStore
from xaytune.storage.database import write_transaction
from xaytune.storage.errors import AggregateNotFoundError, StorageError
from xaytune.storage.graph import ExperimentGraph
from xaytune.storage.journal import (
    EventJournal,
    IdempotencyConflictError,
    OperationJournal,
)
from xaytune.storage.repository import AggregateStore

__all__ = [
    "ControlPlaneRepository",
    "DecisionConflictError",
    "EvaluationReconciliation",
    "ProvenanceError",
    "UnknownOperationTargetError",
]


class _Transitionable(Protocol):
    """What ``_transition`` needs of an aggregate.

    Structural rather than a base class: the four aggregates already share
    these through :class:`AggregateModel`, and naming the requirement here is
    what lets the loader, the writer and the return type be one type.
    """

    revision: int

    @property
    def id(self) -> Any: ...

    def with_status(self, new_status: Any) -> Any: ...


AggregateT = TypeVar("AggregateT", bound=_Transitionable)

_AGGREGATE_TABLES: dict[str, str] = {
    "Experiment": "experiments",
    "ExperimentNode": "experiment_nodes",
    "Run": "runs",
    "RunAttempt": "run_attempts",
    "EvaluationRun": "evaluation_runs",
    "EvaluationAttempt": "evaluation_attempts",
}

_AttemptKind = Literal["training-attempt", "evaluation-attempt"]

_ATTEMPT_KINDS: frozenset[str] = frozenset({"training-attempt", "evaluation-attempt"})
"""The target kinds a single cancel operation can address.

Run- and experiment-level cancellation fans out over descendants (ADR-013 §6),
so it is a saga rather than one effect, and PR-006a refuses it rather than
writing intent nothing will carry out.
"""

_CANCEL_TYPE_FOR: dict[str, str] = {
    "experiment": "cancel-experiment",
    "run": "cancel-run",
    "training-attempt": "cancel-attempt",
    "evaluation-attempt": "cancel-attempt",
}
"""Which cancellation action a target kind implies.

``cancel-attempt`` covers both attempt kinds: cancelling is the same operation
on the same kind of subject, and only the table differs.
"""

_TARGET_TABLES: dict[str, str] = {
    "training-attempt": "run_attempts",
    "evaluation-attempt": "evaluation_attempts",
}


class EvaluationReconciliation(str, Enum):
    """What reconciling a node in ``EVALUATING`` concluded (ADR-015 §5).

    Exactly one, always -- the point of stating it as three cases rather than
    one invariant is that ordinary completion, which passes through "every
    run terminal, node not yet moved", is the middle case and not an incident.
    """

    WAITING = "waiting"
    """A run of the current cycle is still in flight."""

    DECIDING = "deciding"
    """Every run of the cycle succeeded with a result; the node moved on."""

    STALLED = "stalled"
    """The cycle can never complete as recorded: no runs, or a run that ended
    without a result. Recorded as ``EvaluationStalled`` and left visible."""


class CancellationSagaRequiredError(StorageError):
    """Run- and experiment-level cancellation is a saga, not one operation.

    ADR-013 §6: cancelling an experiment fans out to its descendants, and the
    invariant is that it must not reach ``CANCELLED`` while any descendant is
    unresolved. An experiment's saga is
    :meth:`ControlPlaneRepository.request_experiment_cancellation`; a run's is
    not built yet.

    Refused here rather than half-implemented. Writing the `Action` and no operation
    would leave durable intent that nothing carries out and nothing retries --
    the exact state ADR-005 §5 exists to prevent -- and minting a single
    operation against a run id would ask a runtime to cancel something it has
    no handle on.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        remedy = (
            "use request_experiment_cancellation()"
            if kind == "experiment"
            else "cancel its attempts individually"
        )
        super().__init__(
            f"cancelling a {kind} requires the descendant saga of ADR-013 §6: {remedy}"
        )


class ProvenanceError(StorageError):
    """A record would attribute something to a producer that did not make it."""


class DecisionConflictError(StorageError):
    """A cycle already decided is being decided differently.

    The same decision again -- a restarted controller deciding the cycle it
    had decided before it died -- is recognised and returns the one on
    record. A decision on other inputs, or with another outcome, is a second
    answer to one question, and is refused rather than appended.
    """


class UnknownOperationTargetError(StorageError):
    """An operation names a target that does not exist.

    SQLite cannot enforce this: the journal's target is typed rather than a
    foreign key, because training and evaluation attempts live in different
    tables (ADR-013). So the repository enforces it instead -- ADR-005 §10.1.
    """

    def __init__(self, kind: str, target_id: str) -> None:
        self.kind = kind
        self.target_id = target_id
        super().__init__(f"no {kind} exists with id {target_id}")


def _require_consistent_candidate(node: ExperimentNode) -> None:
    """Refuse a node whose stored fingerprint does not describe its candidate.

    The fingerprint is derived, so trusting a supplied one lets the indexed
    identity disagree with the body it indexes -- the same class as a
    caller-supplied aggregate. Every consumer downstream believes the
    fingerprint: graph comparison calls two nodes the same candidate, reuse
    lookups match the wrong hypothesis, and the run consistency check compares
    against a value that describes nothing.

    Raises:
        StorageError: If the fingerprint does not match the candidate.
    """
    expected = node.candidate.candidate.candidate_fingerprint()
    if node.candidate_fingerprint != expected:
        raise StorageError(
            f"node {node.id} carries fingerprint {node.candidate_fingerprint!r} "
            f"but its candidate fingerprints as {expected!r}: the stored "
            f"identity would not describe the proposition it indexes"
        )


def _require_pristine(aggregate: Any, initial: Any) -> None:
    """Refuse a created aggregate that is not actually new.

    Nothing stopped a caller passing ``Experiment(status=ACTIVE, revision=7)``
    to a create method. The row would then start mid-lifecycle with a revision
    no transition produced, and its creation event would claim a state it never
    entered -- history beginning with a state that has no transition into it.
    """
    if aggregate.status is not initial:
        raise ValueError(
            f"a new {type(aggregate).__name__} must start in "
            f"{initial.value!r}, not {aggregate.status.value!r}: a creation "
            f"event cannot record a state the aggregate never transitioned into"
        )
    if aggregate.revision != 0:
        raise ValueError(
            f"a new {type(aggregate).__name__} must start at revision 0, not "
            f"{aggregate.revision}: every later revision is produced by a "
            f"transition that also wrote an event"
        )


def _assert_same_attempt(
    existing: RunAttempt | EvaluationAttempt, requested: RunAttempt | EvaluationAttempt
) -> None:
    """Refuse a replay whose attempt identity differs.

    Raises:
        IdempotencyConflictError: Naming each differing field.
    """
    run_field = "run_id" if isinstance(existing, RunAttempt) else "evaluation_run_id"
    differing = tuple(
        field
        for field in ("id", run_field, "attempt_number")
        if getattr(existing, field, None) != getattr(requested, field, None)
    )
    if differing:
        raise IdempotencyConflictError(str(existing.id), differing, kind="attempt")


def _require_run_of_cycle(run: EvaluationRun, node: ExperimentNode) -> None:
    """Refuse an evaluation run that does not belong to the node's current cycle.

    Raises:
        StorageError: If the run names another node or experiment, the node is
            not evaluating, or the run belongs to another of its cycles.
    """
    if run.node_id != node.id or run.experiment_id != node.experiment_id:
        raise StorageError(
            f"evaluation run {run.id} names node {run.node_id} of experiment "
            f"{run.experiment_id}, not node {node.id} of {node.experiment_id}"
        )
    if node.status is not ExperimentNodeStatus.EVALUATING:
        raise StorageError(
            f"node {node.id} is {node.status.value}; evaluation runs belong to a node "
            f"that is evaluating"
        )
    if run.evaluation_cycle != node.evaluation_cycle:
        raise StorageError(
            f"evaluation run {run.id} is for cycle {run.evaluation_cycle}, but node "
            f"{node.id} is evaluating cycle {node.evaluation_cycle}: a run of another "
            f"round would be counted, or waited for, by the wrong one"
        )


def _require_result_of_run(result: EvaluationResult, run: EvaluationRun) -> None:
    """Refuse a result whose provenance disagrees with the run it names.

    Raises:
        ProvenanceError: Naming every disagreement
            (:func:`~xaytune.core.domain.evaluation.result_provenance_problems`).
    """
    problems = result_provenance_problems(result, run)
    if problems:
        raise ProvenanceError(
            f"result {result.id} disagrees with run {run.id}, so it would describe an "
            f"evaluation the run did not perform: " + "; ".join(problems)
        )


class ControlPlaneRepository:
    """Atomic writes over the control-plane aggregates and their journals."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self.aggregates = AggregateStore(connection)
        self.events = EventJournal(connection)
        self.operations = OperationJournal(connection)
        self.actions = ActionStore(connection)
        self.graph = ExperimentGraph(connection)

    # ---- ADR-005 §3 ----------------------------------------------------

    def create_experiment(
        self,
        experiment: Experiment,
        *,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> Experiment:
        """Create an experiment, its creation event and any outbox records."""
        _require_pristine(experiment, ExperimentStatus.CREATED)

        with write_transaction(self._connection):
            self.aggregates._insert_experiment(experiment)
            self._emit(experiment, "ExperimentCreated", str(experiment.id), actor, destinations)
        return experiment

    def create_node(
        self,
        node: ExperimentNode,
        *,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> ExperimentNode:
        """Create a node, its creation event and any outbox records.

        Lineage is validated first: a parent that does not exist, sits in
        another experiment, or already descends from this node is refused
        before anything is written.

        Raises:
            LineageError: If the node's parents would make the graph unsound.
        """
        _require_pristine(node, ExperimentNodeStatus.CREATED)

        _require_consistent_candidate(node)

        with write_transaction(self._connection):
            self.graph.validate_parents(node)
            self.aggregates._insert_node(node)
            self._emit(node, "NodeCreated", str(node.experiment_id), actor, destinations)
        return node

    def create_run(self, run: Run, *, actor: Actor, destinations: tuple[str, ...] = ()) -> Run:
        """Create a run, its creation event and any outbox records."""
        _require_pristine(run, RunStatus.CREATED)

        with write_transaction(self._connection):
            self._require_consistent_run(run)
            self.aggregates._insert_run(run)
            self._emit(run, "RunCreated", str(run.experiment_id), actor, destinations)
        return run

    def transition_experiment(
        self,
        experiment_id: ExperimentId,
        *,
        expected_revision: int,
        new_status: ExperimentStatus,
        actor: Actor,
        event_type: str | None = None,
        destinations: tuple[str, ...] = (),
    ) -> Experiment:
        """Move an experiment to *new_status*, with its event, atomically."""
        return self._transition(
            "Experiment",
            str(experiment_id),
            expected_revision,
            new_status,
            self.aggregates.get_experiment,
            self.aggregates._update_experiment,
            actor,
            event_type,
            destinations,
        )

    def transition_node(
        self,
        node_id: ExperimentNodeId,
        *,
        expected_revision: int,
        new_status: ExperimentNodeStatus,
        actor: Actor,
        event_type: str | None = None,
        destinations: tuple[str, ...] = (),
    ) -> ExperimentNode:
        """Move a node to *new_status*, with its event, atomically."""
        return self._transition(
            "ExperimentNode",
            str(node_id),
            expected_revision,
            new_status,
            self.aggregates.get_node,
            self.aggregates._update_node,
            actor,
            event_type,
            destinations,
        )

    def transition_run(
        self,
        run_id: RunId,
        *,
        expected_revision: int,
        new_status: RunStatus,
        actor: Actor,
        event_type: str | None = None,
        destinations: tuple[str, ...] = (),
    ) -> Run:
        """Move a run to *new_status*, with its event, atomically."""
        return self._transition(
            "Run",
            str(run_id),
            expected_revision,
            new_status,
            self.aggregates.get_run,
            self.aggregates._update_run,
            actor,
            event_type,
            destinations,
        )

    def transition_attempt(
        self,
        attempt_id: RunAttemptId,
        *,
        expected_revision: int,
        new_status: RunAttemptStatus,
        actor: Actor,
        event_type: str | None = None,
        destinations: tuple[str, ...] = (),
        telemetry_position: tuple[int, int] | None = None,
    ) -> RunAttempt:
        """Move an attempt to *new_status*, with its event, atomically.

        *telemetry_position* is the ``(generation, sequence)`` of the
        telemetry event that caused this transition. The attempt's durable
        cursor advances to it in the same commit, so after a restart the
        controller resumes from the last event whose effect is recorded.
        """
        return self._transition(
            "RunAttempt",
            str(attempt_id),
            expected_revision,
            new_status,
            self.aggregates.get_attempt,
            self.aggregates._update_attempt,
            actor,
            event_type,
            destinations,
            after_write=(
                None
                if telemetry_position is None
                else lambda moved: self.aggregates._advance_telemetry(
                    str(moved.id), telemetry_position
                )
            ),
        )

    def record_artifact(
        self,
        attempt_id: RunAttemptId,
        artifact: ArtifactRef,
        *,
        expected_revision: int,
        actor: Actor,
        destinations: tuple[str, ...] = (),
        telemetry_position: tuple[int, int] | None = None,
    ) -> RunAttempt:
        """Record an artifact an attempt produced, with its event, atomically.

        *telemetry_position*, if given, advances the attempt's durable cursor
        in the same commit (see :meth:`transition_attempt`).

        The event carries the artifact itself, so the history answers "what
        did this attempt produce" without reading the attempt's current row --
        which a later revision may have moved on from.

        **The producer is made explicit here.** A worker reports an artifact
        without ``producer_attempt_id``, because at that boundary attribution
        travels on the telemetry envelope's target. Stored as reported, the
        durable artifact would carry less provenance than the telemetry that
        produced it, and an artifact read on its own -- in a result, an export
        -- could not say which attempt made it. So a missing producer is filled
        in with *attempt_id*. A different producer is refused rather than
        overwritten: that is a claim this attempt did not make.

        Raises:
            AggregateNotFoundError: If no such attempt exists.
            ConcurrentModificationError: If it has moved since the caller read it.
            ProvenanceError: If the artifact names a different producing attempt.
            ValueError: If the attempt already records this artifact.
        """
        if artifact.producer_attempt_id is None:
            artifact = artifact.model_copy(update={"producer_attempt_id": attempt_id})
        elif artifact.producer_attempt_id != attempt_id:
            raise ProvenanceError(
                f"artifact {artifact.id} names attempt {artifact.producer_attempt_id} as its "
                f"producer, not {attempt_id}; recording it here would misattribute it"
            )
        with write_transaction(self._connection):
            current = self.aggregates.get_attempt(str(attempt_id))
            if current is None:
                raise AggregateNotFoundError("RunAttempt", str(attempt_id))
            if current.revision != expected_revision:
                raise ConcurrentModificationError("RunAttempt", str(attempt_id), expected_revision)
            recorded = current.with_artifact(artifact)
            self.aggregates._update_attempt(recorded)
            self._emit(
                recorded,
                "ArtifactRecorded",
                self._owning_experiment(recorded),
                actor,
                destinations,
                extra={"artifact": artifact.model_dump(mode="json")},
            )
            if telemetry_position is not None:
                self.aggregates._advance_telemetry(str(attempt_id), telemetry_position)
        return recorded

    def record_telemetry_degraded(
        self,
        attempt_id: RunAttemptId | EvaluationAttemptId,
        *,
        reason: str,
        actor: Actor,
        destinations: tuple[str, ...] = (),
        kind: Literal["training-attempt", "evaluation-attempt"] = "training-attempt",
    ) -> tuple[int, int]:
        """A dead stream over a live workload: advance the generation, and say so.

        ADR-014 §1a. The supervisor writing the attempt's telemetry is gone
        while the workload runs on. Nothing it wrote can be continued, so the
        attempt's stream moves to the next generation -- a later supervisor,
        or a later controller, starts clean rather than reading old sequences
        as new -- and a ``TelemetryDegraded`` event records the interval in
        which the controller saw nothing. **No attempt is created**: the
        workload is the same one, and a new attempt would claim a new
        execution that never happened.

        The attempt's revision does not change: its status and its payload
        are what they were, and the generation is a column the controller
        assigns (001). Idempotent per generation -- a controller that finds
        the same dead stream again after a restart does not advance twice.

        Returns:
            The new ``(generation, sequence)`` position.
        """
        with write_transaction(self._connection):
            attempt: RunAttempt | EvaluationAttempt = (
                self.aggregates.load_attempt(str(attempt_id))
                if kind == "training-attempt"
                else self.aggregates.load_evaluation_attempt(str(attempt_id))
            )
            generation, sequence = self.aggregates.telemetry_position(str(attempt_id), kind=kind)
            # Already degraded into this generation, and nothing recorded from
            # it since: the same dead stream, found again. A generation that
            # has since carried events and then died is a new degradation.
            already = sequence == -1 and any(
                event.event_type == "TelemetryDegraded"
                and event.payload.get("to_generation") == generation
                for event in self.events.events_for_aggregate(str(attempt_id))
            )
            if already:
                return generation, -1
            self._connection.execute(
                f"UPDATE {_TARGET_TABLES[kind]} "  # noqa: S608 - table from a literal map
                "SET telemetry_generation = ?, telemetry_sequence = -1 WHERE id = ?",
                (generation + 1, str(attempt_id)),
            )
            self._emit(
                attempt,
                "TelemetryDegraded",
                self._owning_experiment(attempt),
                actor,
                destinations,
                extra={
                    "from_generation": generation,
                    "to_generation": generation + 1,
                    "reason": reason,
                },
            )
        return generation + 1, -1

    # ---- ADR-015: evaluation ------------------------------------------------

    def begin_evaluation_cycle(
        self,
        node_id: ExperimentNodeId,
        *,
        expected_revision: int,
        runs: Sequence[EvaluationRun],
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> tuple[ExperimentNode, tuple[EvaluationRun, ...]]:
        """Move a node into ``EVALUATING`` and record the runs its new cycle requires.

        One commit: the node's transition -- which advances its
        ``evaluation_cycle`` -- and every run of that cycle. Split, a crash
        between them would leave a node evaluating with nothing to evaluate,
        or runs belonging to a round the node never entered. Each run must
        name the cycle the node is entering.

        An empty *runs* is recorded as asked. The node then has nothing to
        wait on, and reconciling it reports the cycle stalled rather than
        pretending it completed.

        Raises:
            AggregateNotFoundError: If the node does not exist.
            ConcurrentModificationError: If it has moved since the caller read it.
            InvalidTransitionError: If the node cannot enter ``EVALUATING``.
            StorageError: If a run names another node, experiment or cycle.
        """
        for run in runs:
            _require_pristine(run, EvaluationRunStatus.CREATED)

        with write_transaction(self._connection):
            current = self.aggregates.get_node(str(node_id))
            if current is None:
                raise AggregateNotFoundError("ExperimentNode", str(node_id))
            if current.revision != expected_revision:
                raise ConcurrentModificationError("ExperimentNode", str(node_id), expected_revision)
            moved = current.with_status(ExperimentNodeStatus.EVALUATING)
            for run in runs:
                _require_run_of_cycle(run, moved)
            self.aggregates._update_node(moved)
            self._emit(
                moved,
                "ExperimentNodeStatusChanged",
                str(moved.experiment_id),
                actor,
                destinations,
                extra={"evaluation_cycle": moved.evaluation_cycle},
            )
            for run in runs:
                self.aggregates._insert_evaluation_run(run)
                self._emit(run, "EvaluationRunCreated", str(run.experiment_id), actor, destinations)
        return moved, tuple(runs)

    def create_evaluation_run(
        self, run: EvaluationRun, *, actor: Actor, destinations: tuple[str, ...] = ()
    ) -> EvaluationRun:
        """Add a run -- a replicate, say -- to the cycle a node is evaluating.

        Raises:
            AggregateNotFoundError: If the node does not exist.
            StorageError: If the node is not evaluating, or is in another cycle.
        """
        _require_pristine(run, EvaluationRunStatus.CREATED)
        with write_transaction(self._connection):
            node = self.aggregates.load_node(str(run.node_id))
            _require_run_of_cycle(run, node)
            self.aggregates._insert_evaluation_run(run)
            self._emit(run, "EvaluationRunCreated", str(run.experiment_id), actor, destinations)
        return run

    def transition_evaluation_run(
        self,
        run_id: EvaluationRunId,
        *,
        expected_revision: int,
        new_status: EvaluationRunStatus,
        actor: Actor,
        destinations: tuple[str, ...] = (),
        reason: str | None = None,
    ) -> EvaluationRun:
        """Move an evaluation run to *new_status*, with its event, atomically.

        *reason*, if given, is carried on the event -- why an evaluation
        failed is part of its history.

        Not to ``SUCCEEDED``: a run succeeds only with its result, through
        :meth:`record_evaluation_result`, so no record can say an evaluation
        succeeded without saying what it measured.
        """
        if new_status is EvaluationRunStatus.SUCCEEDED:
            raise StorageError(
                f"evaluation run {run_id} succeeds only with its result: use "
                f"record_evaluation_result(), which writes both in one commit"
            )
        return self._transition(
            "EvaluationRun",
            str(run_id),
            expected_revision,
            new_status,
            self.aggregates.get_evaluation_run,
            self.aggregates._update_evaluation_run,
            actor,
            None,
            destinations,
            extra=None if reason is None else {"reason": reason},
        )

    def transition_evaluation_attempt(
        self,
        attempt_id: EvaluationAttemptId,
        *,
        expected_revision: int,
        new_status: EvaluationAttemptStatus,
        actor: Actor,
        destinations: tuple[str, ...] = (),
        telemetry_position: tuple[int, int] | None = None,
        reason: str | None = None,
    ) -> EvaluationAttempt:
        """Move an evaluation attempt to *new_status*, with its event, atomically.

        *telemetry_position* advances the attempt's durable cursor in the same
        commit, as for a training attempt. Not to ``SUCCEEDED`` -- see
        :meth:`transition_evaluation_run`.
        """
        if new_status is EvaluationAttemptStatus.SUCCEEDED:
            raise StorageError(
                f"evaluation attempt {attempt_id} succeeds only with its result: use "
                f"record_evaluation_result(), which writes both in one commit"
            )
        return self._transition(
            "EvaluationAttempt",
            str(attempt_id),
            expected_revision,
            new_status,
            self.aggregates.get_evaluation_attempt,
            self.aggregates._update_evaluation_attempt,
            actor,
            None,
            destinations,
            after_write=(
                None
                if telemetry_position is None
                else lambda moved: self.aggregates._advance_telemetry(
                    str(moved.id), telemetry_position, kind="evaluation-attempt"
                )
            ),
            extra=None if reason is None else {"reason": reason},
        )

    def hold_evaluation_completion(
        self,
        attempt_id: EvaluationAttemptId,
        completion: EvaluationCompletedPayload,
        *,
        telemetry_position: tuple[int, int],
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> tuple[EvaluationCompletedPayload, tuple[int, int]]:
        """Keep an ``EvaluationCompleted`` durably until the workload's end decides it.

        A completion is not yet a result: the workload may still fail on its
        way out, so it cannot be recorded as one. Held only in memory,
        though, it could be lost -- a stream that dies over a live workload
        moves the attempt to a new generation (ADR-014 §1a), and the
        completion is in the one the next controller no longer reads. So it
        is written here, with the cursor advanced to it in the same commit:
        the effect of that event is now "the completion is held", and the
        cursor never claims more than the record holds.

        The first completion is kept. A worker reports one; a second is not a
        correction the controller can choose between, and replaying the
        first after a restart must not replace it.

        Returns:
            The completion held and its position -- the first one, if one
            was already held.

        Raises:
            AggregateNotFoundError: If the attempt does not exist.
            StorageError: If the attempt has already ended.
        """
        with write_transaction(self._connection):
            attempt = self.aggregates.load_evaluation_attempt(str(attempt_id))
            if attempt.is_terminal:
                raise StorageError(
                    f"evaluation attempt {attempt_id} is {attempt.status.value}; a completion "
                    f"held for it now could never be settled"
                )
            held = self.aggregates.pending_completion(str(attempt_id))
            if held is not None:
                return held
            self.aggregates._hold_completion(str(attempt_id), completion, telemetry_position)
            self.aggregates._advance_telemetry(
                str(attempt_id), telemetry_position, kind="evaluation-attempt"
            )
            self._emit(
                attempt,
                "EvaluationCompletionHeld",
                self._owning_experiment(attempt),
                actor,
                destinations,
                extra={
                    "metrics": [metric.name for metric in completion.metrics or ()],
                    "generation": telemetry_position[0],
                    "sequence": telemetry_position[1],
                },
            )
        return completion, telemetry_position

    def create_evaluation_attempt_with_submit_intent(
        self,
        attempt: EvaluationAttempt,
        *,
        request_digest: str,
        actor: Actor,
        operation_id: OperationId | None = None,
        destinations: tuple[str, ...] = (),
    ) -> tuple[EvaluationAttempt, RuntimeOperation]:
        """An evaluation attempt and its ``INTENDED`` submit, in one commit (ADR-015 §4).

        The same unit, the same get-or-create and the same journal as
        :meth:`create_attempt_with_submit_intent`: submitting an evaluation
        is the same problem as submitting training, solved once.

        Raises:
            IdempotencyConflictError: If either half exists against a
                different request.
            StorageError: If the run has already finished.
        """
        created: tuple[EvaluationAttempt, RuntimeOperation] = self._create_with_submit_intent(
            "evaluation-attempt",
            attempt,
            EvaluationAttemptStatus.CREATED,
            request_digest=request_digest,
            actor=actor,
            operation_id=operation_id,
            destinations=destinations,
        )
        return created

    def record_evaluation_result(
        self,
        attempt_id: EvaluationAttemptId,
        result: EvaluationResult,
        *,
        expected_revision: int,
        actor: Actor,
        telemetry_position: tuple[int, int] | None = None,
        destinations: tuple[str, ...] = (),
    ) -> EvaluationResult:
        """Record what an evaluation measured, and that it succeeded, in one commit.

        The result, the attempt's ``SUCCEEDED`` and the run's ``SUCCEEDED``:
        there is no state in between, so no success without a result and no
        result without a success. The completion the result comes from was
        already held durably, with the cursor advanced to it
        (:meth:`hold_evaluation_completion`), so a crash before this commit
        loses nothing -- the next controller records the same result from the
        held completion -- and a crash after it leaves nothing to redo.
        *telemetry_position* only moves the cursor forward, if at all.

        The result's provenance is checked against the run it names -- every
        field, including each metric's evaluator, version and seed
        (``result_provenance_problems``).

        Raises:
            AggregateNotFoundError: If the attempt does not exist.
            ConcurrentModificationError: If it has moved since the caller read it.
            InvalidTransitionError: If the attempt is not running or the run
                not active.
            ProvenanceError: If the result disagrees with its run.
            StorageError: If the run already has a result.
        """
        with write_transaction(self._connection):
            attempt = self.aggregates.get_evaluation_attempt(str(attempt_id))
            if attempt is None:
                raise AggregateNotFoundError("EvaluationAttempt", str(attempt_id))
            if attempt.revision != expected_revision:
                raise ConcurrentModificationError(
                    "EvaluationAttempt", str(attempt_id), expected_revision
                )
            run = self.aggregates.load_evaluation_run(str(attempt.evaluation_run_id))
            _require_result_of_run(result, run)
            if self.aggregates.evaluation_result_for_run(str(run.id)) is not None:
                raise StorageError(
                    f"evaluation run {run.id} already has a result; a second would be a "
                    f"second answer from one execution"
                )

            succeeded_attempt = attempt.with_status(EvaluationAttemptStatus.SUCCEEDED)
            succeeded_run = run.with_status(EvaluationRunStatus.SUCCEEDED)
            self.aggregates._update_evaluation_attempt(succeeded_attempt)
            self.aggregates._update_evaluation_run(succeeded_run)
            self.aggregates._insert_evaluation_result(result)
            if telemetry_position is not None:
                self.aggregates._advance_telemetry(
                    str(attempt.id), telemetry_position, kind="evaluation-attempt"
                )

            experiment_id = str(run.experiment_id)
            self._emit(
                succeeded_attempt,
                "EvaluationAttemptStatusChanged",
                experiment_id,
                actor,
                destinations,
            )
            self._emit(
                succeeded_run, "EvaluationRunStatusChanged", experiment_id, actor, destinations
            )
            self._emit(
                succeeded_run,
                "EvaluationResultRecorded",
                experiment_id,
                actor,
                destinations,
                extra={
                    "result_id": str(result.id),
                    "metrics": [metric.name for metric in result.metrics],
                },
            )
        return result

    def reconcile_evaluating_node(
        self,
        node_id: ExperimentNodeId,
        *,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> EvaluationReconciliation:
        """Resolve a node in ``EVALUATING`` to exactly one of wait, decide or stall.

        ADR-015 §5, over the runs of the node's **current** evaluation cycle
        only -- a run from an earlier round, however successful, is history:

        ```text
        a run of this cycle is not terminal                 -> WAITING
        every run succeeded, each with its result           -> node DECIDING
        otherwise: no runs, or a run ended without a result -> EvaluationStalled
        ```

        The middle case **repairs** a lag rather than reporting it: a
        controller that died between a run's success and the node's
        transition leaves exactly that state, and this moves the node on.

        A stall is recorded as an ``EvaluationStalled`` event on the node --
        once per cycle, however often it is reconciled -- and the node stays
        ``EVALUATING``. It is visible rather than silent, which is the point;
        what to do about it is a decision, not reconciliation's to make.

        Raises:
            AggregateNotFoundError: If the node does not exist.
            StorageError: If the node is not in ``EVALUATING``.
        """
        with write_transaction(self._connection):
            node = self.aggregates.load_node(str(node_id))
            if node.status is not ExperimentNodeStatus.EVALUATING:
                raise StorageError(
                    f"node {node_id} is {node.status.value}, not evaluating; there is no "
                    f"evaluation cycle to reconcile"
                )
            runs = self.aggregates.evaluation_runs_for_node(
                str(node.id), cycle=node.evaluation_cycle
            )
            if any(not run.is_terminal for run in runs):
                return EvaluationReconciliation.WAITING

            unsatisfied = [
                run
                for run in runs
                if run.status is not EvaluationRunStatus.SUCCEEDED
                or self.aggregates.evaluation_result_for_run(str(run.id)) is None
            ]
            if runs and not unsatisfied:
                deciding = node.with_status(ExperimentNodeStatus.DECIDING)
                self.aggregates._update_node(deciding)
                self._emit(
                    deciding,
                    "ExperimentNodeStatusChanged",
                    str(deciding.experiment_id),
                    actor,
                    destinations,
                    extra={"evaluation_cycle": deciding.evaluation_cycle},
                )
                return EvaluationReconciliation.DECIDING

            already = any(
                event.event_type == "EvaluationStalled"
                and event.payload.get("evaluation_cycle") == node.evaluation_cycle
                for event in self.events.events_for_aggregate(str(node.id))
            )
            if not already:
                reason = (
                    "the cycle has no evaluation runs"
                    if not runs
                    else "evaluation runs ended without a result: "
                    + ", ".join(f"{run.id} ({run.status.value})" for run in unsatisfied)
                )
                self._emit(
                    node,
                    "EvaluationStalled",
                    str(node.experiment_id),
                    actor,
                    destinations,
                    extra={
                        "evaluation_cycle": node.evaluation_cycle,
                        "reason": reason,
                        "runs": [str(run.id) for run in unsatisfied],
                    },
                )
            return EvaluationReconciliation.STALLED

    def record_decision(
        self,
        proposal: DecisionProposal,
        *,
        expected_node_revision: int,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> Decision:
        """Record *proposal* as a decision and apply it, in one commit. Idempotent per cycle.

        The decision is given its id, time and *actor* here -- the engine
        mints none -- and written with the transitions its outcome causes:

        ```text
        STOP_SUCCEEDED   node COMPLETED   experiment SUCCEEDED, best_node_id = node
        STOP_FAILED      node REJECTED    experiment FAILED
        REJECT           node REJECTED    experiment unchanged
        ```

        ``REJECT`` is a judgement on the candidate, not the experiment: another
        candidate may yet be proposed. Only the ``STOP`` outcomes end an
        experiment, and only an ``ACTIVE`` one; a paused experiment is
        somebody's to resume or stop. There is no moment at which a decision
        is recorded but not applied, or applied with no decision on record.

        Deciding a cycle already decided, with the same proposal, returns the
        decision on record and writes nothing -- what a controller restarted
        after deciding sees. Anything else is refused.

        The proposal must be about this node's current cycle, in
        ``DECIDING``, and must name exactly that cycle's results: a decision
        drawn from an earlier round's results, or from some of this round's,
        is not attributable to the evidence it claims.

        Raises:
            AggregateNotFoundError: If the node does not exist.
            DecisionConflictError: If the cycle was decided differently.
            ConcurrentModificationError: If the node moved since it was read.
            InvalidTransitionError: If the node is not in ``DECIDING``.
            ProvenanceError: If the proposal is about another experiment or
                cycle, or names results other than the cycle's.
        """
        with write_transaction(self._connection):
            node = self.aggregates.load_node(str(proposal.node_id))
            recorded = self.aggregates.decision_for_cycle(str(node.id), proposal.evaluation_cycle)
            if recorded is not None:
                if recorded.proposal() == proposal:
                    return recorded
                raise DecisionConflictError(
                    f"node {node.id} cycle {proposal.evaluation_cycle} was decided "
                    f"{recorded.outcome.value} by {recorded.engine_name} "
                    f"{recorded.engine_version} on input {recorded.input_fingerprint}; "
                    f"a different decision for the same cycle is refused"
                )

            problems = []
            if proposal.experiment_id != node.experiment_id:
                problems.append(
                    f"names experiment {proposal.experiment_id}, not {node.experiment_id}"
                )
            if proposal.evaluation_cycle != node.evaluation_cycle:
                problems.append(
                    f"decides cycle {proposal.evaluation_cycle}, but the node is in cycle "
                    f"{node.evaluation_cycle}"
                )
            cycle_results = {
                result.id
                for run in self.aggregates.evaluation_runs_for_node(
                    str(node.id), cycle=node.evaluation_cycle
                )
                if (result := self.aggregates.evaluation_result_for_run(str(run.id))) is not None
            }
            if set(proposal.evaluation_result_ids) != cycle_results:
                problems.append(
                    f"names results {sorted(proposal.evaluation_result_ids)}, but the "
                    f"cycle's are {sorted(cycle_results)}"
                )
            if problems:
                raise ProvenanceError(f"decision for node {node.id} " + "; ".join(problems))
            if node.revision != expected_node_revision:
                raise ConcurrentModificationError(
                    "ExperimentNode", str(node.id), expected_node_revision
                )

            decision = Decision.record(proposal, actor=actor)
            decided = node.with_decision(decision.id, decision.outcome.node_status)
            self.aggregates._update_node(decided)
            self.aggregates._insert_decision(decision)
            experiment_id = str(node.experiment_id)
            self._emit(
                decided,
                "DecisionRecorded",
                experiment_id,
                actor,
                destinations,
                extra={
                    "decision_id": str(decision.id),
                    "evaluation_cycle": decision.evaluation_cycle,
                    "outcome": decision.outcome.value,
                    "reason": decision.reason,
                    "engine": f"{decision.engine_name} {decision.engine_version}",
                },
            )
            self._emit(
                decided,
                "ExperimentNodeStatusChanged",
                experiment_id,
                actor,
                destinations,
                extra={"decision_id": str(decision.id)},
            )
            self._conclude_experiment(decision, actor, destinations)
        return decision

    def _conclude_experiment(
        self, decision: Decision, actor: Actor, destinations: tuple[str, ...]
    ) -> None:
        """End an ``ACTIVE`` experiment when the decision says to stop it; otherwise nothing."""
        if decision.outcome is DecisionOutcome.REJECT:
            return
        experiment = self.aggregates.load_experiment(str(decision.experiment_id))
        if experiment.status is not ExperimentStatus.ACTIVE:
            return
        if decision.outcome is DecisionOutcome.STOP_SUCCEEDED:
            ended = experiment.succeeded_with(decision.node_id)
        else:
            ended = experiment.with_status(ExperimentStatus.FAILED)
        self.aggregates._update_experiment(ended)
        self._emit(
            ended,
            "ExperimentStatusChanged",
            str(ended.id),
            actor,
            destinations,
            extra={
                "decision_id": str(decision.id),
                "best_node_id": str(ended.best_node_id) if ended.best_node_id else None,
            },
        )

    def defer_decision(
        self,
        node_id: ExperimentNodeId,
        *,
        engine: str,
        reasons: tuple[str, ...],
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> bool:
        """Record that the node's current cycle could not be decided, and why.

        The node stays ``DECIDING``: an engine that cannot decide must not be
        made to guess. A ``DecisionDeferred`` event on the node makes that
        visible -- once per cycle, however often it is asked, like an
        ``EvaluationStalled``.

        Returns:
            Whether an event was written; ``False`` if the cycle was already
            recorded as deferred.

        Raises:
            AggregateNotFoundError: If the node does not exist.
            StorageError: If the node is not in ``DECIDING``.
        """
        with write_transaction(self._connection):
            node = self.aggregates.load_node(str(node_id))
            if node.status is not ExperimentNodeStatus.DECIDING:
                raise StorageError(
                    f"node {node_id} is {node.status.value}, not deciding; there is no "
                    f"decision to defer"
                )
            already = any(
                event.event_type == "DecisionDeferred"
                and event.payload.get("evaluation_cycle") == node.evaluation_cycle
                for event in self.events.events_for_aggregate(str(node.id))
            )
            if already:
                return False
            self._emit(
                node,
                "DecisionDeferred",
                str(node.experiment_id),
                actor,
                destinations,
                extra={
                    "evaluation_cycle": node.evaluation_cycle,
                    "engine": engine,
                    "reasons": list(reasons),
                },
            )
        return True

    # ---- ADR-005 §4 ----------------------------------------------------

    def create_attempt_with_submit_intent(
        self,
        attempt: RunAttempt,
        *,
        request_digest: str,
        actor: Actor,
        operation_id: OperationId | None = None,
        destinations: tuple[str, ...] = (),
    ) -> tuple[RunAttempt, RuntimeOperation]:
        """Create an attempt and its ``INTENDED`` submit operation in one commit.

        The runtime call happens **after** this returns and outside any
        transaction. A crash in between leaves the operation ``INTENDED``, which
        ADR-013 defines as *reconcile*, not *re-issue blindly* -- so the effect
        may exist, but never without a record that it was intended.

        **Get-or-create.** This is the method a crash retries, so retrying it
        with the same ids and request returns what is already recorded and
        writes nothing new. Idempotency in the journal alone would not be
        enough: the attempt insert runs first, so a naive retry would hit the
        attempt's primary key before the journal ever checked the operation id
        -- which is precisely the crash-and-retry case ADR-013 exists to make
        safe.

        Returns:
            The attempt and the operation, so the caller has the id to pass to
            ``submit_or_get``.

        Raises:
            IdempotencyConflictError: If either half exists against a different
                request -- the operation's target, type or digest, or the
                attempt's run or number.
            ValueError: If *attempt* is not a freshly created aggregate.
        """
        return self._create_with_submit_intent(
            "training-attempt",
            attempt,
            RunAttemptStatus.CREATED,
            request_digest=request_digest,
            actor=actor,
            operation_id=operation_id,
            destinations=destinations,
        )

    def _create_with_submit_intent(
        self,
        kind: Literal["training-attempt", "evaluation-attempt"],
        attempt: Any,
        initial: Any,
        *,
        request_digest: str,
        actor: Actor,
        operation_id: OperationId | None,
        destinations: tuple[str, ...],
    ) -> tuple[Any, RuntimeOperation]:
        """ADR-005 §4 for either kind of attempt: the attempt and its intent, one commit.

        One implementation rather than one per workload, so training and
        evaluation cannot come to record intent differently.
        """
        operation = RuntimeOperation(
            id=operation_id or OperationId.generate(),
            target=RuntimeOperationTarget(kind=kind, id=str(attempt.id)),
            type="submit",
            request_digest=request_digest,
        )

        _require_pristine(attempt, initial)

        if kind == "training-attempt":
            loader: Callable[[str], Any] = self.aggregates.get_attempt
            created_event = "RunAttemptCreated"
        else:
            loader = self.aggregates.get_evaluation_attempt
            created_event = "EvaluationAttemptCreated"

        with write_transaction(self._connection):
            existing = self.operations.get(str(operation.id))
            if existing is not None:
                self.operations._assert_same_request(existing, operation)
                stored_attempt = loader(existing.target.id)
                if stored_attempt is None:
                    raise StorageError(
                        f"operation {existing.id} references attempt "
                        f"{existing.target.id}, which does not exist: the two "
                        f"are written in one transaction, so this is a "
                        f"corrupted record rather than a retry"
                    )
                # Both halves, not just the operation. The attempt half was
                # unchecked, so a retry naming a different run or attempt
                # number returned the original silently -- the same omission as
                # the cancellation replay, on the creation path.
                _assert_same_attempt(stored_attempt, attempt)
                return stored_attempt, existing

            if kind == "training-attempt":
                experiment_id = self._experiment_of_run(str(attempt.run_id))
                self.aggregates._insert_attempt(attempt)
            else:
                run = self.aggregates.load_evaluation_run(str(attempt.evaluation_run_id))
                if run.is_terminal:
                    raise StorageError(
                        f"evaluation run {run.id} is {run.status.value}; a new attempt at "
                        f"a finished run would execute an evaluation nothing will record"
                    )
                experiment_id = str(run.experiment_id)
                self.aggregates._insert_evaluation_attempt(attempt)
            stored = self.operations._insert(operation)
            self._emit(attempt, created_event, experiment_id, actor, destinations)
            self._emit_operation(
                stored, "RuntimeOperationIntended", experiment_id, actor, destinations
            )

        return attempt, stored

    def confirm_operation(
        self,
        operation_id: OperationId,
        *,
        expected_revision: int,
        actor: Actor,
        runtime_ref: RuntimeRef | None = None,
        destinations: tuple[str, ...] = (),
    ) -> RuntimeOperation:
        """Record that an operation took effect, with the runtime's reference.

        A **submit** operation requires one. Confirming a submission without a
        reference records that a workload exists and forgets where -- it leaves
        the unresolved queue, so nothing looks for it again, which is the
        orphaned workload ADR-013 was written to prevent. A cancel needs none:
        it stops something already identified.

        Raises:
            StorageError: If a submit operation is confirmed with no reference.
        """
        return self._settle(
            operation_id,
            "confirmed",
            expected_revision,
            actor,
            runtime_ref=runtime_ref,
            destinations=destinations,
        )

    def mark_operation_sent(
        self,
        operation_id: OperationId,
        *,
        expected_revision: int,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> RuntimeOperation:
        """Record that the request was issued but its outcome is not yet known."""
        return self._settle(
            operation_id, "sent", expected_revision, actor, destinations=destinations
        )

    def fail_operation(
        self,
        operation_id: OperationId,
        *,
        expected_revision: int,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> RuntimeOperation:
        """Record that the request definitively did not take effect.

        Only for a **known** negative outcome. A lost response is not a failure:
        it stays ``INTENDED`` or ``SENT`` so the reconciler keeps looking, which
        is the difference between "it did not happen" and "we do not know"
        (ADR-005 §6).
        """
        return self._settle(
            operation_id, "failed", expected_revision, actor, destinations=destinations
        )

    # ---- ADR-005 §5 ----------------------------------------------------

    def request_cancellation(
        self,
        target: ActionTarget,
        *,
        reason: str,
        actor: Actor,
        request_digest: str,
        action_id: ActionId | None = None,
        operation_id: OperationId | None = None,
        destinations: tuple[str, ...] = (),
    ) -> tuple[Action, RuntimeOperation | None]:
        """Record intent to cancel, and the effect it causes, in one commit.

        This is the §5 unit: the ``Action`` owns the durable intent while the
        ``RuntimeOperation`` carries the external effect, and splitting them
        produces two states ADR-013's model cannot represent -- an intent
        nothing will act on, or an effect with no recorded cause.

        **Get-or-create**, like submission. A caller that loses the response and
        retries with the same ids gets back what is already recorded and writes
        nothing new; without that, the retry would hit the Action primary key
        and cancellation would lack the restart safety submission has.

        Cancellation is intent plus an observed terminal state, never a status
        the target occupies (ADR-013 §4). A target that has already ended gets
        no operation at all: there is nothing to ask a runtime to stop.

        Only attempt targets are accepted. Run- and experiment-level
        cancellation is a saga over descendants (ADR-013 §6) and is refused
        here rather than half-built.

        Returns:
            The action, and the operation it caused -- or ``None`` when the
            target had already ended, in which case the action resolves
            immediately.

        Raises:
            CancellationSagaRequiredError: If the target is a run or experiment.
            UnknownOperationTargetError: If the target does not exist.
            IdempotencyConflictError: If either id exists against a different
                request.
        """
        if target.kind not in _ATTEMPT_KINDS:
            raise CancellationSagaRequiredError(target.kind)

        action_id = action_id or ActionId.generate()
        operation_id = operation_id or OperationId.generate()

        with write_transaction(self._connection):
            experiment_id = self._experiment_of_action_target(target)
            proposed = Action(
                id=action_id,
                experiment_id=ExperimentId(experiment_id),
                type=_CANCEL_TYPE_FOR[target.kind],
                target=target,
                proposed_by=actor,
                reason=reason,
            )

            replayed = self._replay_cancellation(proposed, operation_id)
            if replayed is not None:
                self._assert_same_effect_request(replayed[1], request_digest)
                return replayed

            return self._record_cancellation(
                proposed, operation_id, request_digest, experiment_id, actor, destinations
            )

    def reconcile_cancellation(
        self,
        action_id: ActionId,
        *,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> Action:
        """Settle a cancellation from observed state, or leave it in flight.

        The outcome is **derived, never supplied**. An earlier version took it
        as an argument, which let a caller persist
        ``SUCCEEDED``/``APPLIED`` while the operation was still ``INTENDED`` and
        the workload still running -- an audit record asserting a cancellation
        took effect when nothing had established it. A caller must not be able
        to manufacture provenance, so the repository owns the rule:

        ```text
        target CANCELLED and its cancel operation CONFIRMED  -> SUCCEEDED/APPLIED
        target terminal for some other reason                -> SUCCEEDED/SUPERSEDED
        cancel effect definitively FAILED, target still live -> FAILED
        otherwise                                            -> unchanged
        ```

        A failed effect has to settle the action. It is not unresolved, so
        nothing would revisit it, and the action would hold durable intent
        against an effect known not to have happened for the life of the
        record. Retrying is a **new** effect under an explicit policy, not a
        silent resurrection of this one.

        ``APPLIED`` requires both halves. A target that reached ``CANCELLED``
        while the operation is unresolved may have been stopped by something
        else, and claiming credit for it would be a guess.

        The action is **loaded by id**, not accepted as an object. A caller
        could otherwise hand in a differently-shaped `Action` carrying a real
        id and revision, and reconciliation would act on a record that is not
        the durable one. Loading it here also means a stale caller naturally
        reconciles against the latest state rather than the one it last saw.

        Returns:
            The settled action, or the stored action unchanged when the
            evidence does not yet support a conclusion.

        Raises:
            AggregateNotFoundError: If no such action exists.
        """
        action = self.actions.get(str(action_id))
        if action is None:
            raise AggregateNotFoundError("Action", str(action_id))
        if action.is_terminal:
            return action

        resolution = self._observed_outcome(action)
        if resolution is None:
            return action
        status, outcome = resolution

        settled = action.with_status(status, outcome=outcome)
        with write_transaction(self._connection):
            self.actions._update(settled)
            label = outcome.value if outcome else "failed"
            self._emit_action(settled, f"Cancellation{label.capitalize()}", actor, destinations)
        return settled

    # ---- ADR-013 §6 -------------------------------------------------------

    def request_experiment_cancellation(
        self,
        experiment_id: ExperimentId,
        *,
        reason: str,
        actor: Actor,
        action_id: ActionId | None = None,
        destinations: tuple[str, ...] = (),
    ) -> tuple[Action, tuple[tuple[Action, RuntimeOperation | None], ...]]:
        """Record intent to cancel an experiment, and every effect it needs.

        One commit writes the ``cancel-experiment`` Action and, for each attempt
        still live, a ``cancel-attempt`` child Action with its cancel
        operation. The experiment itself stays ``ACTIVE`` (ADR-013 §6): the
        intent lives in the Action, and ``CANCELLED`` is reached only by
        :meth:`reconcile_experiment_cancellation`, once nothing it owns is
        executing.

        **Get-or-create.** A second request while one is in flight returns the
        first, with its children, and writes nothing: two sagas over the same
        attempts would issue duplicate effects and race to settle one
        experiment. Retrying with the same ``action_id`` returns what was
        recorded, and with a different request is refused.

        Returns:
            The parent action, and each child with the operation it caused --
            ``None`` for a child whose attempt had already ended.

        Raises:
            UnknownOperationTargetError: If the experiment does not exist.
            IdempotencyConflictError: If *action_id* exists against a different
                request.
        """
        target = ActionTarget(kind="experiment", id=str(experiment_id))

        with write_transaction(self._connection):
            experiment = self.aggregates.get_experiment(str(experiment_id))
            if experiment is None:
                raise UnknownOperationTargetError(target.kind, target.id)

            proposed = Action(
                id=action_id or ActionId.generate(),
                experiment_id=experiment.id,
                type="cancel-experiment",
                target=target,
                proposed_by=actor,
                reason=reason,
            )
            existing = self.actions.get(str(proposed.id))
            if existing is not None:
                self.actions._assert_same_request(existing, proposed)
                return existing, self._saga_children(existing)

            in_flight = [
                action
                for action in self.actions.for_target(target.kind, target.id)
                if action.type == "cancel-experiment" and not action.is_terminal
            ]
            if in_flight:
                return in_flight[0], self._saga_children(in_flight[0])

            self.actions._insert(proposed)
            self._emit_action(proposed, "ActionProposed", actor, destinations)
            validating = self._advance(
                proposed, ActionStatus.VALIDATING, "ActionValidating", actor, destinations
            )
            validated = self._advance(
                validating, ActionStatus.VALIDATED, "ActionValidated", actor, destinations
            )
            executing = self._advance(
                validated, ActionStatus.EXECUTING, "ActionExecuting", actor, destinations
            )

            if experiment.is_terminal:
                outcome = (
                    ActionOutcome.NOOP
                    if experiment.status is ExperimentStatus.CANCELLED
                    else ActionOutcome.SUPERSEDED
                )
                settled = executing.with_status(ActionStatus.SUCCEEDED, outcome=outcome)
                self.actions._update(settled)
                self._emit_action(
                    settled, f"Cancellation{outcome.value.capitalize()}", actor, destinations
                )
                return settled, ()

            children = tuple(
                self._record_cancellation(
                    Action(
                        id=ActionId.generate(),
                        experiment_id=experiment.id,
                        type="cancel-attempt",
                        target=ActionTarget(kind=kind, id=attempt_id),
                        proposed_by=actor,
                        reason=reason,
                        parent_action_id=executing.id,
                    ),
                    OperationId.generate(),
                    _cancel_digest(kind, attempt_id),
                    str(experiment.id),
                    actor,
                    destinations,
                )
                for kind, attempt_id in self._live_attempt_targets(str(experiment.id))
            )
            self._emit_action(executing, "CancellationRequested", actor, destinations)

        return executing, children

    def reconcile_experiment_cancellation(
        self,
        action_id: ActionId,
        *,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> Action:
        """Settle an experiment cancellation from observed state, or leave it.

        ```text
        a child still in flight                    -> unchanged, wait
        a child's effect FAILED, its attempt live  -> FAILED; experiment stays
                                                      ACTIVE (ADR-013 AC-8)
        no attempt of the experiment is live       -> runs, nodes and the
                                                      experiment CANCELLED,
                                                      and the action APPLIED
        ```

        The last row commits as one unit, and re-checks inside that commit that
        no attempt is live: ``CANCELLED`` means Xaytune believes no owned
        workload is executing (ADR-013 §6), so it is written only when the
        record shows that at the moment of writing. A run that already
        finished keeps its outcome -- cancelling an experiment does not
        rewrite what its runs did.

        There is deliberately no timeout path. A cancellation that cannot be
        confirmed leaves the experiment ``ACTIVE``; abandoning it is a separate,
        explicit operation that does not exist yet.

        Raises:
            AggregateNotFoundError: If no such action exists.
            StorageError: If the action is not an experiment cancellation.
        """
        parent = self.actions.get(str(action_id))
        if parent is None:
            raise AggregateNotFoundError("Action", str(action_id))
        if parent.type != "cancel-experiment":
            raise StorageError(f"action {action_id} is a {parent.type}, not cancel-experiment")
        if parent.is_terminal:
            return parent

        for child in self.actions.children(str(parent.id)):
            if not child.is_terminal:
                self.reconcile_cancellation(child.id, actor=actor, destinations=destinations)
        children = self.actions.children(str(parent.id))
        if any(not child.is_terminal for child in children):
            return parent

        experiment_id = parent.target.id
        if self._live_attempts(experiment_id):
            if any(child.status is ActionStatus.FAILED for child in children):
                failed = parent.with_status(ActionStatus.FAILED)
                with write_transaction(self._connection):
                    self.actions._update(failed)
                    self._emit_action(failed, "CancellationFailed", actor, destinations)
                return failed
            return parent

        with write_transaction(self._connection):
            if self._live_attempts(experiment_id):
                return parent
            for node in self.aggregates.nodes_for_experiment(experiment_id):
                for run in self.aggregates.runs_for_node(str(node.id)):
                    if not run.is_terminal:
                        cancelled_run = run.with_status(RunStatus.CANCELLED)
                        self.aggregates._update_run(cancelled_run)
                        self._emit(
                            cancelled_run, "RunStatusChanged", experiment_id, actor, destinations
                        )
                for evaluation in self.aggregates.evaluation_runs_for_node(str(node.id)):
                    if not evaluation.is_terminal:
                        cancelled_evaluation = evaluation.with_status(EvaluationRunStatus.CANCELLED)
                        self.aggregates._update_evaluation_run(cancelled_evaluation)
                        self._emit(
                            cancelled_evaluation,
                            "EvaluationRunStatusChanged",
                            experiment_id,
                            actor,
                            destinations,
                        )
                if not node.is_terminal:
                    cancelled_node = node.with_status(ExperimentNodeStatus.CANCELLED)
                    self.aggregates._update_node(cancelled_node)
                    self._emit(
                        cancelled_node,
                        "ExperimentNodeStatusChanged",
                        experiment_id,
                        actor,
                        destinations,
                    )
            experiment = self.aggregates.load_experiment(experiment_id)
            if not experiment.is_terminal:
                cancelled = experiment.with_status(ExperimentStatus.CANCELLED)
                self.aggregates._update_experiment(cancelled)
                self._emit(cancelled, "ExperimentStatusChanged", experiment_id, actor, destinations)
            settled = parent.with_status(ActionStatus.SUCCEEDED, outcome=ActionOutcome.APPLIED)
            self.actions._update(settled)
            self._emit_action(settled, "CancellationApplied", actor, destinations)
        return settled

    def unsettled_work(
        self, experiment_id: str
    ) -> tuple[tuple[RuntimeOperation, ...], tuple[Action, ...]]:
        """The experiment's control work that has not reached an outcome.

        Operations still ``INTENDED`` or ``SENT`` -- an effect requested with
        no known result -- and actions not yet terminal. A run can be terminal
        while either remains: a cancellation that raced natural completion
        leaves its operation unresolved after the attempt has already
        succeeded. Whoever asks "is there anything left to do" has to ask about
        these too, not only about runs.
        """
        operations = tuple(
            operation
            for kind, attempt_id in self._attempt_targets(experiment_id)
            for operation in self.operations.for_target(kind, attempt_id)
            if operation.state in ("intended", "sent")
        )
        actions = tuple(
            action
            for action in self.actions.for_experiment(experiment_id)
            if not action.is_terminal
        )
        return operations, actions

    def _saga_children(self, parent: Action) -> tuple[tuple[Action, RuntimeOperation | None], ...]:
        """Each child of *parent*, with the operation it caused if any."""
        children = []
        for child in self.actions.children(str(parent.id)):
            caused = self.actions.caused_operation_ids(str(child.id))
            children.append((child, self.operations.get(caused[0]) if caused else None))
        return tuple(children)

    def _live_attempts(self, experiment_id: str) -> tuple[RunAttempt | EvaluationAttempt, ...]:
        """Every attempt of the experiment, of either kind, not yet terminal.

        Evaluation attempts count: ``CANCELLED`` means no owned workload is
        executing (ADR-013 §6), and an evaluation is an owned workload.
        """
        training: tuple[RunAttempt | EvaluationAttempt, ...] = self._attempts_of(experiment_id)
        evaluation: tuple[RunAttempt | EvaluationAttempt, ...] = self._evaluation_attempts_of(
            experiment_id
        )
        return tuple(attempt for attempt in (*training, *evaluation) if not attempt.is_terminal)

    def _live_attempt_targets(self, experiment_id: str) -> tuple[tuple[_AttemptKind, str], ...]:
        """``(kind, id)`` of every live attempt, for addressing its cancel operation."""
        return tuple(
            (kind, attempt_id)
            for kind, attempt_id in self._attempt_targets(experiment_id)
            if not self._cancellable_attempt(kind, attempt_id).is_terminal
        )

    def _attempt_targets(self, experiment_id: str) -> tuple[tuple[_AttemptKind, str], ...]:
        """``(kind, id)`` of every attempt of the experiment, training then evaluation."""
        return (
            *(("training-attempt", str(a.id)) for a in self._attempts_of(experiment_id)),
            *(
                ("evaluation-attempt", str(a.id))
                for a in self._evaluation_attempts_of(experiment_id)
            ),
        )

    def _evaluation_attempts_of(self, experiment_id: str) -> tuple[EvaluationAttempt, ...]:
        """Every evaluation attempt of the experiment, through its nodes and runs."""
        return tuple(
            attempt
            for node in self.aggregates.nodes_for_experiment(experiment_id)
            for run in self.aggregates.evaluation_runs_for_node(str(node.id))
            for attempt in self.aggregates.evaluation_attempts_for_run(str(run.id))
        )

    def _attempts_of(self, experiment_id: str) -> tuple[RunAttempt, ...]:
        """Every attempt of the experiment, through its nodes and runs."""
        return tuple(
            attempt
            for node in self.aggregates.nodes_for_experiment(experiment_id)
            for run in self.aggregates.runs_for_node(str(node.id))
            for attempt in self.aggregates.attempts_for_run(str(run.id))
        )

    # ---- machinery ------------------------------------------------------

    def _record_cancellation(
        self,
        proposed: Action,
        operation_id: OperationId,
        request_digest: str,
        experiment_id: str,
        actor: Actor,
        destinations: tuple[str, ...],
    ) -> tuple[Action, RuntimeOperation | None]:
        """Write one attempt cancellation: the Action and, if live, its effect.

        Runs inside the caller's transaction. Shared by a direct attempt
        cancellation and each child of an experiment saga, so the two cannot
        record the same intent differently.
        """
        target = proposed.target

        # Each lifecycle step is persisted with its own event, rather than
        # folded into one write ending at EXECUTING. Batching is what the
        # aggregate writers refuse everywhere else -- the intervening
        # transitions would have no events -- and the aggregate whose whole
        # purpose is recording intent is the wrong place to make that
        # exception. It all commits in the caller's one transaction, so the
        # atomicity ADR-005 section 5 requires is unchanged.
        self.actions._insert(proposed)
        self._emit_action(proposed, "ActionProposed", actor, destinations)

        # Validation happens before any effect is requested, so an action
        # that cannot be carried out never mints an operation.
        validating = self._advance(
            proposed, ActionStatus.VALIDATING, "ActionValidating", actor, destinations
        )
        validated = self._advance(
            validating, ActionStatus.VALIDATED, "ActionValidated", actor, destinations
        )

        settled_outcome = self._terminal_outcome(target)
        if settled_outcome is not None:
            executing = self._advance(
                validated, ActionStatus.EXECUTING, "ActionExecuting", actor, destinations
            )
            settled = executing.with_status(ActionStatus.SUCCEEDED, outcome=settled_outcome)
            self.actions._update(settled)
            self._emit_action(
                settled,
                f"Cancellation{settled_outcome.value.capitalize()}",
                actor,
                destinations,
            )
            return settled, None

        executing = self._advance(
            validated, ActionStatus.EXECUTING, "ActionExecuting", actor, destinations
        )

        operation = RuntimeOperation(
            id=operation_id,
            target=RuntimeOperationTarget.model_validate({"kind": target.kind, "id": target.id}),
            type="cancel",
            request_digest=request_digest,
            caused_by_action_id=executing.id,
        )
        stored = self.operations._insert(operation)

        self._emit_action(executing, "CancellationRequested", actor, destinations)
        self._emit_operation(stored, "RuntimeOperationIntended", experiment_id, actor, destinations)
        return executing, stored

    def _transition(
        self,
        kind: str,
        aggregate_id: str,
        expected_revision: int,
        new_status: Any,
        loader: Callable[[str], AggregateT | None],
        writer: Callable[[AggregateT], None],
        actor: Actor,
        event_type: str | None,
        destinations: tuple[str, ...],
        *,
        after_write: Callable[[AggregateT], None] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AggregateT:
        """Load, transition and persist an aggregate, with its event.

        *after_write* runs inside the same transaction, for a write that must
        commit with the transition and nowhere else.

        The aggregate is **loaded inside the transaction**, never accepted from
        the caller. A caller-supplied object can carry a real id and revision
        alongside different identity fields -- a `RunAttempt` naming a
        different `run_id`, say -- and the row's indexed columns, its payload
        and the event's experiment would then disagree, each silently. Loading
        it here means the only thing the caller chooses is the transition.

        ``expected_revision`` is what the caller last saw. It is checked before
        the state machine runs, so a lost race fails as contention rather than
        as an invalid transition computed from stale state.

        Raises:
            AggregateNotFoundError: If no such aggregate exists.
            ConcurrentModificationError: If it has moved since the caller read it.
            InvalidTransitionError: If the state machine forbids the edge.
        """
        with write_transaction(self._connection):
            current = loader(aggregate_id)
            if current is None:
                raise AggregateNotFoundError(kind, aggregate_id)
            if current.revision != expected_revision:
                raise ConcurrentModificationError(kind, aggregate_id, expected_revision)

            moved: AggregateT = current.with_status(new_status)
            writer(moved)
            if after_write is not None:
                after_write(moved)
            self._emit(
                moved,
                event_type or f"{kind}StatusChanged",
                self._owning_experiment(moved),
                actor,
                destinations,
                extra=extra,
            )
        return moved

    def _settle(
        self,
        operation_id: OperationId,
        state: Literal["sent", "confirmed", "failed"],
        expected_revision: int,
        actor: Actor,
        *,
        runtime_ref: RuntimeRef | None = None,
        destinations: tuple[str, ...] = (),
    ) -> RuntimeOperation:
        with write_transaction(self._connection):
            current = self.operations.get(str(operation_id))
            if current is None:
                raise AggregateNotFoundError("RuntimeOperation", str(operation_id))
            if current.revision != expected_revision:
                raise ConcurrentModificationError(
                    "RuntimeOperation", str(operation_id), expected_revision
                )

            if state == "confirmed" and current.type == "submit":
                reference = runtime_ref or current.runtime_ref
                if reference is None:
                    raise StorageError(
                        f"submit operation {operation_id} cannot be confirmed "
                        f"without a RuntimeRef: it would leave the unresolved "
                        f"queue, so nothing would look for the workload again, "
                        f"and the record would say one exists without saying "
                        f"where (ADR-013)"
                    )

            moved = current.with_state(state, runtime_ref=runtime_ref)
            experiment_id = self._experiment_of_operation(moved)
            self.operations._update(moved)
            self._emit_operation(
                moved,
                f"RuntimeOperation{state.capitalize()}",
                experiment_id,
                actor,
                destinations,
            )
        return moved

    def _require_consistent_run(self, run: Run) -> None:
        """Refuse a run whose ownership contradicts its node's.

        Both foreign keys pass independently: the experiment exists and the
        node exists. Neither establishes that the node belongs to *that*
        experiment, so a run could sit under one candidate's node while its
        payload and events named another experiment -- the same provenance
        split as a caller-supplied aggregate, arriving through creation.

        The fingerprint is checked for the same reason. A `Run` is a
        realization of its node's candidate, so a run claiming a different
        fingerprint is claiming to realize something the node never proposed.

        Raises:
            AggregateNotFoundError: If the node does not exist.
            StorageError: If the node's experiment or fingerprint disagrees.
        """
        node = self.aggregates.get_node(str(run.node_id))
        if node is None:
            raise AggregateNotFoundError("ExperimentNode", str(run.node_id))

        if node.experiment_id != run.experiment_id:
            raise StorageError(
                f"run {run.id} claims experiment {run.experiment_id} but its "
                f"node {node.id} belongs to {node.experiment_id}: lineage and "
                f"provenance would disagree about which experiment owns it"
            )

        # PR-007 renames both fields to candidate_fingerprint; the rule is the
        # same either way, and pinning it here keeps the rename honest.
        if node.candidate_fingerprint != run.candidate_fingerprint:
            raise StorageError(
                f"run {run.id} carries fingerprint {run.candidate_fingerprint!r} "
                f"but its node proposes {node.candidate_fingerprint!r}: a run "
                f"realizes its node's candidate, not a different one"
            )

    def _experiment_of_action_target(self, target: ActionTarget) -> str:
        """Resolve the experiment an action's target belongs to."""
        if target.kind == "experiment":
            if self.aggregates.get_experiment(target.id) is None:
                raise UnknownOperationTargetError(target.kind, target.id)
            return target.id
        if target.kind == "run":
            return self._experiment_of_run(target.id)
        if target.kind in _ATTEMPT_KINDS:
            return self._experiment_of_attempt(target.kind, target.id)
        # node / evaluation-run: neither has a cancellation type yet.
        raise UnknownOperationTargetError(target.kind, target.id)

    def _replay_cancellation(
        self, candidate: Action, operation_id: OperationId
    ) -> tuple[Action, RuntimeOperation | None] | None:
        """Return the already-recorded result of this exact request, if any.

        Both halves are checked. Matching on ids alone would make a retry that
        carried a different reason, actor or digest return the original
        silently, and the durable record would describe a decision nobody made
        -- the same failure the operation journal's digest check prevents, on
        the other half of the write.

        Raises:
            IdempotencyConflictError: If either half exists against a different
                request, or if the action is paired with a different operation.
            StorageError: If the action exists with no caused operation while
                still in flight. The two are written in one transaction, so
                that is a corrupted record rather than a retry, and completing
                it silently would paper over the corruption.
        """
        action_id = candidate.id
        existing_action = self.actions.get(str(action_id))
        existing_operation = self.operations.get(str(operation_id))

        if existing_action is None and existing_operation is None:
            return None

        if existing_action is None:
            # An operation with no matching action is one of two things. If it
            # names a *different* action, a new action is trying to adopt an
            # effect that already has a cause -- a conflict, not a retry.
            if existing_operation is not None and (
                existing_operation.caused_by_action_id != action_id
            ):
                raise IdempotencyConflictError(str(operation_id), ("caused_by_action_id",))
            raise StorageError(
                f"operation {operation_id} exists but action {action_id} does "
                f"not: they are written in one transaction, so this is a "
                f"corrupted record rather than a retry"
            )

        self.actions._assert_same_request(existing_action, candidate)

        if existing_operation is None:
            # The already-ended path legitimately writes no effect.
            if existing_action.is_terminal:
                return existing_action, None

            caused = self.actions.caused_operation_ids(str(action_id))
            if caused:
                # The action is paired with a different operation id. That is a
                # changed request, not corruption: the caller retried with a
                # new effect id for an intent that already has one.
                raise IdempotencyConflictError(str(action_id), ("operation_id",), kind="action")

            raise StorageError(
                f"action {action_id} is in flight but has no caused operation: "
                f"they are written in one transaction, so this is a corrupted "
                f"record rather than a retry"
            )

        if existing_operation.caused_by_action_id != action_id:
            raise IdempotencyConflictError(str(operation_id), ("caused_by_action_id",))

        return existing_action, existing_operation

    @staticmethod
    def _assert_same_effect_request(existing: RuntimeOperation | None, request_digest: str) -> None:
        """Refuse a replay whose external request changed.

        The action half is compared in :meth:`_replay_cancellation`; this is the
        effect half, which only exists when the target was live.
        """
        if existing is not None and existing.request_digest != request_digest:
            raise IdempotencyConflictError(str(existing.id), ("request_digest",))

    def _terminal_outcome(self, target: ActionTarget) -> ActionOutcome | None:
        """How a cancellation resolves against a target that has already ended.

        ``None`` means the target is still live and a real effect is needed.

        The two terminal cases are different facts and are recorded as such: a
        target already `CANCELLED` was **already in the requested state**, which
        is `NOOP`; a target that ended any other way **overtook** the request,
        which is `SUPERSEDED`. Collapsing both into `SUPERSEDED` would lose the
        distinction the outcome enum was introduced to carry.
        """
        attempt = self._cancellable_attempt(target.kind, target.id)
        if not attempt.is_terminal:
            return None
        if attempt.status.value == "cancelled":
            return ActionOutcome.NOOP
        return ActionOutcome.SUPERSEDED

    def _cancellable_attempt(self, kind: str, attempt_id: str) -> RunAttempt | EvaluationAttempt:
        """The attempt a cancellation targets, of either kind.

        Raises:
            UnknownOperationTargetError: If the kind is not an attempt kind.
            AggregateNotFoundError: If the attempt does not exist.
        """
        if kind == "training-attempt":
            return self.aggregates.load_attempt(attempt_id)
        if kind == "evaluation-attempt":
            return self.aggregates.load_evaluation_attempt(attempt_id)
        raise UnknownOperationTargetError(kind, attempt_id)

    def _observed_outcome(self, action: Action) -> tuple[ActionStatus, ActionOutcome | None] | None:
        """Derive a cancellation's resolution from what is actually recorded.

        ``None`` when the evidence does not yet support a conclusion, so the
        action stays `EXECUTING` rather than being resolved on a guess.

        ```text
        effect CONFIRMED and target CANCELLED   -> SUCCEEDED / APPLIED
        target terminal by another path         -> SUCCEEDED / SUPERSEDED
        effect definitively FAILED, target live -> FAILED, no outcome
        effect INTENDED or SENT                 -> unresolved, wait
        ```

        Returns a status rather than an outcome alone because of the third
        row. A failed effect is not unresolved, so nothing would ever revisit
        it: the action would sit in `EXECUTING` for the life of the record,
        holding durable intent against an effect known not to have happened.
        Retrying is a **new** effect under an explicit policy, not a silent
        resurrection of this one.
        """
        attempt = self._cancellable_attempt(action.target.kind, action.target.id)
        caused = [
            op
            for op in self.operations.for_target(action.target.kind, action.target.id)
            if op.caused_by_action_id == action.id
        ]

        if attempt.is_terminal:
            if attempt.status.value != "cancelled":
                return ActionStatus.SUCCEEDED, ActionOutcome.SUPERSEDED
            # Cancelled -- but only claim credit if our effect was confirmed.
            if any(op.state == "confirmed" for op in caused):
                return ActionStatus.SUCCEEDED, ActionOutcome.APPLIED
            return None

        # The target is still live. A definitively failed effect settles the
        # action; anything unresolved leaves it in flight.
        if caused and all(op.state == "failed" for op in caused):
            return ActionStatus.FAILED, None
        return None

    def _advance(
        self,
        action: Action,
        new_status: ActionStatus,
        event_type: str,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> Action:
        """Move an action one step, persisting the transition with its event.

        *destinations* reaches here too. ADR-005 §3 pairs every event with its
        outbox records, so an event a sink never hears about is a hole in the
        contract rather than a detail -- and the intermediate lifecycle steps
        are exactly the ones most easily forgotten.
        """
        moved = action.with_status(new_status)
        self.actions._update(moved)
        self._emit_action(moved, event_type, actor, destinations)
        return moved

    def _emit_action(
        self,
        action: Action,
        event_type: str,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> DomainEvent:
        event = DomainEvent(
            id=EventId.generate(),
            experiment_id=str(action.experiment_id),
            aggregate_type="Action",
            aggregate_id=str(action.id),
            aggregate_revision=action.revision,
            event_type=event_type,
            actor=actor,
            payload=FrozenDict(
                {
                    "type": action.type,
                    "status": action.status.value,
                    "outcome": action.outcome.value if action.outcome else None,
                    "target_kind": action.target.kind,
                    "target_id": action.target.id,
                }
            ),
        )
        return self._write_event(event, destinations)

    def _owning_experiment(self, aggregate: _Transitionable) -> str:
        """Return the experiment an aggregate belongs to, from the record itself.

        Derived rather than accepted from the caller, so an event cannot be
        filed under an experiment that does not own the thing it describes.
        """
        if isinstance(aggregate, Experiment):
            return str(aggregate.id)
        if isinstance(aggregate, (ExperimentNode, Run)):
            return str(aggregate.experiment_id)
        if isinstance(aggregate, RunAttempt):
            return self._experiment_of_run(str(aggregate.run_id))
        if isinstance(aggregate, EvaluationRun):
            return str(aggregate.experiment_id)
        if isinstance(aggregate, EvaluationAttempt):
            return self._experiment_of_evaluation_run(str(aggregate.evaluation_run_id))
        raise StorageError(f"{type(aggregate).__name__} has no owning experiment")

    def _experiment_of_run(self, run_id: str) -> str:
        """Resolve a run's experiment, which an attempt does not carry."""
        row = self._connection.execute(
            "SELECT experiment_id FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise AggregateNotFoundError("Run", run_id)
        return str(row["experiment_id"])

    def _experiment_of_evaluation_run(self, run_id: str) -> str:
        """Resolve an evaluation run's experiment."""
        row = self._connection.execute(
            "SELECT experiment_id FROM evaluation_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise AggregateNotFoundError("EvaluationRun", run_id)
        return str(row["experiment_id"])

    def _experiment_of_operation(self, operation: RuntimeOperation) -> str:
        """Resolve an operation's experiment through the attempt it targets."""
        return self._experiment_of_attempt(operation.target.kind, operation.target.id)

    def _experiment_of_attempt(self, kind: str, attempt_id: str) -> str:
        """Resolve the experiment of a training or evaluation attempt.

        Raises:
            UnknownOperationTargetError: If no such attempt exists.
        """
        if kind == "training-attempt":
            row = self._connection.execute(
                "SELECT run_id FROM run_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise UnknownOperationTargetError(kind, attempt_id)
            return self._experiment_of_run(str(row["run_id"]))
        if kind == "evaluation-attempt":
            row = self._connection.execute(
                "SELECT evaluation_run_id FROM evaluation_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise UnknownOperationTargetError(kind, attempt_id)
            return self._experiment_of_evaluation_run(str(row["evaluation_run_id"]))
        raise UnknownOperationTargetError(kind, attempt_id)

    def _require_target(self, target: RuntimeOperationTarget) -> None:
        """Enforce ADR-005 §10.1, which SQLite cannot."""
        table = _TARGET_TABLES[target.kind]
        try:
            row = self._connection.execute(
                f"SELECT 1 FROM {table} WHERE id = ?",  # noqa: S608 - table from a literal map
                (target.id,),
            ).fetchone()
        except sqlite3.OperationalError as error:
            # The table is not in the schema yet (evaluation_attempts, ADR-015).
            raise UnknownOperationTargetError(target.kind, target.id) from error
        if row is None:
            raise UnknownOperationTargetError(target.kind, target.id)

    def _emit(
        self,
        aggregate: _Transitionable,
        event_type: str,
        experiment_id: str,
        actor: Actor,
        destinations: tuple[str, ...] = (),
        *,
        extra: dict[str, Any] | None = None,
    ) -> DomainEvent:
        event = DomainEvent(
            id=EventId.generate(),
            experiment_id=experiment_id,
            aggregate_type=type(aggregate).__name__,
            aggregate_id=str(aggregate.id),  # type: ignore[attr-defined]
            aggregate_revision=aggregate.revision,  # type: ignore[attr-defined]
            event_type=event_type,
            actor=actor,
            payload=FrozenDict(
                {
                    "status": getattr(getattr(aggregate, "status", None), "value", None),
                    **(extra or {}),
                }
            ),
        )
        return self._write_event(event, destinations)

    def _emit_operation(
        self,
        operation: RuntimeOperation,
        event_type: str,
        experiment_id: str,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> DomainEvent:
        event = DomainEvent(
            id=EventId.generate(),
            experiment_id=experiment_id,
            aggregate_type="RuntimeOperation",
            aggregate_id=str(operation.id),
            aggregate_revision=operation.revision,
            event_type=event_type,
            actor=actor,
            payload=FrozenDict(
                {
                    "state": operation.state,
                    "type": operation.type,
                    "target_kind": operation.target.kind,
                    "target_id": operation.target.id,
                }
            ),
        )
        return self._write_event(event, destinations)

    def _write_event(self, event: DomainEvent, destinations: tuple[str, ...]) -> DomainEvent:
        self.events._append(event)
        now = utc_now()
        for destination in destinations:
            self.events._enqueue(
                OutboxRecord(
                    id=f"obx_{uuid.uuid4().hex}",
                    event_id=event.id,
                    destination=destination,
                    created_at=now,
                    updated_at=now,
                )
            )
        return event


def _cancel_digest(kind: str, target_id: str) -> str:
    """The request digest of a saga's cancel operation.

    Derived from the target alone: cancelling an attempt is the same request
    however many times it is made, so a retry's digest matches.
    """
    return fingerprint({"type": "cancel", "target": {"kind": kind, "id": target_id}})
