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
```

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
from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeVar

from xaytune.core.clock import utc_now
from xaytune.core.domain.action import (
    Action,
    ActionOutcome,
    ActionStatus,
    ActionTarget,
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
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)
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

__all__ = ["ControlPlaneRepository", "ProvenanceError", "UnknownOperationTargetError"]


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
}

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
    # ADR-015's tables arrive with the evaluation lifecycle; until then an
    # evaluation target has nothing to resolve against and is refused rather
    # than accepted unchecked.
    "evaluation-attempt": "evaluation_attempts",
}


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


def _assert_same_attempt(existing: RunAttempt, requested: RunAttempt) -> None:
    """Refuse a replay whose attempt identity differs.

    Raises:
        IdempotencyConflictError: Naming each differing field.
    """
    differing = tuple(
        field
        for field in ("id", "run_id", "attempt_number")
        if getattr(existing, field) != getattr(requested, field)
    )
    if differing:
        raise IdempotencyConflictError(str(existing.id), differing, kind="attempt")


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
        operation = RuntimeOperation(
            id=operation_id or OperationId.generate(),
            target=RuntimeOperationTarget(kind="training-attempt", id=str(attempt.id)),
            type="submit",
            request_digest=request_digest,
        )

        _require_pristine(attempt, RunAttemptStatus.CREATED)

        with write_transaction(self._connection):
            existing = self.operations.get(str(operation.id))
            if existing is not None:
                self.operations._assert_same_request(existing, operation)
                stored_attempt = self.aggregates.get_attempt(existing.target.id)
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

            experiment_id = self._experiment_of_run(str(attempt.run_id))
            self.aggregates._insert_attempt(attempt)
            stored = self.operations._insert(operation)
            self._emit(attempt, "RunAttemptCreated", experiment_id, actor, destinations)
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
                        target=ActionTarget(kind="training-attempt", id=str(attempt.id)),
                        proposed_by=actor,
                        reason=reason,
                        parent_action_id=executing.id,
                    ),
                    OperationId.generate(),
                    _cancel_digest("training-attempt", str(attempt.id)),
                    str(experiment.id),
                    actor,
                    destinations,
                )
                for attempt in self._live_attempts(str(experiment.id))
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
            for attempt in self._attempts_of(experiment_id)
            for operation in self.operations.for_target("training-attempt", str(attempt.id))
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

    def _live_attempts(self, experiment_id: str) -> tuple[RunAttempt, ...]:
        """Every attempt of the experiment that has not reached a terminal state."""
        return tuple(a for a in self._attempts_of(experiment_id) if not a.is_terminal)

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
        if target.kind == "training-attempt":
            attempt = self.aggregates.get_attempt(target.id)
            if attempt is None:
                raise UnknownOperationTargetError(target.kind, target.id)
            return self._experiment_of_run(str(attempt.run_id))
        # node / evaluation-run / evaluation-attempt: the first has no
        # cancellation type yet, the others wait for ADR-015's tables.
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
        if target.kind != "training-attempt":
            raise UnknownOperationTargetError(target.kind, target.id)

        attempt = self.aggregates.load_attempt(target.id)
        if not attempt.is_terminal:
            return None
        if attempt.status is RunAttemptStatus.CANCELLED:
            return ActionOutcome.NOOP
        return ActionOutcome.SUPERSEDED

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
        if action.target.kind != "training-attempt":
            raise UnknownOperationTargetError(action.target.kind, action.target.id)

        caused = [
            op
            for op in self.operations.for_target(action.target.kind, action.target.id)
            if op.caused_by_action_id == action.id
        ]
        attempt = self.aggregates.load_attempt(action.target.id)

        if attempt.is_terminal:
            if attempt.status is not RunAttemptStatus.CANCELLED:
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
        raise StorageError(f"{type(aggregate).__name__} has no owning experiment")

    def _experiment_of_run(self, run_id: str) -> str:
        """Resolve a run's experiment, which an attempt does not carry."""
        row = self._connection.execute(
            "SELECT experiment_id FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise AggregateNotFoundError("Run", run_id)
        return str(row["experiment_id"])

    def _experiment_of_operation(self, operation: RuntimeOperation) -> str:
        """Resolve an operation's experiment through the attempt it targets."""
        table = _TARGET_TABLES.get(operation.target.kind)
        if table != "run_attempts":
            raise UnknownOperationTargetError(operation.target.kind, operation.target.id)
        row = self._connection.execute(
            "SELECT run_id FROM run_attempts WHERE id = ?", (operation.target.id,)
        ).fetchone()
        if row is None:
            raise UnknownOperationTargetError(operation.target.kind, operation.target.id)
        return self._experiment_of_run(str(row["run_id"]))

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
