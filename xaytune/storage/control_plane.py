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

import math
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Literal, Protocol, TypeVar

from xaytune.core.capabilities import CapabilityDocument
from xaytune.core.checkpoint import RecordedCheckpoint, checkpoint_state_refs
from xaytune.core.clock import utc_now
from xaytune.core.domain.action import (
    CANCELLATION_TYPES,
    Action,
    ActionOutcome,
    ActionStatus,
    ActionTarget,
)
from xaytune.core.domain.actions import (
    ActionSpec,
    ChangeLearningRate,
    MutationClass,
    action_descriptor,
    action_from_spec,
    encode_payload,
    spec_of,
)
from xaytune.core.domain.agent_invocation import (
    AgentInvocation,
    AgentInvocationIntent,
    next_invocation,
)
from xaytune.core.domain.budget import (
    BudgetDimension,
    BudgetExhaustedError,
    BudgetLedgerEntry,
    BudgetStatus,
    BudgetSubjectKind,
    CapacityUnavailableError,
    LedgerEntryKind,
    budget_status,
    limits,
)
from xaytune.core.domain.controller_request import ControllerRequest, ControllerRequestState
from xaytune.core.domain.decision import (
    Decision,
    DecisionContext,
    DecisionOutcome,
    DecisionProposal,
)
from xaytune.core.domain.evaluation import (
    EvaluationAttempt,
    EvaluationResult,
    EvaluationRun,
    result_provenance_problems,
)
from xaytune.core.domain.event import DomainEvent, OutboxRecord
from xaytune.core.domain.experiment import CandidateSpecSnapshot, Experiment, ExperimentNode
from xaytune.core.domain.incident import AttemptContext, Incident
from xaytune.core.domain.intervention import (
    IncidentTrigger,
    InterventionApplication,
    InterventionDirective,
    InterventionDirectiveKind,
    InterventionOrigin,
    TrainingIntervention,
    TrainingPosition,
    TriggerEvaluation,
    mutation_for_action,
)
from xaytune.core.domain.intervention_replay import ReplayPlan, plan_intervention_directives
from xaytune.core.domain.numerical_recovery import (
    EffectiveLearningRate,
    NumericalLRProposal,
    NumericalRecoveryActionBinding,
    NumericalRecoveryInputsV1,
    NumericalRecoveryPolicyV1,
    PriorNumericalIntervention,
)
from xaytune.core.domain.oom_recovery import OOMRecoveryInputsV1, OOMResizeProposal
from xaytune.core.domain.operation import (
    RuntimeOperation,
    RuntimeOperationTarget,
)
from xaytune.core.domain.planning import (
    PLANNER_SPEC_IDENTITY_VERSION,
    PLANNING_CONTEXT_IDENTITY_VERSION,
    CandidateBranchOrigin,
    CandidateProposal,
    DecisionSummary,
    EvaluationSummary,
    MetricSummary,
    NodeSummary,
    PlanningContext,
    ProposalProvenance,
    planner_spec_identity_v1,
)
from xaytune.core.domain.policy import (
    GovernedAction,
    PolicyContext,
    PolicyDecision,
    PolicyProposer,
    PolicyVerdict,
    applicability_problems,
    awaits_execution,
)
from xaytune.core.domain.realization import (
    AttemptAncestry,
    CheckpointAncestry,
    RunRealization,
    project_run_realization,
    retained_trajectory,
)
from xaytune.core.domain.recovery import (
    COORDINATOR_NAME,
    COORDINATOR_VERSION,
    RecoveryCheckpointReport,
    RecoveryEpisode,
    RecoveryEvidence,
    RecoveryEvidenceDisposition,
    RecoveryInputsV1,
    RecoveryPlan,
    RecoveryPredecessor,
    RecoveryRepeatCount,
    RecoveryRequest,
    checkpoint_order,
    checkpoint_report_problem,
    decide_recovery,
    execution_state_fingerprint_v1,
    recovery_requires_checkpoints,
)
from xaytune.core.domain.recovery_action import RecoveryActionBinding
from xaytune.core.domain.recovery_execution import (
    RecoveryExecutionOutcome,
    RecoveryExecutionReceipt,
)
from xaytune.core.domain.run import Run, RunAttempt
from xaytune.core.errors import ConcurrentModificationError
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import (
    ActionId,
    ControllerRequestId,
    EvaluationAttemptId,
    EvaluationRunId,
    EventId,
    ExperimentId,
    ExperimentNodeId,
    InterventionApplicationId,
    InterventionId,
    OperationId,
    RecoveryEpisodeId,
    RunAttemptId,
    RunId,
)
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor, ArtifactRef, CheckpointRef, RuntimeRef
from xaytune.core.state.machines import EXPERIMENT_MACHINE
from xaytune.core.state.status import (
    EvaluationAttemptStatus,
    EvaluationRunStatus,
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)
from xaytune.core.telemetry import CheckpointCommittedPayload, EvaluationCompletedPayload
from xaytune.storage.actions import ActionStore
from xaytune.storage.agent_invocations import (
    AgentInvocationStore,
    RepositoryAgentInvocationJournal,
)
from xaytune.storage.budget import BudgetLedgerStore
from xaytune.storage.checkpoints import CheckpointRecordStore
from xaytune.storage.database import write_transaction
from xaytune.storage.errors import AggregateNotFoundError, StorageError
from xaytune.storage.graph import ExperimentGraph
from xaytune.storage.incidents import IncidentStore
from xaytune.storage.interventions import (
    InterventionApplicationStore,
    InterventionDirectiveStore,
    NumericalRecoveryActionBindingStore,
    NumericalRecoveryExecutionStore,
    TrainingInterventionStore,
)
from xaytune.storage.journal import (
    EventJournal,
    IdempotencyConflictError,
    OperationJournal,
)
from xaytune.storage.leases import WriteFence
from xaytune.storage.policy import PolicyDecisionStore
from xaytune.storage.recovery import RecoveryEpisodeStore, RecoveryPlanStore
from xaytune.storage.recovery_actions import RecoveryActionBindingStore
from xaytune.storage.recovery_execution import RecoveryExecutionReceiptStore
from xaytune.storage.repository import AggregateStore
from xaytune.storage.requests import ControllerRequestStore, ControllerRest

__all__ = [
    "AdmissionRefusedError",
    "ApprovalConflictError",
    "ApprovalError",
    "BranchRefusedError",
    "CandidateConflictError",
    "CancellationNotGovernedError",
    "ControlPlaneRepository",
    "DecisionConflictError",
    "ExperimentAdmission",
    "EvaluationReconciliation",
    "InterventionNotAuthorizedError",
    "ProvenanceError",
    "StalePolicyContextError",
    "StaleProposalError",
    "UnknownOperationTargetError",
]


_IN_FLIGHT = frozenset({ActionStatus.PROPOSED, ActionStatus.VALIDATING, ActionStatus.EXECUTING})
"""Action states a controller is working through, as opposed to resting in."""

_GOVERNANCE = Actor(type="system", id="governance")
"""Who validates a proposed action: the control plane itself, by its built-in rules."""

_VERDICT_STATUS: dict[PolicyVerdict, ActionStatus] = {
    PolicyVerdict.ALLOW: ActionStatus.VALIDATED,
    PolicyVerdict.DENY: ActionStatus.REJECTED,
    PolicyVerdict.REQUIRE_APPROVAL: ActionStatus.APPROVAL_PENDING,
}
"""Where each verdict leaves a validated action. ``ALLOW`` moves nothing: it is the decision."""

_VERDICT_EVENT: dict[PolicyVerdict, str] = {
    PolicyVerdict.ALLOW: "ActionAuthorized",
    PolicyVerdict.DENY: "ActionRejected",
    PolicyVerdict.REQUIRE_APPROVAL: "ActionApprovalPending",
}


def _approval_identity(event_type: str, answer: dict[str, Any]) -> tuple[str, str, str, str]:
    """What makes two human answers the same: the answer, who (type and id), and why.

    The approver's metadata is provenance, kept on the event; it is not who
    answered, so a replay differing only in metadata is the same answer.
    """
    approver = answer["approver"]
    return event_type, approver["type"], approver["id"], answer["reason"]


def _plain(value: Any) -> Any:
    """A frozen event payload as plain dicts and lists, for comparison."""
    if isinstance(value, (dict, FrozenDict)):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


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


class AdmissionRefusedError(StorageError):
    """A submission that cannot be admitted as given, however often it is retried.

    The request it names is for another experiment or another kind, its
    payload is not the spec the submission was derived from, or the
    experiment id is already taken (ADR-004 §4).
    """


class BranchRefusedError(StorageError):
    """A candidate proposal cannot be materialized as a node; nothing was written."""


class StaleProposalError(BranchRefusedError):
    """The experiment changed since the proposal was planned. Plan again from a fresh context."""


class CandidateConflictError(BranchRefusedError):
    """The experiment already has this candidate, from another proposal or none."""


class InterventionNotAuthorizedError(StorageError):
    """A TrainingIntervention was requested for an Action governance has not authorized."""


class StaleRecoveryContextError(StorageError):
    """Planning inputs changed before the decision could be recorded."""


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


class CancellationNotGovernedError(StorageError):
    """A cancellation was proposed through the governed path.

    Cancellation is controller-owned and always possible (ADR-013). It
    records its intent together with the cancel operation, through the
    cancellation API; a ``cancel-*`` Action recorded here would be intent
    that nothing carries out.
    """

    def __init__(self, action_type: str) -> None:
        self.action_type = action_type
        super().__init__(
            f"{action_type} is not proposed to policy: cancel through the cancellation API "
            f"(ExperimentHandle.cancel(), request_cancellation(), "
            f"request_experiment_cancellation())"
        )


class CancellationInFlightError(StorageError):
    """A cancellation with its own id was asked for while another is in flight.

    Without an id, a second request joins the cancellation in flight. With
    one -- a daemon request's, minted by its client (PR-029) -- the request
    names the Action it records, and is never carried out through a
    different one: it is refused, and nothing is written.
    """

    def __init__(self, action_id: str, in_flight: str) -> None:
        self.action_id = action_id
        self.in_flight = in_flight
        super().__init__(
            f"cancellation {action_id} was not recorded: cancellation {in_flight} "
            f"of the same experiment is already in flight"
        )


class StalePolicyContextError(StorageError):
    """The state changed between the policy's evaluation and its recording.

    A decision authorizes the snapshot it judged, so it is not recorded
    against a different one. Nothing was written; evaluate again.
    """


class ApprovalError(StorageError):
    """An approval that cannot be given: by a non-human, or of an action not awaiting one."""


class ApprovalConflictError(StorageError):
    """An action already approved or rejected is being resolved differently.

    The same human giving the same answer with the same reason is recognised
    and returns the action as it stands. Anything else would rewrite who
    decided, and is refused.
    """


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


def _require_oom_successor_lineage(
    source: RunAttempt,
    successor: RunAttempt,
    proposal: OOMResizeProposal,
    action_id: ActionId,
    checkpoint: RecoveryCheckpointReport,
) -> None:
    """Bind one governed resize to the successor's complete operational lineage."""
    prior = source.execution_overrides
    if prior and prior[-1].kind == "checkpoint_restore":
        prior = prior[:-1]
    overrides = successor.execution_overrides
    if (
        successor.run_id != source.run_id
        or successor.attempt_number != source.attempt_number + 1
        or successor.checkpoint_ref != checkpoint.checkpoint_ref
        or successor.execution_fingerprint is None
        or successor.execution_fingerprint == source.execution_fingerprint
        or len(overrides) != len(prior) + 3
        or overrides[: len(prior)] != prior
    ):
        raise ProvenanceError("OOM successor does not preserve the source attempt lineage")
    micro, accumulation, restore = overrides[-3:]
    if (
        micro.kind != "micro_batch_resize"
        or micro.values
        != FrozenDict(
            {"from": proposal.old_micro_batch_size, "to": proposal.action_spec.micro_batch_size}
        )
        or micro.preserves != ("effective_batch_size",)
        or accumulation.kind != "gradient_accumulation_adjustment"
        or accumulation.values
        != FrozenDict(
            {
                "from": proposal.old_gradient_accumulation,
                "to": proposal.action_spec.gradient_accumulation,
            }
        )
        or accumulation.preserves != ("effective_batch_size",)
        or restore.kind != "checkpoint_restore"
        or restore.values != FrozenDict({"checkpoint_id": str(checkpoint.checkpoint_ref.id)})
        or any(item.action_id != action_id for item in (micro, accumulation, restore))
    ):
        raise ProvenanceError("OOM successor overrides disagree with the governed Action")


def _require_numerical_successor_lineage(
    source: RunAttempt,
    successor: RunAttempt,
    action_id: ActionId,
    checkpoint: RecoveryCheckpointReport,
) -> None:
    """Bind a numerical successor to its source's operational lineage plus one restore.

    No operational override carries the learning rate: that is the
    intervention's directive, never an ``ExecutionOverride``.
    """
    prior = source.execution_overrides
    if prior and prior[-1].kind == "checkpoint_restore":
        prior = prior[:-1]
    overrides = successor.execution_overrides
    if (
        successor.run_id != source.run_id
        or successor.attempt_number != source.attempt_number + 1
        or successor.checkpoint_ref != checkpoint.checkpoint_ref
        or successor.execution_fingerprint is None
        or len(overrides) != len(prior) + 1
        or overrides[: len(prior)] != prior
    ):
        raise ProvenanceError("numerical successor does not preserve the source attempt lineage")
    restore = overrides[-1]
    if (
        restore.kind != "checkpoint_restore"
        or restore.values != FrozenDict({"checkpoint_id": str(checkpoint.checkpoint_ref.id)})
        or restore.action_id != action_id
    ):
        raise ProvenanceError("numerical successor restore disagrees with the governed Action")


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


@dataclass(frozen=True)
class ExperimentAdmission:
    """What :meth:`ControlPlaneRepository.admit_experiment` committed, in one transaction.

    ``run``, ``attempt`` and ``operation`` are ``None`` when the budget had
    nothing left for the first run: the experiment was admitted and ended
    ``BUDGET_EXHAUSTED`` with no effect, for ``exhausted`` reasons.
    """

    experiment: Experiment
    node: ExperimentNode
    run: Run | None
    attempt: RunAttempt | None
    operation: RuntimeOperation | None
    exhausted: tuple[str, ...] = ()


@dataclass(frozen=True)
class _BoundRecoveryProposal:
    """A recovery decision's Action binding, checked and written with the Action.

    ``replay`` returns the Action already bound to the same decision;
    ``require_current`` re-verifies the decision under the write lock;
    ``insert`` writes the binding in the Action's transaction.
    """

    action_id: ActionId
    spec: ActionSpec
    replay: Callable[[], GovernedAction | None]
    require_current: Callable[[], None]
    insert: Callable[[], None]


class ControlPlaneRepository:
    """Atomic writes over the control-plane aggregates and their journals.

    Args:
        connection: The state database.
        fence: What every write transaction proves, inside it and before its
            first mutation (ADR-004 §8). A daemon's controller passes its
            :class:`~xaytune.storage.leases.ControllerLeaseFence`; an embedded
            host a :class:`~xaytune.storage.leases.NoLiveLeaseFence`. ``None``
            -- a mailbox client, a test -- writes unfenced.
    """

    def __init__(self, connection: sqlite3.Connection, *, fence: WriteFence | None = None) -> None:
        self._connection = connection
        self._fence = fence
        self.aggregates = AggregateStore(connection)
        self.events = EventJournal(connection)
        self.operations = OperationJournal(connection)
        self.actions = ActionStore(connection)
        self.budget = BudgetLedgerStore(connection)
        self.policy = PolicyDecisionStore(connection)
        self.incidents = IncidentStore(connection)
        self.checkpoints = CheckpointRecordStore(connection)
        self.recovery_episodes = RecoveryEpisodeStore(connection)
        self.recovery_plans = RecoveryPlanStore(connection)
        self.recovery_action_bindings = RecoveryActionBindingStore(connection)
        self.recovery_execution_receipts = RecoveryExecutionReceiptStore(connection)
        self.numerical_recovery_bindings = NumericalRecoveryActionBindingStore(connection)
        self.training_interventions = TrainingInterventionStore(connection)
        self.intervention_applications = InterventionApplicationStore(connection)
        self.intervention_directives = InterventionDirectiveStore(connection)
        self.numerical_recovery_executions = NumericalRecoveryExecutionStore(connection)
        self.graph = ExperimentGraph(connection)
        self.controller_requests = ControllerRequestStore(connection)
        self.agent_invocations = AgentInvocationStore(connection)

    @contextmanager
    def _write(self) -> Iterator[None]:
        """One write transaction, fenced: the only way this repository writes.

        The fence runs after ``BEGIN IMMEDIATE`` has taken SQLite's write lock
        and before anything is written, so ownership cannot change between
        the check and the commit: the transaction serializes wholly before a
        takeover or wholly after it, and a refused one writes nothing.

        Raises:
            LeaseLostError: Through a daemon controller's fence.
            ControllerLeaseHeldError: Through an embedded host's fence.
        """
        with write_transaction(self._connection):
            if self._fence is not None:
                self._fence.check(self._connection)
            yield

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

        with self._write():
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

        A ``branch_origin`` is refused. It is repository-issued provenance --
        proof that a proposal passed branching's admission, which the adaptive
        loop trusts enough to run the node (PR-026) -- so only
        :meth:`materialize_candidate_proposal` mints it.

        Raises:
            LineageError: If the node's parents would make the graph unsound.
            ProvenanceError: If the node carries a ``branch_origin``.
        """
        _require_pristine(node, ExperimentNodeStatus.CREATED)

        _require_consistent_candidate(node)
        if node.branch_origin is not None:
            raise ProvenanceError(
                f"node {node.id} carries a branch_origin; that is repository-issued "
                f"provenance, minted only by materialize_candidate_proposal()"
            )

        with self._write():
            self.graph.validate_parents(node)
            self.aggregates._insert_node(node)
            self._emit(node, "NodeCreated", str(node.experiment_id), actor, destinations)
        return node

    def create_run(self, run: Run, *, actor: Actor, destinations: tuple[str, ...] = ()) -> Run:
        """Create a run, its creation event and any outbox records.

        With a ``max_runs`` budget, the run's reservation is written in the
        same commit; a budget with no run left refuses it before anything is
        written.

        Raises:
            BudgetExhaustedError: If a quota the run needs is used up.
        """
        _require_pristine(run, RunStatus.CREATED)

        with self._write():
            self._insert_run_with_reservation(run, actor, destinations)
        return run

    def _insert_run_with_reservation(
        self, run: Run, actor: Actor, destinations: tuple[str, ...]
    ) -> None:
        """:meth:`create_run`'s work, inside a transaction the caller already holds."""
        self._require_consistent_run(run)
        self._require_budget(str(run.experiment_id), new_run=True)
        self.aggregates._insert_run(run)
        self._emit(run, "RunCreated", str(run.experiment_id), actor, destinations)
        self._ledger(
            str(run.experiment_id),
            BudgetDimension.RUNS,
            LedgerEntryKind.RESERVE,
            Decimal(1),
            BudgetSubjectKind.RUN,
            str(run.id),
            actor,
            destinations,
        )

    # ---- initial admission and the daemon mailbox (ADR-004 §3-§4) ------------

    def admit_experiment(
        self,
        experiment: Experiment,
        node: ExperimentNode,
        run: Run,
        attempt: RunAttempt,
        *,
        request_digest: str,
        actor: Actor,
        request_id: ControllerRequestId | str | None = None,
        submitted_digest: str | None = None,
        destinations: tuple[str, ...] = (),
    ) -> ExperimentAdmission | None:
        """Admit a submitted experiment and the intent to start it, in one commit (ADR-004 §4).

        Validation, binding and compilation happen before this; what it
        records is everything a restart needs to carry the submission on:

        ```text
        the request, if any, is a submit for this
          experiment, of exactly this spec          else AdmissionRefusedError
        it is still PENDING                         else None: it moved on
        experiment CREATED → ACTIVE
        root node CREATED → PLANNED → READY → ACTIVE
        its first run CREATED → ACTIVE, max_runs reserved
        attempt 1, parallel-run slot, submit INTENDED
        the request PENDING → ACCEPTED
        ```

        A budget with no run left for the first run admits the experiment and
        ends it ``BUDGET_EXHAUSTED`` in the same commit, with no run and no
        effect. The runtime is called only after this commits; a crash after
        it leaves an ``INTENDED`` operation, which reconciliation resolves
        (ADR-013), and a crash before it leaves nothing but the request.

        *submitted_digest* is the canonical digest of the ``ExperimentSpec``
        the aggregates were derived from
        (:meth:`~xaytune.experiment.ExperimentSpec.submission_payload`), and is
        required with *request_id*: the request is accepted only for the spec
        it carries, so a request can never be marked ``ACCEPTED`` for an
        experiment derived from anything else.

        Returns:
            What was admitted -- or ``None`` if the request is no longer
            ``PENDING``: it was admitted, or failed, before; nothing is written.

        Raises:
            AggregateNotFoundError: If *request_id* names no request.
            AdmissionRefusedError: If the request is another kind, for another
                experiment or of another spec, or the experiment already
                exists. Nothing is written.
            StorageError: If the aggregates are not one fresh submission -- a
                root node of this experiment, a run of it with no seed origin,
                that run's first attempt.
            ValueError: If *request_id* is given without *submitted_digest*.
        """
        if request_id is not None and submitted_digest is None:
            raise ValueError("admitting a controller request needs the submitted spec's digest")
        _require_pristine(experiment, ExperimentStatus.CREATED)
        _require_pristine(node, ExperimentNodeStatus.CREATED)
        _require_pristine(run, RunStatus.CREATED)
        _require_pristine(attempt, RunAttemptStatus.CREATED)
        _require_consistent_candidate(node)
        problems = [
            problem
            for problem, broken in (
                ("the node is not the experiment's", node.experiment_id != experiment.id),
                ("the node is not a root", bool(node.parent_ids)),
                ("the node carries a branch_origin", node.branch_origin is not None),
                (
                    "the run is not the node's",
                    run.node_id != node.id or run.experiment_id != experiment.id,
                ),
                ("the run claims a seed origin", run.seed_origin is not None),
                (
                    "the attempt is not the run's first",
                    attempt.run_id != run.id or attempt.attempt_number != 1,
                ),
            )
            if broken
        ]
        if problems:
            raise StorageError(f"not one fresh submission: {'; '.join(problems)}")

        aggregates = self.aggregates
        with self._write():
            request = None
            if request_id is not None:
                request = self.controller_requests.get(str(request_id))
                if request is None:
                    raise AggregateNotFoundError("ControllerRequest", str(request_id))
                if request.kind != "submit" or request.experiment_id != experiment.id:
                    raise AdmissionRefusedError(
                        f"controller request {request.id} is a {request.kind} for experiment "
                        f"{request.experiment_id}, not the submission of {experiment.id}"
                    )
                if request.payload_digest != submitted_digest:
                    raise AdmissionRefusedError(
                        f"controller request {request.id} carries spec {request.payload_digest}; "
                        f"the submission was derived from {submitted_digest}"
                    )
                if request.state is not ControllerRequestState.PENDING:
                    return None
            if aggregates.get_experiment(str(experiment.id)) is not None:
                raise AdmissionRefusedError(f"experiment {experiment.id} is already recorded")

            aggregates._insert_experiment(experiment)
            self._emit(experiment, "ExperimentCreated", str(experiment.id), actor, destinations)
            experiment = self._apply_transition(
                "Experiment",
                str(experiment.id),
                experiment.revision,
                ExperimentStatus.ACTIVE,
                aggregates.get_experiment,
                aggregates._update_experiment,
                actor,
                None,
                destinations,
            )
            self.graph.validate_parents(node)
            aggregates._insert_node(node)
            self._emit(node, "NodeCreated", str(experiment.id), actor, destinations)
            for status in (
                ExperimentNodeStatus.PLANNED,
                ExperimentNodeStatus.READY,
                ExperimentNodeStatus.ACTIVE,
            ):
                node = self._apply_transition(
                    "ExperimentNode",
                    str(node.id),
                    node.revision,
                    status,
                    aggregates.get_node,
                    aggregates._update_node,
                    actor,
                    None,
                    destinations,
                )

            admission: ExperimentAdmission
            try:
                self._require_budget(str(experiment.id), new_run=True)
            except BudgetExhaustedError as exhausted:
                experiment = self._exhaust(experiment, exhausted.reasons, actor, destinations)
                admission = ExperimentAdmission(
                    experiment, node, None, None, None, exhausted=exhausted.reasons
                )
            else:
                operation = RuntimeOperation(
                    id=OperationId.generate(),
                    target=RuntimeOperationTarget(kind="training-attempt", id=str(attempt.id)),
                    type="submit",
                    request_digest=request_digest,
                )
                self._insert_run_with_reservation(run, actor, destinations)
                active = self._apply_transition(
                    "Run",
                    str(run.id),
                    run.revision,
                    RunStatus.ACTIVE,
                    aggregates.get_run,
                    aggregates._update_run,
                    actor,
                    None,
                    destinations,
                )
                stored = self._insert_training_attempt_with_intent(
                    attempt, operation, actor, destinations
                )
                admission = ExperimentAdmission(experiment, node, active, attempt, stored)

            if request is not None:
                self.controller_requests._update(
                    request.with_state(ControllerRequestState.ACCEPTED)
                )
        return admission

    def record_controller_request(self, request: ControllerRequest) -> ControllerRequest:
        """Commit a client's request to the daemon's mailbox: get-or-create by its id.

        The client's write. Committed, the request is handed off: the client
        may exit, and the daemon finds it.

        Returns:
            The recorded request -- the one already there, in whatever state it
            has reached, if this id was recorded before with the same request.

        Raises:
            IdempotencyConflictError: If the id was recorded with a different
                kind, experiment or payload.
            ValueError: If *request* is not a new ``PENDING`` request.
        """
        if request.state is not ControllerRequestState.PENDING or request.revision != 0:
            raise ValueError("a controller request is recorded PENDING, at revision 0")
        with self._write():
            existing = self.controller_requests.get(str(request.id))
            if existing is not None:
                differing = existing.same_request(request)
                if differing:
                    raise IdempotencyConflictError(
                        str(request.id), differing, kind="controller request"
                    )
                return existing
            self.controller_requests._insert(request)
        return request

    def complete_controller_request(
        self, request_id: ControllerRequestId | str, *, expected_revision: int
    ) -> ControllerRequest:
        """Mark a request's handoff done: submit()/attach() would have returned.

        Raises:
            AggregateNotFoundError: If no such request exists.
            ConcurrentModificationError: If it moved since the caller read it.
            InvalidTransitionError: If its kind does not allow the edge.
        """
        return self._move_request(
            request_id, expected_revision, ControllerRequestState.COMPLETED, None
        )

    def fail_controller_request(
        self,
        request_id: ControllerRequestId | str,
        *,
        expected_revision: int,
        error: FrozenDict,
    ) -> ControllerRequest:
        """Record a definitive failure before anything was admitted.

        Only from ``PENDING``: an ``ACCEPTED`` submission has admitted an
        experiment whose outcome is the operation journal's to settle, never
        this request's (ADR-004 §3).
        """
        return self._move_request(
            request_id, expected_revision, ControllerRequestState.FAILED, error
        )

    def _move_request(
        self,
        request_id: ControllerRequestId | str,
        expected_revision: int,
        new_state: ControllerRequestState,
        error: FrozenDict | None,
    ) -> ControllerRequest:
        with self._write():
            request = self.controller_requests.get(str(request_id))
            if request is None:
                raise AggregateNotFoundError("ControllerRequest", str(request_id))
            if request.revision != expected_revision:
                raise ConcurrentModificationError(
                    "ControllerRequest", str(request_id), expected_revision
                )
            moved = request.with_state(new_state, error=error)
            self.controller_requests._update(moved)
        return moved

    def record_controller_rest(
        self,
        experiment_id: ExperimentId | str,
        *,
        controller_id: str,
        escalation: dict[str, Any] | None = None,
    ) -> ControllerRest:
        """Record that the controller has nothing left it can do for the experiment (PR-029).

        Stamped with the experiment's latest event, read in the same
        transaction: the experiment is at rest for as long as that stays its
        latest event. Rewrites nothing when the rest is the one already
        recorded.
        """
        with self._write():
            sequence = self.events.latest_sequence_for_experiment(str(experiment_id))
            current = self.controller_requests.rest(str(experiment_id))
            if current is not None and (
                current.controller_id,
                current.sequence,
                current.escalation,
            ) == (controller_id, sequence, escalation):
                return current
            rest = ControllerRest(
                experiment_id=ExperimentId(str(experiment_id)),
                controller_id=controller_id,
                sequence=sequence,
                escalation=escalation,
                recorded_at=utc_now(),
            )
            self.controller_requests._put_rest(rest)
        return rest

    # ---- agent invocations (PR-032) ---------------------------------------

    def agent_invocation_journal(self) -> RepositoryAgentInvocationJournal:
        """This record, as the journal a model-backed planner records its invocations in."""
        return RepositoryAgentInvocationJournal(self)

    def begin_agent_invocation(self, intent: AgentInvocationIntent) -> AgentInvocation:
        """The invocation a planning round goes on with, written before the model is asked.

        In one fenced transaction: the round's latest attempt is read; one
        that got an answer is returned to replay, and nothing is written; one
        still ``INTENDED`` -- the process stopped during the call -- is closed
        ``OUTCOME_UNKNOWN``; and a new attempt is written ``INTENDED``
        (:func:`~xaytune.core.domain.agent_invocation.next_invocation`).

        Raises:
            AgentInvocationConflictError: The round recorded another request.
        """
        with self._write():
            latest = self.agent_invocations.latest_for_round(
                str(intent.experiment_id),
                intent.planner_spec_fingerprint,
                intent.context_fingerprint,
            )
            closed, invocation, replayed = next_invocation(latest, intent, at=utc_now())
            if closed is not None:
                self.agent_invocations._update(closed)
            if not replayed:
                self.agent_invocations._insert(invocation)
        return invocation

    def settle_agent_invocation(self, invocation: AgentInvocation) -> AgentInvocation:
        """Write an invocation's transition, guarded on the revision it moved from.

        Raises:
            ConcurrentModificationError: The record moved since *invocation*
                was read.
        """
        with self._write():
            self.agent_invocations._update(invocation)
        return invocation

    def daemon_responsibilities(self) -> tuple[ExperimentId, ...]:
        """Every nonterminal experiment a daemon has durably taken on, oldest first.

        One a daemon admitted (its ``controller_host`` is ``local_daemon``),
        or one a daemon adopted by carrying out a request for it: a
        ``COMPLETED`` request of any kind, since the daemon attaches an
        experiment before it cancels, proposes on or approves for it
        (PR-029). An experiment an embedded host admitted, and no daemon ever
        acted on, is not here: that host may still be driving it.
        """
        open_states = tuple(
            status.value
            for status in ExperimentStatus
            if not EXPERIMENT_MACHINE.is_terminal(status)
        )
        rows = self._connection.execute(
            "SELECT id FROM experiments "  # noqa: S608
            f"WHERE status IN ({', '.join('?' for _ in open_states)}) AND ("
            "json_extract(payload_json, '$.controller_host.kind') = 'local_daemon' "
            "OR id IN (SELECT experiment_id FROM controller_requests "
            "WHERE kind != 'submit' AND state = 'completed')) "
            "ORDER BY created_at, id",
            open_states,
        ).fetchall()
        return tuple(ExperimentId(row["id"]) for row in rows)

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
        """Move a run to *new_status*, with its event, atomically.

        Raises:
            ProvenanceError: *new_status* is ``SUCCEEDED`` while the Run's final
                attempt carries an intervention directive no worker confirmed.
                Migration 015 enforces the same rule inside the write.
        """
        if new_status is RunStatus.SUCCEEDED and self.unconfirmed_final_directives(str(run_id)):
            raise ProvenanceError(
                f"Run {run_id} cannot succeed: its final attempt did not confirm every "
                f"intervention it was directed to apply"
            )
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

    def unconfirmed_final_directives(self, run_id: str) -> tuple[InterventionDirective, ...]:
        """Directives of the Run's final attempt that no confirmed application answers."""
        attempts = self.aggregates.attempts_for_run(run_id)
        if not attempts:
            return ()
        final = max(attempts, key=lambda attempt: attempt.attempt_number)
        return self.intervention_directives.unconfirmed_for_attempt(str(final.id))

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
            after_write=lambda moved: self._after_attempt_transition(
                moved, telemetry_position, actor, destinations
            ),
        )

    def _after_attempt_transition(
        self,
        moved: RunAttempt,
        telemetry_position: tuple[int, int] | None,
        actor: Actor,
        destinations: tuple[str, ...],
    ) -> None:
        if telemetry_position is not None:
            self.aggregates._advance_telemetry(str(moved.id), telemetry_position)
        self._fail_unconfirmed_numerical_action(moved, actor, destinations)

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
        with self._write():
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

    def incident_context(self, target: RuntimeOperationTarget) -> AttemptContext:
        """Resolve the observation's attempt, run, node and experiment from the record."""
        if target.kind == "training-attempt":
            attempt = self.aggregates.load_attempt(target.id)
            run = self.aggregates.load_run(str(attempt.run_id))
        else:
            evaluation_attempt = self.aggregates.load_evaluation_attempt(target.id)
            evaluation_run = self.aggregates.load_evaluation_run(
                str(evaluation_attempt.evaluation_run_id)
            )
            return AttemptContext(
                target=target,
                run_id=str(evaluation_run.id),
                node_id=evaluation_run.node_id,
                experiment_id=evaluation_run.experiment_id,
            )
        return AttemptContext(
            target=target, run_id=str(run.id), node_id=run.node_id, experiment_id=run.experiment_id
        )

    def prepare_recovery_episode(
        self, incident_id: str, request: RecoveryRequest, actor: Actor
    ) -> RecoveryEpisode:
        """Prepare immutable initial provenance without writing a half-episode."""
        with self._write():
            incident = self.incidents.get(incident_id)
            if incident is None:
                raise AggregateNotFoundError("incident", incident_id)
            existing = self.recovery_episodes.for_attempt(incident.context.target)
            if existing is not None:
                return existing
            context = self.incident_context(incident.context.target)
            attempt, run, _ = self._recovery_attempt_state(context)
            node = self.aggregates.load_node(str(context.node_id))
            return RecoveryEpisode(
                context=context,
                attempt_number=attempt.attempt_number,
                candidate_fingerprint=(
                    run.candidate_fingerprint
                    if isinstance(run, Run)
                    else node.candidate_fingerprint
                ),
                request=request,
                created_by=actor,
            )

    def _recovery_attempt_state(
        self, context: AttemptContext
    ) -> tuple[
        RunAttempt | EvaluationAttempt,
        Run | EvaluationRun,
        tuple[RunAttempt, ...] | tuple[EvaluationAttempt, ...],
    ]:
        if context.target.kind == "training-attempt":
            attempt = self.aggregates.load_attempt(context.target.id)
            return (
                attempt,
                self.aggregates.load_run(context.run_id),
                self.aggregates.attempts_for_run(context.run_id),
            )
        evaluation_attempt = self.aggregates.load_evaluation_attempt(context.target.id)
        return (
            evaluation_attempt,
            self.aggregates.load_evaluation_run(context.run_id),
            self.aggregates.evaluation_attempts_for_run(context.run_id),
        )

    def recovery_target_is_superseded(self, context: AttemptContext) -> bool:
        with self._write():
            attempt, _, attempts = self._recovery_attempt_state(context)
            return any(a.attempt_number > attempt.attempt_number for a in attempts)

    def recovery_snapshot(self, episode: RecoveryEpisode) -> RecoveryInputsV1:
        """Typed decision-only projection. Byte inspection happens outside this lock."""
        with self._write():
            return self._recovery_snapshot(episode)

    def _recovery_snapshot(self, episode: RecoveryEpisode) -> RecoveryInputsV1:
        context = self.incident_context(episode.context.target)
        attempt, run, attempts = self._recovery_attempt_state(context)
        experiment = self.aggregates.load_experiment(str(context.experiment_id))
        node = self.aggregates.load_node(str(context.node_id))
        candidate = (
            run.candidate_fingerprint if isinstance(run, Run) else node.candidate_fingerprint
        )
        if (
            context != episode.context
            or episode.attempt_number != attempt.attempt_number
            or candidate != episode.candidate_fingerprint
        ):
            raise ProvenanceError("episode does not describe its authoritative attempt")
        if (episode.coordinator_name, episode.coordinator_version) != (
            COORDINATOR_NAME,
            COORDINATOR_VERSION,
        ):
            raise ProvenanceError("unsupported episode coordinator identity")
        previous = self.recovery_plans.effective_for_episode(str(episode.id))
        memberships = self.recovery_episodes.memberships(str(episode.id))
        evidence: list[RecoveryEvidence] = []
        if self.recovery_episodes.get(str(episode.id)) is None:
            for sequence, incident in enumerate(self.incidents.for_attempt(context.target), 1):
                evidence.append(RecoveryEvidence.from_incident(incident, sequence, candidate))
        else:
            for membership in memberships:
                if membership.disposition is RecoveryEvidenceDisposition.ACCEPTED_FOR_DECISION:
                    member_incident = self.incidents.get(str(membership.incident_id))
                    assert member_incident is not None
                    evidence.append(
                        RecoveryEvidence.from_incident(
                            member_incident, membership.membership_sequence, candidate
                        )
                    )
        all_evidence = tuple(sorted(evidence, key=lambda e: str(e.incident_id)))
        # Repair every accepted extension, even if several arrived before planning resumed.
        through = len(evidence) if previous is None else previous.accepted_through_sequence + 1
        prefix = tuple(e for e in all_evidence if e.membership_sequence <= through)
        signatures = sorted({e.signature for e in prefix})
        inputs = RecoveryInputsV1(
            context=context,
            experiment_status=experiment.status,
            experiment_revision=experiment.revision,
            node_status=node.status,
            node_revision=node.revision,
            run_status=run.status,
            run_revision=run.revision,
            attempt_status=attempt.status,
            attempt_revision=attempt.revision,
            attempt_number=attempt.attempt_number,
            actual_attempt_count=len(attempts),
            successor_exists=any(a.attempt_number > attempt.attempt_number for a in attempts),
            pending_other_run_reservations=self.recovery_episodes.pending_excluding(
                context.target, context.run_id, str(episode.id)
            ),
            candidate_fingerprint=candidate,
            execution_state_fingerprint=execution_state_fingerprint_v1(
                FrozenDict(attempt.model_dump(mode="json"))
            ),
            episode_id=episode.id,
            request_fingerprint=episode.request_fingerprint,
            coordinator_name=episode.coordinator_name,
            coordinator_version=episode.coordinator_version,
            accepted_evidence=prefix,
            accepted_membership_count=len(all_evidence),
            accepted_membership_fingerprint=fingerprint(all_evidence),
            predecessor=None
            if previous is None
            else RecoveryPredecessor(
                plan_id=previous.id,
                sequence=previous.sequence,
                accepted_through_sequence=previous.accepted_through_sequence,
                accepted_evidence_fingerprint=previous.accepted_evidence_fingerprint,
            ),
            experiment_recovery_usage_excluding_target=self.recovery_episodes.usage_excluding(
                str(context.experiment_id), str(episode.id)
            ),
            repeat_counts=tuple(
                RecoveryRepeatCount(
                    signature=signature,
                    prior_matching_episodes=self.recovery_episodes.prior_matching(
                        episode, signature
                    ),
                )
                for signature in signatures
            ),
        )
        if recovery_requires_checkpoints(inputs, episode.request):
            reports = []
            for producer in attempts:
                if producer.attempt_number <= attempt.attempt_number:
                    for record in self.checkpoints.for_attempt(str(producer.id)):
                        reports.append(
                            RecoveryCheckpointReport.from_record(record, producer.attempt_number)
                        )
            inputs = inputs.model_copy(
                update={
                    "checkpoint_reports": tuple(sorted(reports, key=checkpoint_order, reverse=True))
                }
            )
        return inputs

    def attach_recovery_evidence(
        self, incident_id: str, *, actor: Actor, destinations: tuple[str, ...] = ()
    ) -> None:
        """Repair audit linkage only; accepted coverage gaps intentionally fail closed."""
        with self._write():
            incident = self.incidents.get(incident_id)
            if incident is None:
                raise AggregateNotFoundError("incident", incident_id)
            episode = self.recovery_episodes.for_attempt(incident.context.target)
            if episode is not None and self.recovery_episodes.membership(incident_id) is None:
                self._attach_recovery_evidence(episode, incident, actor, destinations)

    def _attach_recovery_evidence(
        self,
        episode: RecoveryEpisode,
        incident: Incident,
        actor: Actor,
        destinations: tuple[str, ...],
    ) -> None:
        membership = self.recovery_episodes._attach(episode, incident, actor)
        aggregate = (
            self.aggregates.load_attempt(incident.context.target.id)
            if incident.context.target.kind == "training-attempt"
            else self.aggregates.load_evaluation_attempt(incident.context.target.id)
        )
        self._emit(
            aggregate,
            "RecoveryEvidenceAttached",
            str(episode.context.experiment_id),
            actor,
            destinations,
            extra={"membership": membership.model_dump(mode="json")},
        )

    def record_recovery_plan(
        self,
        plan: RecoveryPlan,
        *,
        actor: Actor,
        destinations: tuple[str, ...] = (),
        episode: RecoveryEpisode | None = None,
    ) -> RecoveryPlan:
        """Authoritatively bind DB inputs and trusted coordinator eligibility.

        This lock cannot prove byte integrity: inspection is outside the lock.
        A future executor MUST revalidate checkpoint bytes before using them.
        """
        plan = RecoveryPlan.model_validate_json(plan.model_dump_json())
        with self._write():
            revisions = self.recovery_plans.revisions_for_episode(str(plan.episode_id))
            existing = next((p for p in revisions if p.sequence == plan.sequence), None)
            reused = self.recovery_plans.get(str(plan.id))
            if reused is not None and (reused.episode_id, reused.sequence) != (
                plan.episode_id,
                plan.sequence,
            ):
                raise IdempotencyConflictError(
                    str(plan.id), ("episode_id", "sequence"), kind="recovery plan"
                )
            if existing is not None:
                if existing.semantic_fingerprint() != plan.semantic_fingerprint():
                    raise IdempotencyConflictError(
                        str(plan.episode_id), ("recovery_plan",), kind="episode revision"
                    )
                return existing
            stored = self.recovery_episodes.get(str(plan.episode_id))
            if stored is None:
                if episode is None or episode.id != plan.episode_id:
                    raise ProvenanceError(
                        "first plan requires explicit immutable episode provenance"
                    )
                owner = self.recovery_episodes.for_attempt(episode.context.target)
                if owner is not None:
                    raise StaleRecoveryContextError("another planner created the target episode")
                stored = RecoveryEpisode.model_validate_json(episode.model_dump_json())
            elif episode is not None and episode != stored:
                raise ProvenanceError("episode provenance is immutable")
            inputs = self._recovery_snapshot(stored)
            if inputs != plan.inputs or inputs.successor_exists:
                raise StaleRecoveryContextError("decision-relevant recovery inputs changed")
            expected_reports = inputs.checkpoint_reports
            eligibility = plan.checkpoint_eligibility
            if tuple((e.checkpoint_ref, e.report_fingerprint) for e in eligibility) != tuple(
                (r.checkpoint_ref, r.report_fingerprint) for r in expected_reports
            ):
                raise ProvenanceError(
                    "eligibility must bind every relevant report once in canonical order"
                )
            if any(
                e.eligible
                and (
                    checkpoint_report_problem(r, inputs) is not None
                    or stored.request.restore_context is None
                )
                for e, r in zip(eligibility, expected_reports)
            ):
                raise ProvenanceError(
                    "eligible checkpoint contradicts authoritative structured provenance"
                )
            expected = decide_recovery(inputs, stored.request, eligibility)
            for field in (
                "strategy",
                "recoverability",
                "checkpoint_ref",
                "reason",
                "requires_approval",
            ):
                if getattr(plan, field) != getattr(expected, field):
                    raise ProvenanceError("recovery decision disagrees with deterministic planning")
            initial = self.recovery_episodes.get(str(stored.id)) is None
            if initial:
                self.recovery_episodes._insert(stored)
                for incident in self.incidents.for_attempt(stored.context.target):
                    self.recovery_episodes._attach(stored, incident, actor)
            self.recovery_plans._insert(plan)
            aggregate = (
                self.aggregates.load_attempt(stored.context.target.id)
                if stored.context.target.kind == "training-attempt"
                else self.aggregates.load_evaluation_attempt(stored.context.target.id)
            )
            extra: dict[str, Any] = {
                "episode_id": str(stored.id),
                "plan_id": str(plan.id),
                "sequence": plan.sequence,
                "supersedes_plan_id": None
                if plan.supersedes_plan_id is None
                else str(plan.supersedes_plan_id),
                "accepted_through_sequence": plan.accepted_through_sequence,
                "accepted_evidence_fingerprint": plan.accepted_evidence_fingerprint,
                "recovery_plan": plan.model_dump(mode="json"),
            }
            if initial:
                extra["recovery_episode"] = stored.model_dump(mode="json")
                extra["memberships"] = [
                    m.model_dump(mode="json")
                    for m in self.recovery_episodes.memberships(str(stored.id))
                ]
            self._emit(
                aggregate,
                "RecoveryPlanned",
                str(stored.context.experiment_id),
                actor,
                destinations,
                extra=extra,
            )
        return plan

    def record_incident(
        self, incident: Incident, *, actor: Actor, destinations: tuple[str, ...] = ()
    ) -> Incident:
        """Record an incident, its event/outbox and replay cursor in one commit.

        A replay returns the original diagnosis and provenance, even if the
        caller's classifier has changed. Changed evidence at the same stream
        position is a conflict. Observations do not change attempt state or
        revision; the incident table is the authority for attempt membership.
        """
        incident = Incident.model_validate_json(incident.model_dump_json())
        target = incident.context.target
        with self._write():
            context = self.incident_context(target)
            if context != incident.context:
                raise ProvenanceError("incident context does not match its recorded attempt")
            correlation = incident.evidence.get("context") or {}
            claims = {
                "experiment_id": str(context.experiment_id),
                "node_id": str(context.node_id),
                "run_id"
                if target.kind == "training-attempt"
                else "evaluation_run_id": context.run_id,
                "attempt_id"
                if target.kind == "training-attempt"
                else "evaluation_attempt_id": target.id,
            }
            other_fields = (
                ("evaluation_run_id", "evaluation_attempt_id")
                if target.kind == "training-attempt"
                else ("run_id", "attempt_id")
            )
            for name in other_fields:
                if correlation.get(name) is not None:
                    raise ProvenanceError(f"incident evidence claims an unrelated {name}")
            for name, value in claims.items():
                if correlation.get(name) is not None and correlation[name] != value:
                    raise ProvenanceError(f"incident evidence claims a different {name}")
            existing = self.incidents.for_observation(incident.observation_key)
            if existing is not None:
                if existing.evidence_fingerprint != incident.evidence_fingerprint:
                    raise IdempotencyConflictError(
                        incident.observation_key, ("evidence",), kind="incident observation"
                    )
                return existing
            self.incidents._insert(incident)
            episode = self.recovery_episodes.for_attempt(target)
            if episode is not None:
                self._attach_recovery_evidence(episode, incident, actor, destinations)
            aggregate = (
                self.aggregates.load_attempt(target.id)
                if target.kind == "training-attempt"
                else self.aggregates.load_evaluation_attempt(target.id)
            )
            self._emit(
                aggregate,
                "IncidentRecorded",
                str(context.experiment_id),
                actor,
                destinations,
                extra={"incident": incident.model_dump(mode="json")},
            )
            self.aggregates._advance_telemetry(
                target.id, (incident.stream_generation, incident.sequence), kind=target.kind
            )
        return incident

    def record_checkpoint(
        self,
        attempt_id: RunAttemptId,
        payload: CheckpointCommittedPayload,
        *,
        evidence: FrozenDict,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> RecordedCheckpoint:
        """Persist a commit report, event/outbox, replay receipt and cursor atomically.

        Reports never establish byte integrity or an achieved restore. A
        future coordinator must use the checkpoint manager before selecting
        one for resume. Re-emission at a new position adds a receipt, not a
        second checkpoint/event. The original provenance remains authoritative.
        """
        target = RuntimeOperationTarget(kind="training-attempt", id=str(attempt_id))
        with self._write():
            context = self.incident_context(target)
            attempt = self.aggregates.load_attempt(str(attempt_id))
            run = self.aggregates.load_run(str(attempt.run_id))
            if attempt.execution_fingerprint is None:
                raise ProvenanceError("checkpoint producer has no recorded execution fingerprint")
            correlation = evidence.get("context") or {}
            claims = {
                "experiment_id": str(context.experiment_id),
                "node_id": str(context.node_id),
                "run_id": context.run_id,
                "attempt_id": target.id,
            }
            if any(
                correlation.get(name) is not None
                for name in ("evaluation_run_id", "evaluation_attempt_id")
            ):
                raise ProvenanceError("checkpoint evidence claims an evaluation producer")
            if any(correlation.get(name) not in (None, value) for name, value in claims.items()):
                raise ProvenanceError("checkpoint evidence claims another owner")
            record = RecordedCheckpoint.model_validate(
                {
                    "context": context,
                    "candidate_fingerprint": run.candidate_fingerprint,
                    "execution_fingerprint": attempt.execution_fingerprint,
                    "payload": payload.model_dump(mode="json"),
                    "evidence": evidence,
                }
            )
            refs = checkpoint_state_refs(record.payload.state_manifest, record.payload.data_cursor)
            if any(
                ref is not None
                and (
                    ref.producer_attempt_id != attempt_id or ref.producer_evaluation_id is not None
                )
                for ref in refs
            ):
                raise ProvenanceError("checkpoint state names another or unknown producer")
            generation, sequence = evidence["stream_generation"], evidence["sequence"]
            receipt = self.checkpoints._receipt(str(attempt_id), generation, sequence)
            if receipt is not None:
                if receipt["evidence_digest"] != fingerprint(evidence):
                    raise IdempotencyConflictError(
                        str(attempt_id), ("evidence",), kind="checkpoint observation"
                    )
                existing = self.checkpoints.get(receipt["checkpoint_id"])
                assert existing is not None
                return existing
            checkpoint_id = str(record.payload.checkpoint_ref.id)
            existing = self.checkpoints.get(checkpoint_id)
            if existing is not None:
                if existing.context != context or (
                    fingerprint(existing.payload) != fingerprint(record.payload)
                ):
                    raise IdempotencyConflictError(
                        checkpoint_id, ("producer or payload",), kind="checkpoint"
                    )
            else:
                self.checkpoints._insert(record)
                self._emit(
                    attempt,
                    "CheckpointRecorded",
                    str(context.experiment_id),
                    actor,
                    destinations,
                    extra={"checkpoint": record.model_dump(mode="json")},
                )
            self.checkpoints._insert_receipt(record)
            self.aggregates._advance_telemetry(str(attempt_id), (generation, sequence))
        return existing or record

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
        with self._write():
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
            BudgetExhaustedError: If a quota the evaluation would spend is
                used up; nothing is written, and the node stays where it is.
        """
        for run in runs:
            _require_pristine(run, EvaluationRunStatus.CREATED)

        with self._write():
            current = self.aggregates.get_node(str(node_id))
            if current is None:
                raise AggregateNotFoundError("ExperimentNode", str(node_id))
            if current.revision != expected_revision:
                raise ConcurrentModificationError("ExperimentNode", str(node_id), expected_revision)
            if runs:
                self._require_budget(str(current.experiment_id), new_run=False)
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
        with self._write():
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
        with self._write():
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
        with self._write():
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
            self._settle_budget(succeeded_attempt, actor, destinations)
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
        with self._write():
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
        BRANCH           node COMPLETED   experiment unchanged
        ```

        ``REJECT`` and ``BRANCH`` are judgements on the candidate, not the
        experiment: another candidate may yet be proposed. Only the ``STOP`` outcomes end an
        experiment, and only an ``ACTIVE`` one; a paused experiment is
        somebody's to resume or stop. There is no moment at which a decision
        is recorded but not applied, or applied with no decision on record.

        Deciding a cycle already decided, with the same proposal, returns the
        decision on record and writes nothing -- what a controller restarted
        after deciding sees. Anything else is refused.

        The proposal must be about this node's current cycle, in
        ``DECIDING``, and must name exactly that cycle's results, each once:
        a decision drawn from an earlier round's results, or from some of
        this round's, is not attributable to the evidence it claims. Its
        evidence may cite only those results, and its ``input_fingerprint``
        must be the one the experiment's objective and those results give
        (:func:`~xaytune.core.domain.decision.decision_input_identity_v1`),
        recomputed here, not taken on trust. Which outcome those inputs
        warrant is the engine's to say and is not checked: a custom engine
        may decide by rules of its own.

        Raises:
            AggregateNotFoundError: If the node does not exist.
            DecisionConflictError: If the cycle was decided differently.
            ConcurrentModificationError: If the node moved since it was read.
            InvalidTransitionError: If the node is not in ``DECIDING``.
            ProvenanceError: If the proposal is about another experiment or
                cycle, names results other than the cycle's or one twice,
                cites a result outside the cycle, or claims an input
                fingerprint the durable inputs do not give.
        """
        with self._write():
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
            results = tuple(
                result
                for run in self.aggregates.evaluation_runs_for_node(
                    str(node.id), cycle=node.evaluation_cycle
                )
                if (result := self.aggregates.evaluation_result_for_run(str(run.id))) is not None
            )
            cycle_results = {result.id for result in results}
            named = proposal.evaluation_result_ids
            if len(set(named)) != len(named):
                repeated = sorted({str(i) for i in named if named.count(i) > 1})
                problems.append(f"names results {repeated} more than once")
            if set(named) != cycle_results:
                problems.append(
                    f"names results {sorted(named)}, but the cycle's are {sorted(cycle_results)}"
                )
            cited = {evidence.evaluation_result_id for evidence in proposal.evidence}
            if foreign := cited - cycle_results:
                problems.append(f"cites results {sorted(foreign)} from outside the cycle")
            if not problems:
                # The fingerprint is the record's claim about what was decided
                # on. Recompute it from the durable inputs rather than trust
                # the caller's: which outcome follows from those inputs is the
                # engine's business, but what the inputs were is ours.
                experiment = self.aggregates.load_experiment(str(node.experiment_id))
                expected = DecisionContext(
                    experiment_id=experiment.id,
                    node_id=node.id,
                    evaluation_cycle=node.evaluation_cycle,
                    objective=experiment.objective,
                    results=results,
                ).input_fingerprint()
                if proposal.input_fingerprint != expected:
                    problems.append(
                        f"claims input fingerprint {proposal.input_fingerprint}, but the "
                        f"objective and the cycle's results fingerprint to {expected}"
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
        """End an ``ACTIVE`` experiment when the decision says to stop it; otherwise nothing.

        Every outcome is handled by name. One with no transition here fails
        closed, inside the decision's transaction, so an outcome added later
        can never fall through to failing the experiment.

        Raises:
            StorageError: If the outcome has no experiment transition.
        """
        outcome = decision.outcome
        if outcome in (DecisionOutcome.REJECT, DecisionOutcome.BRANCH):
            return
        if outcome not in (DecisionOutcome.STOP_SUCCEEDED, DecisionOutcome.STOP_FAILED):
            raise StorageError(f"decision outcome {outcome.value} has no experiment transition")
        experiment = self.aggregates.load_experiment(str(decision.experiment_id))
        if experiment.status is not ExperimentStatus.ACTIVE:
            return
        if outcome is DecisionOutcome.STOP_SUCCEEDED:
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
        with self._write():
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

    # ---- planning (PR-024) ----------------------------------------------------

    def planning_context(self, experiment_id: ExperimentId | str) -> PlanningContext:
        """The experiment as a planner may see it: a curated projection, read-only.

        Topology comes from the relationships alone (open question 15):
        nodes of the experiment, each node's decisions and its evaluation
        results with the cycle each belongs to. Every candidate's identity is
        recomputed under the current projection from its stored snapshot.
        Nothing is written.
        """
        aggregates = self.aggregates
        experiment = aggregates.load_experiment(str(experiment_id))
        nodes = []
        for node in aggregates.nodes_for_experiment(str(experiment.id)):
            evaluations = []
            for run in aggregates.evaluation_runs_for_node(str(node.id)):
                result = aggregates.evaluation_result_for_run(str(run.id))
                if result is None:
                    continue
                evaluations.append(
                    EvaluationSummary(
                        evaluation_result_id=result.id,
                        evaluation_cycle=run.evaluation_cycle,
                        metrics=tuple(
                            MetricSummary(
                                name=metric.name,
                                value=metric.value,
                                slice=metric.slice,
                                evaluator_name=metric.evaluator_name,
                                evaluator_version=metric.evaluator_version,
                            )
                            for metric in result.metrics
                        ),
                    )
                )
            candidate = node.candidate.candidate
            nodes.append(
                NodeSummary(
                    node_id=node.id,
                    status=node.status,
                    parent_ids=node.parent_ids,
                    candidate=candidate,
                    candidate_fingerprint=candidate.candidate_fingerprint(),
                    decisions=tuple(
                        DecisionSummary(
                            decision_id=decision.id,
                            evaluation_cycle=decision.evaluation_cycle,
                            outcome=decision.outcome,
                            engine_name=decision.engine_name,
                            engine_version=decision.engine_version,
                            input_fingerprint=decision.input_fingerprint,
                            evaluation_result_ids=decision.evaluation_result_ids,
                        )
                        for decision in aggregates.decisions_for_node(str(node.id))
                    ),
                    evaluations=tuple(evaluations),
                )
            )
        return PlanningContext(
            experiment_id=experiment.id,
            experiment_status=experiment.status,
            objective=experiment.objective,
            nodes=tuple(nodes),
            budget=self.budget_status(experiment.id),
        )

    def materialize_candidate_proposal(
        self,
        experiment_id: ExperimentId | str,
        proposal: CandidateProposal,
        *,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> ExperimentNode:
        """Turn a candidate proposal into a ``PLANNED`` child node, in one commit (PR-025).

        Everything is checked inside the write transaction that creates the
        node, so no other writer can change the experiment in between:

        ```text
        the candidate's current fingerprint is the proposal's     else ProvenanceError
        the experiment already has this candidate:
            from this very proposal                 → that node; nothing written
            from anything else                      → CandidateConflictError
        the experiment is ACTIVE                                   else BranchRefusedError
        the proposal was made by the recorded planner, configured
          exactly as recorded (spec fingerprint recomputed)        else ProvenanceError
        the planning context is unchanged since (fingerprint)      else StaleProposalError
        no budget quota is exhausted                               else BranchRefusedError
        each evidence ref is a decision or result of a parent      else ProvenanceError
        the parents exist, in this experiment, with no cycle       else LineageError
        ```

        Then the node is created -- id minted here, candidate stored as
        proposed, ``branch_origin`` recording the proposal -- and moved to
        ``PLANNED``, with ``NodeCreated`` and ``ExperimentNodeStatusChanged``,
        in the same commit. No run, attempt, action, operation or ledger entry
        is written: realizing the node is PR-026's, and ``max_runs`` is
        reserved only when a run is created.

        The repository checks the durable planner spec; that the proposal's
        descriptor identity is the planner this host actually binds is the
        controller's to check before calling (layering: storage never imports
        a planner implementation).

        Raises:
            AggregateNotFoundError: If the experiment does not exist.
            ProvenanceError: See above.
            CandidateConflictError: See above.
            StaleProposalError: See above.
            BranchRefusedError: See above.
            LineageError: See above.
        """
        with self._write():
            aggregates = self.aggregates
            experiment = aggregates.load_experiment(str(experiment_id))
            candidate = proposal.candidate
            identity = candidate.candidate_fingerprint()
            if identity != proposal.candidate_fingerprint:
                raise ProvenanceError(
                    f"the proposal claims candidate {proposal.candidate_fingerprint}, but its "
                    f"candidate fingerprints as {identity}"
                )
            origin = CandidateBranchOrigin.of(proposal)
            for existing in aggregates.nodes_for_experiment(str(experiment.id)):
                if existing.candidate.candidate.candidate_fingerprint() != identity:
                    continue
                recorded = existing.branch_origin
                if recorded is not None and recorded.proposal_fingerprint == (
                    origin.proposal_fingerprint
                ):
                    return existing
                raise CandidateConflictError(
                    f"experiment {experiment.id} already has candidate {identity} as node "
                    f"{existing.id}, from "
                    + (
                        f"proposal {recorded.proposal_fingerprint}"
                        if recorded is not None
                        else "no proposal"
                    )
                    + "; the same scientific candidate is never a second node"
                )
            if experiment.status is not ExperimentStatus.ACTIVE:
                raise BranchRefusedError(
                    f"experiment {experiment.id} is {experiment.status.value}; only an ACTIVE "
                    f"experiment branches"
                )
            self._require_recorded_planner(experiment, proposal.provenance)
            context = self.planning_context(experiment.id)
            provenance = proposal.provenance
            if (
                provenance.context_identity_version != PLANNING_CONTEXT_IDENTITY_VERSION
                or provenance.context_fingerprint != context.input_fingerprint()
            ):
                raise StaleProposalError(
                    f"the proposal was planned against context {provenance.context_fingerprint} "
                    f"(identity v{provenance.context_identity_version}), but experiment "
                    f"{experiment.id} is now {context.input_fingerprint()} "
                    f"(v{PLANNING_CONTEXT_IDENTITY_VERSION}); plan again"
                )
            if context.budget is not None and context.budget.exhausted:
                raise BranchRefusedError(
                    f"experiment {experiment.id} has an exhausted quota "
                    f"({', '.join(d.dimension.value for d in context.budget.exhausted)}); "
                    f"no run of a new candidate could follow"
                )
            self._require_parent_evidence(proposal)

            node = ExperimentNode(
                id=ExperimentNodeId.generate(),
                experiment_id=experiment.id,
                parent_ids=proposal.parent_ids,
                hypothesis=proposal.hypothesis,
                reason=proposal.reason,
                candidate=CandidateSpecSnapshot(candidate=candidate),
                candidate_fingerprint=identity,
                branch_origin=origin,
                created_by=actor,
            )
            self.graph.validate_parents(node)
            aggregates._insert_node(node)
            self._emit(
                node,
                "NodeCreated",
                str(experiment.id),
                actor,
                destinations,
                extra={
                    "proposal_fingerprint": origin.proposal_fingerprint,
                    "parent_ids": [str(parent) for parent in node.parent_ids],
                },
            )
            planned = node.with_status(ExperimentNodeStatus.PLANNED)
            aggregates._update_node(planned)
            self._emit(
                planned, "ExperimentNodeStatusChanged", str(experiment.id), actor, destinations
            )
        return planned

    def realize_planned_node(
        self,
        node_id: ExperimentNodeId | str,
        *,
        expected_revision: int,
        run: Run,
        attempt: RunAttempt,
        request_digest: str,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> tuple[ExperimentNode, Run, RunAttempt, RuntimeOperation] | None:
        """Accept a branched node's first run, and the intent to submit it, in one commit (PR-026).

        The ``PLANNED`` node is the gate. Inside one write transaction:

        ```text
        the node is still PLANNED                     else None: already realized,
                                                      or no longer eligible; nothing written
        its revision is the caller's                  else ConcurrentModificationError
        it was branched (branch_origin) by the
          experiment's recorded planner, exactly      else ProvenanceError
        the experiment is ACTIVE                      else BranchRefusedError
        the run and attempt are new, the node's,
          its first, replicate 1; one parent with
          one SUCCEEDED, seeded run, named as the
          seed origin, whose seed the run has         else StorageError
        a run is left (max_runs), failures are not
          used up, a parallel-run slot is free        else BudgetExhaustedError /
                                                      CapacityUnavailableError
        ```

        then: node ``PLANNED → READY → ACTIVE``, ``RunCreated`` with its
        ``max_runs`` reservation, run ``CREATED → ACTIVE``, the attempt
        ``CREATED`` with its ``INTENDED`` submit and its parallel-run slot --
        each transition through its state machine, with its event. The runtime
        is called only after this commits; a crash after it leaves an
        ``INTENDED`` operation, which reconciliation resolves (ADR-013).

        Returns:
            The active node, the active run, the attempt and its submit
            operation -- or ``None`` if the node is no longer ``PLANNED``.
        """
        with self._write():
            aggregates = self.aggregates
            node = aggregates.get_node(str(node_id))
            if node is None:
                raise AggregateNotFoundError("ExperimentNode", str(node_id))
            if node.status is not ExperimentNodeStatus.PLANNED:
                return None
            if node.revision != expected_revision:
                raise ConcurrentModificationError("ExperimentNode", str(node_id), expected_revision)
            if node.branch_origin is None:
                raise ProvenanceError(
                    f"node {node.id} was not branched from a planner's proposal; only a "
                    f"planner-created node is realized automatically"
                )
            experiment = aggregates.load_experiment(str(node.experiment_id))
            if experiment.status is not ExperimentStatus.ACTIVE:
                raise BranchRefusedError(
                    f"experiment {experiment.id} is {experiment.status.value}; only an ACTIVE "
                    f"experiment starts a run"
                )
            self._require_recorded_planner(experiment, node.branch_origin.provenance)
            self._require_first_realization(node, run, attempt)

            operation = RuntimeOperation(
                id=OperationId.generate(),
                target=RuntimeOperationTarget(kind="training-attempt", id=str(attempt.id)),
                type="submit",
                request_digest=request_digest,
            )
            for status in (ExperimentNodeStatus.READY, ExperimentNodeStatus.ACTIVE):
                node = self._apply_transition(
                    "ExperimentNode",
                    str(node.id),
                    node.revision,
                    status,
                    aggregates.get_node,
                    aggregates._update_node,
                    actor,
                    None,
                    destinations,
                )
            self._insert_run_with_reservation(run, actor, destinations)
            active = self._apply_transition(
                "Run",
                str(run.id),
                run.revision,
                RunStatus.ACTIVE,
                aggregates.get_run,
                aggregates._update_run,
                actor,
                None,
                destinations,
            )
            stored = self._insert_training_attempt_with_intent(
                attempt, operation, actor, destinations
            )
        return node, active, attempt, stored

    def _require_first_realization(
        self, node: ExperimentNode, run: Run, attempt: RunAttempt
    ) -> None:
        """The v1 contract for a branched node's first run, nothing inferred.

        ```text
        the run and attempt are new; the run is the node's (and its first);
          the attempt is the run's first
        the node has exactly one parent, and that parent exactly one run
        the new run names that run as its seed source (seed_origin required),
          has its seed, and is replicate 1
        the source run SUCCEEDED, with a seed
        ```

        The comparison the adaptive loop draws -- the child against its parent,
        differing by the mutation and not by chance -- holds only under these;
        a caller that would have to choose a parent or a run, or supply a
        seed of its own, is refused rather than trusted. That the seed claimed
        is honest is checked again, for any run, by ``_require_consistent_run``.
        """
        _require_pristine(run, RunStatus.CREATED)
        _require_pristine(attempt, RunAttemptStatus.CREATED)
        problems = []
        if run.node_id != node.id or run.experiment_id != node.experiment_id:
            problems.append(f"run {run.id} is not node {node.id}'s")
        if self.aggregates.runs_for_node(str(node.id)):
            problems.append(f"node {node.id} already has a run")
        if attempt.run_id != run.id or attempt.attempt_number != 1:
            problems.append(f"attempt {attempt.id} is not run {run.id}'s first")
        if run.replicate != 1:
            problems.append(f"run {run.id} is replicate {run.replicate}, not 1")
        if len(node.parent_ids) != 1:
            problems.append(
                f"node {node.id} has {len(node.parent_ids)} parents; a first run inherits "
                f"its seed from a single parent"
            )
        else:
            (parent,) = node.parent_ids
            parent_runs = self.aggregates.runs_for_node(str(parent))
            if len(parent_runs) != 1:
                problems.append(
                    f"parent {parent} has {len(parent_runs)} runs; which one's seed to "
                    f"inherit is not defined"
                )
            else:
                (source,) = parent_runs
                if run.seed_origin is None:
                    problems.append(
                        f"run {run.id} records no seed origin; a branched node's first run "
                        f"inherits its parent run's seed"
                    )
                elif run.seed_origin.source_run_id != source.id:
                    problems.append(
                        f"run {run.id} names seed source {run.seed_origin.source_run_id}, not "
                        f"parent {parent}'s run {source.id}"
                    )
                if source.status is not RunStatus.SUCCEEDED or source.seed is None:
                    problems.append(
                        f"parent run {source.id} is {source.status.value} with seed "
                        f"{source.seed}; only a successful, seeded run's seed is inherited"
                    )
                elif run.seed != source.seed:
                    problems.append(
                        f"run {run.id} has seed {run.seed}, not parent run {source.id}'s "
                        f"{source.seed}"
                    )
        if problems:
            raise StorageError("; ".join(problems))

    def _require_recorded_planner(
        self, experiment: Experiment, provenance: ProposalProvenance
    ) -> None:
        """The proposal was made under the experiment's recorded, bound planner spec, exactly."""
        spec = experiment.planner
        if spec is None:
            raise ProvenanceError(
                f"experiment {experiment.id} records no planner; a proposal cannot be "
                f"attributed to one"
            )
        problems = []
        if (provenance.planner_spec_kind, provenance.planner_name) != (spec.kind, spec.kind):
            problems.append(
                f"names planner {provenance.planner_name!r} "
                f"(spec {provenance.planner_spec_kind!r}), not the recorded {spec.kind!r}"
            )
        if (provenance.planner_spec_version, provenance.planner_version) != (
            spec.version,
            spec.version,
        ):
            problems.append(
                f"names version {provenance.planner_version} (spec "
                f"{provenance.planner_spec_version}), not the recorded {spec.version}"
            )
        if provenance.planner_spec_identity_version != PLANNER_SPEC_IDENTITY_VERSION:
            problems.append(
                f"uses planner-spec identity v{provenance.planner_spec_identity_version}, "
                f"not v{PLANNER_SPEC_IDENTITY_VERSION}"
            )
        else:
            assert spec.version is not None
            expected = fingerprint(
                planner_spec_identity_v1(
                    spec,
                    provider=provenance.planner_provider,
                    name=spec.kind,
                    plugin_version=spec.version,
                    api_version=provenance.planner_api_version,
                )
            )
            if provenance.planner_spec_fingerprint != expected:
                problems.append(
                    "its planner-spec fingerprint is not the recorded spec's: it was "
                    "configured differently"
                )
        if problems:
            raise ProvenanceError(f"proposal for experiment {experiment.id} " + "; ".join(problems))

    def _require_parent_evidence(self, proposal: CandidateProposal) -> None:
        """Every evidence ref is a durable decision or result of one of the proposal's parents."""
        decisions: set[str] = set()
        results: set[str] = set()
        for parent in proposal.parent_ids:
            decisions.update(str(d.id) for d in self.aggregates.decisions_for_node(str(parent)))
            results.update(
                str(r.id) for r in self.aggregates.evaluation_results_for_node(str(parent))
            )
        foreign = [
            f"{ref.kind}:{ref.id}"
            for ref in proposal.evidence_refs
            if ref.id not in (decisions if ref.kind == "decision" else results)
        ]
        if foreign:
            raise ProvenanceError(
                f"evidence {foreign} is not a decision or evaluation result of the proposal's "
                f"parents {[str(p) for p in proposal.parent_ids]}"
            )

    # ---- the budget ledger (PR-016) ---------------------------------------

    def budget_status(self, experiment_id: ExperimentId | str) -> BudgetStatus | None:
        """The experiment's budget as the ledger stands, or ``None`` if nothing is limited."""
        experiment = self.aggregates.load_experiment(str(experiment_id))
        if not limits(experiment.budget):
            return None
        return budget_status(experiment.budget, self.budget.entries(str(experiment_id)))

    def exhaust_budget(
        self,
        experiment_id: ExperimentId | str,
        *,
        reasons: tuple[str, ...],
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> bool:
        """End an ``ACTIVE`` experiment as ``BUDGET_EXHAUSTED`` -- once nothing is running.

        A quota being used up stops the *next* effect at once, but not one
        already running: an experiment is terminal only when no workload it
        owns is still executing, so ``wait()`` never sees it ended while a
        worker lives. With an attempt still live this writes nothing and
        returns ``False``; asked again once it has ended, it ends the
        experiment.

        Returns:
            Whether the experiment is now ``BUDGET_EXHAUSTED`` by this call.
        """
        with self._write():
            experiment = self.aggregates.load_experiment(str(experiment_id))
            if experiment.status is not ExperimentStatus.ACTIVE:
                return False
            if self._live_attempts(str(experiment_id)):
                return False
            self._exhaust(experiment, reasons, actor, destinations)
        return True

    def _exhaust(
        self,
        experiment: Experiment,
        reasons: tuple[str, ...],
        actor: Actor,
        destinations: tuple[str, ...],
    ) -> Experiment:
        """:meth:`exhaust_budget`'s write, inside a transaction the caller already holds."""
        moved = experiment.with_status(ExperimentStatus.BUDGET_EXHAUSTED)
        self.aggregates._update_experiment(moved)
        self._emit(moved, "ExperimentStatusChanged", str(moved.id), actor, destinations)
        self._emit(
            moved,
            "BudgetExhausted",
            str(moved.id),
            actor,
            destinations,
            extra={"reasons": list(reasons)},
        )
        return moved

    def settle_budget(
        self,
        experiment_id: ExperimentId | str,
        *,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> int:
        """Write any settlement the record implies and the ledger lacks. A safety net.

        Every entry is written with the change that causes it, so this
        normally finds nothing. It exists for a record the ledger cannot
        already account for -- one written before the ledger, say -- and is
        idempotent: each entry it writes is the one the change would have.

        Returns:
            How many entries it wrote.
        """
        with self._write():
            before = len(self.budget.entries(str(experiment_id)))
            for node in self.aggregates.nodes_for_experiment(str(experiment_id)):
                for run in self.aggregates.runs_for_node(str(node.id)):
                    for attempt in self.aggregates.attempts_for_run(str(run.id)):
                        for operation in self.operations.for_target(
                            "training-attempt", str(attempt.id)
                        ):
                            if operation.type == "submit" and operation.state == "confirmed":
                                self._commit_budget(
                                    str(experiment_id), str(attempt.id), actor, destinations
                                )
                        self._settle_budget(attempt, actor, destinations)
                    self._settle_budget(run, actor, destinations)
                for evaluation in self.aggregates.evaluation_runs_for_node(str(node.id)):
                    for measured in self.aggregates.evaluation_attempts_for_run(str(evaluation.id)):
                        self._settle_budget(measured, actor, destinations)
            return len(self.budget.entries(str(experiment_id))) - before

    # ---- governed actions (PR-023) -----------------------------------------------------

    def policy_context(
        self,
        spec: ActionSpec,
        *,
        experiment_id: ExperimentId | str,
        proposed_by: Actor,
        capabilities: CapabilityDocument | None,
    ) -> PolicyContext:
        """The snapshot validation and policy judge *spec* against, from the record.

        *capabilities* is what the experiment's runtime declares, supplied by
        the host; it becomes part of the snapshot.

        Raises:
            AggregateNotFoundError: If the experiment does not exist.
            UnknownActionTypeError: If the spec's type is not registered.
        """
        experiment = self.aggregates.load_experiment(str(experiment_id))
        descriptor = action_descriptor(spec.type, spec.version)  # type: ignore[attr-defined]
        provider = encode_payload(spec).get("provider")
        status, revision = self._governed_target_state(spec.target, str(experiment.id))
        return PolicyContext(
            experiment_id=experiment.id,
            experiment_status=experiment.status,
            experiment_revision=experiment.revision,
            action_type=descriptor.type,
            action_version=descriptor.version,
            provider=FrozenDict(provider) if provider is not None else None,
            mutation_class=descriptor.mutation_class,
            target=spec.target,
            parameters=FrozenDict(spec.parameters()),
            target_status=status,
            target_revision=revision,
            proposed_by=PolicyProposer.of(proposed_by),
            budget=self.budget_status(experiment.id),
            capabilities=capabilities,
        )

    def propose_action(
        self,
        spec: ActionSpec,
        *,
        experiment_id: ExperimentId | str,
        proposed_by: Actor,
        reason: str,
        policy: Any,
        capabilities: CapabilityDocument | None,
        action_id: ActionId | None = None,
        destinations: tuple[str, ...] = (),
        _bound: _BoundRecoveryProposal | None = None,
    ) -> GovernedAction:
        """Validate, authorize and record one proposed action, in one commit. Applies nothing.

        ```text
        PROPOSED → VALIDATING ─ not applicable ─────────▶ REJECTED          (no policy decision)
                              └ VALIDATED → policy
                                  ALLOW ─────────────────▶ VALIDATED         (+ decision)
                                  DENY ──────────────────▶ REJECTED          (+ decision)
                                  REQUIRE_APPROVAL ──────▶ APPROVAL_PENDING  (+ decision)
        ```

        *policy* is a :class:`~xaytune.policy.PolicyEngine`. It is evaluated
        on a snapshot read before the transaction; the transaction reads the
        snapshot again and records the decision only if it is unchanged, so a
        decision is never about a state other than the one it is filed
        against. The Action, the decision and every transition's event
        commit together, or none does.

        Proposing an action id already recorded, with the same request,
        returns it as governance left it and evaluates nothing.

        Raises:
            CancellationNotGovernedError: For a ``cancel-*`` spec.
            UnknownActionTypeError: If the spec's type is not registered.
            UnsupportedActionError: If the type's plugin refuses the spec.
            AggregateNotFoundError: If the experiment does not exist.
            IdempotencyConflictError: If *action_id* was recorded for a
                different request.
            ProvenanceError: If the engine's proposal names inputs other than
                the snapshot it was given, or an engine other than itself.
            StalePolicyContextError: If the state changed meanwhile.
        """
        if spec.type in CANCELLATION_TYPES:  # type: ignore[attr-defined]
            raise CancellationNotGovernedError(spec.type)  # type: ignore[attr-defined]
        if _bound is not None and (action_id != _bound.action_id or spec != _bound.spec):
            raise ProvenanceError("recovery Action spec or id disagrees with its binding")
        action = action_from_spec(
            spec,
            experiment_id=ExperimentId(str(experiment_id)),
            proposed_by=proposed_by,
            reason=reason,
            action_id=action_id,
        )
        if _bound is not None:
            bound = _bound.replay()
            if bound is not None:
                return bound
        replayed = self._replay_governed(action)
        if replayed is not None:
            if _bound is not None:
                raise IdempotencyConflictError(str(action.id), ("recovery binding",), kind="Action")
            return replayed

        context = self.policy_context(
            spec, experiment_id=experiment_id, proposed_by=proposed_by, capabilities=capabilities
        )
        problems = applicability_problems(spec, context)
        proposal = None if problems else policy.evaluate(spec, context)
        if proposal is not None and (proposal.engine_name, proposal.engine_version) != (
            policy.name,
            policy.version,
        ):
            raise ProvenanceError(
                f"policy {policy.name} {policy.version} returned a proposal signed "
                f"{proposal.engine_name} {proposal.engine_version}; a decision is recorded "
                f"under the name of the engine that made it, and nothing was written"
            )
        if proposal is not None and proposal.input_fingerprint != context.input_fingerprint():
            raise ProvenanceError(
                f"{proposal.engine_name} {proposal.engine_version} claims input "
                f"{proposal.input_fingerprint}, but it was given {context.input_fingerprint()}"
            )

        with self._write():
            if _bound is not None:
                bound = _bound.replay()
                if bound is not None:
                    return bound
            replayed = self._replay_governed(action)
            if replayed is not None:
                if _bound is not None:
                    raise IdempotencyConflictError(
                        str(action.id), ("recovery binding",), kind="Action"
                    )
                return replayed
            if _bound is not None:
                _bound.require_current()
            current = self.policy_context(
                spec,
                experiment_id=experiment_id,
                proposed_by=proposed_by,
                capabilities=capabilities,
            )
            if current.input_fingerprint() != context.input_fingerprint():
                raise StalePolicyContextError(
                    f"the state {action.type} was judged against changed before it could be "
                    f"recorded ({context.input_fingerprint()} → {current.input_fingerprint()}); "
                    f"nothing was written"
                )

            self.actions._insert(action)
            if _bound is not None:
                _bound.insert()
            self._emit_action(
                action, "ActionProposed", proposed_by, destinations, extra={"reason": reason}
            )
            validating = self._advance(
                action, ActionStatus.VALIDATING, "ActionValidating", proposed_by, destinations
            )
            if problems:
                rejected = validating.with_status(ActionStatus.REJECTED)
                self.actions._update(rejected)
                self._emit_action(
                    rejected,
                    "ActionRejected",
                    _GOVERNANCE,
                    destinations,
                    extra={"stage": "validation", "problems": list(problems)},
                )
                return GovernedAction(action=rejected, problems=problems)

            assert proposal is not None
            validated = self._advance(
                validating, ActionStatus.VALIDATED, "ActionValidated", _GOVERNANCE, destinations
            )
            engine = Actor(
                type="rule",
                id=f"policy:{proposal.engine_name}",
                metadata=FrozenDict({"engine_version": proposal.engine_version}),
            )
            decision = PolicyDecision.record(
                proposal, action_id=action.id, context=context, actor=engine
            )
            self.policy._insert(decision)
            governed = validated.governed_by(str(decision.id), _VERDICT_STATUS[proposal.verdict])
            self.actions._update(governed)
            self._emit_action(
                governed,
                _VERDICT_EVENT[proposal.verdict],
                engine,
                destinations,
                extra={
                    "policy_decision_id": str(decision.id),
                    "verdict": proposal.verdict.value,
                    "reasons": list(proposal.reasons),
                    "rule_ids": list(proposal.rule_ids),
                    "engine": f"{proposal.engine_name} {proposal.engine_version}",
                },
            )
            return GovernedAction(action=governed, decision=decision)

    def propose_oom_recovery_action(
        self,
        inputs: OOMRecoveryInputsV1,
        proposal: OOMResizeProposal,
        *,
        proposed_by: Actor,
        reason: str,
        policy: Any,
        capabilities: CapabilityDocument | None,
        action_id: ActionId | None = None,
        destinations: tuple[str, ...] = (),
    ) -> GovernedAction:
        """Govern one decision-bound resize and atomically bind its Action.

        The binding is written with the Action and any policy decision, before
        approval or execution. Replay of the same plan/proposal returns that
        Action without using current policy or configuration. This method
        creates no successor attempt, override, operation or receipt.
        """
        binding = RecoveryActionBinding.for_proposal(action_id or ActionId.generate(), proposal)
        return self.propose_action(
            proposal.action_spec,
            experiment_id=inputs.plan.inputs.context.experiment_id,
            proposed_by=proposed_by,
            reason=reason,
            policy=policy,
            capabilities=capabilities,
            action_id=binding.action_id,
            destinations=destinations,
            _bound=_BoundRecoveryProposal(
                action_id=binding.action_id,
                spec=binding.proposal.action_spec,
                replay=lambda: self._replay_recovery_action(binding),
                require_current=lambda: self._require_current_oom_action(binding, inputs),
                insert=lambda: self.recovery_action_bindings._insert(binding),
            ),
        )

    def _replay_recovery_action(self, requested: RecoveryActionBinding) -> GovernedAction | None:
        existing = self.recovery_action_bindings.for_plan(str(requested.plan_id))
        if existing is None:
            return None
        if existing.proposal != requested.proposal:
            raise IdempotencyConflictError(
                str(requested.plan_id), ("OOM proposal",), kind="recovery plan"
            )
        return self.governed_action(existing.action_id)

    def _require_current_oom_action(
        self, binding: RecoveryActionBinding, inputs: OOMRecoveryInputsV1
    ) -> None:
        proposal = binding.proposal
        if (
            proposal.input_fingerprint != inputs.input_fingerprint
            or proposal.run_id != inputs.run_id
            or proposal.candidate_fingerprint != inputs.candidate_fingerprint
            or proposal.source_execution_state_fingerprint != inputs.execution_state_fingerprint
            or proposal.old_micro_batch_size != inputs.current_micro_batch_size
            or proposal.old_gradient_accumulation != inputs.current_gradient_accumulation
            or proposal.world_size != inputs.world_size
            or proposal.effective_batch_size != inputs.effective_batch_size
        ):
            raise ProvenanceError("OOM proposal disagrees with its typed planning inputs")
        plan = self.recovery_plans.get(str(binding.plan_id))
        if (
            plan is None
            or plan != inputs.plan
            or plan.episode_id != binding.episode_id
            or plan.sequence != binding.plan_sequence
        ):
            raise ProvenanceError("OOM proposal does not describe the recorded RecoveryPlan")
        if not self.recovery_plans.is_effective_and_fresh(str(plan.id)):
            raise StaleRecoveryContextError("OOM RecoveryPlan is no longer open and fresh")
        context = plan.inputs.context
        run = self.aggregates.load_run(context.run_id)
        attempt = self.aggregates.load_attempt(context.target.id)
        if (
            run.status is not RunStatus.ACTIVE
            or attempt.status not in (RunAttemptStatus.FAILED, RunAttemptStatus.PREEMPTED)
            or attempt.run_id != run.id
            or run.candidate_fingerprint != proposal.candidate_fingerprint
            or execution_state_fingerprint_v1(FrozenDict(attempt.model_dump(mode="json")))
            != proposal.source_execution_state_fingerprint
        ):
            raise StaleRecoveryContextError("OOM source execution changed before Action proposal")

    # ---- PR-021: numerical recovery and training interventions ----------------------------

    def run_ancestry(
        self, run_id: RunId | str
    ) -> tuple[tuple[AttemptAncestry, ...], tuple[CheckpointAncestry, ...]]:
        """The immutable attempt/checkpoint facts that retained-trajectory lineage needs."""
        attempts = self.aggregates.attempts_for_run(str(run_id))
        ancestry = tuple(
            AttemptAncestry(
                attempt_id=attempt.id,
                attempt_number=attempt.attempt_number,
                restored_from=None if attempt.checkpoint_ref is None else attempt.checkpoint_ref.id,
            )
            for attempt in attempts
        )
        checkpoints = tuple(
            CheckpointAncestry(
                checkpoint_id=record.payload.checkpoint_ref.id,
                producer_attempt_id=RunAttemptId(record.context.target.id),
                embodied_application_ids=None
                if record.payload.state_manifest is None
                else record.payload.state_manifest.applied_intervention_application_ids,
            )
            for attempt in attempts
            for record in self.checkpoints.for_attempt(str(attempt.id))
        )
        return ancestry, checkpoints

    def events_for_run(self, run_id: RunId | str) -> tuple[DomainEvent, ...]:
        """The run's intervention history as recorded in the event log, in sequence order."""
        run = self.aggregates.load_run(str(run_id))
        attempts = {str(a.id) for a in self.aggregates.attempts_for_run(str(run.id))}
        return tuple(
            event
            for event in self.events.events_for_experiment(str(run.experiment_id))
            if event.aggregate_id == str(run.id) or event.aggregate_id in attempts
        )

    def get_run_realization(self, run_id: RunId | str) -> RunRealization:
        """Project the run's realization from the intervention and application tables.

        A projection, never stored: ``rebuild_run_realization`` over
        :meth:`events_for_run` must give the same value.
        """
        run = self.aggregates.load_run(str(run_id))
        attempts, checkpoints = self.run_ancestry(run.id)
        return project_run_realization(
            run,
            self.training_interventions.for_run(str(run.id)),
            self.intervention_applications.for_run(str(run.id)),
            attempts,
            checkpoints,
        )

    def _effective_learning_rate(self, run: Run, head: RunAttemptId) -> EffectiveLearningRate:
        """The base learning rate on *head*'s retained trajectory, from durable records."""
        attempts, checkpoints = self.run_ancestry(run.id)
        applications = self.intervention_applications.for_run(str(run.id))
        trajectory = retained_trajectory(head, attempts, checkpoints, applications)
        if trajectory is None:
            raise ProvenanceError(
                f"attempt {head} descends from a checkpoint that does not record which "
                f"intervention applications it embodies; its learning rate is unknown"
            )
        if trajectory.application_ids:
            latest = next(a for a in applications if a.id == trajectory.application_ids[-1])
            return EffectiveLearningRate(
                value=latest.applied_value,
                application_id=latest.id,
                intervention_id=latest.intervention_id,
            )
        node = self.aggregates.load_node(str(run.node_id))
        declared = node.candidate.candidate.training.optimization.learning_rate
        if declared is None or not math.isfinite(declared) or declared <= 0:
            raise ProvenanceError(f"run {run.id}'s candidate declares no learning rate")
        return EffectiveLearningRate(value=float(declared))

    def _prior_numerical_interventions(
        self, run_id: RunId | str
    ) -> tuple[PriorNumericalIntervention, ...]:
        priors = []
        for intervention in self.training_interventions.for_run(str(run_id)):
            binding = self.numerical_recovery_bindings.for_action(str(intervention.action_id))
            if binding is None:
                continue
            priors.append(
                PriorNumericalIntervention(
                    intervention_id=intervention.id,
                    action_id=intervention.action_id,
                    episode_id=binding.episode_id,
                    previous_learning_rate=binding.proposal.previous_learning_rate.value,
                    promised_learning_rate=intervention.mutation.learning_rate,
                )
            )
        return tuple(priors)

    def numerical_recovery_inputs(
        self, plan_id: str, policy: NumericalRecoveryPolicyV1 | None
    ) -> NumericalRecoveryInputsV1:
        """Derive the planner's typed inputs from durable records alone.

        The learning rate is the source attempt's *retained-trajectory* value:
        the latest recorded application it descends from, else the candidate's
        declared value. Never a proposed or approved-but-unapplied value.

        Raises:
            AggregateNotFoundError: If the plan does not exist.
            ProvenanceError: If the trajectory's learning rate cannot be established.
            pydantic.ValidationError: If the plan is not a training-attempt episode.
        """
        plan = self.recovery_plans.get(plan_id)
        if plan is None:
            raise AggregateNotFoundError("recovery plan", plan_id)
        context = plan.inputs.context
        run = self.aggregates.load_run(str(context.run_id))
        source = RunAttemptId(context.target.id)
        return NumericalRecoveryInputsV1(
            plan=plan,
            run_id=run.id,
            candidate_fingerprint=run.candidate_fingerprint,
            source_attempt_id=source,
            execution_state_fingerprint=plan.execution_state_fingerprint,
            current_learning_rate=self._effective_learning_rate(run, source),
            prior_interventions=self._prior_numerical_interventions(run.id),
            policy=policy,
        )

    def propose_numerical_recovery_action(
        self,
        inputs: NumericalRecoveryInputsV1,
        proposal: NumericalLRProposal,
        *,
        proposed_by: Actor,
        reason: str,
        policy: Any,
        capabilities: CapabilityDocument | None,
        action_id: ActionId | None = None,
        destinations: tuple[str, ...] = (),
    ) -> GovernedAction:
        """Govern one decision-bound ``ChangeLearningRate`` and bind it atomically.

        Goes through the ordinary Action path: validation, applicability and
        *policy* decide ALLOW, DENY or REQUIRE_APPROVAL. This records no
        intervention and no application; see :meth:`record_numerical_intervention`.
        """
        binding = NumericalRecoveryActionBinding.for_proposal(
            action_id or ActionId.generate(), proposal
        )
        return self.propose_action(
            proposal.action_spec,
            experiment_id=inputs.plan.inputs.context.experiment_id,
            proposed_by=proposed_by,
            reason=reason,
            policy=policy,
            capabilities=capabilities,
            action_id=binding.action_id,
            destinations=destinations,
            _bound=_BoundRecoveryProposal(
                action_id=binding.action_id,
                spec=binding.proposal.action_spec,
                replay=lambda: self._replay_numerical_recovery_action(binding),
                require_current=lambda: self._require_current_numerical_action(binding, inputs),
                insert=lambda: self.numerical_recovery_bindings._insert(binding),
            ),
        )

    def _replay_numerical_recovery_action(
        self, requested: NumericalRecoveryActionBinding
    ) -> GovernedAction | None:
        existing = self.numerical_recovery_bindings.for_plan(str(requested.plan_id))
        if existing is None:
            return None
        if existing.proposal != requested.proposal:
            raise IdempotencyConflictError(
                str(requested.plan_id), ("numerical proposal",), kind="recovery plan"
            )
        return self.governed_action(existing.action_id)

    def _require_current_numerical_action(
        self, binding: NumericalRecoveryActionBinding, inputs: NumericalRecoveryInputsV1
    ) -> None:
        proposal = binding.proposal
        if (
            proposal.input_fingerprint != inputs.input_fingerprint
            or proposal.run_id != inputs.run_id
            or proposal.candidate_fingerprint != inputs.candidate_fingerprint
            or proposal.source_attempt_id != inputs.source_attempt_id
            or proposal.source_execution_state_fingerprint != inputs.execution_state_fingerprint
            or proposal.previous_learning_rate != inputs.current_learning_rate
            or inputs.policy is None
            or proposal.policy_fingerprint != inputs.policy.policy_fingerprint
        ):
            raise ProvenanceError("numerical proposal disagrees with its typed planning inputs")
        self._require_fresh_numerical_decision(binding)
        if self.numerical_recovery_inputs(str(binding.plan_id), inputs.policy) != inputs:
            raise StaleRecoveryContextError("numerical recovery inputs changed before proposal")

    def _require_fresh_numerical_decision(self, binding: NumericalRecoveryActionBinding) -> None:
        """The bound plan is still the episode's open, fresh decision for this source."""
        proposal = binding.proposal
        plan = self.recovery_plans.get(str(binding.plan_id))
        if (
            plan is None
            or plan.episode_id != binding.episode_id
            or plan.sequence != binding.plan_sequence
            or proposal.trigger.incident_id not in plan.accepted_incident_ids
        ):
            raise ProvenanceError("numerical proposal does not describe the recorded RecoveryPlan")
        if not self.recovery_plans.is_effective_and_fresh(str(plan.id)):
            raise StaleRecoveryContextError("numerical RecoveryPlan is no longer open and fresh")
        run = self.aggregates.load_run(str(proposal.run_id))
        attempt = self.aggregates.load_attempt(str(proposal.source_attempt_id))
        if (
            run.status is not RunStatus.ACTIVE
            or attempt.run_id != run.id
            or run.candidate_fingerprint != proposal.candidate_fingerprint
            or execution_state_fingerprint_v1(FrozenDict(attempt.model_dump(mode="json")))
            != proposal.source_execution_state_fingerprint
        ):
            raise StaleRecoveryContextError("numerical source changed before the decision was used")
        if self._effective_learning_rate(run, attempt.id) != proposal.previous_learning_rate:
            raise StaleRecoveryContextError("the trajectory's learning rate changed meanwhile")

    def record_numerical_intervention(
        self,
        action_id: ActionId | str,
        *,
        actor: Actor,
        rationale: str,
        intervention_id: InterventionId | None = None,
        destinations: tuple[str, ...] = (),
    ) -> TrainingIntervention:
        """Record the intervention an authorized numerical-recovery Action decided.

        Provenance is copied from the bound proposal, never inferred:
        ``REACTIVE_POLICY`` origin, the exact nonfinite ``IncidentTrigger`` and
        the proposal's explicit replay policy. The plan must still be the
        episode's open, fresh decision. No application is recorded: that
        requires confirmation that the change actually took effect.
        """
        binding = self.numerical_recovery_bindings.for_action(str(action_id))
        if binding is None:
            raise ProvenanceError(f"Action {action_id} has no numerical recovery binding")
        proposal = binding.proposal
        spec = proposal.action_spec
        intervention = TrainingIntervention(
            id=intervention_id or InterventionId.generate(),
            run_id=proposal.run_id,
            action_id=binding.action_id,
            origin=InterventionOrigin.REACTIVE_POLICY,
            trigger=proposal.trigger,
            replay_policy=proposal.replay_policy,
            mutation=mutation_for_action(spec),
            rationale=rationale,
            evidence_refs=(str(proposal.trigger.incident_id),),
        )
        return self.record_training_intervention(
            intervention, actor=actor, destinations=destinations
        )

    def record_training_intervention(
        self,
        intervention: TrainingIntervention,
        *,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> TrainingIntervention:
        """Record the scientific decision of one authorized Action, with its event.

        Only an Action awaiting execution (VALIDATED + ALLOW or APPROVED +
        REQUIRE_APPROVAL) whose type is a scientific intervention on this same
        active Run qualifies, and its mutation must be exactly the Action's.
        One intervention per Action; an identical replay returns the original.

        Raises:
            InterventionNotAuthorizedError: If governance has not authorized it.
            ProvenanceError: If the intervention disagrees with its Action,
                Run, trigger or numerical decision.
            StaleRecoveryContextError: If a numerical decision is no longer fresh.
            IdempotencyConflictError: If the Action already has a different one.
        """
        intervention = TrainingIntervention.model_validate_json(intervention.model_dump_json())
        with self._write():
            existing = self.training_interventions.for_action(
                str(intervention.action_id)
            ) or self.training_interventions.get(str(intervention.id))
            if existing is not None:
                if existing.semantic_fingerprint() != intervention.semantic_fingerprint():
                    raise IdempotencyConflictError(
                        str(intervention.action_id), ("training_intervention",), kind="Action"
                    )
                return existing
            action = self.actions.get(str(intervention.action_id))
            if action is None:
                raise AggregateNotFoundError("Action", str(intervention.action_id))
            decision = self.policy.for_action(str(action.id))
            if not awaits_execution(action, decision):
                raise InterventionNotAuthorizedError(
                    f"Action {action.id} is {action.status.value}; only an authorized "
                    f"Action becomes a training intervention"
                )
            spec = spec_of(action)
            if (
                action_descriptor(action.type).mutation_class
                is not MutationClass.SCIENTIFIC_INTERVENTION
                or not isinstance(spec, ChangeLearningRate)
                or mutation_for_action(spec) != intervention.mutation
            ):
                raise ProvenanceError(
                    f"Action {action.id} ({action.type}) does not decide this scientific mutation"
                )
            run = self.aggregates.load_run(str(intervention.run_id))
            if (
                action.target != ActionTarget(kind="run", id=str(run.id))
                or action.experiment_id != run.experiment_id
            ):
                raise ProvenanceError(f"Action {action.id} does not target run {run.id}")
            if run.status is not RunStatus.ACTIVE:
                raise ProvenanceError(f"run {run.id} is {run.status.value}, not active")
            self._require_intervention_provenance(intervention, action, run)

            sequence = self._write_sequenced_event(
                DomainEvent(
                    id=EventId.generate(),
                    experiment_id=str(run.experiment_id),
                    aggregate_type="Run",
                    aggregate_id=str(run.id),
                    aggregate_revision=run.revision,
                    event_type="TrainingInterventionRecorded",
                    actor=actor,
                    payload=FrozenDict(
                        {
                            "status": run.status.value,
                            "intervention_id": str(intervention.id),
                            "action_id": str(action.id),
                            "training_intervention": intervention.model_dump(mode="json"),
                        }
                    ),
                ),
                destinations,
            )
            self.training_interventions._insert(
                intervention, experiment_id=str(run.experiment_id), event_sequence=sequence
            )
            return intervention

    def _require_intervention_provenance(
        self, intervention: TrainingIntervention, action: Action, run: Run
    ) -> None:
        binding = self.numerical_recovery_bindings.for_action(str(action.id))
        trigger = intervention.trigger
        if binding is not None:
            proposal = binding.proposal
            if (
                intervention.origin is not InterventionOrigin.REACTIVE_POLICY
                or trigger != proposal.trigger
                or intervention.replay_policy is not proposal.replay_policy
                or intervention.run_id != proposal.run_id
            ):
                raise ProvenanceError(
                    "a numerical-recovery intervention copies its bound decision's provenance"
                )
            self._require_fresh_numerical_decision(binding)
            return
        if intervention.origin is InterventionOrigin.REACTIVE_POLICY:
            raise ProvenanceError("a policy-origin intervention requires a bound recovery decision")
        if intervention.origin is InterventionOrigin.SCHEDULED:
            raise ProvenanceError("scheduled interventions are not executed in this release")
        proposer = {
            InterventionOrigin.REACTIVE_HUMAN: "human",
            InterventionOrigin.REACTIVE_AGENT: "llm_agent",
        }[intervention.origin]
        if action.proposed_by.type != proposer:
            raise ProvenanceError(
                f"a {intervention.origin.value} intervention is proposed by a {proposer}, "
                f"not {action.proposed_by.type}"
            )
        if isinstance(trigger, IncidentTrigger) and trigger.incident_id is not None:
            incident = self.incidents.get(str(trigger.incident_id))
            if (
                incident is None
                or incident.context.run_id != run.id
                or incident.category is not trigger.incident_category
            ):
                raise ProvenanceError("the triggering incident is not this run's")

    def _require_directive(
        self,
        intervention: TrainingIntervention,
        attempt: RunAttempt,
        application_id: InterventionApplicationId,
        observed_previous_value: float,
    ) -> InterventionDirective | None:
        """An application made under a directive must be exactly what was directed.

        A numerical-recovery intervention takes effect only through a directive,
        which pins it to its episode's checkpoint-backed successor (first effect)
        or to a later restore that dropped every earlier application (replay).
        Directives apply in ordinal order. Other interventions without a
        directive keep the generic rules.
        """
        directive = self.intervention_directives.get(str(application_id))
        if directive is None:
            if self.numerical_recovery_bindings.for_action(str(intervention.action_id)):
                raise ProvenanceError(
                    f"numerical intervention {intervention.id} takes effect only through a "
                    f"directive recorded with its checkpoint-backed successor"
                )
            return None
        if directive.attempt_id != attempt.id or directive.intervention_id != intervention.id:
            raise ProvenanceError(f"application {application_id} does not match its directive")
        if observed_previous_value != directive.expected_previous_value:
            raise ProvenanceError(
                f"the confirmed previous learning rate {observed_previous_value!r} is not the "
                f"directive's expected {directive.expected_previous_value!r}"
            )
        earlier = [
            item
            for item in self.intervention_directives.for_attempt(str(attempt.id))
            if item.ordinal < directive.ordinal
            and self.intervention_applications.get(str(item.application_id)) is None
        ]
        if earlier:
            raise ProvenanceError("directives take effect in their recorded order")
        return directive

    def record_intervention_application(
        self,
        intervention_id: InterventionId | str,
        *,
        application_id: InterventionApplicationId,
        attempt_id: RunAttemptId | str,
        position: TrainingPosition,
        observed_previous_value: float,
        applied_value: float,
        actor: Actor,
        trigger_evaluation: TriggerEvaluation | None = None,
        destinations: tuple[str, ...] = (),
        telemetry_position: tuple[int, int] | None = None,
    ) -> InterventionApplication:
        """Append one confirmed effect of an intervention, ordered by its event sequence.

        Only confirmation that the mutation actually took effect may call
        this: approval, an intervention row, or an intended runtime request is
        not an application.

        Provenance is derived, never trusted. ``previous_value`` is the base
        learning rate on *attempt_id*'s retained trajectory immediately before
        this application; the effect confirmation's *observed_previous_value*
        must equal it. ``checkpoint_ancestor`` is the attempt's own restore
        checkpoint (``None`` for a fresh start), which must be a recorded
        checkpoint of the same run. An identical replay of *application_id*
        returns the original with its original sequence.

        *telemetry_position* is the ``(generation, sequence)`` of the
        ``InterventionApplied`` event that confirmed the effect. The attempt's
        cursor advances to it in the same commit -- on an identical replay too,
        so a confirmation recorded before a crash is not redelivered forever.

        Raises:
            ProvenanceError: If the attempt, values or ancestry disagree with
                the durable record, or the trajectory's rate is unknown.
            IdempotencyConflictError: If *application_id* was recorded differently.
        """
        with self._write():
            intervention = self.training_interventions.get(str(intervention_id))
            if intervention is None:
                raise AggregateNotFoundError("training intervention", str(intervention_id))
            attempt = self.aggregates.load_attempt(str(attempt_id))
            if attempt.run_id != intervention.run_id:
                raise ProvenanceError(
                    f"attempt {attempt.id} does not belong to run {intervention.run_id}"
                )
            existing = self.intervention_applications.get(str(application_id))
            if existing is not None:
                if (
                    existing.intervention_id != intervention.id
                    or existing.attempt_id != attempt.id
                    or existing.position != position
                    or existing.trigger_evaluation != trigger_evaluation
                    or existing.applied_value != applied_value
                    or existing.previous_value != observed_previous_value
                ):
                    raise IdempotencyConflictError(
                        str(application_id), ("intervention_application",), kind="application"
                    )
                if telemetry_position is not None:
                    self.aggregates._advance_telemetry(str(attempt.id), telemetry_position)
                return existing
            if applied_value != intervention.mutation.learning_rate:
                raise ProvenanceError("the applied value is not the intervention's decided value")
            directive = self._require_directive(
                intervention, attempt, application_id, observed_previous_value
            )
            run = self.aggregates.load_run(str(intervention.run_id))
            previous = self._effective_learning_rate(run, attempt.id)
            if observed_previous_value != previous.value:
                raise ProvenanceError(
                    f"the confirmed previous learning rate {observed_previous_value!r} is not "
                    f"attempt {attempt.id}'s retained-trajectory rate {previous.value!r}"
                )
            ancestor = attempt.checkpoint_ref
            if ancestor is not None:
                record = self.checkpoints.get(str(ancestor.id))
                if record is None or record.context.run_id != str(run.id):
                    raise ProvenanceError(
                        f"attempt {attempt.id} restores from a checkpoint this run never recorded"
                    )
            preview = InterventionApplication(
                id=application_id,
                intervention_id=intervention.id,
                attempt_id=attempt.id,
                event_sequence=1,
                position=position,
                trigger_evaluation=trigger_evaluation,
                previous_value=previous.value,
                applied_value=applied_value,
                checkpoint_ancestor=ancestor,
            )
            sequence = self._write_sequenced_event(
                DomainEvent(
                    id=EventId.generate(),
                    experiment_id=str(run.experiment_id),
                    aggregate_type="RunAttempt",
                    aggregate_id=str(attempt.id),
                    aggregate_revision=attempt.revision,
                    event_type="InterventionApplied",
                    actor=actor,
                    payload=FrozenDict(
                        {
                            "status": attempt.status.value,
                            "application_id": str(application_id),
                            "intervention_id": str(intervention.id),
                            "run_id": str(run.id),
                            "intervention_application": preview.model_dump(
                                mode="json", exclude={"event_sequence"}
                            ),
                        }
                    ),
                ),
                destinations,
            )
            application = preview.model_copy(update={"event_sequence": sequence})
            self.intervention_applications._insert(application, run_id=str(run.id))
            if directive is not None:
                self._settle_numerical_action_on_effect(
                    intervention, directive, actor, destinations
                )
            if telemetry_position is not None:
                self.aggregates._advance_telemetry(str(attempt.id), telemetry_position)
            return application

    # ---- PR-021 executor: successor directives and numerical execution ----------------------

    def plan_successor_interventions(
        self,
        run_id: RunId | str,
        restore: CheckpointRef | None,
        *,
        initial: TrainingIntervention | None = None,
    ) -> ReplayPlan:
        """Plan which interventions a successor restoring *restore* must apply.

        Pure planning (:func:`plan_intervention_directives`) over durable
        records: every intervention and application on the run, the declared
        learning rate, and the restore checkpoint's embodied applications.

        Raises:
            InterventionReplayError: When the restored lineage is unknown or
                ambiguous, or a replay this release cannot perform is needed.
            ProvenanceError: If *restore* is not a recorded checkpoint of the run.
        """
        run = self.aggregates.load_run(str(run_id))
        embodied: tuple[str, ...] | None = None
        if restore is not None:
            record = self.checkpoints.get(str(restore.id))
            if record is None or record.context.run_id != str(run.id):
                raise ProvenanceError(f"checkpoint {restore.id} is not a recorded one of {run.id}")
            manifest = record.payload.state_manifest
            embodied = None if manifest is None else manifest.applied_intervention_application_ids
        node = self.aggregates.load_node(str(run.node_id))
        declared = node.candidate.candidate.training.optimization.learning_rate
        interventions = self.training_interventions.for_run(str(run.id))
        applications = self.intervention_applications.for_run(str(run.id))
        if declared is None:
            if interventions or initial is not None:
                raise ProvenanceError(f"run {run.id}'s candidate declares no learning rate")
            declared = 1.0  # unused: nothing to plan without interventions
        return plan_intervention_directives(
            declared_learning_rate=float(declared),
            interventions=interventions,
            applications=applications,
            restored=restore is not None,
            embodied_application_ids=embodied,
            initial=initial,
        )

    def _require_and_insert_directives(
        self,
        successor: RunAttempt,
        directives: tuple[InterventionDirective, ...],
        *,
        initial: TrainingIntervention | None,
    ) -> None:
        """Re-derive the successor's replay plan under the lock; refuse any disagreement."""
        planned = self.plan_successor_interventions(
            successor.run_id, successor.checkpoint_ref, initial=initial
        ).directives
        if len(planned) != len(directives) or any(
            directive.attempt_id != successor.id
            or directive.ordinal != ordinal
            or (
                directive.intervention_id,
                directive.kind,
                directive.mutation,
                directive.expected_previous_value,
            )
            != (plan.intervention_id, plan.kind, plan.mutation, plan.expected_previous_value)
            for ordinal, (directive, plan) in enumerate(zip(directives, planned, strict=False))
        ):
            raise StaleRecoveryContextError(
                "successor intervention directives disagree with the restored lineage"
            )
        for directive in directives:
            self.intervention_directives._insert(directive, run_id=str(successor.run_id))

    def _record_numerical_recovery_execution(
        self,
        action_id: ActionId,
        successor: RunAttempt,
        checkpoint: RecoveryCheckpointReport,
        directives: tuple[InterventionDirective, ...],
        *,
        request_digest: str,
        actor: Actor,
        capabilities: CapabilityDocument | None = None,
        operation_id: OperationId | None = None,
        destinations: tuple[str, ...] = (),
    ) -> tuple[RunAttempt, RuntimeOperation, RecoveryExecutionReceipt]:
        """Commit one numerical successor, its directives, submit intent and receipt.

        Like :meth:`_record_oom_recovery_execution`, a trusted executor has
        already validated checkpoint bytes and resolved the plans; this method
        rechecks database authority and the restored intervention lineage under
        the write lock. It performs no checkpoint or runtime I/O. The Action
        moves to ``EXECUTING``; it succeeds only when the worker confirms the
        intervention's first application.
        """
        _require_pristine(successor, RunAttemptStatus.CREATED)
        operation = RuntimeOperation(
            id=operation_id or OperationId.generate(),
            target=RuntimeOperationTarget(kind="training-attempt", id=str(successor.id)),
            type="submit",
            request_digest=request_digest,
            caused_by_action_id=action_id,
        )
        with self._write():
            binding = self.numerical_recovery_bindings.for_action(str(action_id))
            if binding is None:
                raise ProvenanceError("numerical Action has no recovery decision binding")
            intervention = self.training_interventions.for_action(str(action_id))
            if intervention is None:
                raise ProvenanceError("numerical Action has no recorded TrainingIntervention")
            prior = self.numerical_recovery_executions.executed_for_episode(str(binding.episode_id))
            if prior is not None:
                if (
                    prior.action_id != action_id
                    or prior.successor_attempt_id != successor.id
                    or prior.runtime_operation_id != operation.id
                    or prior.checkpoint_ref != checkpoint.checkpoint_ref
                    or self.intervention_directives.for_attempt(str(successor.id)) != directives
                ):
                    raise IdempotencyConflictError(
                        str(binding.episode_id), ("successor execution",), kind="recovery episode"
                    )
                recorded_attempt = self.aggregates.load_attempt(str(successor.id))
                recorded_operation = self.operations.get(str(operation.id))
                assert recorded_operation is not None
                self.operations._assert_same_request(recorded_operation, operation)
                _assert_same_attempt(recorded_attempt, successor)
                return recorded_attempt, recorded_operation, prior

            self._require_fresh_numerical_decision(binding)
            episode = self.recovery_episodes.get(str(binding.episode_id))
            plan = self.recovery_plans.get(str(binding.plan_id))
            assert episode is not None and plan is not None
            action = self.actions.get(str(action_id))
            decision = self.policy.for_action(str(action_id))
            spec = binding.proposal.action_spec
            if (
                action is None
                or spec_of(action) != spec
                or not awaits_execution(action, decision)
                or decision is None
            ):
                raise ProvenanceError("numerical Action is not authorized for its bound plan")
            current_policy_context = self.policy_context(
                spec,
                experiment_id=episode.context.experiment_id,
                proposed_by=action.proposed_by,
                capabilities=capabilities,
            )
            if capabilities != decision.context.capabilities or applicability_problems(
                spec, current_policy_context
            ):
                raise StalePolicyContextError(
                    "numerical Action capabilities or applicability changed"
                )
            inputs = self._recovery_snapshot(episode)
            source = self.aggregates.load_attempt(episode.context.target.id)
            if inputs.successor_exists or source.status not in (
                RunAttemptStatus.FAILED,
                RunAttemptStatus.PREEMPTED,
            ):
                raise StaleRecoveryContextError("numerical source attempt changed before execution")
            limits = episode.request.limits
            if (
                inputs.actual_attempt_count + inputs.pending_other_run_reservations + 1
                > limits.max_attempts_per_run
                or inputs.experiment_recovery_usage_excluding_target + 1
                > limits.max_recoveries_per_experiment
            ):
                raise StaleRecoveryContextError(
                    "numerical recovery limits no longer admit a successor"
                )
            record = self.checkpoints.get(str(checkpoint.checkpoint_ref.id))
            producer = (
                None if record is None else self.aggregates.load_attempt(record.context.target.id)
            )
            if (
                record is None
                or producer is None
                or checkpoint
                != RecoveryCheckpointReport.from_record(record, producer.attempt_number)
                or checkpoint_report_problem(checkpoint, inputs) is not None
            ):
                raise StaleRecoveryContextError("validated numerical checkpoint report changed")
            _require_numerical_successor_lineage(source, successor, action_id, checkpoint)
            experiment_id = str(episode.context.experiment_id)
            self._require_budget(experiment_id, new_run=False)
            self._require_capacity(experiment_id)
            self.aggregates._insert_attempt(successor)
            self._require_and_insert_directives(successor, directives, initial=intervention)
            stored_operation = self.operations._insert(operation)
            self._emit(successor, "RunAttemptCreated", experiment_id, actor, destinations)
            self._ledger(
                experiment_id,
                BudgetDimension.PARALLEL_RUNS,
                LedgerEntryKind.RESERVE,
                Decimal(1),
                BudgetSubjectKind.TRAINING_ATTEMPT,
                str(successor.id),
                actor,
                destinations,
            )
            self._emit_operation(
                stored_operation, "RuntimeOperationIntended", experiment_id, actor, destinations
            )
            executing = action.with_status(ActionStatus.EXECUTING)
            self.actions._update(executing)
            self._emit_action(executing, "ActionExecuting", actor, destinations)
            receipt = RecoveryExecutionReceipt(
                episode_id=episode.id,
                plan_id=plan.id,
                plan_sequence=plan.sequence,
                action_id=action_id,
                outcome=RecoveryExecutionOutcome.EXECUTED,
                successor_attempt_id=successor.id,
                runtime_operation_id=stored_operation.id,
                checkpoint_ref=checkpoint.checkpoint_ref,
                created_by=actor,
            )
            self.numerical_recovery_executions._insert(
                receipt, intervention_id=str(intervention.id)
            )
            self._emit(
                source,
                "RecoveryExecutionRecorded",
                experiment_id,
                actor,
                destinations,
                extra={
                    "episode_id": str(episode.id),
                    "plan_id": str(plan.id),
                    "plan_sequence": plan.sequence,
                    "action_id": str(action_id),
                    "intervention_id": str(intervention.id),
                    "successor_attempt_id": str(successor.id),
                    "runtime_operation_id": str(stored_operation.id),
                    "checkpoint_id": str(checkpoint.checkpoint_ref.id),
                    "directives": [d.model_dump(mode="json") for d in directives],
                },
            )
            return successor, stored_operation, receipt

    def abandon_numerical_recovery_action(
        self,
        action_id: ActionId,
        *,
        actor: Actor,
        reason: str,
        destinations: tuple[str, ...] = (),
    ) -> RecoveryExecutionReceipt:
        """Record a definitive refusal of numerical recovery and fail the Run atomically.

        For a rejected Action, or one no eligible checkpoint can carry. Mirrors
        :meth:`abandon_oom_recovery_action`. Creates no successor or operation.
        """
        with self._write():
            binding = self.numerical_recovery_bindings.for_action(str(action_id))
            if binding is None:
                raise ProvenanceError("numerical Action has no recovery decision binding")
            existing = self.numerical_recovery_executions.for_episode(str(binding.episode_id))
            same = next((r for r in existing if r.action_id == action_id), None)
            if same is not None:
                if same.outcome is RecoveryExecutionOutcome.ABANDONED:
                    return same
                raise IdempotencyConflictError(
                    str(binding.episode_id), ("recovery execution",), kind="recovery episode"
                )
            if any(r.outcome is RecoveryExecutionOutcome.EXECUTED for r in existing):
                raise IdempotencyConflictError(
                    str(binding.episode_id), ("recovery execution",), kind="recovery episode"
                )
            episode = self.recovery_episodes.get(str(binding.episode_id))
            plan = self.recovery_plans.get(str(binding.plan_id))
            action = self.actions.get(str(action_id))
            if (
                episode is None
                or plan is None
                or not self.recovery_plans.is_effective_and_fresh(str(plan.id))
                or action is None
                or action.status
                not in (
                    ActionStatus.VALIDATED,
                    ActionStatus.APPROVAL_PENDING,
                    ActionStatus.APPROVED,
                    ActionStatus.REJECTED,
                )
            ):
                raise StaleRecoveryContextError(
                    "numerical Action no longer governs the open episode"
                )
            run = self.aggregates.load_run(episode.context.run_id)
            if run.status is not RunStatus.ACTIVE:
                raise StaleRecoveryContextError("numerical Run has already settled")
            if action.status is not ActionStatus.REJECTED:
                rejected = action.with_status(ActionStatus.REJECTED)
                self.actions._update(rejected)
                self._emit_action(rejected, "ActionRejected", actor, destinations)
            receipt = RecoveryExecutionReceipt(
                episode_id=episode.id,
                plan_id=plan.id,
                plan_sequence=plan.sequence,
                action_id=action_id,
                outcome=RecoveryExecutionOutcome.ABANDONED,
                created_by=actor,
            )
            self.numerical_recovery_executions._insert(receipt, intervention_id=None)
            failed = run.with_status(RunStatus.FAILED)
            self.aggregates._update_run(failed)
            self._settle_budget(failed, actor, destinations)
            self._emit(
                failed, "RunStatusChanged", str(episode.context.experiment_id), actor, destinations
            )
            self._emit(
                self.aggregates.load_attempt(episode.context.target.id),
                "RecoveryExecutionRecorded",
                str(episode.context.experiment_id),
                actor,
                destinations,
                extra={
                    "episode_id": str(episode.id),
                    "plan_id": str(plan.id),
                    "plan_sequence": plan.sequence,
                    "action_id": str(action_id),
                    "outcome": "ABANDONED",
                    "reason": reason,
                },
            )
            return receipt

    def _settle_numerical_action_on_effect(
        self,
        intervention: TrainingIntervention,
        directive: InterventionDirective,
        actor: Actor,
        destinations: tuple[str, ...],
    ) -> None:
        """The first confirmed effect completes the governing Action."""
        if directive.kind is not InterventionDirectiveKind.INITIAL:
            return
        action = self.actions.get(str(intervention.action_id))
        if action is not None and action.status is ActionStatus.EXECUTING:
            settled = action.with_status(ActionStatus.SUCCEEDED, outcome=ActionOutcome.APPLIED)
            self.actions._update(settled)
            self._emit_action(settled, "ActionApplied", actor, destinations)

    def _fail_unconfirmed_numerical_action(
        self, attempt: RunAttempt, actor: Actor, destinations: tuple[str, ...]
    ) -> None:
        """A successor that ended without confirming its first effect fails the Action.

        The intervention stays as the immutable decision; it simply never took
        effect, so nothing will re-apply it.
        """
        if not attempt.is_terminal:
            return
        receipt = self.numerical_recovery_executions.for_successor(str(attempt.id))
        if receipt is None or receipt.outcome is not RecoveryExecutionOutcome.EXECUTED:
            return
        action = self.actions.get(str(receipt.action_id))
        if action is None or action.status is not ActionStatus.EXECUTING:
            return
        initial = next(
            (
                directive
                for directive in self.intervention_directives.for_attempt(str(attempt.id))
                if directive.kind is InterventionDirectiveKind.INITIAL
            ),
            None,
        )
        if initial is not None and self.intervention_applications.get(str(initial.application_id)):
            return
        failed = action.with_status(ActionStatus.FAILED)
        self.actions._update(failed)
        self._emit_action(
            failed,
            "ActionFailed",
            actor,
            destinations,
            extra={"reason": "successor ended before confirming the intervention's effect"},
        )

    def _record_oom_recovery_execution(
        self,
        action_id: ActionId,
        successor: RunAttempt,
        checkpoint: RecoveryCheckpointReport,
        *,
        request_digest: str,
        actor: Actor,
        capabilities: CapabilityDocument | None = None,
        operation_id: OperationId | None = None,
        destinations: tuple[str, ...] = (),
        directives: tuple[InterventionDirective, ...] = (),
    ) -> tuple[RunAttempt, RuntimeOperation, RecoveryExecutionReceipt]:
        """Commit one governed OOM successor, submit intent and receipt atomically.

        *directives* re-apply interventions that the restore dropped from the
        retained trajectory (ADR-011 ``REAPPLY_AFTER_ROLLBACK``); they are
        re-derived under the lock. A run without interventions has none.

        A trusted executor must validate checkpoint bytes and resolve the
        source/successor training plans before this call. Under the write lock,
        this method rechecks database authority and the validated report's
        identity. It performs no checkpoint or runtime I/O.
        """
        _require_pristine(successor, RunAttemptStatus.CREATED)
        operation = RuntimeOperation(
            id=operation_id or OperationId.generate(),
            target=RuntimeOperationTarget(kind="training-attempt", id=str(successor.id)),
            type="submit",
            request_digest=request_digest,
            caused_by_action_id=action_id,
        )
        with self._write():
            binding = self.recovery_action_bindings.for_action(str(action_id))
            if binding is None:
                raise ProvenanceError("OOM Action has no recovery decision binding")
            prior_receipt = self.recovery_execution_receipts.executed_for_episode(
                str(binding.episode_id)
            )
            if prior_receipt is not None:
                if (
                    prior_receipt.action_id != action_id
                    or prior_receipt.successor_attempt_id != successor.id
                    or prior_receipt.runtime_operation_id != operation.id
                    or prior_receipt.checkpoint_ref != checkpoint.checkpoint_ref
                ):
                    raise IdempotencyConflictError(
                        str(binding.episode_id), ("successor execution",), kind="recovery episode"
                    )
                recorded_attempt = self.aggregates.load_attempt(str(successor.id))
                recorded_operation = self.operations.get(str(operation.id))
                assert recorded_operation is not None
                self.operations._assert_same_request(recorded_operation, operation)
                _assert_same_attempt(recorded_attempt, successor)
                return recorded_attempt, recorded_operation, prior_receipt

            episode = self.recovery_episodes.get(str(binding.episode_id))
            plan = self.recovery_plans.get(str(binding.plan_id))
            if (
                episode is None
                or plan is None
                or plan.episode_id != episode.id
                or plan.sequence != binding.plan_sequence
                or not self.recovery_plans.is_effective_and_fresh(str(plan.id))
            ):
                raise StaleRecoveryContextError("OOM recovery decision is no longer open and fresh")
            action = self.actions.get(str(action_id))
            decision = self.policy.for_action(str(action_id))
            if (
                action is None
                or decision is None
                or spec_of(action) != binding.proposal.action_spec
                or action.experiment_id != episode.context.experiment_id
                or action.target.kind != "run"
                or action.target.id != episode.context.run_id
                or not (
                    (
                        decision.verdict is PolicyVerdict.ALLOW
                        and action.status is ActionStatus.VALIDATED
                    )
                    or (
                        decision.verdict is PolicyVerdict.REQUIRE_APPROVAL
                        and action.status is ActionStatus.APPROVED
                    )
                )
            ):
                raise ProvenanceError("OOM Action is not authorized for its bound recovery plan")
            current_policy_context = self.policy_context(
                binding.proposal.action_spec,
                experiment_id=episode.context.experiment_id,
                proposed_by=action.proposed_by,
                capabilities=capabilities,
            )
            if capabilities != decision.context.capabilities or applicability_problems(
                binding.proposal.action_spec, current_policy_context
            ):
                raise StalePolicyContextError("OOM Action capabilities or applicability changed")
            inputs = self._recovery_snapshot(episode)
            source = self.aggregates.load_attempt(episode.context.target.id)
            run = self.aggregates.load_run(episode.context.run_id)
            if (
                inputs.successor_exists
                or run.status is not RunStatus.ACTIVE
                or source.status not in (RunAttemptStatus.FAILED, RunAttemptStatus.PREEMPTED)
                or inputs.execution_state_fingerprint != binding.source_execution_state_fingerprint
                or inputs.candidate_fingerprint != binding.proposal.candidate_fingerprint
            ):
                raise StaleRecoveryContextError(
                    "OOM source attempt or Run changed before execution"
                )
            limits = episode.request.limits
            if (
                inputs.actual_attempt_count + inputs.pending_other_run_reservations + 1
                > limits.max_attempts_per_run
                or inputs.experiment_recovery_usage_excluding_target + 1
                > limits.max_recoveries_per_experiment
            ):
                raise StaleRecoveryContextError("OOM recovery limits no longer admit a successor")
            record = self.checkpoints.get(str(checkpoint.checkpoint_ref.id))
            producer = (
                None if record is None else self.aggregates.load_attempt(record.context.target.id)
            )
            if (
                record is None
                or producer is None
                or checkpoint
                != RecoveryCheckpointReport.from_record(record, producer.attempt_number)
                or checkpoint_report_problem(checkpoint, inputs) is not None
            ):
                raise StaleRecoveryContextError("validated OOM checkpoint report changed")
            _require_oom_successor_lineage(
                source, successor, binding.proposal, action_id, checkpoint
            )
            experiment_id = str(episode.context.experiment_id)
            self._require_budget(experiment_id, new_run=False)
            self._require_capacity(experiment_id)
            self.aggregates._insert_attempt(successor)
            self._require_and_insert_directives(successor, directives, initial=None)
            stored_operation = self.operations._insert(operation)
            self._emit(successor, "RunAttemptCreated", experiment_id, actor, destinations)
            self._ledger(
                experiment_id,
                BudgetDimension.PARALLEL_RUNS,
                LedgerEntryKind.RESERVE,
                Decimal(1),
                BudgetSubjectKind.TRAINING_ATTEMPT,
                str(successor.id),
                actor,
                destinations,
            )
            self._emit_operation(
                stored_operation, "RuntimeOperationIntended", experiment_id, actor, destinations
            )
            executing = action.with_status(ActionStatus.EXECUTING)
            self.actions._update(executing)
            self._emit_action(executing, "ActionExecuting", actor, destinations)
            receipt = RecoveryExecutionReceipt(
                episode_id=episode.id,
                plan_id=plan.id,
                plan_sequence=plan.sequence,
                action_id=action_id,
                outcome=RecoveryExecutionOutcome.EXECUTED,
                successor_attempt_id=successor.id,
                runtime_operation_id=stored_operation.id,
                checkpoint_ref=checkpoint.checkpoint_ref,
                created_by=actor,
            )
            self.recovery_execution_receipts._insert(receipt)
            self._emit(
                source,
                "RecoveryExecutionRecorded",
                experiment_id,
                actor,
                destinations,
                extra={
                    "episode_id": str(episode.id),
                    "plan_id": str(plan.id),
                    "plan_sequence": plan.sequence,
                    "action_id": str(action_id),
                    "successor_attempt_id": str(successor.id),
                    "runtime_operation_id": str(stored_operation.id),
                    "checkpoint_id": str(checkpoint.checkpoint_ref.id),
                },
            )
            return successor, stored_operation, receipt

    def abandon_oom_recovery_action(
        self,
        action_id: ActionId,
        *,
        actor: Actor,
        reason: str,
        destinations: tuple[str, ...] = (),
    ) -> RecoveryExecutionReceipt:
        """Record a definitive refusal and fail the unresolved Run atomically.

        A stale plan cannot terminalize a Run whose current episode decision
        may already have changed. This path creates no successor or operation.
        """
        with self._write():
            binding = self.recovery_action_bindings.for_action(str(action_id))
            if binding is None:
                raise ProvenanceError("OOM Action has no recovery decision binding")
            existing = self.recovery_execution_receipts.for_episode(str(binding.episode_id))
            same_action = next(
                (receipt for receipt in existing if receipt.action_id == action_id), None
            )
            if same_action is not None:
                if same_action.outcome is RecoveryExecutionOutcome.ABANDONED:
                    return same_action
                raise IdempotencyConflictError(
                    str(binding.episode_id), ("recovery execution",), kind="recovery episode"
                )
            if any(receipt.outcome is RecoveryExecutionOutcome.EXECUTED for receipt in existing):
                raise IdempotencyConflictError(
                    str(binding.episode_id), ("recovery execution",), kind="recovery episode"
                )
            episode = self.recovery_episodes.get(str(binding.episode_id))
            plan = self.recovery_plans.get(str(binding.plan_id))
            action = self.actions.get(str(action_id))
            if (
                episode is None
                or plan is None
                or plan.sequence != binding.plan_sequence
                or not self.recovery_plans.is_effective_and_fresh(str(plan.id))
                or action is None
                or action.status
                not in (
                    ActionStatus.VALIDATED,
                    ActionStatus.APPROVAL_PENDING,
                    ActionStatus.APPROVED,
                    ActionStatus.REJECTED,
                )
            ):
                raise StaleRecoveryContextError("OOM Action no longer governs the open episode")
            run = self.aggregates.load_run(episode.context.run_id)
            if run.status is not RunStatus.ACTIVE:
                raise StaleRecoveryContextError("OOM Run has already settled")
            if action.status is not ActionStatus.REJECTED:
                rejected = action.with_status(ActionStatus.REJECTED)
                self.actions._update(rejected)
                self._emit_action(rejected, "ActionRejected", actor, destinations)
            receipt = RecoveryExecutionReceipt(
                episode_id=episode.id,
                plan_id=plan.id,
                plan_sequence=plan.sequence,
                action_id=action_id,
                outcome=RecoveryExecutionOutcome.ABANDONED,
                created_by=actor,
            )
            self.recovery_execution_receipts._insert(receipt)
            failed = run.with_status(RunStatus.FAILED)
            self.aggregates._update_run(failed)
            self._settle_budget(failed, actor, destinations)
            self._emit(
                failed,
                "RunStatusChanged",
                str(episode.context.experiment_id),
                actor,
                destinations,
            )
            self._emit(
                self.aggregates.load_attempt(episode.context.target.id),
                "RecoveryExecutionRecorded",
                str(episode.context.experiment_id),
                actor,
                destinations,
                extra={
                    "episode_id": str(episode.id),
                    "plan_id": str(plan.id),
                    "plan_sequence": plan.sequence,
                    "action_id": str(action_id),
                    "outcome": "ABANDONED",
                    "reason": reason,
                },
            )
            return receipt

    def supersede_stale_oom_actions(
        self,
        episode_id: RecoveryEpisodeId,
        *,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> tuple[RecoveryExecutionReceipt, ...]:
        """Retire unconsumed Actions bound to obsolete episode revisions."""
        with self._write():
            current = self.recovery_plans.effective_for_episode(str(episode_id))
            if current is None or not self.recovery_plans.is_effective_and_fresh(str(current.id)):
                raise StaleRecoveryContextError("OOM episode has no fresh effective decision")
            episode = self.recovery_episodes.get(str(episode_id))
            assert episode is not None
            recorded: list[RecoveryExecutionReceipt] = []
            for binding in self.recovery_action_bindings.for_episode(str(episode_id)):
                if binding.plan_id == current.id:
                    continue
                prior = self.recovery_execution_receipts.for_episode(str(episode_id))
                if any(receipt.action_id == binding.action_id for receipt in prior):
                    continue
                action = self.actions.get(str(binding.action_id))
                assert action is not None
                if action.status in (
                    ActionStatus.VALIDATED,
                    ActionStatus.APPROVAL_PENDING,
                    ActionStatus.APPROVED,
                ):
                    rejected = action.with_status(ActionStatus.REJECTED)
                    self.actions._update(rejected)
                    self._emit_action(rejected, "ActionSuperseded", actor, destinations)
                elif action.status is not ActionStatus.REJECTED:
                    raise StaleRecoveryContextError(
                        "an executing OOM Action cannot be superseded by a new plan"
                    )
                receipt = RecoveryExecutionReceipt(
                    episode_id=episode.id,
                    plan_id=binding.plan_id,
                    plan_sequence=binding.plan_sequence,
                    action_id=binding.action_id,
                    outcome=RecoveryExecutionOutcome.SUPERSEDED,
                    created_by=actor,
                )
                self.recovery_execution_receipts._insert(receipt)
                self._emit(
                    self.aggregates.load_attempt(episode.context.target.id),
                    "RecoveryExecutionRecorded",
                    str(episode.context.experiment_id),
                    actor,
                    destinations,
                    extra={
                        "episode_id": str(episode.id),
                        "plan_id": str(binding.plan_id),
                        "plan_sequence": binding.plan_sequence,
                        "action_id": str(binding.action_id),
                        "outcome": "SUPERSEDED",
                    },
                )
                recorded.append(receipt)
            return tuple(recorded)

    def governed_action(self, action_id: ActionId | str) -> GovernedAction:
        """An action as governance left it: its decision, or why validation refused it.

        Raises:
            AggregateNotFoundError: If it does not exist.
        """
        action = self.actions.get(str(action_id))
        if action is None:
            raise AggregateNotFoundError("Action", str(action_id))
        return self._governed(action)

    def approve_action(
        self,
        action_id: ActionId | str,
        *,
        approver: Actor,
        reason: str,
        destinations: tuple[str, ...] = (),
    ) -> Action:
        """A human approves the recorded proposal: ``APPROVAL_PENDING → APPROVED``.

        Approves what policy judged; it does not judge again. Whether the
        action still applies is for execution to check, when it happens.

        Raises:
            ApprovalError: If *approver* is not a human, or the action is not
                awaiting approval and was never resolved by a human.
            ApprovalConflictError: If a human already resolved it otherwise.
        """
        return self._resolve_approval(
            action_id, approver, reason, ActionStatus.APPROVED, "ActionApproved", destinations
        )

    def reject_action(
        self,
        action_id: ActionId | str,
        *,
        approver: Actor,
        reason: str,
        destinations: tuple[str, ...] = (),
    ) -> Action:
        """A human refuses the recorded proposal: ``APPROVAL_PENDING → REJECTED``.

        Raises:
            ApprovalError: As :meth:`approve_action`.
            ApprovalConflictError: As :meth:`approve_action`.
        """
        return self._resolve_approval(
            action_id,
            approver,
            reason,
            ActionStatus.REJECTED,
            "ActionApprovalDenied",
            destinations,
        )

    def _resolve_approval(
        self,
        action_id: ActionId | str,
        approver: Actor,
        reason: str,
        new_status: ActionStatus,
        event_type: str,
        destinations: tuple[str, ...],
    ) -> Action:
        if approver.type != "human":
            raise ApprovalError(
                f"only a human approves or rejects an action awaiting approval; "
                f"{approver.type} {approver.id} cannot"
            )
        if not reason.strip():
            raise ApprovalError("an approval or rejection says why")
        answer = {"approver": approver.model_dump(mode="json"), "reason": reason}
        identity = _approval_identity(event_type, answer)
        with self._write():
            action = self.actions.get(str(action_id))
            if action is None:
                raise AggregateNotFoundError("Action", str(action_id))
            if action.status is ActionStatus.APPROVAL_PENDING:
                resolved = action.with_status(new_status)
                self.actions._update(resolved)
                self._emit_action(
                    resolved,
                    event_type,
                    approver,
                    destinations,
                    extra={**answer, "policy_decision_id": action.policy_decision_id},
                )
                return resolved
            given = self._human_resolution(str(action.id))
            if given is None:
                raise ApprovalError(
                    f"action {action.id} is {action.status.value}, not awaiting approval"
                )
            kind, recorded = given
            if _approval_identity(kind, recorded) == identity:
                return action
            raise ApprovalConflictError(
                f"action {action.id} was already {action.status.value} by "
                f"{recorded['approver']['type']} {recorded['approver']['id']} "
                f"({recorded['reason']!r}); it is not resolved again"
            )

    def _human_resolution(self, action_id: str) -> tuple[str, dict[str, Any]] | None:
        for event in self.events.events_for_aggregate(action_id):
            if event.event_type in ("ActionApproved", "ActionApprovalDenied"):
                payload = _plain(event.payload)
                return event.event_type, {
                    "approver": payload["approver"],
                    "reason": payload["reason"],
                }
        return None

    def _replay_governed(self, action: Action) -> GovernedAction | None:
        """The recorded action for *action*'s id, if the same request was recorded."""
        existing = self.actions.get(str(action.id))
        if existing is None:
            return None
        ActionStore._assert_same_request(existing, action)
        return self._governed(existing)

    def _governed(self, action: Action) -> GovernedAction:
        decision = self.policy.for_action(str(action.id))
        problems: tuple[str, ...] = ()
        if decision is None:
            for event in self.events.events_for_aggregate(str(action.id)):
                if event.event_type == "ActionRejected" and "problems" in event.payload:
                    problems = tuple(event.payload["problems"])
        return GovernedAction(action=action, decision=decision, problems=problems)

    def _governed_target_state(
        self, target: ActionTarget, experiment_id: str
    ) -> tuple[str | None, int | None]:
        """The target's status and revision, if it exists in *experiment_id*."""
        try:
            if target.kind == "experiment":
                if target.id != experiment_id:
                    return None, None
                aggregate: Any = self.aggregates.load_experiment(target.id)
                owner = experiment_id
            elif target.kind == "node":
                aggregate = self.aggregates.load_node(target.id)
                owner = str(aggregate.experiment_id)
            elif target.kind == "run":
                aggregate = self.aggregates.load_run(target.id)
                owner = str(aggregate.experiment_id)
            elif target.kind == "training-attempt":
                aggregate = self.aggregates.load_attempt(target.id)
                owner = self._experiment_of_run(str(aggregate.run_id))
            elif target.kind == "evaluation-run":
                aggregate = self.aggregates.load_evaluation_run(target.id)
                owner = str(aggregate.experiment_id)
            else:
                aggregate = self.aggregates.load_evaluation_attempt(target.id)
                owner = self._experiment_of_evaluation_run(str(aggregate.evaluation_run_id))
        except AggregateNotFoundError:
            return None, None
        if owner != experiment_id:
            return None, None
        return aggregate.status.value, aggregate.revision

    def _require_budget(self, experiment_id: str, *, new_run: bool) -> None:
        """Refuse a new effect a used-up quota cannot cover. Capacity is not judged here.

        A new run needs one run left; any effect needs failures not yet used
        up. Reaching a limit exactly is enough to stop the next
        effect.

        Raises:
            BudgetExhaustedError: With every used-up quota.
        """
        experiment = self.aggregates.load_experiment(experiment_id)
        if not limits(experiment.budget):
            return
        status = budget_status(experiment.budget, self.budget.entries(experiment_id))
        reasons = []
        for dimension in status.dimensions:
            if dimension.kind == "capacity":
                continue
            if dimension.dimension is BudgetDimension.RUNS:
                if new_run and dimension.remaining < 1:
                    reasons.append(
                        f"runs: {dimension.consumed} consumed and {dimension.outstanding} "
                        f"reserved of {dimension.limit}; no run is left for another"
                    )
            elif dimension.remaining <= 0:
                reasons.append(
                    f"{dimension.dimension.value}: {dimension.consumed} consumed of "
                    f"{dimension.limit}"
                )
        if reasons:
            raise BudgetExhaustedError(experiment_id, tuple(reasons))

    def _require_capacity(self, experiment_id: str) -> None:
        """Refuse a training attempt while every parallel-run slot is held.

        Raises:
            CapacityUnavailableError: If none is free; the caller waits.
        """
        experiment = self.aggregates.load_experiment(experiment_id)
        if BudgetDimension.PARALLEL_RUNS not in limits(experiment.budget):
            return
        status = budget_status(experiment.budget, self.budget.entries(experiment_id))
        slots = status.of(BudgetDimension.PARALLEL_RUNS)
        assert slots is not None
        if slots.outstanding >= slots.limit:
            raise CapacityUnavailableError(
                experiment_id, BudgetDimension.PARALLEL_RUNS, slots.limit
            )

    def _commit_budget(
        self, experiment_id: str, attempt_id: str, actor: Actor, destinations: tuple[str, ...]
    ) -> None:
        """A training submission took effect: its slot and its run are committed.

        Provenance only -- a commit subtracts nothing again. The run's is
        written once, by its first confirmed attempt.
        """
        attempt = self.aggregates.load_attempt(attempt_id)
        for subject_kind, subject_id, dimension in (
            (BudgetSubjectKind.TRAINING_ATTEMPT, attempt_id, BudgetDimension.PARALLEL_RUNS),
            (BudgetSubjectKind.RUN, str(attempt.run_id), BudgetDimension.RUNS),
        ):
            reserved = self.budget.entry(
                subject_kind, subject_id, dimension, LedgerEntryKind.RESERVE
            )
            if reserved is not None:
                self._ledger(
                    experiment_id,
                    dimension,
                    LedgerEntryKind.COMMIT,
                    reserved.amount,
                    subject_kind,
                    subject_id,
                    actor,
                    destinations,
                )

    def _settle_budget(
        self, aggregate: object, actor: Actor, destinations: tuple[str, ...]
    ) -> None:
        """Write what a run or an attempt reaching a terminal state costs. Idempotent.

        ```text
        training or evaluation attempt ends   FAILED → failures +1   (not PREEMPTED, CANCELLED)
        training attempt ends                 its parallel-run slot is released
        run ends                              committed → runs consumed; never submitted → released
        ```

        Called inside the transaction that made it terminal, so a record can
        never hold a terminal attempt whose cost is missing.
        """
        if isinstance(aggregate, RunAttempt):
            if not aggregate.is_terminal:
                return
            experiment_id = self._experiment_of_run(str(aggregate.run_id))
            self._settle_attempt(
                experiment_id,
                BudgetSubjectKind.TRAINING_ATTEMPT,
                aggregate,
                aggregate.status is RunAttemptStatus.FAILED,
                actor,
                destinations,
            )
            self._release(
                experiment_id,
                BudgetDimension.PARALLEL_RUNS,
                BudgetSubjectKind.TRAINING_ATTEMPT,
                str(aggregate.id),
                actor,
                destinations,
            )
        elif isinstance(aggregate, EvaluationAttempt):
            if not aggregate.is_terminal:
                return
            experiment_id = self._experiment_of_evaluation_run(str(aggregate.evaluation_run_id))
            self._settle_attempt(
                experiment_id,
                BudgetSubjectKind.EVALUATION_ATTEMPT,
                aggregate,
                aggregate.status is EvaluationAttemptStatus.FAILED,
                actor,
                destinations,
            )
        elif isinstance(aggregate, Run):
            if not aggregate.is_terminal:
                return
            run_id = str(aggregate.id)
            reserved = self.budget.entry(
                BudgetSubjectKind.RUN, run_id, BudgetDimension.RUNS, LedgerEntryKind.RESERVE
            )
            if reserved is None:
                return
            committed = self.budget.entry(
                BudgetSubjectKind.RUN, run_id, BudgetDimension.RUNS, LedgerEntryKind.COMMIT
            )
            if committed is None:
                self._release(
                    str(aggregate.experiment_id),
                    BudgetDimension.RUNS,
                    BudgetSubjectKind.RUN,
                    run_id,
                    actor,
                    destinations,
                )
            else:
                self._ledger(
                    str(aggregate.experiment_id),
                    BudgetDimension.RUNS,
                    LedgerEntryKind.CONSUME,
                    reserved.amount,
                    BudgetSubjectKind.RUN,
                    run_id,
                    actor,
                    destinations,
                )

    def _settle_attempt(
        self,
        experiment_id: str,
        subject_kind: BudgetSubjectKind,
        attempt: RunAttempt | EvaluationAttempt,
        failed: bool,
        actor: Actor,
        destinations: tuple[str, ...],
    ) -> None:
        if failed:
            self._ledger(
                experiment_id,
                BudgetDimension.FAILURES,
                LedgerEntryKind.CONSUME,
                Decimal(1),
                subject_kind,
                str(attempt.id),
                actor,
                destinations,
            )

    def _release(
        self,
        experiment_id: str,
        dimension: BudgetDimension,
        subject_kind: BudgetSubjectKind,
        subject_id: str,
        actor: Actor,
        destinations: tuple[str, ...],
    ) -> None:
        """Give back a subject's reservation on *dimension*, if it holds one."""
        reserved = self.budget.entry(subject_kind, subject_id, dimension, LedgerEntryKind.RESERVE)
        if reserved is not None:
            self._ledger(
                experiment_id,
                dimension,
                LedgerEntryKind.RELEASE,
                reserved.amount,
                subject_kind,
                subject_id,
                actor,
                destinations,
            )

    def _ledger(
        self,
        experiment_id: str,
        dimension: BudgetDimension,
        kind: LedgerEntryKind,
        amount: Decimal,
        subject_kind: BudgetSubjectKind,
        subject_id: str,
        actor: Actor,
        destinations: tuple[str, ...],
    ) -> None:
        """Append one entry, if the dimension is limited and the amount is not nothing.

        A consumption that takes a quota past its limit -- not merely to it
        -- is recorded as a ``BudgetOverrun`` event, once, when it crosses.
        Nothing is stopped by it: what to do about an overrun is policy's.
        """
        experiment = self.aggregates.load_experiment(experiment_id)
        bounded = limits(experiment.budget)
        if dimension not in bounded or amount <= 0:
            return
        before = budget_status(experiment.budget, self.budget.entries(experiment_id))
        entry = self.budget._append(
            BudgetLedgerEntry(
                id=f"ledger_{uuid.uuid4().hex}",
                experiment_id=experiment_id,
                dimension=dimension,
                kind=kind,
                amount=amount,
                subject_kind=subject_kind,
                subject_id=subject_id,
            )
        )
        if kind is not LedgerEntryKind.CONSUME:
            return
        after = budget_status(experiment.budget, self.budget.entries(experiment_id))
        was, now = before.of(dimension), after.of(dimension)
        assert was is not None and now is not None
        if now.overrun and not was.overrun:
            self._emit(
                experiment,
                "BudgetOverrun",
                experiment_id,
                actor,
                destinations,
                extra={
                    "dimension": dimension.value,
                    "limit": str(now.limit),
                    "consumed": str(now.consumed),
                    "subject": f"{subject_kind.value} {subject_id}",
                    "entry": entry.id,
                },
            )

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

        With a ``max_parallel_runs`` budget, the attempt holds a slot from
        this commit until it ends. A retry that finds the attempt already
        recorded takes no new slot and passes no budget check: it is the same
        effect, already authorized.

        Raises:
            IdempotencyConflictError: If either half exists against a different
                request -- the operation's target, type or digest, or the
                attempt's run or number.
            ValueError: If *attempt* is not a freshly created aggregate.
            BudgetExhaustedError: If a quota is used up.
            CapacityUnavailableError: If every parallel-run slot is held;
                nothing is written, and the caller waits for one.
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

        with self._write():
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
                stored = self._insert_training_attempt_with_intent(
                    attempt, operation, actor, destinations
                )
                return attempt, stored
            else:
                run = self.aggregates.load_evaluation_run(str(attempt.evaluation_run_id))
                if run.is_terminal:
                    raise StorageError(
                        f"evaluation run {run.id} is {run.status.value}; a new attempt at "
                        f"a finished run would execute an evaluation nothing will record"
                    )
                experiment_id = str(run.experiment_id)
                self._require_budget(experiment_id, new_run=False)
                self.aggregates._insert_evaluation_attempt(attempt)
            stored = self.operations._insert(operation)
            self._emit(attempt, created_event, experiment_id, actor, destinations)
            self._emit_operation(
                stored, "RuntimeOperationIntended", experiment_id, actor, destinations
            )

        return attempt, stored

    def _insert_training_attempt_with_intent(
        self,
        attempt: RunAttempt,
        operation: RuntimeOperation,
        actor: Actor,
        destinations: tuple[str, ...],
    ) -> RuntimeOperation:
        """A new training attempt and its ``INTENDED`` submit, in the caller's transaction.

        The one place a training attempt's intent is written, for a first
        attempt and for a branched node's realization alike: the failure
        quota, the parallel-run slot, ``RunAttemptCreated`` and
        ``RuntimeOperationIntended``.
        """
        experiment_id = self._experiment_of_run(str(attempt.run_id))
        self._require_budget(experiment_id, new_run=False)
        self._require_capacity(experiment_id)
        self.aggregates._insert_attempt(attempt)
        stored = self.operations._insert(operation)
        self._emit(attempt, "RunAttemptCreated", experiment_id, actor, destinations)
        self._ledger(
            experiment_id,
            BudgetDimension.PARALLEL_RUNS,
            LedgerEntryKind.RESERVE,
            Decimal(1),
            BudgetSubjectKind.TRAINING_ATTEMPT,
            str(attempt.id),
            actor,
            destinations,
        )
        self._emit_operation(stored, "RuntimeOperationIntended", experiment_id, actor, destinations)
        return stored

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

        with self._write():
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
        with self._write():
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
        recorded, and with a different request is refused. A request that
        names its own ``action_id`` is carried out through that Action or not
        at all: with another cancellation in flight it is refused, rather
        than answered with an Action it does not name (PR-029).

        Returns:
            The parent action, and each child with the operation it caused --
            ``None`` for a child whose attempt had already ended.

        Raises:
            UnknownOperationTargetError: If the experiment does not exist.
            IdempotencyConflictError: If *action_id* exists against a different
                request.
            CancellationInFlightError: If *action_id* is given, is not
                recorded, and another cancellation is in flight.
        """
        target = ActionTarget(kind="experiment", id=str(experiment_id))

        with self._write():
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
                if action_id is not None:
                    raise CancellationInFlightError(str(action_id), str(in_flight[0].id))
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
                with self._write():
                    self.actions._update(failed)
                    self._emit_action(failed, "CancellationFailed", actor, destinations)
                return failed
            return parent

        with self._write():
            if self._live_attempts(experiment_id):
                return parent
            for node in self.aggregates.nodes_for_experiment(experiment_id):
                for run in self.aggregates.runs_for_node(str(node.id)):
                    if not run.is_terminal:
                        cancelled_run = run.with_status(RunStatus.CANCELLED)
                        self.aggregates._update_run(cancelled_run)
                        self._settle_budget(cancelled_run, actor, destinations)
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
        """Control work actively in progress, whose outcome must still be driven or reconciled.

        Operations still ``INTENDED`` or ``SENT`` -- an effect requested with
        no known result -- and actions in flight: ``PROPOSED``,
        ``VALIDATING`` or ``EXECUTING``. A run can be terminal while either
        remains: a cancellation that raced natural completion leaves its
        operation unresolved after the attempt has already succeeded. Whoever
        asks "is a controller still working" has to ask about these too, not
        only about runs.

        Not every non-terminal action. A governed action at rest (PR-023) --
        ``VALIDATED`` with its decision, ``APPROVAL_PENDING``, ``APPROVED`` --
        waits for a person or an executor; no controller is driving it, and
        :meth:`resting_actions` reports it instead. A cancellation never rests
        in those states: it is recorded through to ``EXECUTING`` in one
        transaction.
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
            if action.status in _IN_FLIGHT
        )
        return operations, actions

    def resting_actions(self, experiment_id: str) -> tuple[tuple[Action, ...], tuple[Action, ...]]:
        """Governed actions at rest: awaiting a human, and awaiting an executor.

        ``APPROVAL_PENDING`` awaits approval. Awaiting execution is exactly
        :func:`~xaytune.core.domain.policy.awaits_execution`: ``VALIDATED``
        with an ``ALLOW`` decision, or ``APPROVED`` with a
        ``REQUIRE_APPROVAL`` one.
        """
        approval, execution = [], []
        for action in self.actions.for_experiment(experiment_id):
            if action.status is ActionStatus.APPROVAL_PENDING:
                approval.append(action)
            elif awaits_execution(action, self.policy.for_action(str(action.id))):
                execution.append(action)
        return tuple(approval), tuple(execution)

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
        with self._write():
            return self._apply_transition(
                kind,
                aggregate_id,
                expected_revision,
                new_status,
                loader,
                writer,
                actor,
                event_type,
                destinations,
                after_write=after_write,
                extra=extra,
            )

    def _apply_transition(
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
        """:meth:`_transition`'s work, inside a transaction the caller already holds.

        For a unit that moves several aggregates in one commit -- realizing a
        planned node moves the node and its new run -- so each move still goes
        through the state machine, the revision check and its event.
        """
        current = loader(aggregate_id)
        if current is None:
            raise AggregateNotFoundError(kind, aggregate_id)
        if current.revision != expected_revision:
            raise ConcurrentModificationError(kind, aggregate_id, expected_revision)

        moved: AggregateT = current.with_status(new_status)
        writer(moved)
        if after_write is not None:
            after_write(moved)
        self._settle_budget(moved, actor, destinations)
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
        with self._write():
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
            if (
                state == "confirmed"
                and moved.type == "submit"
                and moved.target.kind == "training-attempt"
            ):
                self._commit_budget(experiment_id, moved.target.id, actor, destinations)
            self._emit_operation(
                moved,
                f"RuntimeOperation{state.capitalize()}",
                experiment_id,
                actor,
                destinations,
            )
            if (
                moved.type == "submit"
                and moved.caused_by_action_id is not None
                and state == "failed"
                and self.numerical_recovery_executions.for_action(str(moved.caused_by_action_id))
                is not None
            ):
                # A numerical Action succeeds only on confirmed effect, never on
                # submission; a submission that definitively failed fails it.
                numerical = self.actions.get(str(moved.caused_by_action_id))
                if numerical is not None and numerical.status is ActionStatus.EXECUTING:
                    failed_action = numerical.with_status(ActionStatus.FAILED)
                    self.actions._update(failed_action)
                    self._emit_action(failed_action, "ActionFailed", actor, destinations)
            if moved.type == "submit" and moved.caused_by_action_id is not None:
                binding = self.recovery_action_bindings.for_action(str(moved.caused_by_action_id))
                if binding is not None and state in ("confirmed", "failed"):
                    action = self.actions.get(str(moved.caused_by_action_id))
                    assert action is not None
                    if action.status is ActionStatus.EXECUTING:
                        settled = (
                            action.with_status(
                                ActionStatus.SUCCEEDED, outcome=ActionOutcome.APPLIED
                            )
                            if state == "confirmed"
                            else action.with_status(ActionStatus.FAILED)
                        )
                        self.actions._update(settled)
                        self._emit_action(
                            settled,
                            "ActionApplied" if state == "confirmed" else "ActionFailed",
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

        An inherited seed is checked for honesty wherever a run is created:
        the source run exists, is a run of one of the node's parents, and has
        the seed claimed. Which source is *allowed* is stricter, and the
        caller's: see :meth:`_require_first_realization`.

        Raises:
            AggregateNotFoundError: If the node does not exist.
            StorageError: If the node's experiment, fingerprint or seed
                provenance disagrees.
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

        origin = run.seed_origin
        if origin is not None:
            source = self.aggregates.get_run(str(origin.source_run_id))
            if source is None or source.node_id not in node.parent_ids:
                raise StorageError(
                    f"run {run.id} inherits its seed from run {origin.source_run_id}, which "
                    f"is not a run of node {node.id}'s parents"
                )
            if source.seed != run.seed:
                raise StorageError(
                    f"run {run.id} claims seed {run.seed} from run {source.id}, whose seed is "
                    f"{source.seed}"
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
        *,
        extra: dict[str, Any] | None = None,
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
                    **(extra or {}),
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
        self._write_sequenced_event(event, destinations)
        return event

    def _write_sequenced_event(self, event: DomainEvent, destinations: tuple[str, ...]) -> int:
        """Append *event* and its outbox records; return the sequence the database assigned."""
        sequence = self.events._append(event)
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
        return sequence


def _cancel_digest(kind: str, target_id: str) -> str:
    """The request digest of a saga's cancel operation.

    Derived from the target alone: cancelling an attempt is the same request
    however many times it is made, so a retry's digest matches.
    """
    return fingerprint({"type": "cancel", "target": {"kind": kind, "id": target_id}})
