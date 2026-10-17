"""Experiment and ExperimentNode aggregates.

An :class:`Experiment` owns an optimization objective. An
:class:`ExperimentNode` is one scientific candidate within it — a hypothesis,
not an infrastructure attempt. Worker restarts, preemptions and checkpoint
restores never create nodes (ADR-003); they create run attempts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final

from pydantic import (
    AliasChoices,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from xaytune.core.clock import utc_now
from xaytune.core.domain.candidate import CandidateSpec, TrainingKind
from xaytune.core.domain.candidate_envelope import (
    CandidateEnvelopeError,
    CandidateKind,
    ExperimentCandidate,
    NotATrainingCandidateError,
    TrainingCandidate,
)
from xaytune.core.domain.evaluation import EvaluationSpec
from xaytune.core.domain.numerical_recovery import NumericalRecoveryPolicyV1
from xaytune.core.domain.objective import BudgetSpec, Objective
from xaytune.core.domain.planning import CandidateBranchOrigin
from xaytune.core.domain.resilience import ResilienceSpec
from xaytune.core.domain.specs import CompilerSpec, PlannerSpec, RuntimeSpec
from xaytune.core.errors import InvalidTransitionError
from xaytune.core.ids import (
    DecisionId,
    ExperimentId,
    ExperimentNodeId,
)
from xaytune.core.immutable import (
    AggregateModel,
    FrozenDict,
    FrozenDomainModel,
    drop_retired_fields,
)
from xaytune.core.refs import Actor, ControllerHostRef
from xaytune.core.state.machines import EXPERIMENT_MACHINE, NODE_MACHINE
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus

__all__ = [
    "SNAPSHOT_VERSIONS",
    "CandidateSpecSnapshot",
    "Experiment",
    "ExperimentNode",
]


SNAPSHOT_VERSIONS: Final = ("1", "2")
"""``"1"``: a ``CandidateSpec`` (every node before ADR-018 D10, and every training
node since). ``"2"``: an ``ExperimentCandidate`` envelope of any kind."""


class CandidateSpecSnapshot(FrozenDomainModel):
    """Immutable snapshot of the scientific proposition a node tests.

    Frozen on purpose: a scientific change creates a child node rather than
    editing an existing snapshot (Rule 5).

    Wraps the candidate rather than being one, so a node keeps the exact
    bytes it was created with even if the candidate's own schema later gains
    fields. ``spec_version`` records which shape those bytes are:

    - ``"1"``: ``candidate``, a :class:`~xaytune.core.domain.candidate.CandidateSpec`.
      Training nodes are still written this way, so their payloads are
      byte-identical to every earlier build's.
    - ``"2"``: ``envelope``, an :class:`~xaytune.core.domain.candidate_envelope.ExperimentCandidate`
      of any kind (ADR-018 D10). How a node holds a harness.

    A stored ``"1"`` is never rewritten as ``"2"``. Read the candidate through
    :meth:`experiment_candidate`, or :meth:`training_spec` where only training
    makes sense.
    """

    candidate: CandidateSpec | None = None
    envelope: ExperimentCandidate | None = None
    spec_version: str = "1"
    metadata: FrozenDict = Field(default_factory=FrozenDict)

    @model_validator(mode="after")
    def _one_shape(self) -> CandidateSpecSnapshot:
        if self.spec_version not in SNAPSHOT_VERSIONS:
            raise CandidateEnvelopeError(
                f"candidate snapshot version {self.spec_version!r} is not one this build reads; "
                f"it reads {list(SNAPSHOT_VERSIONS)}"
            )
        if self.spec_version == "1" and (self.candidate is None or self.envelope is not None):
            raise ValueError("a version 1 snapshot holds a CandidateSpec, and nothing else")
        if self.spec_version == "2" and (self.envelope is None or self.candidate is not None):
            raise ValueError("a version 2 snapshot holds a candidate envelope, and nothing else")
        return self

    @model_serializer(mode="wrap")
    def _only_its_shape(self, handler: SerializerFunctionWrapHandler) -> Any:
        # A version 1 snapshot serializes exactly as it always has: no
        # ``envelope: null`` appears in a training node's payload.
        data = handler(self)
        if isinstance(data, dict):
            data.pop("envelope" if self.spec_version == "1" else "candidate", None)
        return data

    @classmethod
    def of(
        cls, candidate: ExperimentCandidate, *, metadata: FrozenDict | None = None
    ) -> CandidateSpecSnapshot:
        """The snapshot a node stores for *candidate*: ``"1"`` for training, ``"2"`` otherwise."""
        recorded = FrozenDict() if metadata is None else metadata
        if isinstance(candidate.payload, TrainingCandidate):
            return cls(candidate=candidate.payload.spec, metadata=recorded)
        return cls(envelope=candidate, spec_version="2", metadata=recorded)

    def experiment_candidate(self) -> ExperimentCandidate:
        """The candidate, of whichever kind, as an envelope."""
        if self.envelope is not None:
            return self.envelope
        assert self.candidate is not None
        return ExperimentCandidate.training(self.candidate)

    @property
    def candidate_kind(self) -> CandidateKind:
        return self.experiment_candidate().kind

    def training_spec(self) -> CandidateSpec:
        """The ``CandidateSpec``, for work only a training candidate can do.

        Raises:
            NotATrainingCandidateError: If the snapshot holds another kind.
        """
        candidate = self.experiment_candidate()
        if not isinstance(candidate.payload, TrainingCandidate):
            raise NotATrainingCandidateError(
                f"this is a {candidate.kind.value} candidate: training-only work cannot run on it"
            )
        return candidate.payload.spec

    def scientific_fingerprint(self) -> str:
        """The candidate's own identity: training v2, or harness v1."""
        return self.experiment_candidate().scientific_fingerprint()

    @property
    def kind(self) -> TrainingKind:
        """The training program this candidate runs.

        Raises:
            NotATrainingCandidateError: If the snapshot holds another kind.
        """
        return self.training_spec().training.kind


class Experiment(AggregateModel):
    """The complete optimization objective and its control-plane state.

    Frozen: status changes go through :meth:`with_status`, never through
    attribute assignment (Rule 7).
    """

    @model_validator(mode="before")
    @classmethod
    def _retired_topology(cls, data: Any) -> Any:
        return drop_retired_fields(data, "Experiment", "active_node_ids")

    id: ExperimentId
    name: str
    objective: Objective
    policy_ref: str | None = None
    budget: BudgetSpec | None = None
    numerical_recovery: NumericalRecoveryPolicyV1 | None = None
    """Arms managed numerical recovery, with its explicit algorithm parameters.

    ``None`` -- the default, and every record written before PR-021 -- means
    numerical recovery is not armed: a nonfinite loss is reported and the
    worker keeps its report-and-continue behaviour.
    """

    status: ExperimentStatus = ExperimentStatus.CREATED
    # A decision, not topology: which nodes the experiment has comes from
    # ``nodes_for_experiment`` (open question 15).
    best_node_id: ExperimentNodeId | None = None

    controller_host: ControllerHostRef

    # ADR-016: what executes this experiment, as specs a restarted controller
    # resolves again, never as implementation objects. Optional only because
    # experiments recorded before a host drove them have none; a host refuses
    # to drive an experiment that lacks them.
    compiler: CompilerSpec | None = None
    runtime: RuntimeSpec | None = None
    artifact_root: str | None = None

    planner: PlannerSpec | None = None
    """Which planner proposes the experiment's next candidate, bound (ADR-016).

    ``None`` -- the default, and every record written before PR-024 -- means
    no planner was chosen: an experiment that reaches the planning stage
    waits for someone to propose by hand.
    """

    resilience: ResilienceSpec | None = None
    """Which resilience provider recovers workers inside an attempt, bound (PR-035).

    ``None`` -- the default, and every record written before PR-035 -- means
    nothing is delegated: every plan is exactly what it was before.
    """

    evaluation: EvaluationSpec | None = None
    """How each trained candidate is evaluated, with its evaluator bound.

    Orchestration, not identity: it never enters a candidate, its fingerprint
    or its compilation, so changing it does not mean retraining anything.
    ``None`` leaves a trained node ``ACTIVE``, with evaluation as the next
    stage nothing has taken on.
    """

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    revision: int = 0
    metadata: FrozenDict = Field(default_factory=FrozenDict)

    def with_status(self, new_status: ExperimentStatus) -> Experiment:
        """Return a copy in *new_status*, with the revision bumped.

        Raises:
            InvalidTransitionError: If the transition is not permitted.
        """
        EXPERIMENT_MACHINE.validate(self.status, new_status)
        return self._validated_copy(
            {
                "status": new_status,
                "revision": self.revision + 1,
                "updated_at": utc_now(),
            }
        )

    def succeeded_with(self, node_id: ExperimentNodeId) -> Experiment:
        """Return a copy ``SUCCEEDED``, with *node_id* recorded as its best candidate.

        The candidate a decision completed is the answer the experiment found;
        recording it is part of the same transition, so an experiment never
        succeeds without naming what succeeded.

        Raises:
            InvalidTransitionError: If the experiment cannot succeed from its status.
        """
        succeeded = self.with_status(ExperimentStatus.SUCCEEDED)
        return succeeded._validated_copy({"best_node_id": node_id})

    @property
    def is_terminal(self) -> bool:
        """Whether this experiment has reached a final state."""
        return EXPERIMENT_MACHINE.is_terminal(self.status)


class ExperimentNode(AggregateModel):
    """One scientific candidate within an experiment.

    ``parent_ids`` is a list rather than a single parent so that a candidate
    can be derived from more than one predecessor.
    """

    @model_validator(mode="before")
    @classmethod
    def _retired_topology(cls, data: Any) -> Any:
        return drop_retired_fields(data, "ExperimentNode", "run_ids", "evaluation_run_ids")

    id: ExperimentNodeId
    experiment_id: ExperimentId

    parent_ids: tuple[ExperimentNodeId, ...] = Field(default_factory=tuple)

    hypothesis: str | None = None
    reason: str | None = None

    candidate: CandidateSpecSnapshot
    candidate_fingerprint: str = Field(
        validation_alias=AliasChoices("candidate_fingerprint", "training_fingerprint"),
        serialization_alias="candidate_fingerprint",
    )

    status: ExperimentNodeStatus = ExperimentNodeStatus.CREATED

    branch_origin: CandidateBranchOrigin | None = None
    """The planner proposal this node materializes, when it was branched (PR-025).

    ``None`` for a node submitted directly, and for every node recorded
    before branching existed. Set once, at creation, and never changed.
    """

    evaluation_cycle: int = Field(default=0, ge=0)
    """Which evaluation round the node is in, or last was in; 0 before any.

    Advanced by entering ``EVALUATING``, in the same write as the transition,
    so every round has its own number. Each ``EvaluationRun`` records the
    round it belongs to, and reconciling a node looks only at runs of its
    current one: a node evaluated, decided and evaluated again must not have
    its second round satisfied by the results of the first (ADR-015 §5).
    """
    decision_ids: tuple[DecisionId, ...] = Field(default_factory=tuple)

    created_by: Actor
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    revision: int = 0

    def with_status(self, new_status: ExperimentNodeStatus) -> ExperimentNode:
        """Return a copy in *new_status*, with the revision bumped.

        Entering ``EVALUATING`` also starts the next evaluation cycle.

        Raises:
            InvalidTransitionError: If the transition is not permitted.
        """
        NODE_MACHINE.validate(self.status, new_status)
        update: dict[str, object] = {
            "status": new_status,
            "revision": self.revision + 1,
            "updated_at": utc_now(),
        }
        if new_status is ExperimentNodeStatus.EVALUATING:
            update["evaluation_cycle"] = self.evaluation_cycle + 1
        return self._validated_copy(update)

    def with_decision(
        self, decision_id: DecisionId, new_status: ExperimentNodeStatus
    ) -> ExperimentNode:
        """Return a copy in *new_status*, with *decision_id* recorded as the reason.

        Only from ``DECIDING``: a decision is about an evaluated candidate.
        The id and the transition are one change, so a node never holds a
        decision it was not moved by, nor was moved by one it does not hold.

        Raises:
            InvalidTransitionError: If the node is not deciding, or the
                transition is not permitted.
        """
        if self.status is not ExperimentNodeStatus.DECIDING:
            # Only a node in DECIDING is decided.
            raise InvalidTransitionError("ExperimentNode", self.status, new_status)
        moved = self.with_status(new_status)
        return moved._validated_copy({"decision_ids": (*self.decision_ids, decision_id)})

    def training_candidate(self) -> CandidateSpec:
        """The node's ``CandidateSpec``, for compiling, planning and other training-only work.

        Raises:
            NotATrainingCandidateError: If the node holds another kind of candidate.
        """
        return self.candidate.training_spec()

    @property
    def candidate_kind(self) -> CandidateKind:
        return self.candidate.candidate_kind

    @property
    def is_root(self) -> bool:
        """Whether this candidate has no predecessor."""
        return not self.parent_ids

    @property
    def is_terminal(self) -> bool:
        """Whether this candidate has reached a final state."""
        return NODE_MACHINE.is_terminal(self.status)
