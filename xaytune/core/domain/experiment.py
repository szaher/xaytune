"""Experiment and ExperimentNode aggregates.

An :class:`Experiment` owns an optimization objective. An
:class:`ExperimentNode` is one scientific candidate within it — a hypothesis,
not an infrastructure attempt. Worker restarts, preemptions and checkpoint
restores never create nodes (ADR-003); they create run attempts.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import AliasChoices, Field

from xaytune.core.clock import utc_now
from xaytune.core.domain.candidate import CandidateSpec, TrainingKind
from xaytune.core.domain.evaluation import EvaluationSpec
from xaytune.core.domain.objective import BudgetSpec, Objective
from xaytune.core.domain.specs import CompilerSpec, RuntimeSpec
from xaytune.core.ids import (
    DecisionId,
    EvaluationRunId,
    ExperimentId,
    ExperimentNodeId,
    RunId,
)
from xaytune.core.immutable import AggregateModel, FrozenDict, FrozenDomainModel
from xaytune.core.refs import Actor, ControllerHostRef
from xaytune.core.state.machines import EXPERIMENT_MACHINE, NODE_MACHINE
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus

__all__ = [
    "CandidateSpecSnapshot",
    "Experiment",
    "ExperimentNode",
]


class CandidateSpecSnapshot(FrozenDomainModel):
    """Immutable snapshot of the scientific proposition a node tests.

    Frozen on purpose: a scientific change creates a child node rather than
    editing an existing snapshot (Rule 5).

    Wraps a :class:`~xaytune.core.domain.candidate.CandidateSpec` rather than
    being one, so a node keeps the exact bytes it was created with even if the
    spec's own schema later gains fields. ``spec_version`` records which shape
    those bytes are.
    """

    candidate: CandidateSpec
    spec_version: str = "1"
    metadata: FrozenDict = Field(default_factory=FrozenDict)

    @property
    def kind(self) -> TrainingKind:
        """The training program this candidate runs."""
        return self.candidate.training.kind


class Experiment(AggregateModel):
    """The complete optimization objective and its control-plane state.

    Frozen: status changes go through :meth:`with_status`, never through
    attribute assignment (Rule 7).
    """

    id: ExperimentId
    name: str
    objective: Objective
    policy_ref: str | None = None
    budget: BudgetSpec | None = None

    status: ExperimentStatus = ExperimentStatus.CREATED
    active_node_ids: tuple[ExperimentNodeId, ...] = Field(default_factory=tuple)
    best_node_id: ExperimentNodeId | None = None

    controller_host: ControllerHostRef

    # ADR-016: what executes this experiment, as specs a restarted controller
    # resolves again, never as implementation objects. Optional only because
    # experiments recorded before a host drove them have none; a host refuses
    # to drive an experiment that lacks them.
    compiler: CompilerSpec | None = None
    runtime: RuntimeSpec | None = None
    artifact_root: str | None = None

    evaluation: EvaluationSpec | None = None
    """How each trained candidate is evaluated, with its evaluators bound.

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

    @property
    def is_terminal(self) -> bool:
        """Whether this experiment has reached a final state."""
        return EXPERIMENT_MACHINE.is_terminal(self.status)


class ExperimentNode(AggregateModel):
    """One scientific candidate within an experiment.

    ``parent_ids`` is a list rather than a single parent so that a candidate
    can be derived from more than one predecessor.
    """

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

    run_ids: tuple[RunId, ...] = Field(default_factory=tuple)
    evaluation_run_ids: tuple[EvaluationRunId, ...] = Field(default_factory=tuple)

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

    @property
    def is_root(self) -> bool:
        """Whether this candidate has no predecessor."""
        return not self.parent_ids

    @property
    def is_terminal(self) -> bool:
        """Whether this candidate has reached a final state."""
        return NODE_MACHINE.is_terminal(self.status)
