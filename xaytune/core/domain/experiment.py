"""Experiment and ExperimentNode aggregates.

An :class:`Experiment` owns an optimization objective. An
:class:`ExperimentNode` is one scientific candidate within it — a hypothesis,
not an infrastructure attempt. Worker restarts, preemptions and checkpoint
restores never create nodes (ADR-003); they create run attempts.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from xaytune.core.clock import utc_now
from xaytune.core.domain.objective import BudgetSpec, Objective
from xaytune.core.ids import (
    DecisionId,
    EvaluationId,
    ExperimentId,
    ExperimentNodeId,
    RunId,
)
from xaytune.core.immutable import AggregateModel, FrozenDict, FrozenDomainModel
from xaytune.core.refs import Actor, ControllerHostRef, DatasetRef, ModelRef
from xaytune.core.state.machines import EXPERIMENT_MACHINE, NODE_MACHINE
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus

__all__ = [
    "Experiment",
    "ExperimentNode",
    "TrainingSpecSnapshot",
]


class TrainingSpecSnapshot(FrozenDomainModel):
    """Immutable snapshot of the training intent attached to a node.

    Frozen on purpose: a scientific change creates a child node rather than
    editing an existing snapshot (Rule 5).

    Phase 1 placeholder. The typed SFT/pretrain/DPO/GRPO schemas and the
    fingerprint framework arrive with the TrainingSpec work; until then the
    payload stays opaque so nothing in the core depends on trainer-specific
    field names.
    """

    kind: str
    spec_version: str = "0"
    model: ModelRef | None = None
    dataset: DatasetRef | None = None
    payload: FrozenDict = Field(default_factory=FrozenDict)


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

    training_spec: TrainingSpecSnapshot
    training_fingerprint: str

    status: ExperimentNodeStatus = ExperimentNodeStatus.CREATED

    run_ids: tuple[RunId, ...] = Field(default_factory=tuple)
    evaluation_ids: tuple[EvaluationId, ...] = Field(default_factory=tuple)
    decision_ids: tuple[DecisionId, ...] = Field(default_factory=tuple)

    created_by: Actor
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    revision: int = 0

    def with_status(self, new_status: ExperimentNodeStatus) -> ExperimentNode:
        """Return a copy in *new_status*, with the revision bumped.

        Raises:
            InvalidTransitionError: If the transition is not permitted.
        """
        NODE_MACHINE.validate(self.status, new_status)
        return self._validated_copy(
            {
                "status": new_status,
                "revision": self.revision + 1,
                "updated_at": utc_now(),
            }
        )

    @property
    def is_root(self) -> bool:
        """Whether this candidate has no predecessor."""
        return not self.parent_ids

    @property
    def is_terminal(self) -> bool:
        """Whether this candidate has reached a final state."""
        return NODE_MACHINE.is_terminal(self.status)
