"""Control-plane domain aggregates."""

from __future__ import annotations

from xaytune.core.domain.event import (
    DomainEvent,
    OutboxRecord,
    OutboxState,
)
from xaytune.core.domain.experiment import (
    Experiment,
    ExperimentNode,
    TrainingSpecSnapshot,
)
from xaytune.core.domain.objective import (
    BudgetSpec,
    MetricConstraint,
    Objective,
    ObjectiveMetric,
)
from xaytune.core.domain.operation import (
    OperationState,
    OperationTargetKind,
    OperationType,
    RuntimeOperation,
    RuntimeOperationTarget,
)
from xaytune.core.domain.run import (
    ExecutionOverride,
    ExecutionOverrideKind,
    Run,
    RunAttempt,
)

__all__ = [
    "BudgetSpec",
    "DomainEvent",
    "ExecutionOverride",
    "ExecutionOverrideKind",
    "Experiment",
    "ExperimentNode",
    "MetricConstraint",
    "Objective",
    "ObjectiveMetric",
    "OperationState",
    "OperationTargetKind",
    "OperationType",
    "OutboxRecord",
    "OutboxState",
    "Run",
    "RunAttempt",
    "RuntimeOperation",
    "RuntimeOperationTarget",
    "TrainingSpecSnapshot",
]
