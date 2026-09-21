"""Control-plane domain aggregates."""

from __future__ import annotations

from xaytune.core.domain.action import (
    Action,
    ActionOutcome,
    ActionTarget,
    ActionTargetKind,
    ActionType,
    UnknownActionTypeError,
    register_action_type,
    registered_action_types,
)
from xaytune.core.domain.event import (
    DomainEvent,
    OutboxRecord,
    OutboxState,
)
from xaytune.core.domain.experiment import (
    CandidateSpecSnapshot,
    Experiment,
    ExperimentNode,
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
    "Action",
    "ActionOutcome",
    "ActionTarget",
    "ActionTargetKind",
    "ActionType",
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
    "register_action_type",
    "registered_action_types",
    "Run",
    "RunAttempt",
    "RuntimeOperation",
    "RuntimeOperationTarget",
    "CandidateSpecSnapshot",
    "UnknownActionTypeError",
]
