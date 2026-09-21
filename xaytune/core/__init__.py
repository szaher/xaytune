"""Xaytune control-plane core.

Domain types, identifiers, state machines and errors for the experiment control
plane. This package is deliberately free of ML and runtime dependencies — no
torch, transformers, peft, trl, ray, torchft, kubernetes, mlflow or wandb — so
that a controller host or client can import it without the training stack
installed (ADR-010).

That boundary is enforced by ``tests/test_core/test_architecture.py``.
"""

from __future__ import annotations

from xaytune.core.clock import utc_now
from xaytune.core.domain import (
    Action,
    ActionOutcome,
    ActionTarget,
    BudgetSpec,
    CandidateSpecSnapshot,
    DomainEvent,
    ExecutionOverride,
    Experiment,
    ExperimentNode,
    MetricConstraint,
    Objective,
    ObjectiveMetric,
    OutboxRecord,
    Run,
    RunAttempt,
    RuntimeOperation,
    RuntimeOperationTarget,
)
from xaytune.core.errors import (
    ConcurrentModificationError,
    DomainError,
    InvalidIdError,
    InvalidTransitionError,
    XaytuneError,
)
from xaytune.core.ids import (
    ActionId,
    ArtifactId,
    CheckpointId,
    DecisionId,
    EvaluationId,
    EventId,
    ExperimentId,
    ExperimentNodeId,
    IncidentId,
    OperationId,
    RunAttemptId,
    RunId,
    TypedId,
)
from xaytune.core.refs import (
    Actor,
    ArtifactRef,
    CheckpointRef,
    ControllerHostRef,
    DatasetRef,
    ModelRef,
    ResourceUsage,
    RuntimeRef,
)
from xaytune.core.state import (
    ACTION_MACHINE,
    ATTEMPT_MACHINE,
    EXPERIMENT_MACHINE,
    NODE_MACHINE,
    RUN_MACHINE,
    ActionStatus,
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
    StateMachine,
)

__all__ = [
    "Action",
    "ACTION_MACHINE",
    "ActionId",
    "ActionOutcome",
    "ActionStatus",
    "ActionTarget",
    "Actor",
    "ArtifactId",
    "ArtifactRef",
    "ATTEMPT_MACHINE",
    "BudgetSpec",
    "CheckpointId",
    "CheckpointRef",
    "ConcurrentModificationError",
    "ControllerHostRef",
    "DatasetRef",
    "DecisionId",
    "DomainError",
    "DomainEvent",
    "EvaluationId",
    "EventId",
    "ExecutionOverride",
    "Experiment",
    "EXPERIMENT_MACHINE",
    "ExperimentId",
    "ExperimentNode",
    "ExperimentNodeId",
    "ExperimentNodeStatus",
    "ExperimentStatus",
    "IncidentId",
    "InvalidIdError",
    "InvalidTransitionError",
    "MetricConstraint",
    "ModelRef",
    "NODE_MACHINE",
    "Objective",
    "ObjectiveMetric",
    "OperationId",
    "OutboxRecord",
    "ResourceUsage",
    "Run",
    "RUN_MACHINE",
    "RunAttempt",
    "RunAttemptId",
    "RunAttemptStatus",
    "RunId",
    "RunStatus",
    "RuntimeOperation",
    "RuntimeOperationTarget",
    "RuntimeRef",
    "StateMachine",
    "CandidateSpecSnapshot",
    "TypedId",
    "utc_now",
    "XaytuneError",
]
