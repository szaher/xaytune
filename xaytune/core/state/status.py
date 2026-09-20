"""Status enums for each control-plane aggregate.

Each aggregate has its own lifecycle (ADR-002). There is deliberately no single
"experiment status" covering training and evaluation: with concurrent branches,
an experiment-wide ``EVALUATING`` would be meaningless.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "ExperimentNodeStatus",
    "ExperimentStatus",
    "RunAttemptStatus",
    "RunStatus",
]


class ExperimentStatus(str, Enum):
    """Coarse lifecycle of an experiment.

    Stays ``ACTIVE`` while individual nodes train, evaluate and decide
    concurrently.
    """

    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ExperimentNodeStatus(str, Enum):
    """Lifecycle of a scientific candidate."""

    CREATED = "created"
    PLANNED = "planned"
    READY = "ready"
    ACTIVE = "active"
    EVALUATING = "evaluating"
    DECIDING = "deciding"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class RunStatus(str, Enum):
    """Lifecycle of a logical run.

    Intentionally coarse — infrastructure detail belongs to
    :class:`RunAttemptStatus`.
    """

    CREATED = "created"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunAttemptStatus(str, Enum):
    """Lifecycle of a single infrastructure attempt."""

    CREATED = "created"
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    RECOVERING = "recovering"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PREEMPTED = "preempted"
    CANCELLED = "cancelled"
