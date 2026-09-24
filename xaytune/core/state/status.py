"""Status enums for each control-plane aggregate.

Each aggregate has its own lifecycle (ADR-002). There is deliberately no single
"experiment status" covering training and evaluation: with concurrent branches,
an experiment-wide ``EVALUATING`` would be meaningless.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "ActionStatus",
    "EvaluationAttemptStatus",
    "EvaluationRunStatus",
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
    """Lifecycle of a scientific candidate.

    ``REJECTED`` and ``CANCELLED`` are different outcomes: a candidate is
    rejected on its merits by a decision, and cancelled when the work was
    stopped before that judgement could be made.
    """

    CREATED = "created"
    PLANNED = "planned"
    READY = "ready"
    ACTIVE = "active"
    EVALUATING = "evaluating"
    DECIDING = "deciding"
    COMPLETED = "completed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
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


class EvaluationRunStatus(str, Enum):
    """Lifecycle of one logical evaluation (ADR-015 §1).

    Coarse, like :class:`RunStatus`: infrastructure detail belongs to
    :class:`EvaluationAttemptStatus`.
    """

    CREATED = "created"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationAttemptStatus(str, Enum):
    """Lifecycle of one infrastructure attempt at an evaluation (ADR-015 §1).

    Not :class:`RunAttemptStatus`: evaluation writes no checkpoints, so it has
    no ``CHECKPOINTING``, and has nothing to recover into, so no
    ``RECOVERING``. A failed evaluation is retried as a new attempt.
    """

    CREATED = "created"
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PREEMPTED = "preempted"
    CANCELLED = "cancelled"


class ActionStatus(str, Enum):
    """Lifecycle of a typed control-plane action.

    Approval is a **branch**, not a stage every action passes through:
    ``VALIDATED`` may go straight to ``EXECUTING`` when no policy requires
    approval. Routing everything through ``APPROVAL_PENDING -> APPROVED`` would
    mean a controller-owned cancellation had to be marked approved by nobody,
    which is a fiction in the audit record -- and it becomes load-bearing in
    band B, where cancellations exist before any ``PolicyEngine`` does.

    The three concepts stay separate::

        validation      is this well-formed and applicable?   always
        authorization   is this permitted by policy?          when a policy applies
        approval        does a human have to say yes?         when policy says so
    """

    PROPOSED = "proposed"
    VALIDATING = "validating"
    VALIDATED = "validated"
    APPROVAL_PENDING = "approval_pending"
    APPROVED = "approved"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
