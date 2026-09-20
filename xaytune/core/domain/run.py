"""Run and RunAttempt aggregates, plus operational execution overrides.

A :class:`Run` is a logical execution of one scientific candidate; a node may
own several when seeds or replicates are wanted. A :class:`RunAttempt` is one
infrastructure attempt at that run. Retry, preemption and checkpoint restore
produce new attempts under the same run — they are operational events and never
branch the scientific graph (ADR-003).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from xaytune.core.clock import utc_now
from xaytune.core.ids import (
    ExperimentId,
    ExperimentNodeId,
    IncidentId,
    RunAttemptId,
    RunId,
)
from xaytune.core.refs import ArtifactRef, CheckpointRef, ResourceUsage, RuntimeRef
from xaytune.core.state.machines import ATTEMPT_MACHINE, RUN_MACHINE
from xaytune.core.state.status import RunAttemptStatus, RunStatus

__all__ = [
    "ExecutionOverride",
    "ExecutionOverrideKind",
    "Run",
    "RunAttempt",
]

ExecutionOverrideKind = Literal[
    "micro_batch_resize",
    "gradient_accumulation_adjustment",
    "worker_count_adjustment",
    "placement_retry",
    "checkpoint_restore",
    "timeout_increase",
    "runtime_native_recovery",
]
"""Operational adjustments that preserve declared training intent.

Changes to learning rate, optimizer, LoRA rank, data, scheduler, reward,
algorithm or model revision are *not* execution overrides — those are
scientific mutations and create a new node.
"""


class ExecutionOverride(BaseModel):
    """A policy-approved operational modification to an attempt.

    Attributes:
        preserves: The invariants this override claims to hold, e.g.
            ``["effective_batch_size"]`` when halving the micro-batch and
            doubling gradient accumulation. Recorded so that a reviewer can
            tell an intent-preserving change from a scientific one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: ExecutionOverrideKind
    reason: str
    values: dict[str, Any] = Field(default_factory=dict)
    preserves: list[str] = Field(default_factory=list)
    incident_id: IncidentId | None = None
    created_at: datetime = Field(default_factory=utc_now)


class Run(BaseModel):
    """A logical execution of a scientific candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: RunId
    node_id: ExperimentNodeId
    experiment_id: ExperimentId

    seed: int | None = None
    replicate: int | None = None

    training_fingerprint: str
    execution_plan_ref: str | None = None

    attempt_ids: list[RunAttemptId] = Field(default_factory=list)
    final_attempt_id: RunAttemptId | None = None

    status: RunStatus = RunStatus.CREATED

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    revision: int = 0

    def with_status(self, new_status: RunStatus) -> Run:
        """Return a copy in *new_status*, with the revision bumped.

        Raises:
            InvalidTransitionError: If the transition is not permitted.
        """
        RUN_MACHINE.validate(self.status, new_status)
        return self.model_copy(
            update={
                "status": new_status,
                "revision": self.revision + 1,
                "updated_at": utc_now(),
            }
        )

    @property
    def is_terminal(self) -> bool:
        """Whether this run has reached a final state."""
        return RUN_MACHINE.is_terminal(self.status)


class RunAttempt(BaseModel):
    """One infrastructure attempt at a run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: RunAttemptId
    run_id: RunId
    attempt_number: int = Field(ge=1)

    status: RunAttemptStatus = RunAttemptStatus.CREATED

    runtime_ref: RuntimeRef | None = None
    execution_fingerprint: str | None = None

    execution_overrides: list[ExecutionOverride] = Field(default_factory=list)

    checkpoint_ref: CheckpointRef | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)

    incident_ids: list[IncidentId] = Field(default_factory=list)

    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)

    started_at: datetime | None = None
    ended_at: datetime | None = None
    revision: int = 0

    def with_status(self, new_status: RunAttemptStatus) -> RunAttempt:
        """Return a copy in *new_status*, with the revision bumped.

        ``started_at`` is stamped on entering ``RUNNING`` and ``ended_at`` on
        reaching a terminal state, in both cases only if not already set, so
        that a ``CHECKPOINTING -> RUNNING`` round trip does not reset the
        attempt's start time.

        Raises:
            InvalidTransitionError: If the transition is not permitted.
        """
        ATTEMPT_MACHINE.validate(self.status, new_status)

        update: dict[str, Any] = {
            "status": new_status,
            "revision": self.revision + 1,
        }
        if new_status is RunAttemptStatus.RUNNING and self.started_at is None:
            update["started_at"] = utc_now()
        if ATTEMPT_MACHINE.is_terminal(new_status) and self.ended_at is None:
            update["ended_at"] = utc_now()

        return self.model_copy(update=update)

    @property
    def is_terminal(self) -> bool:
        """Whether this attempt has reached a final state."""
        return ATTEMPT_MACHINE.is_terminal(self.status)
