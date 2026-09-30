"""Append-only provenance for consuming a recovery decision.

A receipt records the outcome of governed intent; it does not itself grant
execution authority. The executor must bind it atomically to durable effects.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, model_validator

from xaytune.core.clock import utc_now
from xaytune.core.ids import (
    ActionId,
    OperationId,
    RecoveryEpisodeId,
    RecoveryExecutionReceiptId,
    RecoveryPlanId,
    RunAttemptId,
)
from xaytune.core.immutable import FrozenDomainModel
from xaytune.core.refs import Actor, CheckpointRef


class RecoveryExecutionOutcome(str, Enum):
    EXECUTED = "EXECUTED"
    ABANDONED = "ABANDONED"
    SUPERSEDED = "SUPERSEDED"


class RecoveryExecutionReceipt(FrozenDomainModel):
    """Immutable link between an episode decision, governed Action and effect.

    ``EXECUTED`` means an INTENDED submit operation and successor attempt were
    committed, not that the external runtime confirmed submission or training
    succeeded. The latter are recorded by their own operation/attempt lifecycle.
    """

    id: RecoveryExecutionReceiptId = Field(default_factory=RecoveryExecutionReceiptId.generate)
    episode_id: RecoveryEpisodeId
    plan_id: RecoveryPlanId
    plan_sequence: int = Field(ge=1, strict=True)
    action_id: ActionId
    outcome: RecoveryExecutionOutcome
    successor_attempt_id: RunAttemptId | None = None
    runtime_operation_id: OperationId | None = None
    checkpoint_ref: CheckpointRef | None = None
    created_by: Actor
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _links(self) -> RecoveryExecutionReceipt:
        executed = self.outcome is RecoveryExecutionOutcome.EXECUTED
        if executed != (self.successor_attempt_id is not None):
            raise ValueError("executed receipt requires a successor attempt")
        if executed != (self.runtime_operation_id is not None):
            raise ValueError("executed receipt requires a runtime operation")
        if not executed and self.checkpoint_ref is not None:
            raise ValueError("unexecuted receipt cannot bind a checkpoint")
        return self
