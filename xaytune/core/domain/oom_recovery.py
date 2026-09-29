"""Immutable inputs and outputs for deterministic CUDA OOM resize planning.

An OOM resize proposal is an intent for governance, never execution authority.
The current configuration must come from the failed attempt's resolved execution
spec; a future executor must verify that binding before creating a successor.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from xaytune.core.checkpoint import Digest
from xaytune.core.domain.actions.builtin import ResizeMicrobatch
from xaytune.core.domain.incident import IncidentCategory
from xaytune.core.domain.recovery import (
    Recoverability,
    RecoveryPlan,
    RecoveryStrategy,
    diagnosis_requirement,
)
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import ActionId, RecoveryEpisodeId, RecoveryPlanId, RunAttemptId, RunId
from xaytune.core.immutable import FrozenDomainModel

__all__ = [
    "OOMRecoveryInputsV1",
    "OOMRecoveryPlannerResult",
    "OOMResizeProposal",
    "OOMEscalation",
    "OOMEscalationCode",
    "PriorOOMResize",
]


class PriorOOMResize(FrozenDomainModel):
    """An executed prior resize whose successor is the currently failed attempt."""

    action_id: ActionId
    successor_attempt_id: RunAttemptId
    prior_execution_state_fingerprint: Digest
    promised_micro_batch_size: StrictInt = Field(ge=1)
    promised_gradient_accumulation: StrictInt = Field(ge=1)


class OOMRecoveryInputsV1(FrozenDomainModel):
    """Decision-bound execution facts supplied by the intended consumer.

    The caller must derive the current values from the failed attempt's *actual*
    resolved spec, not infer them from a proposed action or candidate defaults.
    The execution path will re-read those durable facts under its write lock.
    """

    schema_version: Literal["xaytune.oom-recovery-inputs/v1alpha1"] = (
        "xaytune.oom-recovery-inputs/v1alpha1"
    )
    plan: RecoveryPlan
    run_id: RunId
    candidate_fingerprint: str = Field(min_length=1)
    execution_state_fingerprint: Digest
    current_micro_batch_size: StrictInt = Field(ge=1)
    current_gradient_accumulation: StrictInt = Field(ge=1)
    world_size: StrictInt = Field(ge=1)
    min_micro_batch_size: StrictInt = Field(default=1, ge=1)
    max_gradient_accumulation: StrictInt | None = Field(default=None, ge=1)
    preserve_effective_batch: bool = Field(default=True, strict=True)
    prior_resize: PriorOOMResize | None = None

    @model_validator(mode="after")
    def _bound_to_oom_episode(self) -> OOMRecoveryInputsV1:
        plan = self.plan
        if (
            plan.strategy is not RecoveryStrategy.PAUSE_FOR_APPROVAL
            or plan.recoverability is not Recoverability.REQUIRES_HUMAN
            or plan.inputs.context.target.kind != "training-attempt"
            or plan.reason != "specialised recovery required: adaptive-execution"
            or self.run_id != plan.inputs.context.run_id
            or self.candidate_fingerprint != plan.inputs.candidate_fingerprint
            or self.execution_state_fingerprint != plan.execution_state_fingerprint
        ):
            raise ValueError("OOM inputs must bind a paused training episode decision")
        categories = (
            category
            for evidence in plan.inputs.accepted_evidence
            for category in evidence.categories
        )
        requirements = tuple((category, diagnosis_requirement(category)) for category in categories)
        if not any(category is IncidentCategory.CUDA_OOM for category, _ in requirements):
            raise ValueError("accepted evidence does not contain CUDA OOM")
        if any(
            requirement.specialised_family != "adaptive-execution"
            and requirement.recoverability is not Recoverability.RECOVERABLE_NEW_ATTEMPT
            for _, requirement in requirements
        ):
            raise ValueError("other accepted evidence blocks autonomous OOM recovery")
        if self.prior_resize is not None and (
            str(self.prior_resize.successor_attempt_id) != plan.inputs.context.target.id
        ):
            raise ValueError("prior resize does not produce the failed attempt")
        return self

    @property
    def effective_batch_size(self) -> int:
        return self.current_micro_batch_size * self.current_gradient_accumulation * self.world_size

    @property
    def input_fingerprint(self) -> str:
        return fingerprint(self)


class OOMRecoveryPlannerResult(FrozenDomainModel):
    """Common freshness binding for a governed proposal or an escalation."""

    episode_id: RecoveryEpisodeId
    plan_id: RecoveryPlanId
    plan_sequence: StrictInt = Field(ge=1)
    input_fingerprint: Digest


class OOMResizeProposal(OOMRecoveryPlannerResult):
    """One ResizeMicrobatch intent preserving the candidate's effective batch."""

    run_id: RunId
    candidate_fingerprint: str = Field(min_length=1)
    source_execution_state_fingerprint: Digest
    old_micro_batch_size: StrictInt = Field(ge=1)
    old_gradient_accumulation: StrictInt = Field(ge=1)
    world_size: StrictInt = Field(ge=1)
    effective_batch_size: StrictInt = Field(ge=1)
    action_spec: ResizeMicrobatch
    preserves: tuple[Literal["effective_batch_size"], ...] = ("effective_batch_size",)

    @model_validator(mode="after")
    def _preserves_and_reduces(self) -> OOMResizeProposal:
        spec = self.action_spec
        if (
            spec.micro_batch_size >= self.old_micro_batch_size
            or spec.target.kind != "run"
            or spec.target.id != str(self.run_id)
            or spec.gradient_accumulation is None
            or spec.micro_batch_size * spec.gradient_accumulation * self.world_size
            != self.effective_batch_size
            or self.old_micro_batch_size * self.old_gradient_accumulation * self.world_size
            != self.effective_batch_size
            or self.preserves != ("effective_batch_size",)
        ):
            raise ValueError("OOM resize must reduce micro-batch and preserve effective batch")
        return self


class OOMEscalationCode(str, Enum):
    PRIOR_OVERRIDE_NOT_APPLIED = "prior-override-not-applied"
    MINIMUM_MICRO_BATCH_REACHED = "minimum-micro-batch-reached"
    PRESERVATION_NOT_AUTHORIZED = "preservation-not-authorized"
    NONINTEGRAL_ACCUMULATION = "nonintegral-accumulation"
    ACCUMULATION_LIMIT = "accumulation-limit"


class OOMEscalation(OOMRecoveryPlannerResult):
    """No autonomous resize can be proposed under the supplied constraints."""

    code: OOMEscalationCode
    reason: str = Field(min_length=1)
