"""Pure deterministic CUDA OOM resize planning; no action or runtime effects."""

from __future__ import annotations

from xaytune.core.domain.action import ActionTarget
from xaytune.core.domain.actions.builtin import ResizeMicrobatch
from xaytune.core.domain.oom_recovery import (
    OOMEscalation,
    OOMEscalationCode,
    OOMRecoveryInputsV1,
    OOMResizeProposal,
)

__all__ = ["OOMRecoveryPlanner"]


class OOMRecoveryPlanner:
    """Propose one governed resize, or a deterministic reason to escalate."""

    name = "deterministic-cuda-oom"
    version = "1"

    def plan(self, inputs: OOMRecoveryInputsV1) -> OOMResizeProposal | OOMEscalation:
        plan = inputs.plan

        def escalate(code: OOMEscalationCode, reason: str) -> OOMEscalation:
            return OOMEscalation(
                episode_id=plan.episode_id,
                plan_id=plan.id,
                plan_sequence=plan.sequence,
                input_fingerprint=inputs.input_fingerprint,
                code=code,
                reason=reason,
            )

        prior = inputs.prior_resize
        if prior is not None and (
            inputs.execution_state_fingerprint == prior.prior_execution_state_fingerprint
            or inputs.current_micro_batch_size != prior.promised_micro_batch_size
            or inputs.current_gradient_accumulation != prior.promised_gradient_accumulation
        ):
            return escalate(
                OOMEscalationCode.PRIOR_OVERRIDE_NOT_APPLIED,
                "the previous adaptive resize is absent from the failed attempt's execution state",
            )
        if not inputs.preserve_effective_batch:
            return escalate(
                OOMEscalationCode.PRESERVATION_NOT_AUTHORIZED,
                "autonomous OOM recovery must preserve the declared effective batch",
            )
        if inputs.current_micro_batch_size <= inputs.min_micro_batch_size:
            return escalate(
                OOMEscalationCode.MINIMUM_MICRO_BATCH_REACHED,
                "micro-batch is already at the configured minimum",
            )
        new_micro_batch = max(inputs.min_micro_batch_size, inputs.current_micro_batch_size // 2)
        divisor = new_micro_batch * inputs.world_size
        if inputs.effective_batch_size % divisor:
            return escalate(
                OOMEscalationCode.NONINTEGRAL_ACCUMULATION,
                "effective batch cannot be preserved with integral gradient accumulation",
            )
        new_accumulation = inputs.effective_batch_size // divisor
        if (
            inputs.max_gradient_accumulation is not None
            and new_accumulation > inputs.max_gradient_accumulation
        ):
            return escalate(
                OOMEscalationCode.ACCUMULATION_LIMIT,
                "preserving effective batch exceeds the gradient accumulation limit",
            )
        return OOMResizeProposal(
            episode_id=plan.episode_id,
            plan_id=plan.id,
            plan_sequence=plan.sequence,
            input_fingerprint=inputs.input_fingerprint,
            run_id=inputs.run_id,
            candidate_fingerprint=inputs.candidate_fingerprint,
            source_execution_state_fingerprint=inputs.execution_state_fingerprint,
            old_micro_batch_size=inputs.current_micro_batch_size,
            old_gradient_accumulation=inputs.current_gradient_accumulation,
            world_size=inputs.world_size,
            effective_batch_size=inputs.effective_batch_size,
            action_spec=ResizeMicrobatch(
                target=ActionTarget(kind="run", id=str(inputs.run_id)),
                micro_batch_size=new_micro_batch,
                gradient_accumulation=new_accumulation,
            ),
        )
