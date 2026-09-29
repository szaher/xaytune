"""Deterministically resolve a compiled candidate for one durable training attempt."""

from __future__ import annotations

from typing import Any

from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.domain.run import RunAttempt
from xaytune.core.execution import ResolvedExecutionPlan, TrainingExecutionSpec
from xaytune.core.immutable import FrozenDict, thaw


def resolve_training_attempt(
    spec: TrainingExecutionSpec, attempt: RunAttempt, runtime: str
) -> ResolvedExecutionPlan:
    """Apply recorded operational lineage after trainer-neutral compilation.

    Each attempt carries the cumulative lineage from the candidate: ordered
    micro-batch/accumulation pairs, then its own checkpoint restore binding.
    Every ``from`` value must match the configuration reached so far, and each
    pair must preserve effective batch. No overrides means the legacy plan is
    byte-equivalent to the pre-recovery path. Inconsistent lineage fails closed.
    """
    config: dict[str, Any] = thaw(spec.config)
    optimization: dict[str, Any] | None = None
    pair_before: tuple[int, int] | None = None
    resized = False
    restore_id: str | None = None

    for override in attempt.execution_overrides:
        if override.kind in ("micro_batch_resize", "gradient_accumulation_adjustment"):
            if restore_id is not None:
                raise ValueError("checkpoint restore must follow the complete resize lineage")
            if optimization is None:
                optimization = config.get("optimization")
                if not isinstance(optimization, dict):
                    raise ValueError("compiled training spec has no optimization section")
            micro = optimization.get("micro_batch_size")
            accumulation = optimization.get("gradient_accumulation")
            if (
                type(micro) is not int
                or micro < 1
                or type(accumulation) is not int
                or accumulation < 1
            ):
                raise ValueError("compiled training spec has invalid batch configuration")
            key = (
                "micro_batch_size"
                if override.kind == "micro_batch_resize"
                else "gradient_accumulation"
            )
            if key == "micro_batch_size":
                if pair_before is not None:
                    raise ValueError("a resize pair must adjust gradient accumulation next")
                pair_before = (micro, accumulation)
            elif pair_before is None:
                raise ValueError("gradient accumulation adjustment requires a preceding resize")
            values = override.values
            before, after = values.get("from"), values.get("to")
            if (
                set(values) != {"from", "to"}
                or type(before) is not int
                or type(after) is not int
                or after < 1
                or optimization.get(key) != before
                or override.preserves != ("effective_batch_size",)
            ):
                raise ValueError(f"attempt has invalid {key} override provenance")
            optimization[key] = after
            if key == "gradient_accumulation":
                assert pair_before is not None
                if pair_before[0] * pair_before[1] != micro * after:
                    raise ValueError("execution overrides do not preserve effective batch")
                pair_before = None
                resized = True
        elif override.kind == "checkpoint_restore":
            if pair_before is not None:
                raise ValueError("checkpoint restore cannot interrupt a resize pair")
            if restore_id is not None or set(override.values) != {"checkpoint_id"}:
                raise ValueError("attempt has ambiguous checkpoint restore override")
            restore_id = override.values["checkpoint_id"]
            if not isinstance(restore_id, str):
                raise ValueError("checkpoint restore override requires a checkpoint id")
        else:
            raise ValueError(f"unsupported training execution override {override.kind}")

    if pair_before is not None:
        raise ValueError("batch-preserving resize requires both operational overrides")
    checkpoint = attempt.checkpoint_ref
    if (checkpoint is None) != (restore_id is None):
        raise ValueError("checkpoint reference and restore override must occur together")
    if checkpoint is not None and restore_id != str(checkpoint.id):
        raise ValueError("checkpoint restore override names another checkpoint")

    resolved_spec = spec if not resized else spec.model_copy(update={"config": FrozenDict(config)})
    options = (
        FrozenDict()
        if checkpoint is None
        else FrozenDict({"checkpoint_restore": checkpoint.model_dump(mode="json")})
    )
    return ResolvedExecutionPlan(
        spec=resolved_spec,
        runtime=runtime,
        target=RuntimeOperationTarget(kind="training-attempt", id=str(attempt.id)),
        runtime_options=options,
    )
