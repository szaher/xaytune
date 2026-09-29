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

    No override or checkpoint means the legacy plan is byte-equivalent to the
    pre-recovery path. Unsupported or inconsistent overrides fail closed rather
    than quietly recompiling the candidate's original execution settings.
    """
    config: dict[str, Any] = thaw(spec.config)
    original: dict[str, Any] | None = None
    optimization: dict[str, Any] | None = None
    seen: set[str] = set()
    restore_id: str | None = None

    for override in attempt.execution_overrides:
        if override.kind in ("micro_batch_resize", "gradient_accumulation_adjustment"):
            if optimization is None:
                optimization = config.get("optimization")
                if not isinstance(optimization, dict):
                    raise ValueError("compiled training spec has no optimization section")
                original = dict(optimization)
            key = (
                "micro_batch_size"
                if override.kind == "micro_batch_resize"
                else "gradient_accumulation"
            )
            if key in seen:
                raise ValueError(f"attempt repeats the {key} execution override")
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
            seen.add(key)
        elif override.kind == "checkpoint_restore":
            if restore_id is not None or set(override.values) != {"checkpoint_id"}:
                raise ValueError("attempt has ambiguous checkpoint restore override")
            restore_id = override.values["checkpoint_id"]
            if not isinstance(restore_id, str):
                raise ValueError("checkpoint restore override requires a checkpoint id")
        else:
            raise ValueError(f"unsupported training execution override {override.kind}")

    if seen:
        if seen != {"micro_batch_size", "gradient_accumulation"}:
            raise ValueError("batch-preserving resize requires both operational overrides")
        assert original is not None and optimization is not None
        if (
            type(original.get("micro_batch_size")) is not int
            or type(original.get("gradient_accumulation")) is not int
            or original["micro_batch_size"] * original["gradient_accumulation"]
            != optimization["micro_batch_size"] * optimization["gradient_accumulation"]
        ):
            raise ValueError("execution overrides do not preserve effective batch")
    checkpoint = attempt.checkpoint_ref
    if (checkpoint is None) != (restore_id is None):
        raise ValueError("checkpoint reference and restore override must occur together")
    if checkpoint is not None and restore_id != str(checkpoint.id):
        raise ValueError("checkpoint restore override names another checkpoint")

    resolved_spec = spec if not seen else spec.model_copy(update={"config": FrozenDict(config)})
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
