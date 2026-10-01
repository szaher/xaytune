"""Shared I/O half of checkpoint-backed recovery: capability checks and restore selection.

Both the OOM and the numerical executor resolve and validate outside SQLite, then
hand the repository a successor to commit atomically. These helpers hold the
parts they share, so the two can never disagree about what makes a checkpoint
safe to restore.
"""

from __future__ import annotations

from pathlib import Path

from xaytune.checkpoints import (
    CheckpointCompatibilityError,
    CheckpointCorruptionError,
    CheckpointManager,
    LocalCheckpointStore,
)
from xaytune.core.capabilities import CapabilityDocument
from xaytune.core.domain.recovery import (
    RecoveryCheckpointReport,
    RecoveryEpisode,
    checkpoint_order,
    checkpoint_report_problem,
)
from xaytune.core.execution import (
    PythonModuleEntrypoint,
    ResolvedExecutionPlan,
    TrainingExecutionSpec,
)
from xaytune.storage.control_plane import ControlPlaneRepository, StaleRecoveryContextError

__all__ = ["require_managed_restore", "select_restore_checkpoint"]


def require_managed_restore(
    capabilities: CapabilityDocument | None,
    source_plan: ResolvedExecutionPlan,
    checkpoint_manager: CheckpointManager,
    unavailable: type[Exception],
) -> None:
    """The runtime applies FULL+EXACT restore for this worker, from the shared store."""
    assert isinstance(source_plan.spec, TrainingExecutionSpec)
    if (
        capabilities is None
        or capabilities.checkpoint is None
        or capabilities.checkpoint.full_exact_restore is not True
    ):
        raise unavailable("runtime has not declared FULL+EXACT checkpoint application")
    supported_entrypoints = capabilities.extensions.get("checkpoint_restore_entrypoints", ())
    if supported_entrypoints and (
        not isinstance(source_plan.spec.entrypoint, PythonModuleEntrypoint)
        or source_plan.spec.entrypoint.module not in supported_entrypoints
    ):
        raise unavailable("runtime has not declared FULL+EXACT restore for this worker")
    if supported_entrypoints:
        store_uri = source_plan.spec.checkpoint.store_uri
        store = checkpoint_manager.store
        if (
            source_plan.spec.checkpoint.format != "native-torch/v1"
            or store_uri is None
            or not isinstance(store, LocalCheckpointStore)
            or store.root != Path(store_uri).resolve()
        ):
            raise unavailable(
                "managed worker and checkpoint manager do not share the declared store"
            )


async def select_restore_checkpoint(
    repository: ControlPlaneRepository,
    checkpoint_manager: CheckpointManager,
    episode: RecoveryEpisode,
    unavailable: type[Exception],
) -> RecoveryCheckpointReport:
    """The newest structurally eligible checkpoint whose bytes validate, or *unavailable*."""
    request = episode.request
    if request.restore_context is None or not checkpoint_manager.supports_validation:
        raise unavailable("recovery requires consumer restore context and validation capability")
    inputs = repository.recovery_snapshot(episode)
    if inputs.successor_exists:
        raise StaleRecoveryContextError("episode closed before checkpoint assessment")
    candidates = []
    for attempt in repository.aggregates.attempts_for_run(episode.context.run_id):
        if attempt.attempt_number > episode.attempt_number:
            continue
        for record in repository.checkpoints.for_attempt(str(attempt.id)):
            report = RecoveryCheckpointReport.from_record(record, attempt.attempt_number)
            candidates.append((report, record))
    candidates.sort(key=lambda pair: checkpoint_order(pair[0]), reverse=True)
    for report, record in candidates:
        if checkpoint_report_problem(report, inputs) is not None:
            continue
        try:
            await checkpoint_manager.validate_recorded(record, request.restore_context)
        except (CheckpointCompatibilityError, CheckpointCorruptionError, OSError):
            continue
        return report
    raise unavailable("no validated FULL+EXACT checkpoint is eligible")
