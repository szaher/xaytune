"""Governed CUDA OOM effect, with checkpoint I/O before the durable write."""

from __future__ import annotations

from collections.abc import Callable

from xaytune.checkpoints import (
    CheckpointCompatibilityError,
    CheckpointCorruptionError,
    CheckpointManager,
)
from xaytune.compilation.attempt_resolution import training_execution_fingerprint
from xaytune.core.capabilities import CapabilityDocument
from xaytune.core.domain.operation import RuntimeOperation
from xaytune.core.domain.recovery import (
    RecoveryCheckpointReport,
    checkpoint_order,
    checkpoint_report_problem,
)
from xaytune.core.domain.recovery_execution import RecoveryExecutionReceipt
from xaytune.core.domain.run import ExecutionOverride, RunAttempt
from xaytune.core.execution import ResolvedExecutionPlan, TrainingExecutionSpec
from xaytune.core.ids import ActionId, OperationId, RunAttemptId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor
from xaytune.storage.control_plane import (
    ControlPlaneRepository,
    ProvenanceError,
    StaleRecoveryContextError,
)


class OOMCheckpointUnavailableError(RuntimeError):
    """No current FULL+EXACT checkpoint can safely support adaptive recovery."""


class OOMRecoveryExecutor:
    """Resolve and validate outside SQLite, then atomically consume governed intent.

    ``resolve_plan`` recompiles the unchanged candidate and applies a durable
    attempt's operational overrides. The host provides its existing resolver;
    this class contains no trainer-specific OOM rule or runtime submission.
    """

    def __init__(
        self,
        repository: ControlPlaneRepository,
        checkpoint_manager: CheckpointManager,
        resolve_plan: Callable[[RunAttempt], ResolvedExecutionPlan],
        *,
        capabilities: CapabilityDocument | None = None,
        destinations: tuple[str, ...] = (),
    ) -> None:
        self.repository = repository
        self.checkpoint_manager = checkpoint_manager
        self.resolve_plan = resolve_plan
        self.capabilities = capabilities
        self.destinations = destinations

    async def execute(
        self, action_id: ActionId, *, actor: Actor
    ) -> tuple[RunAttempt, RuntimeOperation, RecoveryExecutionReceipt]:
        """Consume an approved/current OOM Action; never call the runtime here."""
        binding = self.repository.recovery_action_bindings.for_action(str(action_id))
        if binding is None:
            raise ProvenanceError("OOM Action has no recovery decision binding")
        prior = self.repository.recovery_execution_receipts.executed_for_episode(
            str(binding.episode_id)
        )
        if prior is not None:
            if prior.action_id != action_id:
                raise ProvenanceError("another Action already consumed this recovery episode")
            assert prior.successor_attempt_id is not None
            assert prior.runtime_operation_id is not None
            attempt = self.repository.aggregates.load_attempt(str(prior.successor_attempt_id))
            operation = self.repository.operations.get(str(prior.runtime_operation_id))
            assert operation is not None
            return attempt, operation, prior

        episode = self.repository.recovery_episodes.get(str(binding.episode_id))
        if episode is None or not self.repository.recovery_plans.is_effective_and_fresh(
            str(binding.plan_id)
        ):
            raise StaleRecoveryContextError("OOM recovery decision is no longer open and fresh")
        source = self.repository.aggregates.load_attempt(episode.context.target.id)
        source_plan = self.resolve_plan(source)
        proposal = binding.proposal
        if not isinstance(source_plan.spec, TrainingExecutionSpec):
            raise ProvenanceError("OOM source does not resolve to a training execution")
        if (
            self.capabilities is None
            or self.capabilities.checkpoint is None
            or self.capabilities.checkpoint.full_exact_restore is not True
        ):
            raise OOMCheckpointUnavailableError(
                "runtime has not declared FULL+EXACT checkpoint application"
            )
        optimization = source_plan.spec.config.get("optimization")
        workers = source_plan.spec.resources.workers or 1
        if (
            source_plan.target.kind != "training-attempt"
            or source_plan.target.id != str(source.id)
            or source_plan.spec.candidate_fingerprint != proposal.candidate_fingerprint
            or source.execution_fingerprint != training_execution_fingerprint(source_plan)
            or not isinstance(optimization, FrozenDict)
            or optimization.get("micro_batch_size") != proposal.old_micro_batch_size
            or optimization.get("gradient_accumulation") != proposal.old_gradient_accumulation
            or workers != proposal.world_size
        ):
            raise StaleRecoveryContextError("OOM proposal no longer describes source execution")

        checkpoint = await self._select_checkpoint(episode)
        new_id = RunAttemptId.generate()
        operation_id = OperationId.generate()
        prior_overrides = source.execution_overrides
        if prior_overrides and prior_overrides[-1].kind == "checkpoint_restore":
            prior_overrides = prior_overrides[:-1]
        overrides = (
            *prior_overrides,
            ExecutionOverride(
                id=f"{action_id}:micro-batch",
                kind="micro_batch_resize",
                reason="governed adaptive CUDA OOM recovery",
                values=FrozenDict(
                    {
                        "from": proposal.old_micro_batch_size,
                        "to": proposal.action_spec.micro_batch_size,
                    }
                ),
                preserves=("effective_batch_size",),
                action_id=action_id,
            ),
            ExecutionOverride(
                id=f"{action_id}:gradient-accumulation",
                kind="gradient_accumulation_adjustment",
                reason="preserve effective batch size",
                values=FrozenDict(
                    {
                        "from": proposal.old_gradient_accumulation,
                        "to": proposal.action_spec.gradient_accumulation,
                    }
                ),
                preserves=("effective_batch_size",),
                action_id=action_id,
            ),
            ExecutionOverride(
                id=f"{action_id}:checkpoint-restore",
                kind="checkpoint_restore",
                reason="restore validated FULL+EXACT checkpoint",
                values=FrozenDict({"checkpoint_id": str(checkpoint.checkpoint_ref.id)}),
                action_id=action_id,
            ),
        )
        draft = RunAttempt(
            id=new_id,
            run_id=source.run_id,
            attempt_number=source.attempt_number + 1,
            execution_overrides=overrides,
            checkpoint_ref=checkpoint.checkpoint_ref,
        )
        resolved = self.resolve_plan(draft)
        if (
            not isinstance(resolved.spec, TrainingExecutionSpec)
            or resolved.target.kind != "training-attempt"
            or resolved.target.id != str(draft.id)
            or resolved.spec.candidate_fingerprint != proposal.candidate_fingerprint
            or resolved.spec.config["optimization"]["micro_batch_size"]
            != proposal.action_spec.micro_batch_size
            or resolved.spec.config["optimization"]["gradient_accumulation"]
            != proposal.action_spec.gradient_accumulation
        ):
            raise ProvenanceError("OOM successor does not resolve to the governed configuration")
        successor = RunAttempt.model_validate(
            {
                **draft.model_dump(mode="json"),
                "execution_fingerprint": training_execution_fingerprint(resolved),
            }
        )
        return self.repository._record_oom_recovery_execution(
            action_id,
            successor,
            checkpoint,
            request_digest=resolved.request_digest("submit"),
            actor=actor,
            capabilities=self.capabilities,
            operation_id=operation_id,
            destinations=self.destinations,
        )

    async def _select_checkpoint(self, episode) -> RecoveryCheckpointReport:
        request = episode.request
        if request.restore_context is None or not self.checkpoint_manager.supports_validation:
            raise OOMCheckpointUnavailableError(
                "OOM recovery requires consumer restore context and validation capability"
            )
        inputs = self.repository.recovery_snapshot(episode)
        if inputs.successor_exists:
            raise StaleRecoveryContextError("OOM episode closed before checkpoint assessment")
        candidates = []
        for attempt in self.repository.aggregates.attempts_for_run(episode.context.run_id):
            if attempt.attempt_number > episode.attempt_number:
                continue
            for record in self.repository.checkpoints.for_attempt(str(attempt.id)):
                report = RecoveryCheckpointReport.from_record(record, attempt.attempt_number)
                candidates.append((report, record))
        candidates.sort(key=lambda pair: checkpoint_order(pair[0]), reverse=True)
        for report, record in candidates:
            if checkpoint_report_problem(report, inputs) is not None:
                continue
            try:
                await self.checkpoint_manager.validate_recorded(record, request.restore_context)
            except (CheckpointCompatibilityError, CheckpointCorruptionError, OSError):
                continue
            return report
        raise OOMCheckpointUnavailableError("no validated FULL+EXACT checkpoint is eligible")
