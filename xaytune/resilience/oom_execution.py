"""Governed CUDA OOM effect, with checkpoint I/O before the durable write."""

from __future__ import annotations

from collections.abc import Callable

from xaytune.checkpoints import (
    CheckpointManager,
)
from xaytune.compilation.attempt_resolution import training_execution_fingerprint
from xaytune.core.capabilities import CapabilityDocument
from xaytune.core.domain.intervention import InterventionDirective
from xaytune.core.domain.operation import RuntimeOperation
from xaytune.core.domain.recovery_execution import RecoveryExecutionReceipt
from xaytune.core.domain.run import ExecutionOverride, RunAttempt
from xaytune.core.execution import (
    ResolvedExecutionPlan,
    TrainingExecutionSpec,
)
from xaytune.core.ids import ActionId, InterventionApplicationId, OperationId, RunAttemptId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor
from xaytune.resilience.successor import require_managed_restore, select_restore_checkpoint
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
        resolve_plan: Callable[..., ResolvedExecutionPlan],
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
        require_managed_restore(
            self.capabilities, source_plan, self.checkpoint_manager, OOMCheckpointUnavailableError
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

        checkpoint = await select_restore_checkpoint(
            self.repository, self.checkpoint_manager, episode, OOMCheckpointUnavailableError
        )
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
        # A restore that drops an earlier intervention's effect re-applies it
        # (ADR-011 REAPPLY_AFTER_ROLLBACK); a run without interventions has none,
        # and its successor resolves exactly as before.
        replay = self.repository.plan_successor_interventions(
            source.run_id, checkpoint.checkpoint_ref
        )
        directives = tuple(
            InterventionDirective(
                application_id=InterventionApplicationId.generate(),
                intervention_id=planned.intervention_id,
                attempt_id=draft.id,
                ordinal=ordinal,
                kind=planned.kind,
                mutation=planned.mutation,
                expected_previous_value=planned.expected_previous_value,
            )
            for ordinal, planned in enumerate(replay.directives)
        )
        resolved = (
            self.resolve_plan(draft, directives=directives)
            if directives
            else self.resolve_plan(draft)
        )
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
            directives=directives,
        )
