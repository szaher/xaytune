"""Governed numerical recovery effect: checkpoint I/O first, then one durable write.

```text
authorized ChangeLearningRate + recorded TrainingIntervention
  → validated FULL+EXACT checkpoint from before the unsafe step
  → replay plan: re-apply what the restore dropped, then this intervention
  → successor attempt N+1 + directives + INTENDED submit + receipt, atomically
```

No runtime call happens here, and nothing claims the change took effect: the
worker applies each directive on restore and reports it, and only that report
becomes an ``InterventionApplication`` and completes the Action. The learning
rate never becomes an ``ExecutionOverride``; the successor carries only a
checkpoint-restore override.
"""

from __future__ import annotations

from collections.abc import Callable

from xaytune.checkpoints import CheckpointManager
from xaytune.compilation.attempt_resolution import training_execution_fingerprint
from xaytune.core.capabilities import CapabilityDocument
from xaytune.core.domain.intervention import InterventionDirective, InterventionDirectiveKind
from xaytune.core.domain.operation import RuntimeOperation
from xaytune.core.domain.recovery_execution import RecoveryExecutionReceipt
from xaytune.core.domain.run import ExecutionOverride, RunAttempt
from xaytune.core.execution import ResolvedExecutionPlan, TrainingExecutionSpec
from xaytune.core.execution_controls import TRAINING_INTERVENTIONS, TrainingInterventionDirectives
from xaytune.core.ids import ActionId, InterventionApplicationId, OperationId, RunAttemptId
from xaytune.core.immutable import FrozenDict, thaw
from xaytune.core.refs import Actor
from xaytune.resilience.successor import require_managed_restore, select_restore_checkpoint
from xaytune.storage.control_plane import (
    ControlPlaneRepository,
    ProvenanceError,
    StaleRecoveryContextError,
)

__all__ = ["NumericalCheckpointUnavailableError", "NumericalRecoveryExecutor"]


class NumericalCheckpointUnavailableError(RuntimeError):
    """No current FULL+EXACT checkpoint can safely carry the intervention."""


class NumericalRecoveryExecutor:
    """Consume one authorized numerical-recovery Action as a checkpoint-backed successor.

    ``resolve_plan(attempt, directives=..., restore_action_id=...)`` recompiles
    the unchanged candidate and resolves the attempt; the host supplies its
    resolver. This class holds no trainer-specific rule and calls no runtime.
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
        """Record the successor, its directives and submit intent; never call the runtime.

        Raises:
            NumericalCheckpointUnavailableError: No eligible validated checkpoint.
            InterventionReplayError: The restored lineage is unknown or ambiguous.
            StaleRecoveryContextError: The decision or its source changed.
            ProvenanceError: The Action is not a recorded, authorized decision.
        """
        repository = self.repository
        binding = repository.numerical_recovery_bindings.for_action(str(action_id))
        if binding is None:
            raise ProvenanceError("numerical Action has no recovery decision binding")
        prior = repository.numerical_recovery_executions.executed_for_episode(
            str(binding.episode_id)
        )
        if prior is not None:
            if prior.action_id != action_id:
                raise ProvenanceError("another Action already consumed this recovery episode")
            assert prior.successor_attempt_id is not None
            assert prior.runtime_operation_id is not None
            attempt = repository.aggregates.load_attempt(str(prior.successor_attempt_id))
            operation = repository.operations.get(str(prior.runtime_operation_id))
            assert operation is not None
            return attempt, operation, prior
        intervention = repository.training_interventions.for_action(str(action_id))
        if intervention is None:
            raise ProvenanceError("numerical Action has no recorded TrainingIntervention")

        episode = repository.recovery_episodes.get(str(binding.episode_id))
        if episode is None or not repository.recovery_plans.is_effective_and_fresh(
            str(binding.plan_id)
        ):
            raise StaleRecoveryContextError("numerical recovery decision is no longer fresh")
        proposal = binding.proposal
        source = repository.aggregates.load_attempt(episode.context.target.id)
        source_plan = self.resolve_plan(source)
        if not isinstance(source_plan.spec, TrainingExecutionSpec):
            raise ProvenanceError("numerical source does not resolve to a training execution")
        require_managed_restore(
            self.capabilities,
            source_plan,
            self.checkpoint_manager,
            NumericalCheckpointUnavailableError,
        )
        if (
            source_plan.target.kind != "training-attempt"
            or source_plan.target.id != str(source.id)
            or source_plan.spec.candidate_fingerprint != proposal.candidate_fingerprint
            or source.execution_fingerprint != training_execution_fingerprint(source_plan)
        ):
            raise StaleRecoveryContextError("numerical proposal no longer describes the source")

        checkpoint = await select_restore_checkpoint(
            repository, self.checkpoint_manager, episode, NumericalCheckpointUnavailableError
        )
        prior_overrides = source.execution_overrides
        if prior_overrides and prior_overrides[-1].kind == "checkpoint_restore":
            prior_overrides = prior_overrides[:-1]
        draft = RunAttempt(
            id=RunAttemptId.generate(),
            run_id=source.run_id,
            attempt_number=source.attempt_number + 1,
            execution_overrides=(
                *prior_overrides,
                ExecutionOverride(
                    id=f"{action_id}:checkpoint-restore",
                    kind="checkpoint_restore",
                    reason="restore validated FULL+EXACT checkpoint before the unsafe step",
                    values=FrozenDict({"checkpoint_id": str(checkpoint.checkpoint_ref.id)}),
                    action_id=action_id,
                ),
            ),
            checkpoint_ref=checkpoint.checkpoint_ref,
        )
        replay = repository.plan_successor_interventions(
            source.run_id, checkpoint.checkpoint_ref, initial=intervention
        )
        if replay.directives[-1].expected_previous_value != (proposal.previous_learning_rate.value):
            raise StaleRecoveryContextError(
                "the restored trajectory does not reach the rate the proposal reduced"
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
        assert directives[-1].kind is InterventionDirectiveKind.INITIAL
        resolved = self.resolve_plan(draft, directives=directives, restore_action_id=str(action_id))
        carried = resolved.runtime_options.get(TRAINING_INTERVENTIONS)
        if (
            not isinstance(resolved.spec, TrainingExecutionSpec)
            or resolved.target.id != str(draft.id)
            or resolved.spec.candidate_fingerprint != proposal.candidate_fingerprint
            or carried is None
            or TrainingInterventionDirectives.model_validate(thaw(carried)).directives != directives
        ):
            raise ProvenanceError("numerical successor does not resolve to its directives")
        successor = RunAttempt.model_validate(
            {
                **draft.model_dump(mode="json"),
                "execution_fingerprint": training_execution_fingerprint(resolved),
            }
        )
        return repository._record_numerical_recovery_execution(
            action_id,
            successor,
            checkpoint,
            directives,
            request_digest=resolved.request_digest("submit"),
            actor=actor,
            capabilities=self.capabilities,
            operation_id=OperationId.generate(),
            destinations=self.destinations,
        )
