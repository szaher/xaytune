"""Numerical recovery with real Native workers and the built-in LocalRuntime.

The first worker is made to fail as an armed numerical guard would -- the NaN
observation, then ``TrainingFailed(numerical-nan)`` -- after its first managed
checkpoint. Everything after that is real: the governed ChangeLearningRate, the
TrainingIntervention, the checkpoint-backed successor carrying the directive,
the worker restoring and applying it, its confirmation, and the successor's
checkpoints recording the application they embody.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import torch

from tests.test_experiment.test_host_behaviour import _spec
from tests.test_experiment.test_real_local_oom import _InjectOneOOM
from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
from xaytune.compilation.native import NativeCompiler
from xaytune.core.domain.action import ActionStatus
from xaytune.core.domain.candidate import CheckpointIntent
from xaytune.core.domain.intervention import InterventionDirectiveKind, InterventionOrigin
from xaytune.core.domain.numerical_recovery import NumericalRecoveryPolicyV1
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.domain.policy import PolicyVerdict
from xaytune.core.domain.realization import rebuild_run_realization
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.core.execution_controls import MANAGED_NUMERICAL_RECOVERY
from xaytune.core.ids import OperationId
from xaytune.core.state.status import RunAttemptStatus, RunStatus
from xaytune.core.telemetry import (
    CheckpointCommittedPayload,
    NumericalInstabilityObserved,
    TrainingFailedPayload,
)
from xaytune.experiment import EmbeddedControllerHost
from xaytune.policy import RulePolicyEngine
from xaytune.runtimes import RuntimeEventEnvelope, TrainingEventPayload
from xaytune.runtimes.local import LocalRuntime
from xaytune.workers.native_checkpoint import native_restore_context

DECLARED_LR = 1e-3
HALVE = NumericalRecoveryPolicyV1(learning_rate_multiplier=0.5, minimum_learning_rate=None)


class _InjectOneNaN(_InjectOneOOM):
    """Fail the first worker after its first commit, exactly as the armed guard reports."""

    async def watch(self, reference, cursor=None):
        async for event in self.inner.watch(reference, cursor):
            yield event
            if (
                reference.external_id == self.first_external_id
                and not self.failed
                and isinstance(event.payload.data, CheckpointCommittedPayload)
                and event.payload.data.optimizer_step == 1
            ):
                self.failed = True
                await self.inner.cancel(reference, OperationId.generate())
                for offset, data in enumerate(
                    (
                        NumericalInstabilityObserved(
                            optimizer_step=2, quantity="loss", observation="nan"
                        ),
                        TrainingFailedPayload(reason="numerical-nan"),
                    ),
                    start=1,
                ):
                    yield RuntimeEventEnvelope(
                        event_id=f"injected-nan-{self.first_attempt}-{offset}",
                        target=event.target,
                        stream_generation=event.stream_generation,
                        sequence=event.sequence + offset,
                        payload=TrainingEventPayload(data=data),
                    )
                return


def test_real_local_native_nan_recovers_with_a_training_intervention(tmp_path: Path) -> None:
    async def scenario() -> None:
        spec = _spec(
            tmp_path,
            budget=BudgetSpec(max_runs=1, max_parallel_runs=1, max_failures=2),
            numerical_recovery=HALVE,
        )
        optimization = spec.candidate.training.optimization.model_copy(
            update={"micro_batch_size": 2, "gradient_accumulation": 1}
        )
        assert optimization.learning_rate == DECLARED_LR
        training = spec.candidate.training.model_copy(
            update={
                "optimization": optimization,
                "checkpoint": CheckpointIntent(every_optimizer_steps=1),
            }
        )
        spec = spec.model_copy(
            update={"candidate": spec.candidate.model_copy(update={"training": training})}
        )
        manager = CheckpointManager(
            SerializedStateCodec(), LocalCheckpointStore(tmp_path / "artifacts" / "checkpoints")
        )
        runtime = _InjectOneNaN(LocalRuntime(tmp_path / "runtime"))

        def request_for_incident(incident) -> RecoveryRequest:
            experiment = host.repository.aggregates.load_experiment(
                str(incident.context.experiment_id)
            )
            run = host.repository.aggregates.load_run(incident.context.run_id)
            attempt = host.repository.aggregates.load_attempt(incident.context.target.id)
            source_plan = host._plan(experiment, run, attempt, NativeCompiler())
            assert run.seed is not None
            return RecoveryRequest(
                restore_context=native_restore_context(
                    source_plan,
                    Path(source_plan.spec.config["data"]["path"]),
                    seed=run.seed,
                )
            )

        host = EmbeddedControllerHost(
            tmp_path / "state.db",
            runtimes={"local": lambda _config: runtime},
            policy=RulePolicyEngine(default=PolicyVerdict.ALLOW),
            checkpoint_manager=manager,
            recovery_request_for_incident=request_for_incident,
        )
        try:
            handle = await host.submit(spec)
            await asyncio.wait_for(handle.wait(), timeout=90)
            repository = host.repository
            (node,) = repository.aggregates.nodes_for_experiment(str(handle.experiment_id))
            (run,) = repository.aggregates.runs_for_node(str(node.id))
            attempts = repository.aggregates.attempts_for_run(str(run.id))
            assert run.status is RunStatus.SUCCEEDED
            assert [a.status for a in attempts] == [
                RunAttemptStatus.FAILED,
                RunAttemptStatus.SUCCEEDED,
            ]
            first, second = attempts
            # Same node, same candidate, same run; the LR is not an override.
            assert run.candidate_fingerprint == node.candidate_fingerprint
            assert [o.kind for o in second.execution_overrides] == ["checkpoint_restore"]
            experiment = repository.aggregates.load_experiment(str(handle.experiment_id))
            plan = host._plan(experiment, run, first, NativeCompiler())
            assert MANAGED_NUMERICAL_RECOVERY in plan.runtime_options

            (intervention,) = repository.training_interventions.for_run(str(run.id))
            assert intervention.origin is InterventionOrigin.REACTIVE_POLICY
            assert intervention.mutation.learning_rate == DECLARED_LR * 0.5
            assert repository.actions.get(str(intervention.action_id)).status is (
                ActionStatus.SUCCEEDED
            )
            (directive,) = repository.intervention_directives.for_attempt(str(second.id))
            assert directive.kind is InterventionDirectiveKind.INITIAL
            (application,) = repository.intervention_applications.for_run(str(run.id))
            assert application.id == directive.application_id
            assert application.attempt_id == second.id
            assert (application.previous_value, application.applied_value) == (
                DECLARED_LR,
                DECLARED_LR * 0.5,
            )
            assert application.checkpoint_ancestor == second.checkpoint_ref

            # The successor's checkpoints embody the application, and really run at it.
            captures = repository.checkpoints.for_attempt(str(second.id))
            assert captures
            for capture in captures:
                assert capture.payload.state_manifest is not None
                assert capture.payload.state_manifest.applied_intervention_application_ids == (
                    str(application.id),
                )
            restored = await manager.store.get(captures[-1].payload.checkpoint_ref)
            optimizer = torch.load(
                restored.directory / "optimizer.pt", weights_only=True, map_location="cpu"
            )
            assert {group["initial_lr"] for group in optimizer["param_groups"]} == {
                DECLARED_LR * 0.5
            }
            assert {group["lr"] for group in optimizer["param_groups"]} == {DECLARED_LR * 0.5}
            first_manifest = repository.checkpoints.for_attempt(str(first.id))[0]
            assert first_manifest.payload.state_manifest is not None
            assert first_manifest.payload.state_manifest.applied_intervention_application_ids == ()

            # Restart reconstruction: the durable record rebuilds the same requests.
            for attempt in (first, second):
                (submitted,) = [
                    operation
                    for operation in repository.operations.for_target(
                        "training-attempt", str(attempt.id)
                    )
                    if operation.type == "submit"
                ]
                rebuilt = host._plan(experiment, run, attempt, NativeCompiler())
                assert rebuilt.request_digest("submit") == submitted.request_digest

            stored = repository.get_run_realization(run.id)
            ancestry, checkpoints = repository.run_ancestry(run.id)
            assert stored == rebuild_run_realization(
                run, repository.events_for_run(run.id), ancestry, checkpoints
            )
            assert stored.trajectory is not None
            assert stored.trajectory.application_ids == (application.id,)
        finally:
            await host.close()

    asyncio.run(scenario())
