"""CUDA OOM recovery with real Native workers and the built-in LocalRuntime."""

from __future__ import annotations

import asyncio
from pathlib import Path

from tests.test_experiment.test_host_behaviour import _spec
from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
from xaytune.compilation.native import NativeCompiler
from xaytune.core.domain.candidate import CheckpointIntent
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.domain.policy import PolicyVerdict
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.core.ids import OperationId
from xaytune.core.state.status import RunAttemptStatus, RunStatus
from xaytune.core.telemetry import CheckpointCommittedPayload, IncidentObservedPayload
from xaytune.experiment import EmbeddedControllerHost
from xaytune.policy import RulePolicyEngine
from xaytune.runtimes import RuntimeEventEnvelope, RuntimeStatus, TrainingEventPayload
from xaytune.runtimes.local import LocalRuntime
from xaytune.workers.native_checkpoint import native_restore_context


class _InjectOneOOM:
    """Fail one real worker after its first managed commit; run successor normally."""

    descriptor = LocalRuntime.descriptor

    def __init__(self, inner: LocalRuntime) -> None:
        self.inner = inner
        self.first_attempt: str | None = None
        self.first_external_id: str | None = None
        self.failed = False

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    async def submit_or_get(self, operation_id, plan):
        if self.first_attempt is None:
            self.first_attempt = plan.target.id
        reference = await self.inner.submit_or_get(operation_id, plan)
        if plan.target.id == self.first_attempt and self.first_external_id is None:
            self.first_external_id = reference.external_id
        return reference

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
                yield RuntimeEventEnvelope(
                    event_id=f"injected-oom-{self.first_attempt}",
                    target=event.target,
                    stream_generation=event.stream_generation,
                    sequence=event.sequence + 1,
                    payload=TrainingEventPayload(data=IncidentObservedPayload(reason="cuda-oom")),
                )
                return

    async def get_status(self, reference):
        status = await self.inner.get_status(reference)
        if reference.external_id == self.first_external_id and self.failed:
            while status.state in ("pending", "running", "cancelling"):
                await asyncio.sleep(0.02)
                status = await self.inner.get_status(reference)
            return RuntimeStatus(state="failed", exit_code=1)
        return status


def test_real_local_native_oom_restores_and_continues(tmp_path: Path) -> None:
    async def scenario() -> None:
        spec = _spec(
            tmp_path,
            budget=BudgetSpec(max_runs=1, max_parallel_runs=1, max_failures=2),
        )
        optimization = spec.candidate.training.optimization.model_copy(
            update={"micro_batch_size": 2, "gradient_accumulation": 1}
        )
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
        runtime = _InjectOneOOM(LocalRuntime(tmp_path / "runtime"))

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
            await asyncio.wait_for(handle.wait(), timeout=60)
            (node,) = host.repository.aggregates.nodes_for_experiment(str(handle.experiment_id))
            (run,) = host.repository.aggregates.runs_for_node(str(node.id))
            attempts = host.repository.aggregates.attempts_for_run(str(run.id))
            assert run.status is RunStatus.SUCCEEDED
            assert [attempt.status for attempt in attempts] == [
                RunAttemptStatus.FAILED,
                RunAttemptStatus.SUCCEEDED,
            ]
            assert attempts[1].checkpoint_ref is not None
            assert attempts[1].execution_fingerprint != attempts[0].execution_fingerprint
            assert attempts[1].execution_overrides[0].values["to"] == 1
            assert attempts[1].execution_overrides[1].values["to"] == 2
            receipt = host.repository.recovery_execution_receipts.for_successor(str(attempts[1].id))
            assert receipt is not None
            assert receipt.checkpoint_ref == attempts[1].checkpoint_ref
            first_capture = host.repository.checkpoints.for_attempt(str(attempts[0].id))[0]
            successor_captures = host.repository.checkpoints.for_attempt(str(attempts[1].id))
            assert first_capture.payload.data_cursor is not None
            assert first_capture.payload.data_cursor.next_sample_offset == 2
            assert len(successor_captures) == 1
            assert successor_captures[0].payload.optimizer_step == 2
            assert successor_captures[0].payload.data_cursor is not None
            assert successor_captures[0].payload.data_cursor.next_sample_offset == 4
            first_bytes = await manager.store.get(first_capture.payload.checkpoint_ref)
            successor_bytes = await manager.store.get(successor_captures[0].payload.checkpoint_ref)
            import torch

            initial_optimizer = torch.load(
                first_bytes.directory / "optimizer.pt", weights_only=True, map_location="cpu"
            )
            resumed_optimizer = torch.load(
                successor_bytes.directory / "optimizer.pt", weights_only=True, map_location="cpu"
            )
            assert {int(state["step"]) for state in initial_optimizer["state"].values()} == {1}
            assert {int(state["step"]) for state in resumed_optimizer["state"].values()} == {2}
        finally:
            await host.close()

    asyncio.run(scenario())
