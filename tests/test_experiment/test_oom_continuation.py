"""A CUDA OOM produces one governed resized successor that continues the Run."""

from __future__ import annotations

import asyncio

import pytest

from tests.test_checkpoints.helpers import make_bundle
from tests.test_experiment.test_host_behaviour import _spec
from tests.test_storage.test_recovery import incident as record_test_incident
from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
from xaytune.compilation.attempt_resolution import training_execution_fingerprint
from xaytune.core.domain.action import ActionOutcome, ActionStatus
from xaytune.core.domain.budget import BudgetDimension
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.domain.policy import PolicyVerdict
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.core.domain.recovery_execution import RecoveryExecutionOutcome
from xaytune.core.refs import Actor, CheckpointRef, RuntimeRef
from xaytune.core.state.status import RunAttemptStatus, RunStatus
from xaytune.core.telemetry import IncidentObservedPayload
from xaytune.experiment import EmbeddedControllerHost, ReconciliationEscalatedError
from xaytune.policy import RulePolicyEngine
from xaytune.runtimes import RuntimeEventEnvelope, RuntimeStatus, TrainingEventPayload
from xaytune.runtimes.local.runtime import LocalRuntime


class InjectedOOMRuntime:
    descriptor = LocalRuntime.descriptor

    def __init__(
        self, manager, tmp_path, *, publish_checkpoint=True, failures=1, restore_supported=True
    ):
        self.manager = manager
        self.tmp_path = tmp_path
        self.publish_checkpoint = publish_checkpoint
        self.failures = failures
        self.restore_supported = restore_supported
        self.plans = []
        self.events = {}
        self.outcomes = {}
        self.restore_context = None
        self.restored = []

    def capabilities(self):
        base = LocalRuntime.capabilities(self)
        assert base.checkpoint is not None
        return base.model_copy(
            update={
                "extensions": {},
                "checkpoint": base.checkpoint.model_copy(
                    update={
                        "atomic_commit": True,
                        "full_exact_restore": self.restore_supported,
                        "formats": (),
                    }
                ),
            }
        )

    async def submit_or_get(self, operation_id, plan):
        self.plans.append(plan)
        attempt_id = plan.target.id
        reference = RuntimeRef(backend="local", external_id=attempt_id)
        if len(self.plans) <= self.failures:
            state, context, restore = make_bundle(
                self.tmp_path / f"capture-{attempt_id}",
                attempt_id=attempt_id,
                candidate=plan.spec.candidate_fingerprint,
                execution=training_execution_fingerprint(plan),
            )
            self.restore_context = restore
            events = []
            if self.publish_checkpoint:
                checkpoint = await self.manager.save(state, context)
                manifest = (await self.manager.store.get(checkpoint)).manifest
                events.append(
                    RuntimeEventEnvelope(
                        event_id="oom-checkpoint",
                        sequence=0,
                        target=plan.target,
                        payload=TrainingEventPayload(data=manifest.committed_payload(checkpoint)),
                    )
                )
            events.append(
                RuntimeEventEnvelope(
                    event_id="injected-cuda-oom",
                    sequence=len(events),
                    target=plan.target,
                    payload=TrainingEventPayload(data=IncidentObservedPayload(reason="cuda-oom")),
                )
            )
            self.events[attempt_id] = tuple(events)
            self.outcomes[attempt_id] = RuntimeStatus(state="failed", exit_code=1)
        else:
            reference_payload = plan.runtime_options.get("checkpoint_restore")
            assert reference_payload is not None
            assert self.restore_context is not None
            checkpoint_ref = CheckpointRef.model_validate(reference_payload)
            restored = await self.manager.restore(checkpoint_ref, self.restore_context)
            assert restored.manifest.context.checkpoint_id == checkpoint_ref.id
            self.restored.append(checkpoint_ref)
            self.events[attempt_id] = ()
            self.outcomes[attempt_id] = RuntimeStatus(state="succeeded", exit_code=0)
        return reference

    async def watch(self, reference, cursor):
        for event in self.events[reference.external_id]:
            yield event

    async def get_status(self, reference):
        return self.outcomes[reference.external_id]

    def close(self):
        pass


def _oom_spec(tmp_path, *, max_failures=2):
    base = _spec(
        tmp_path,
        budget=BudgetSpec(max_runs=1, max_parallel_runs=1, max_failures=max_failures),
    )
    optimization = base.candidate.training.optimization.model_copy(
        update={"micro_batch_size": 4, "gradient_accumulation": 8}
    )
    candidate = base.candidate.model_copy(
        update={
            "training": base.candidate.training.model_copy(update={"optimization": optimization})
        }
    )
    return base.model_copy(update={"candidate": candidate})


def test_injected_cuda_oom_resizes_and_continues_training(tmp_path):
    async def scenario():
        manager = CheckpointManager(
            SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles")
        )
        runtime = InjectedOOMRuntime(manager, tmp_path)

        def request_for_incident(incident):
            assert incident.context.target.kind == "training-attempt"
            assert runtime.restore_context is not None
            return RecoveryRequest(restore_context=runtime.restore_context)

        host = EmbeddedControllerHost(
            tmp_path / "state.db",
            runtimes={"local": lambda config: runtime},
            policy=RulePolicyEngine(default=PolicyVerdict.ALLOW),
            checkpoint_manager=manager,
            recovery_request_for_incident=request_for_incident,
        )
        try:
            handle = await host.submit(_oom_spec(tmp_path))
            await asyncio.wait_for(handle.wait(), timeout=10)
            (node,) = host.repository.aggregates.nodes_for_experiment(str(handle.experiment_id))
            (run,) = host.repository.aggregates.runs_for_node(str(node.id))
            attempts = host.repository.aggregates.attempts_for_run(str(run.id))
            assert run.status is RunStatus.SUCCEEDED
            assert len(attempts) == 2
            assert attempts[0].status is RunAttemptStatus.FAILED
            assert attempts[1].status is RunAttemptStatus.SUCCEEDED
            assert attempts[0].execution_fingerprint != attempts[1].execution_fingerprint
            assert attempts[1].checkpoint_ref is not None
            assert runtime.restored == [attempts[1].checkpoint_ref]
            assert attempts[1].execution_overrides[-1].kind == "checkpoint_restore"
            assert runtime.plans[0].spec.config["optimization"]["micro_batch_size"] == 4
            assert runtime.plans[1].spec.config["optimization"]["micro_batch_size"] == 2
            assert runtime.plans[1].spec.config["optimization"]["gradient_accumulation"] == 16
            assert (
                host.repository.recovery_episodes.for_attempt(
                    RuntimeOperationTarget(kind="training-attempt", id=str(attempts[0].id))
                )
                is not None
            )
            (action,) = host.repository.actions.for_target("run", str(run.id))
            assert action.status is ActionStatus.SUCCEEDED
            assert action.outcome is ActionOutcome.APPLIED
            receipt = host.repository.recovery_execution_receipts.for_successor(str(attempts[1].id))
            assert receipt is not None and receipt.action_id == action.id
            operation = host.repository.operations.get(str(receipt.runtime_operation_id))
            assert operation is not None and operation.caused_by_action_id == action.id
            budget = host.repository.budget_status(handle.experiment_id)
            assert budget is not None
            assert budget.of(BudgetDimension.RUNS).consumed == 1
            assert budget.of(BudgetDimension.FAILURES).consumed == 1
        finally:
            await host.close()

    asyncio.run(scenario())


def test_oom_approval_waits_then_executes_same_bound_action(tmp_path):
    async def scenario():
        manager = CheckpointManager(
            SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles")
        )
        runtime = InjectedOOMRuntime(manager, tmp_path)
        host = EmbeddedControllerHost(
            tmp_path / "state.db",
            runtimes={"local": lambda config: runtime},
            policy=RulePolicyEngine(default=PolicyVerdict.REQUIRE_APPROVAL),
            checkpoint_manager=manager,
            recovery_request_for_incident=lambda incident: RecoveryRequest(
                restore_context=runtime.restore_context
            ),
        )
        try:
            handle = await host.submit(_oom_spec(tmp_path))
            resting = await asyncio.wait_for(handle.wait(), timeout=10)
            assert resting.quiescent and resting.next_stage == "action-approval"
            (node,) = host.repository.aggregates.nodes_for_experiment(str(handle.experiment_id))
            (run,) = host.repository.aggregates.runs_for_node(str(node.id))
            assert run.status is RunStatus.ACTIVE
            assert len(host.repository.aggregates.attempts_for_run(str(run.id))) == 1
            (action,) = host.repository.actions.for_target("run", str(run.id))
            assert action.status is ActionStatus.APPROVAL_PENDING
            await host.approve_action(
                action.id, approver=Actor(type="human", id="reviewer"), reason="approve resize"
            )
            complete = await asyncio.wait_for(handle.wait(), timeout=10)
            assert complete.quiescent
            assert host.repository.aggregates.load_run(str(run.id)).status is RunStatus.SUCCEEDED
            assert len(host.repository.aggregates.attempts_for_run(str(run.id))) == 2
        finally:
            await host.close()

    asyncio.run(scenario())


def test_rejected_oom_approval_abandons_without_successor(tmp_path):
    async def scenario():
        manager = CheckpointManager(
            SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles")
        )
        runtime = InjectedOOMRuntime(manager, tmp_path)
        host = EmbeddedControllerHost(
            tmp_path / "state.db",
            runtimes={"local": lambda config: runtime},
            policy=RulePolicyEngine(default=PolicyVerdict.REQUIRE_APPROVAL),
            checkpoint_manager=manager,
            recovery_request_for_incident=lambda observed: RecoveryRequest(
                restore_context=runtime.restore_context
            ),
        )
        try:
            handle = await host.submit(_oom_spec(tmp_path))
            assert (await handle.wait()).next_stage == "action-approval"
            (node,) = host.repository.aggregates.nodes_for_experiment(str(handle.experiment_id))
            (run,) = host.repository.aggregates.runs_for_node(str(node.id))
            (action,) = host.repository.actions.for_target("run", str(run.id))
            await host.reject_action(
                action.id,
                approver=Actor(type="human", id="reviewer"),
                reason="do not resize this run",
            )
            assert (await handle.wait()).quiescent
            assert host.repository.aggregates.load_run(str(run.id)).status is RunStatus.FAILED
            assert len(host.repository.aggregates.attempts_for_run(str(run.id))) == 1
            binding = host.repository.recovery_action_bindings.for_action(str(action.id))
            assert binding is not None
            (receipt,) = host.repository.recovery_execution_receipts.for_episode(
                str(binding.episode_id)
            )
            assert receipt.outcome is RecoveryExecutionOutcome.ABANDONED
        finally:
            await host.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("publish_checkpoint", "restore_supported"),
    [(False, True), (True, False)],
)
def test_oom_without_checkpoint_or_restore_capability_abandons_action_and_fails_run(
    tmp_path, publish_checkpoint, restore_supported
):
    async def scenario():
        manager = CheckpointManager(
            SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles")
        )
        runtime = InjectedOOMRuntime(
            manager,
            tmp_path,
            publish_checkpoint=publish_checkpoint,
            restore_supported=restore_supported,
        )
        host = EmbeddedControllerHost(
            tmp_path / "state.db",
            runtimes={"local": lambda config: runtime},
            policy=RulePolicyEngine(default=PolicyVerdict.ALLOW),
            checkpoint_manager=manager,
            recovery_request_for_incident=lambda incident: RecoveryRequest(
                restore_context=runtime.restore_context
            ),
        )
        try:
            handle = await host.submit(_oom_spec(tmp_path))
            result = await asyncio.wait_for(handle.wait(), timeout=10)
            assert result.quiescent
            (node,) = host.repository.aggregates.nodes_for_experiment(str(handle.experiment_id))
            (run,) = host.repository.aggregates.runs_for_node(str(node.id))
            assert run.status is RunStatus.FAILED
            assert len(host.repository.aggregates.attempts_for_run(str(run.id))) == 1
            (action,) = host.repository.actions.for_target("run", str(run.id))
            assert action.status is ActionStatus.REJECTED
            (receipt,) = host.repository.recovery_execution_receipts.for_episode(
                str(host.repository.recovery_action_bindings.for_action(str(action.id)).episode_id)
            )
            assert receipt.outcome is RecoveryExecutionOutcome.ABANDONED
        finally:
            await host.close()

    asyncio.run(scenario())


def test_second_oom_replays_cumulative_lineage_and_recovers_again(tmp_path):
    async def scenario():
        manager = CheckpointManager(
            SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles")
        )
        runtime = InjectedOOMRuntime(manager, tmp_path, failures=2)
        host = EmbeddedControllerHost(
            tmp_path / "state.db",
            runtimes={"local": lambda config: runtime},
            policy=RulePolicyEngine(default=PolicyVerdict.ALLOW),
            checkpoint_manager=manager,
            recovery_request_for_incident=lambda incident: RecoveryRequest(
                restore_context=runtime.restore_context
            ),
        )
        try:
            handle = await host.submit(_oom_spec(tmp_path, max_failures=3))
            result = await asyncio.wait_for(handle.wait(), timeout=10)
            assert result.quiescent
            (node,) = host.repository.aggregates.nodes_for_experiment(str(handle.experiment_id))
            (run,) = host.repository.aggregates.runs_for_node(str(node.id))
            attempts = host.repository.aggregates.attempts_for_run(str(run.id))
            assert run.status is RunStatus.SUCCEEDED
            assert len(attempts) == 3
            micro_batches = [
                plan.spec.config["optimization"]["micro_batch_size"] for plan in runtime.plans
            ]
            assert micro_batches == [4, 2, 1]
            assert [
                plan.spec.config["optimization"]["gradient_accumulation"] for plan in runtime.plans
            ] == [8, 16, 32]
            assert [override.kind for override in attempts[-1].execution_overrides] == [
                "micro_batch_resize",
                "gradient_accumulation_adjustment",
                "micro_batch_resize",
                "gradient_accumulation_adjustment",
                "checkpoint_restore",
            ]
            assert len(host.repository.actions.for_target("run", str(run.id))) == 2
        finally:
            await host.close()

    asyncio.run(scenario())


def test_restart_reuses_stored_request_and_pending_approval(tmp_path):
    async def scenario():
        manager = CheckpointManager(
            SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles")
        )
        runtime = InjectedOOMRuntime(manager, tmp_path)
        database = tmp_path / "state.db"
        first = EmbeddedControllerHost(
            database,
            runtimes={"local": lambda config: runtime},
            policy=RulePolicyEngine(default=PolicyVerdict.REQUIRE_APPROVAL),
            checkpoint_manager=manager,
            recovery_request_for_incident=lambda incident: RecoveryRequest(
                restore_context=runtime.restore_context
            ),
        )
        handle = await first.submit(_oom_spec(tmp_path))
        awaiting = await asyncio.wait_for(handle.wait(), timeout=10)
        assert awaiting.next_stage == "action-approval"
        experiment_id = handle.experiment_id
        (node,) = first.repository.aggregates.nodes_for_experiment(str(experiment_id))
        (run,) = first.repository.aggregates.runs_for_node(str(node.id))
        (action,) = first.repository.actions.for_target("run", str(run.id))
        await first.close()

        second = EmbeddedControllerHost(
            database,
            runtimes={"local": lambda config: runtime},
            policy=RulePolicyEngine(default=PolicyVerdict.REQUIRE_APPROVAL),
            checkpoint_manager=manager,
            # The original RecoveryRequest is already in the episode.
        )
        try:
            resumed = await second.attach(experiment_id)
            assert (await resumed.wait()).next_stage == "action-approval"
            assert second.repository.actions.for_target("run", str(run.id)) == (action,)
            await second.approve_action(
                action.id, approver=Actor(type="human", id="reviewer"), reason="approve resize"
            )
            assert (await asyncio.wait_for(resumed.wait(), timeout=10)).quiescent
            assert second.repository.aggregates.load_run(str(run.id)).status is RunStatus.SUCCEEDED
            assert len(second.repository.aggregates.attempts_for_run(str(run.id))) == 2
        finally:
            await second.close()

    asyncio.run(scenario())


def test_missing_first_request_leaves_recovery_gap_repairable_after_restart(tmp_path):
    async def scenario():
        manager = CheckpointManager(
            SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles")
        )
        runtime = InjectedOOMRuntime(manager, tmp_path)
        database = tmp_path / "state.db"
        first = EmbeddedControllerHost(
            database,
            runtimes={"local": lambda config: runtime},
            policy=RulePolicyEngine(default=PolicyVerdict.ALLOW),
            checkpoint_manager=manager,
        )
        handle = await first.submit(_oom_spec(tmp_path))
        with pytest.raises(ReconciliationEscalatedError, match="explicit RecoveryRequest"):
            await handle.wait()
        experiment_id = handle.experiment_id
        (node,) = first.repository.aggregates.nodes_for_experiment(str(experiment_id))
        (run,) = first.repository.aggregates.runs_for_node(str(node.id))
        (attempt,) = first.repository.aggregates.attempts_for_run(str(run.id))
        target = RuntimeOperationTarget(kind="training-attempt", id=str(attempt.id))
        assert first.repository.aggregates.load_run(str(run.id)).status is RunStatus.ACTIVE
        assert first.repository.recovery_episodes.for_attempt(target) is None
        assert first.repository.actions.for_target("run", str(run.id)) == ()
        await first.close()

        second = EmbeddedControllerHost(
            database,
            runtimes={"local": lambda config: runtime},
            policy=RulePolicyEngine(default=PolicyVerdict.ALLOW),
            checkpoint_manager=manager,
            recovery_request_for_incident=lambda observed: RecoveryRequest(
                restore_context=runtime.restore_context
            ),
        )
        try:
            repaired = await second.attach(experiment_id)
            assert (await asyncio.wait_for(repaired.wait(), timeout=10)).quiescent
            assert second.repository.aggregates.load_run(str(run.id)).status is RunStatus.SUCCEEDED
            assert len(second.repository.aggregates.attempts_for_run(str(run.id))) == 2
        finally:
            await second.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("old_action_response", ["approve", "reject"])
@pytest.mark.parametrize("current_action_response", ["approve", "reject"])
def test_new_evidence_supersedes_pending_oom_action_before_execution(
    tmp_path, old_action_response, current_action_response
):
    async def scenario():
        manager = CheckpointManager(
            SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles")
        )
        runtime = InjectedOOMRuntime(manager, tmp_path)
        host = EmbeddedControllerHost(
            tmp_path / "state.db",
            runtimes={"local": lambda config: runtime},
            policy=RulePolicyEngine(default=PolicyVerdict.REQUIRE_APPROVAL),
            checkpoint_manager=manager,
            recovery_request_for_incident=lambda observed: RecoveryRequest(
                restore_context=runtime.restore_context
            ),
        )
        try:
            handle = await host.submit(_oom_spec(tmp_path))
            assert (await handle.wait()).next_stage == "action-approval"
            (node,) = host.repository.aggregates.nodes_for_experiment(str(handle.experiment_id))
            (run,) = host.repository.aggregates.runs_for_node(str(node.id))
            (attempt,) = host.repository.aggregates.attempts_for_run(str(run.id))
            (old_action,) = host.repository.actions.for_target("run", str(run.id))
            record_test_incident(host.repository, attempt, sequence=9)
            old_plan = host.repository.recovery_action_bindings.for_action(str(old_action.id))
            assert old_plan is not None
            assert not host.repository.recovery_plans.is_effective_and_fresh(str(old_plan.plan_id))

            resolve_old = (
                host.approve_action if old_action_response == "approve" else host.reject_action
            )
            await resolve_old(
                old_action.id,
                approver=Actor(type="human", id="reviewer"),
                reason="human decision raced new evidence",
            )
            actions = host.repository.actions.for_target("run", str(run.id))
            assert len(actions) == 2
            assert host.repository.actions.get(str(old_action.id)).status is ActionStatus.REJECTED
            (receipt,) = host.repository.recovery_execution_receipts.for_episode(
                str(old_plan.episode_id)
            )
            assert receipt.outcome is RecoveryExecutionOutcome.SUPERSEDED
            current_action = next(action for action in actions if action.id != old_action.id)
            assert current_action.status is ActionStatus.APPROVAL_PENDING
            assert len(host.repository.aggregates.attempts_for_run(str(run.id))) == 1
            resolve_current = (
                host.approve_action if current_action_response == "approve" else host.reject_action
            )
            await resolve_current(
                current_action.id,
                approver=Actor(type="human", id="reviewer"),
                reason="reviewed all evidence",
            )
            assert (await handle.wait()).quiescent
            expected = (
                RunStatus.SUCCEEDED if current_action_response == "approve" else RunStatus.FAILED
            )
            assert host.repository.aggregates.load_run(str(run.id)).status is expected
            if current_action_response == "reject":
                receipts = host.repository.recovery_execution_receipts.for_episode(
                    str(old_plan.episode_id)
                )
                assert {item.outcome for item in receipts} == {
                    RecoveryExecutionOutcome.SUPERSEDED,
                    RecoveryExecutionOutcome.ABANDONED,
                }
        finally:
            await host.close()

    asyncio.run(scenario())
