"""Checkpoint-backed OOM execution reconstructs the same successor after restart."""

from __future__ import annotations

import asyncio

from tests.test_checkpoints.helpers import make_bundle
from tests.test_compilation.test_attempt_resolution import _candidate
from tests.test_storage.conftest import make_experiment, make_node, make_run
from tests.test_storage.test_recovery import incident
from tests.test_storage.test_recovery_episodes import signal
from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
from xaytune.compilation import CompilationContext
from xaytune.compilation.attempt_resolution import (
    resolve_training_attempt,
    training_execution_fingerprint,
)
from xaytune.compilation.native import NativeCompiler
from xaytune.core.capabilities import CapabilityDocument, CheckpointCapabilities
from xaytune.core.domain.experiment import CandidateSpecSnapshot, Experiment, ExperimentNode
from xaytune.core.domain.oom_recovery import OOMRecoveryInputsV1, OOMResizeProposal
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.domain.policy import PolicyVerdict
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.core.domain.run import Run, RunAttempt
from xaytune.core.domain.specs import CompilerSpec, RuntimeSpec
from xaytune.core.ids import RunAttemptId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor
from xaytune.core.state.status import ExperimentStatus, RunAttemptStatus, RunStatus
from xaytune.policy import RulePolicyEngine
from xaytune.resilience.oom import OOMRecoveryPlanner
from xaytune.resilience.oom_execution import OOMRecoveryExecutor
from xaytune.resilience.recovery import RecoveryCoordinator
from xaytune.runtimes import RuntimeEventEnvelope, TrainingEventPayload
from xaytune.storage import ControlPlaneRepository, connect, migrate, write_transaction

ACTOR = Actor(type="system", id="oom-executor-test")


def test_executor_validates_checkpoint_and_records_restart_stable_successor(
    connection, tmp_path, db_path
):
    repo = ControlPlaneRepository(connection)
    candidate = _candidate(tmp_path)
    base = make_experiment()
    compiler = NativeCompiler()
    experiment = Experiment.model_validate(
        {
            **base.model_dump(mode="json"),
            "compiler": CompilerSpec(
                name=compiler.descriptor.name, version=compiler.descriptor.plugin_version
            ),
            "runtime": RuntimeSpec(kind="local", version="1"),
            "artifact_root": str(tmp_path / "artifacts"),
        }
    )
    base_node = make_node(experiment)
    snapshot = CandidateSpecSnapshot(candidate=candidate)
    node = ExperimentNode.model_validate(
        {
            **base_node.model_dump(mode="json"),
            "candidate": snapshot,
            "candidate_fingerprint": candidate.candidate_fingerprint(),
        }
    )
    base_run = make_run(node)
    run = Run.model_validate({**base_run.model_dump(mode="json"), "seed": 7})
    compiled = compiler.compile(
        candidate,
        CompilationContext(
            run_id=str(run.id), seed=7, output_uri=str(tmp_path / "artifacts" / str(run.id))
        ),
    )

    def resolve(attempt):
        return resolve_training_attempt(compiled, attempt, "local")

    first_id = RunAttemptId.generate()
    first_draft = RunAttempt(id=first_id, run_id=run.id, attempt_number=1)
    first = RunAttempt.model_validate(
        {
            **first_draft.model_dump(mode="json"),
            "execution_fingerprint": training_execution_fingerprint(resolve(first_draft)),
        }
    )
    with write_transaction(connection):
        repo.aggregates._insert_experiment(experiment)
        repo.aggregates._insert_node(node)
        repo.aggregates._insert_run(run)
        repo.aggregates._insert_attempt(first)
    experiment = repo.transition_experiment(
        experiment.id,
        expected_revision=experiment.revision,
        new_status=ExperimentStatus.ACTIVE,
        actor=ACTOR,
    )
    run = repo.transition_run(
        run.id, expected_revision=run.revision, new_status=RunStatus.ACTIVE, actor=ACTOR
    )
    store = LocalCheckpointStore(tmp_path / "bundles")
    manager = CheckpointManager(SerializedStateCodec(), store)
    state, context, restore = make_bundle(
        tmp_path / "checkpoint-source",
        attempt_id=first.id,
        candidate=run.candidate_fingerprint,
        execution=first.execution_fingerprint,
    )
    reference = asyncio.run(manager.save(state, context))
    manifest = asyncio.run(store.get(reference)).manifest
    event = RuntimeEventEnvelope(
        event_id="checkpoint-commit",
        sequence=3,
        target=RuntimeOperationTarget(kind="training-attempt", id=str(first.id)),
        payload=TrainingEventPayload(data=manifest.committed_payload(reference)),
    )
    repo.record_checkpoint(
        first.id,
        event.payload.data,
        evidence=FrozenDict(event.model_dump(mode="json")),
        actor=ACTOR,
    )
    first = repo.transition_attempt(
        first.id,
        expected_revision=first.revision,
        new_status=RunAttemptStatus.FAILED,
        actor=ACTOR,
    )
    observed = incident(repo, first, signal=signal("cuda-oom"))
    request = RecoveryRequest(restore_context=restore)
    coordinator = RecoveryCoordinator(repo, manager)
    plan = asyncio.run(coordinator.plan(str(observed.id), request))
    inputs = OOMRecoveryInputsV1(
        plan=plan,
        run_id=run.id,
        candidate_fingerprint=run.candidate_fingerprint,
        execution_state_fingerprint=plan.execution_state_fingerprint,
        current_micro_batch_size=4,
        current_gradient_accumulation=8,
        world_size=1,
    )
    proposal = OOMRecoveryPlanner().plan(inputs)
    assert isinstance(proposal, OOMResizeProposal)
    capabilities = CapabilityDocument(checkpoint=CheckpointCapabilities(full_exact_restore=True))
    governed = repo.propose_oom_recovery_action(
        inputs,
        proposal,
        proposed_by=ACTOR,
        reason="recover CUDA OOM without changing effective batch",
        policy=RulePolicyEngine(default=PolicyVerdict.ALLOW),
        capabilities=capabilities,
    )
    executor = OOMRecoveryExecutor(repo, manager, resolve, capabilities=capabilities)
    successor, operation, receipt = asyncio.run(executor.execute(governed.action.id, actor=ACTOR))
    successor_plan = resolve(successor)
    assert successor.run_id == run.id
    assert successor.attempt_number == 2
    assert successor.execution_fingerprint != first.execution_fingerprint
    assert successor_plan.spec.config["optimization"]["micro_batch_size"] == 2
    assert successor_plan.spec.config["optimization"]["gradient_accumulation"] == 16
    assert successor.checkpoint_ref == reference
    assert operation.request_digest == successor_plan.request_digest("submit")
    assert receipt.action_id == governed.action.id
    assert {override.action_id for override in successor.execution_overrides} == {
        governed.action.id
    }
    with connect(db_path) as reopened:
        migrate(reopened)
        restored = ControlPlaneRepository(reopened)
        replayed_attempt = restored.aggregates.load_attempt(str(successor.id))
        assert resolve(replayed_attempt).request_digest("submit") == operation.request_digest
        replay = OOMRecoveryExecutor(restored, manager, resolve, capabilities=capabilities)
        assert asyncio.run(replay.execute(governed.action.id, actor=ACTOR)) == (
            replayed_attempt,
            operation,
            receipt,
        )
