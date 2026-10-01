"""A run whose candidate declares LR 2e-4, failed on a structured nonfinite incident."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from tests.test_checkpoints.helpers import make_bundle
from tests.test_storage.conftest import make_attempt, make_experiment, make_run
from tests.test_storage.test_recovery_episodes import decide, incident, signal
from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
from xaytune.core import (
    CandidateSpecSnapshot,
    ExperimentNode,
    ExperimentNodeId,
    RunAttempt,
    RunAttemptId,
)
from xaytune.core.domain.action import ActionTarget
from xaytune.core.domain.actions import ChangeLearningRate
from xaytune.core.domain.candidate import (
    CandidateSpec,
    DataSpec,
    ModelSpec,
    OptimizationSpec,
    TrainingKind,
    TrainingSpec,
)
from xaytune.core.domain.intervention import (
    InterventionDirective,
    InterventionOrigin,
    InterventionReplayPolicy,
    LearningRateMutation,
    ManualTrigger,
    TrainingIntervention,
)
from xaytune.core.domain.numerical_recovery import NumericalRecoveryPolicyV1
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.domain.policy import PolicyVerdict
from xaytune.core.domain.recovery import RecoveryCheckpointReport, RecoveryRequest
from xaytune.core.domain.run import ExecutionOverride
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import InterventionApplicationId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor, DatasetRef, ModelRef
from xaytune.core.state.status import ExperimentStatus, RunAttemptStatus, RunStatus
from xaytune.policy import RulePolicyEngine
from xaytune.runtimes import RuntimeEventEnvelope, TrainingEventPayload
from xaytune.storage import ControlPlaneRepository, write_transaction

ACTOR = Actor(type="system", id="numerical-coordinator")
REVIEWER = Actor(type="human", id="reviewer")
ALLOW = RulePolicyEngine(default=PolicyVerdict.ALLOW)
APPROVAL = RulePolicyEngine(default=PolicyVerdict.REQUIRE_APPROVAL)
DENY = RulePolicyEngine(default=PolicyVerdict.DENY)
HALVE = NumericalRecoveryPolicyV1(learning_rate_multiplier=0.5, minimum_learning_rate=None)
DECLARED_LR = 2e-4


def lr_node(experiment, learning_rate=DECLARED_LR):
    snapshot = CandidateSpecSnapshot(
        candidate=CandidateSpec(
            model=ModelSpec(model=ModelRef(uri="Qwen/Qwen3-8B")),
            data=DataSpec(dataset=DatasetRef(uri="./data/support-lr.jsonl")),
            training=TrainingSpec(
                kind=TrainingKind.SFT,
                optimization=OptimizationSpec(learning_rate=learning_rate),
            ),
        )
    )
    return ExperimentNode(
        id=ExperimentNodeId.generate(),
        experiment_id=experiment.id,
        candidate=snapshot,
        candidate_fingerprint=snapshot.candidate.candidate_fingerprint(),
        created_by=Actor(type="system", id="controller"),
    )


def seeded_lr_run(connection):
    """Experiment ACTIVE, run ACTIVE, attempt 1 FAILED. Returns (repo, world)."""
    repo = ControlPlaneRepository(connection)
    experiment = make_experiment()
    node = lr_node(experiment)
    run = make_run(node)
    attempt = make_attempt(run)
    with write_transaction(connection):
        repo.aggregates._insert_experiment(experiment)
        repo.aggregates._insert_node(node)
        repo.aggregates._insert_run(run)
        repo.aggregates._insert_attempt(attempt)
    repo.transition_experiment(
        experiment.id,
        expected_revision=experiment.revision,
        new_status=ExperimentStatus.ACTIVE,
        actor=ACTOR,
    )
    run = repo.transition_run(
        run.id, expected_revision=run.revision, new_status=RunStatus.ACTIVE, actor=ACTOR
    )
    attempt = repo.transition_attempt(
        attempt.id,
        expected_revision=attempt.revision,
        new_status=RunAttemptStatus.FAILED,
        actor=ACTOR,
    )
    return repo, {"experiment": experiment, "node": node, "run": run, "attempt": attempt}


def nonfinite_plan(repo, attempt, reason="numerical-nan", sequence=9):
    observed = incident(repo, attempt, sequence=sequence, signal=signal(reason))
    return observed, decide(repo, observed, RecoveryRequest())


def _record_checkpoint(repo, attempt, tmp_path, name, step, sequence, embodied=()):
    run = repo.aggregates.load_run(str(attempt.run_id))
    state, context, _ = make_bundle(
        tmp_path / name, attempt_id=attempt.id, candidate=run.candidate_fingerprint
    )
    manager = CheckpointManager(SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles"))
    manifest = state.state_manifest.model_copy(
        update={"applied_intervention_application_ids": tuple(str(item) for item in embodied)}
    )
    ref = asyncio.run(
        manager.save(replace(state, optimizer_step=step, state_manifest=manifest), context)
    )
    manifest = asyncio.run(manager.store.get(ref)).manifest
    event = RuntimeEventEnvelope(
        event_id=name,
        sequence=sequence,
        target=RuntimeOperationTarget(kind="training-attempt", id=str(attempt.id)),
        payload=TrainingEventPayload(data=manifest.committed_payload(ref)),
    )
    return repo.record_checkpoint(
        attempt.id,
        event.payload.data,
        evidence=FrozenDict(event.model_dump(mode="json")),
        actor=ACTOR,
    )


def checkpointed_lr_run(connection, tmp_path):
    """As seeded_lr_run, with two committed checkpoints on attempt 1 before it failed.

    ``c100`` (optimizer step 100) and ``c50`` (step 50) both embody no applications.
    """
    repo = ControlPlaneRepository(connection)
    experiment = make_experiment()
    node = lr_node(experiment)
    run = make_run(node)
    attempt = RunAttempt(
        id=RunAttemptId.generate(),
        run_id=run.id,
        attempt_number=1,
        execution_fingerprint="execution-a",
    )
    with write_transaction(connection):
        repo.aggregates._insert_experiment(experiment)
        repo.aggregates._insert_node(node)
        repo.aggregates._insert_run(run)
        repo.aggregates._insert_attempt(attempt)
    repo.transition_experiment(
        experiment.id,
        expected_revision=experiment.revision,
        new_status=ExperimentStatus.ACTIVE,
        actor=ACTOR,
    )
    run = repo.transition_run(
        run.id, expected_revision=run.revision, new_status=RunStatus.ACTIVE, actor=ACTOR
    )
    c50 = _record_checkpoint(repo, attempt, tmp_path, "c50", 50, 2)
    c100 = _record_checkpoint(repo, attempt, tmp_path, "c100", 100, 3)
    attempt = repo.transition_attempt(
        attempt.id,
        expected_revision=repo.aggregates.load_attempt(str(attempt.id)).revision,
        new_status=RunAttemptStatus.FAILED,
        actor=ACTOR,
    )
    return repo, {
        "experiment": experiment,
        "node": node,
        "run": run,
        "attempt": attempt,
        "c50": c50,
        "c100": c100,
    }


def restored_successor(repo, run, source, checkpoint, number=2):
    """A successor attempt restored from *checkpoint*, as the executor gate will create it."""
    successor = RunAttempt(
        id=RunAttemptId.generate(),
        run_id=run.id,
        attempt_number=number,
        execution_fingerprint=source.execution_fingerprint,
        checkpoint_ref=None if checkpoint is None else checkpoint.payload.checkpoint_ref,
    )
    with write_transaction(repo._connection):
        repo.aggregates._insert_attempt(successor)
    return successor


def human_intervention(repo, run, learning_rate=5e-5):
    """A generic, non-numerical intervention: a researcher's governed LR change."""
    action = repo.propose_action(
        ChangeLearningRate(
            target=ActionTarget(kind="run", id=str(run.id)), learning_rate=learning_rate
        ),
        experiment_id=run.experiment_id,
        proposed_by=REVIEWER,
        reason="researcher lowers LR",
        policy=ALLOW,
        capabilities=None,
    ).action
    return repo.record_training_intervention(
        TrainingIntervention(
            run_id=run.id,
            action_id=action.id,
            origin=InterventionOrigin.REACTIVE_HUMAN,
            trigger=ManualTrigger(actor=REVIEWER),
            replay_policy=InterventionReplayPolicy.APPLY_ONCE,
            mutation=LearningRateMutation(learning_rate=learning_rate),
            rationale="researcher judgement",
        ),
        actor=ACTOR,
    )


record_checkpoint = _record_checkpoint


def executed_successor(repo, run, source, action_id, checkpoint):
    """Commit a numerical successor through the real successor transaction.

    Stands in for the executor's I/O half (byte validation and plan resolution):
    the directives come from the repository's own replay plan, with fresh
    application ids, exactly as the executor assigns them.
    """
    intervention = repo.training_interventions.for_action(str(action_id))
    reference = checkpoint.payload.checkpoint_ref
    prior = source.execution_overrides
    if prior and prior[-1].kind == "checkpoint_restore":
        prior = prior[:-1]
    successor = RunAttempt(
        id=RunAttemptId.generate(),
        run_id=run.id,
        attempt_number=source.attempt_number + 1,
        execution_fingerprint=source.execution_fingerprint,
        execution_overrides=(
            *prior,
            ExecutionOverride(
                id=f"{action_id}:checkpoint-restore",
                kind="checkpoint_restore",
                reason="restore validated FULL+EXACT checkpoint",
                values=FrozenDict({"checkpoint_id": str(reference.id)}),
                action_id=action_id,
            ),
        ),
        checkpoint_ref=reference,
    )
    plan = repo.plan_successor_interventions(run.id, reference, initial=intervention)
    directives = tuple(
        InterventionDirective(
            application_id=InterventionApplicationId.generate(),
            intervention_id=planned.intervention_id,
            attempt_id=successor.id,
            ordinal=ordinal,
            kind=planned.kind,
            mutation=planned.mutation,
            expected_previous_value=planned.expected_previous_value,
        )
        for ordinal, planned in enumerate(plan.directives)
    )
    producer = repo.aggregates.load_attempt(checkpoint.context.target.id)
    successor, _, receipt = repo._record_numerical_recovery_execution(
        action_id,
        successor,
        RecoveryCheckpointReport.from_record(checkpoint, producer.attempt_number),
        directives,
        request_digest=fingerprint({"submit": str(successor.id)}),
        actor=ACTOR,
    )
    return successor, directives, receipt
