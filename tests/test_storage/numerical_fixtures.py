"""A run whose candidate declares LR 2e-4, failed on a structured nonfinite incident."""

from __future__ import annotations

from tests.test_storage.conftest import make_attempt, make_experiment, make_run
from tests.test_storage.test_recovery_episodes import decide, incident, signal
from xaytune.core import CandidateSpecSnapshot, ExperimentNode, ExperimentNodeId
from xaytune.core.domain.candidate import (
    CandidateSpec,
    DataSpec,
    ModelSpec,
    OptimizationSpec,
    TrainingKind,
    TrainingSpec,
)
from xaytune.core.domain.numerical_recovery import NumericalRecoveryPolicyV1
from xaytune.core.domain.policy import PolicyVerdict
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.core.refs import Actor, DatasetRef, ModelRef
from xaytune.core.state.status import ExperimentStatus, RunAttemptStatus, RunStatus
from xaytune.policy import RulePolicyEngine
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
