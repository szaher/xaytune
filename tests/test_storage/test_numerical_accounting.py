"""An executed numerical recovery consumes the experiment's recovery budget.

``max_recoveries_per_experiment`` counts episodes, once each, whichever
execution family consumed them: a generic RETRY/RESUME, an executed OOM
receipt, or an executed numerical receipt.
"""

from __future__ import annotations

import pytest

from tests.test_storage.conftest import make_run
from tests.test_storage.numerical_fixtures import (
    ACTOR,
    ALLOW,
    HALVE,
    checkpointed_lr_run,
    executed_successor,
    record_checkpoint,
)
from tests.test_storage.test_numerical_successor_model import PLANNER, confirm
from tests.test_storage.test_recovery_episodes import decide, incident, signal
from xaytune.core.domain.numerical_recovery import NumericalLRProposal
from xaytune.core.domain.recovery import RecoveryLimits, RecoveryRequest
from xaytune.core.domain.run import RunAttempt
from xaytune.core.ids import RunAttemptId
from xaytune.core.state.status import RunAttemptStatus, RunStatus
from xaytune.storage import write_transaction
from xaytune.storage.control_plane import StaleRecoveryContextError


def governed(repo, attempt, request, sequence=9):
    observed = incident(repo, attempt, sequence=sequence, signal=signal("numerical-nan"))
    plan = decide(repo, observed, request)
    inputs = repo.numerical_recovery_inputs(str(plan.id), HALVE)
    proposal = PLANNER.plan(inputs)
    assert isinstance(proposal, NumericalLRProposal)
    action = repo.propose_numerical_recovery_action(
        inputs,
        proposal,
        proposed_by=ACTOR,
        reason="stabilise the continuing trajectory",
        policy=ALLOW,
        capabilities=None,
    ).action
    repo.record_numerical_intervention(action.id, actor=ACTOR, rationale="loss became nonfinite")
    return plan, action


def fail(repo, attempt):
    return repo.transition_attempt(
        attempt.id,
        expected_revision=repo.aggregates.load_attempt(str(attempt.id)).revision,
        new_status=RunAttemptStatus.FAILED,
        actor=ACTOR,
    )


def second_run(repo, w, tmp_path):
    """Another run of the same node, its attempt 1 checkpointed then failed."""
    run = make_run(w["node"])
    repo.create_run(run, actor=ACTOR)
    run = repo.transition_run(
        run.id, expected_revision=run.revision, new_status=RunStatus.ACTIVE, actor=ACTOR
    )
    attempt = RunAttempt(
        id=RunAttemptId.generate(),
        run_id=run.id,
        attempt_number=1,
        execution_fingerprint="execution-a",
    )
    with write_transaction(repo._connection):
        repo.aggregates._insert_attempt(attempt)
    checkpoint = record_checkpoint(repo, attempt, tmp_path, "b100", 100, 3)
    return run, fail(repo, attempt), checkpoint


def usage(repo, w):
    return repo.recovery_episodes.usage_excluding(str(w["experiment"].id), "other")


def test_an_executed_numerical_episode_counts_once_toward_the_next_snapshot(connection, tmp_path):
    repo, w = checkpointed_lr_run(connection, tmp_path)
    _, action = governed(repo, w["attempt"], RecoveryRequest())
    assert usage(repo, w) == 0, "a pending decision is not usage"
    attempt2, (directive,), _ = executed_successor(
        repo, w["run"], w["attempt"], action.id, w["c100"]
    )
    assert usage(repo, w) == 1
    confirm(repo, directive)
    assert usage(repo, w) == 1, "confirming the effect does not count the episode again"

    plan2, _ = governed(repo, fail(repo, attempt2), RecoveryRequest(), sequence=5)
    episode2 = repo.recovery_episodes.get(str(plan2.episode_id))
    assert repo.recovery_snapshot(episode2).experiment_recovery_usage_excluding_target == 1


def test_the_experiment_limit_refuses_a_second_numerical_execution_across_runs(
    connection, tmp_path
):
    """Both episodes are decided while the budget is free; only one may execute."""
    one = RecoveryRequest(limits=RecoveryLimits(max_recoveries_per_experiment=1))
    repo, w = checkpointed_lr_run(connection, tmp_path)
    run_b, attempt_b, checkpoint_b = second_run(repo, w, tmp_path)
    _, first = governed(repo, w["attempt"], one)
    _, second = governed(repo, attempt_b, one, sequence=4)

    executed_successor(repo, w["run"], w["attempt"], first.id, w["c100"])
    before = repo.numerical_recovery_executions.for_action(str(second.id))
    with pytest.raises(StaleRecoveryContextError, match="limits"):
        executed_successor(repo, run_b, attempt_b, second.id, checkpoint_b)
    assert repo.numerical_recovery_executions.for_action(str(second.id)) == before
    assert len(repo.aggregates.attempts_for_run(str(run_b.id))) == 1
    assert usage(repo, w) == 1


def test_oom_and_numerical_executions_are_two_units_of_experiment_usage(connection, tmp_path):
    from tests.test_storage.test_oom_execution import successor as oom_successor
    from tests.test_storage.test_recovery_action_bindings import propose
    from xaytune.core.domain.oom_recovery import OOMRecoveryInputsV1, OOMResizeProposal
    from xaytune.core.domain.recovery import RecoveryCheckpointReport
    from xaytune.resilience.oom import OOMRecoveryPlanner

    repo, w = checkpointed_lr_run(connection, tmp_path)
    _, numerical = governed(repo, w["attempt"], RecoveryRequest())
    executed_successor(repo, w["run"], w["attempt"], numerical.id, w["c100"])

    run_b, attempt_b, checkpoint_b = second_run(repo, w, tmp_path)
    observed = incident(repo, attempt_b, sequence=4, signal=signal("cuda-oom"))
    plan = decide(repo, observed, RecoveryRequest())
    inputs = OOMRecoveryInputsV1(
        plan=plan,
        run_id=run_b.id,
        candidate_fingerprint=plan.inputs.candidate_fingerprint,
        execution_state_fingerprint=plan.execution_state_fingerprint,
        current_micro_batch_size=4,
        current_gradient_accumulation=8,
        world_size=8,
    )
    proposal = OOMRecoveryPlanner().plan(inputs)
    assert isinstance(proposal, OOMResizeProposal)
    action = propose(repo, inputs, proposal).action
    report = RecoveryCheckpointReport.from_record(checkpoint_b, attempt_b.attempt_number)
    repo._record_oom_recovery_execution(
        action.id,
        oom_successor(attempt_b, proposal, action, report),
        report,
        request_digest="sha256:" + "2" * 64,
        actor=ACTOR,
    )
    assert usage(repo, w) == 2
