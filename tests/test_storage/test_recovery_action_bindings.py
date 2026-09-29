"""OOM Action governance is durably bound to one fresh recovery revision."""

from __future__ import annotations

import sqlite3

import pytest
from pydantic import ValidationError

from tests.test_storage.conftest import make_attempt
from tests.test_storage.test_recovery_episodes import decide, incident, signal
from xaytune.core.domain.action import ActionStatus, ActionTarget
from xaytune.core.domain.actions import ResizeMicrobatch, action_from_spec
from xaytune.core.domain.oom_recovery import OOMRecoveryInputsV1, OOMResizeProposal
from xaytune.core.domain.policy import PolicyVerdict
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.core.domain.recovery_action import RecoveryActionBinding
from xaytune.core.ids import RunId
from xaytune.core.refs import Actor
from xaytune.core.state.status import ExperimentStatus, RunAttemptStatus, RunStatus
from xaytune.policy import RulePolicyEngine
from xaytune.resilience.oom import OOMRecoveryPlanner
from xaytune.storage import ControlPlaneRepository, connect, migrate, write_transaction
from xaytune.storage.control_plane import StaleRecoveryContextError
from xaytune.storage.journal import IdempotencyConflictError

ACTOR = Actor(type="system", id="oom-coordinator")
ALLOW = RulePolicyEngine(default=PolicyVerdict.ALLOW)
APPROVAL = RulePolicyEngine(default=PolicyVerdict.REQUIRE_APPROVAL)
DENY = RulePolicyEngine(default=PolicyVerdict.DENY)


def prepared(connection, seeded):
    repo = ControlPlaneRepository(connection)
    experiment = seeded["experiment"]
    run = seeded["run"]
    attempt = seeded["attempt"]
    repo.transition_experiment(
        experiment.id,
        expected_revision=experiment.revision,
        new_status=ExperimentStatus.ACTIVE,
        actor=ACTOR,
    )
    repo.transition_run(
        run.id, expected_revision=run.revision, new_status=RunStatus.ACTIVE, actor=ACTOR
    )
    attempt = repo.transition_attempt(
        attempt.id,
        expected_revision=attempt.revision,
        new_status=RunAttemptStatus.FAILED,
        actor=ACTOR,
    )
    observed = incident(repo, attempt, signal=signal("cuda-oom"))
    plan = decide(repo, observed, RecoveryRequest())
    inputs = OOMRecoveryInputsV1(
        plan=plan,
        run_id=run.id,
        candidate_fingerprint=plan.inputs.candidate_fingerprint,
        execution_state_fingerprint=plan.execution_state_fingerprint,
        current_micro_batch_size=4,
        current_gradient_accumulation=8,
        world_size=8,
    )
    proposal = OOMRecoveryPlanner().plan(inputs)
    assert isinstance(proposal, OOMResizeProposal)
    return repo, inputs, proposal, attempt


def propose(repo, inputs, proposal, *, policy=ALLOW, **kwargs):
    return repo.propose_oom_recovery_action(
        inputs,
        proposal,
        proposed_by=ACTOR,
        reason="preserve effective batch after CUDA OOM",
        policy=policy,
        capabilities=None,
        destinations=("audit",),
        **kwargs,
    )


def count(repo, table):
    return int(repo._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_allowed_action_and_binding_are_atomic_and_survive_restart(connection, seeded, db_path):
    repo, inputs, proposal, _ = prepared(connection, seeded)
    before_outbox = count(repo, "outbox")
    governed = propose(repo, inputs, proposal)
    action = governed.action
    binding = repo.recovery_action_bindings.for_action(str(action.id))
    assert action.status is ActionStatus.VALIDATED
    assert governed.decision is not None and governed.decision.verdict is PolicyVerdict.ALLOW
    assert binding == RecoveryActionBinding.for_proposal(action.id, proposal).model_copy(
        update={"created_at": binding.created_at}
    )
    assert repo.recovery_action_bindings.for_plan(str(inputs.plan.id)) == binding
    assert count(repo, "outbox") > before_outbox
    with connect(db_path) as reopened:
        migrate(reopened)
        restored = ControlPlaneRepository(reopened)
        assert restored.recovery_action_bindings.for_action(str(action.id)) == binding
        assert restored.governed_action(action.id) == governed
        assert propose(restored, inputs, proposal, policy=None) == governed


def test_replay_uses_recorded_action_without_current_policy(connection, seeded):
    repo, inputs, proposal, _ = prepared(connection, seeded)
    first = propose(repo, inputs, proposal)
    before = tuple(count(repo, table) for table in ("actions", "policy_decisions", "outbox"))
    replayed = propose(repo, inputs, proposal, policy=None)
    assert replayed == first
    assert (
        tuple(count(repo, table) for table in ("actions", "policy_decisions", "outbox")) == before
    )


def test_two_proposers_converge_when_the_other_commits_during_policy_evaluation(
    connection, seeded, db_path
):
    repo, inputs, proposal, _ = prepared(connection, seeded)
    winner = None

    class RacingPolicy:
        name = ALLOW.name
        version = ALLOW.version

        def evaluate(self, spec, context):
            nonlocal winner
            other_connection = connect(db_path)
            try:
                migrate(other_connection)
                other = ControlPlaneRepository(other_connection)
                winner = propose(other, inputs, proposal)
            finally:
                other_connection.close()
            return ALLOW.evaluate(spec, context)

    replayed = propose(repo, inputs, proposal, policy=RacingPolicy())
    assert winner is not None and replayed == winner
    assert count(repo, "actions") == 1
    assert count(repo, "recovery_action_bindings") == 1


def test_approval_path_has_binding_before_human_decision(connection, seeded):
    repo, inputs, proposal, _ = prepared(connection, seeded)
    governed = propose(repo, inputs, proposal, policy=APPROVAL)
    assert governed.action.status is ActionStatus.APPROVAL_PENDING
    assert repo.recovery_action_bindings.for_action(str(governed.action.id)) is not None
    approved = repo.approve_action(
        governed.action.id,
        approver=Actor(type="human", id="reviewer"),
        reason="approved for this episode",
    )
    assert approved.status is ActionStatus.APPROVED
    assert repo.recovery_action_bindings.for_action(str(approved.id)).plan_id == inputs.plan.id


def test_denied_action_still_has_immutable_recovery_origin(connection, seeded):
    repo, inputs, proposal, _ = prepared(connection, seeded)
    governed = propose(repo, inputs, proposal, policy=DENY)
    assert governed.action.status is ActionStatus.REJECTED
    assert (
        repo.recovery_action_bindings.for_action(str(governed.action.id)).plan_id == inputs.plan.id
    )


def test_new_evidence_stales_the_plan_before_an_action_can_be_bound(connection, seeded):
    repo, inputs, proposal, attempt = prepared(connection, seeded)
    incident(repo, attempt, sequence=10, signal=signal("process-failure"))
    before = tuple(
        count(repo, table)
        for table in ("actions", "recovery_action_bindings", "policy_decisions", "outbox")
    )
    with pytest.raises(StaleRecoveryContextError, match="open and fresh"):
        propose(repo, inputs, proposal)
    assert (
        tuple(
            count(repo, table)
            for table in ("actions", "recovery_action_bindings", "policy_decisions", "outbox")
        )
        == before
    )


def test_evidence_arriving_during_policy_evaluation_is_rechecked_under_write_lock(
    connection, seeded
):
    repo, inputs, proposal, attempt = prepared(connection, seeded)

    class RacingPolicy:
        name = ALLOW.name
        version = ALLOW.version

        def evaluate(self, spec, context):
            incident(repo, attempt, sequence=10, signal=signal("process-failure"))
            return ALLOW.evaluate(spec, context)

    with pytest.raises(StaleRecoveryContextError, match="open and fresh"):
        propose(repo, inputs, proposal, policy=RacingPolicy())
    assert count(repo, "actions") == 0
    assert count(repo, "recovery_action_bindings") == 0


def test_superseded_plan_revision_cannot_propose_its_old_action(connection, seeded):
    repo, inputs, proposal, attempt = prepared(connection, seeded)
    later = incident(repo, attempt, sequence=10, signal=signal("process-failure"))
    effective = decide(repo, later)
    assert effective.sequence == inputs.plan.sequence + 1
    with pytest.raises(StaleRecoveryContextError, match="open and fresh"):
        propose(repo, inputs, proposal)
    assert repo.recovery_action_bindings.for_plan(str(inputs.plan.id)) is None


def test_successor_closure_refuses_new_action_binding(connection, seeded):
    repo, inputs, proposal, _ = prepared(connection, seeded)
    successor = make_attempt(seeded["run"], attempt_number=2)
    with write_transaction(connection):
        repo.aggregates._insert_attempt(successor)
    before = (count(repo, "actions"), count(repo, "recovery_action_bindings"))
    with pytest.raises(StaleRecoveryContextError, match="open and fresh"):
        propose(repo, inputs, proposal)
    assert (count(repo, "actions"), count(repo, "recovery_action_bindings")) == before


def test_conflicting_proposal_for_one_plan_is_refused(connection, seeded):
    repo, inputs, proposal, _ = prepared(connection, seeded)
    first = propose(repo, inputs, proposal)
    different = proposal.model_copy(
        update={
            "action_spec": ResizeMicrobatch(
                target=ActionTarget(kind="run", id=str(inputs.run_id)),
                micro_batch_size=1,
                gradient_accumulation=32,
            )
        }
    )
    with pytest.raises(IdempotencyConflictError, match="OOM proposal"):
        propose(repo, inputs, different)
    assert repo.recovery_action_bindings.for_plan(str(inputs.plan.id)).action_id == first.action.id


def test_binding_rows_are_append_only_and_receipt_requires_exact_binding(connection, seeded):
    repo, inputs, proposal, _ = prepared(connection, seeded)
    action = propose(repo, inputs, proposal).action
    with pytest.raises(sqlite3.IntegrityError, match="append-only"), write_transaction(connection):
        connection.execute(
            "UPDATE recovery_action_bindings SET plan_sequence = 2 WHERE action_id = ?",
            (str(action.id),),
        )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"), write_transaction(connection):
        connection.execute(
            "DELETE FROM recovery_action_bindings WHERE action_id = ?", (str(action.id),)
        )
    another = action_from_spec(
        proposal.action_spec,
        experiment_id=inputs.plan.inputs.context.experiment_id,
        proposed_by=ACTOR,
        reason="second Action for the same recovery revision",
    )
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"), write_transaction(connection):
        repo.actions._insert(another)
        repo.recovery_action_bindings._insert(
            RecoveryActionBinding.for_proposal(another.id, proposal)
        )
    assert repo.actions.get(str(another.id)) is None
    binding = repo.recovery_action_bindings.for_action(str(action.id))
    with pytest.raises(ValidationError, match="disagrees with its OOM proposal"):
        binding.model_copy(update={"input_fingerprint": "sha256:" + "0" * 64})


def test_schema_rejects_binding_action_targeting_another_run(connection, seeded):
    repo, inputs, proposal, _ = prepared(connection, seeded)
    wrong_action = action_from_spec(
        ResizeMicrobatch(
            target=ActionTarget(kind="run", id=str(RunId.generate())),
            micro_batch_size=2,
            gradient_accumulation=16,
        ),
        experiment_id=inputs.plan.inputs.context.experiment_id,
        proposed_by=ACTOR,
        reason="wrong Run",
    )
    with (
        pytest.raises(sqlite3.IntegrityError, match="Action ownership"),
        write_transaction(connection),
    ):
        repo.actions._insert(wrong_action)
        repo.recovery_action_bindings._insert(
            RecoveryActionBinding.for_proposal(wrong_action.id, proposal)
        )
    assert repo.actions.get(str(wrong_action.id)) is None


def test_action_event_failure_rolls_back_action_binding_and_policy(connection, seeded, monkeypatch):
    repo, inputs, proposal, _ = prepared(connection, seeded)
    before = tuple(
        count(repo, table)
        for table in ("actions", "recovery_action_bindings", "policy_decisions", "outbox")
    )

    original = repo._emit_action

    def fail(*args, **kwargs):
        if args[1] == "ActionAuthorized":
            raise RuntimeError("injected Action event failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(repo, "_emit_action", fail)
    with pytest.raises(RuntimeError, match="injected"):
        propose(repo, inputs, proposal)
    assert (
        tuple(
            count(repo, table)
            for table in ("actions", "recovery_action_bindings", "policy_decisions", "outbox")
        )
        == before
    )


def test_binding_insert_failure_rolls_back_action_and_events(connection, seeded, monkeypatch):
    repo, inputs, proposal, _ = prepared(connection, seeded)
    before = tuple(
        count(repo, table) for table in ("actions", "recovery_action_bindings", "outbox")
    )

    def fail(*args, **kwargs):
        raise RuntimeError("injected binding failure")

    monkeypatch.setattr(repo.recovery_action_bindings, "_insert", fail)
    with pytest.raises(RuntimeError, match="injected binding failure"):
        propose(repo, inputs, proposal)
    assert (
        tuple(count(repo, table) for table in ("actions", "recovery_action_bindings", "outbox"))
        == before
    )
