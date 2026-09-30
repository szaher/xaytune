"""A governed OOM effect consumes one fresh episode in one durable commit."""

from __future__ import annotations

import asyncio

import pytest

from tests.test_storage.test_recovery import incident, prepare
from tests.test_storage.test_recovery_action_bindings import ACTOR, APPROVAL, propose
from tests.test_storage.test_recovery_episodes import decide, signal
from xaytune.core.domain.action import ActionStatus
from xaytune.core.domain.oom_recovery import OOMRecoveryInputsV1, OOMResizeProposal
from xaytune.core.domain.recovery import RecoveryCheckpointReport
from xaytune.core.domain.run import ExecutionOverride, RunAttempt
from xaytune.core.ids import OperationId, RunAttemptId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor
from xaytune.core.state.status import ExperimentStatus, RunAttemptStatus, RunStatus
from xaytune.resilience.oom import OOMRecoveryPlanner
from xaytune.storage.control_plane import ProvenanceError, StaleRecoveryContextError


def prepared_execution(connection, seeded, tmp_path, *, policy=None):
    repo, source, recorded, request, coordinator = prepare(connection, seeded, tmp_path)
    experiment, run = seeded["experiment"], seeded["run"]
    repo.transition_experiment(
        experiment.id,
        expected_revision=experiment.revision,
        new_status=ExperimentStatus.ACTIVE,
        actor=ACTOR,
    )
    repo.transition_run(
        run.id, expected_revision=run.revision, new_status=RunStatus.ACTIVE, actor=ACTOR
    )
    source = repo.transition_attempt(
        source.id,
        expected_revision=source.revision,
        new_status=RunAttemptStatus.FAILED,
        actor=ACTOR,
    )
    observed = incident(repo, source, signal=signal("cuda-oom"))
    plan = decide(repo, observed, request)
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
    action = (
        propose(repo, inputs, proposal, policy=policy).action
        if policy
        else propose(repo, inputs, proposal).action
    )
    checkpoint = RecoveryCheckpointReport.from_record(recorded, source.attempt_number)
    assert request.restore_context is not None
    asyncio.run(coordinator.checkpoint_manager.validate_recorded(recorded, request.restore_context))
    return repo, source, plan, proposal, action, checkpoint


def successor(source, proposal, action, checkpoint):
    return RunAttempt(
        id=RunAttemptId.generate(),
        run_id=source.run_id,
        attempt_number=source.attempt_number + 1,
        execution_fingerprint="execution-resized",
        checkpoint_ref=checkpoint.checkpoint_ref,
        execution_overrides=(
            ExecutionOverride(
                id="oom-micro",
                kind="micro_batch_resize",
                reason="governed OOM resize",
                values=FrozenDict(
                    {
                        "from": proposal.old_micro_batch_size,
                        "to": proposal.action_spec.micro_batch_size,
                    }
                ),
                preserves=("effective_batch_size",),
                action_id=action.id,
            ),
            ExecutionOverride(
                id="oom-accumulation",
                kind="gradient_accumulation_adjustment",
                reason="preserve effective batch",
                values=FrozenDict(
                    {
                        "from": proposal.old_gradient_accumulation,
                        "to": proposal.action_spec.gradient_accumulation,
                    }
                ),
                preserves=("effective_batch_size",),
                action_id=action.id,
            ),
            ExecutionOverride(
                id="oom-restore",
                kind="checkpoint_restore",
                reason="resume exact state",
                values=FrozenDict({"checkpoint_id": str(checkpoint.checkpoint_ref.id)}),
                action_id=action.id,
            ),
        ),
    )


def count(connection, table):
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_authorized_oom_execution_commits_successor_operation_and_receipt_together(
    connection, seeded, tmp_path
):
    repo, source, plan, proposal, action, checkpoint = prepared_execution(
        connection, seeded, tmp_path
    )
    attempt = successor(source, proposal, action, checkpoint)
    operation_id = OperationId.generate()
    before_usage = repo.recovery_episodes.usage_excluding(
        str(plan.inputs.context.experiment_id), "not-the-target"
    )
    recorded, operation, receipt = repo._record_oom_recovery_execution(
        action.id,
        attempt,
        checkpoint,
        request_digest="sha256:" + "1" * 64,
        actor=ACTOR,
        operation_id=operation_id,
        destinations=("audit",),
    )
    assert recorded == attempt
    assert operation.state == "intended"
    assert operation.caused_by_action_id == action.id
    assert receipt.successor_attempt_id == attempt.id
    assert receipt.runtime_operation_id == operation.id
    assert receipt.checkpoint_ref == checkpoint.checkpoint_ref
    assert repo.actions.get(str(action.id)).status is ActionStatus.EXECUTING
    assert (
        repo.recovery_episodes.usage_excluding(
            str(plan.inputs.context.experiment_id), "not-the-target"
        )
        == before_usage + 1
    )
    assert repo.aggregates.load_run(str(source.run_id)).status is RunStatus.ACTIVE
    assert repo.aggregates.load_attempt(str(source.id)).status is RunAttemptStatus.FAILED
    assert count(connection, "recovery_execution_receipts") == 1
    assert repo._record_oom_recovery_execution(
        action.id,
        attempt,
        checkpoint,
        request_digest="sha256:" + "1" * 64,
        actor=ACTOR,
        operation_id=operation_id,
    ) == (recorded, operation, receipt)
    assert count(connection, "recovery_execution_receipts") == 1


def test_oom_execution_refuses_unapproved_action(connection, seeded, tmp_path):
    repo, source, _, proposal, action, checkpoint = prepared_execution(
        connection, seeded, tmp_path, policy=APPROVAL
    )
    attempt = successor(source, proposal, action, checkpoint)
    with pytest.raises(ProvenanceError, match="not authorized"):
        repo._record_oom_recovery_execution(
            action.id, attempt, checkpoint, request_digest="request", actor=ACTOR
        )
    assert count(connection, "run_attempts") == 1
    assert count(connection, "recovery_execution_receipts") == 0
    repo.approve_action(
        action.id, approver=Actor(type="human", id="reviewer"), reason="allow recovery"
    )
    _, operation, receipt = repo._record_oom_recovery_execution(
        action.id, attempt, checkpoint, request_digest="request", actor=ACTOR
    )
    assert operation.caused_by_action_id == action.id
    assert receipt.action_id == action.id


def test_new_evidence_or_checkpoint_report_change_refuses_oom_execution(
    connection, seeded, tmp_path
):
    repo, source, _, proposal, action, checkpoint = prepared_execution(connection, seeded, tmp_path)
    later = incident(repo, source, sequence=10, signal=signal("process-failure"))
    assert later is not None
    with pytest.raises(StaleRecoveryContextError, match="open and fresh"):
        repo._record_oom_recovery_execution(
            action.id,
            successor(source, proposal, action, checkpoint),
            checkpoint,
            request_digest="request",
            actor=ACTOR,
        )
    assert count(connection, "run_attempts") == 1
    assert count(connection, "recovery_execution_receipts") == 0


def test_receipt_event_failure_rolls_back_successor_and_submit_intent(
    connection, seeded, tmp_path, monkeypatch
):
    repo, source, _, proposal, action, checkpoint = prepared_execution(connection, seeded, tmp_path)
    attempt = successor(source, proposal, action, checkpoint)
    before = tuple(
        count(connection, table)
        for table in ("run_attempts", "runtime_operations", "recovery_execution_receipts", "outbox")
    )

    def fail(*args, **kwargs):
        raise RuntimeError("injected receipt failure")

    monkeypatch.setattr(repo.recovery_execution_receipts, "_insert", fail)
    with pytest.raises(RuntimeError, match="injected receipt failure"):
        repo._record_oom_recovery_execution(
            action.id, attempt, checkpoint, request_digest="request", actor=ACTOR
        )
    assert (
        tuple(
            count(connection, table)
            for table in (
                "run_attempts",
                "runtime_operations",
                "recovery_execution_receipts",
                "outbox",
            )
        )
        == before
    )
    assert repo.actions.get(str(action.id)).status is ActionStatus.VALIDATED
