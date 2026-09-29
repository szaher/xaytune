"""The receipt is durable provenance, not a standalone execution API."""

from __future__ import annotations

import sqlite3

import pytest
from pydantic import ValidationError

from xaytune.core.domain.action import ActionTarget
from xaytune.core.domain.actions import ResizeMicrobatch, action_from_spec
from xaytune.core.domain.operation import RuntimeOperation, RuntimeOperationTarget
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.core.domain.recovery_execution import (
    RecoveryExecutionOutcome,
    RecoveryExecutionReceipt,
)
from xaytune.core.ids import OperationId, RecoveryExecutionReceiptId, RunAttemptId
from xaytune.storage import ControlPlaneRepository, connect, migrate, write_transaction

from .conftest import make_attempt
from .test_recovery import ACTOR, draft, incident, record_plan


def prepared(connection, seeded):
    repo = ControlPlaneRepository(connection)
    observed = incident(repo, seeded["attempt"])
    plan = record_plan(repo, draft(repo, observed, RecoveryRequest()), actor=ACTOR)
    action = action_from_spec(
        ResizeMicrobatch(
            target=ActionTarget(kind="run", id=str(seeded["run"].id)),
            micro_batch_size=2,
            gradient_accumulation=16,
        ),
        experiment_id=seeded["experiment"].id,
        proposed_by=ACTOR,
        reason="test governed resize intent",
    )
    with write_transaction(connection):
        repo.actions._insert(action)
    return repo, plan, action


def receipt(plan, action, outcome=RecoveryExecutionOutcome.ABANDONED, **kwargs):
    return RecoveryExecutionReceipt(
        episode_id=plan.episode_id,
        plan_id=plan.id,
        plan_sequence=plan.sequence,
        action_id=action.id,
        outcome=outcome,
        created_by=ACTOR,
        **kwargs,
    )


def test_receipt_shape_requires_bound_effect_only_for_executed(connection, seeded):
    _, plan, action = prepared(connection, seeded)
    with pytest.raises(ValidationError, match="successor attempt"):
        receipt(plan, action, RecoveryExecutionOutcome.EXECUTED)
    with pytest.raises(ValidationError, match="runtime operation"):
        receipt(
            plan,
            action,
            RecoveryExecutionOutcome.EXECUTED,
            successor_attempt_id=RunAttemptId.generate(),
        )
    with pytest.raises(ValidationError, match="successor attempt"):
        receipt(plan, action, successor_attempt_id=RunAttemptId.generate())


def test_abandoned_receipt_is_append_only_and_survives_restart(connection, seeded, db_path):
    repo, plan, action = prepared(connection, seeded)
    abandoned = receipt(plan, action)
    with write_transaction(connection):
        repo.recovery_execution_receipts._insert(abandoned)
    assert repo.recovery_execution_receipts.for_episode(str(plan.episode_id)) == (abandoned,)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"), write_transaction(connection):
        connection.execute(
            "UPDATE recovery_execution_receipts SET outcome = 'SUPERSEDED' WHERE id = ?",
            (str(abandoned.id),),
        )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"), write_transaction(connection):
        connection.execute(
            "DELETE FROM recovery_execution_receipts WHERE id = ?", (str(abandoned.id),)
        )
    with connect(db_path) as reopened:
        migrate(reopened)
        restored = ControlPlaneRepository(reopened).recovery_execution_receipts.get(
            str(abandoned.id)
        )
    assert restored == abandoned


def test_receipt_refuses_wrong_plan_sequence_and_duplicate_action(connection, seeded):
    repo, plan, action = prepared(connection, seeded)
    first = receipt(plan, action)
    with write_transaction(connection):
        repo.recovery_execution_receipts._insert(first)
    wrong = receipt(plan, action).model_copy(
        update={"id": RecoveryExecutionReceiptId.generate(), "plan_sequence": plan.sequence + 1}
    )
    with pytest.raises(sqlite3.IntegrityError), write_transaction(connection):
        repo.recovery_execution_receipts._insert(wrong)
    with pytest.raises(sqlite3.IntegrityError), write_transaction(connection):
        repo.recovery_execution_receipts._insert(receipt(plan, action))
    assert repo.recovery_execution_receipts.for_episode(str(plan.episode_id)) == (first,)


def test_executed_receipt_requires_bound_successor_and_submit_intent(connection, seeded):
    repo, plan, action = prepared(connection, seeded)
    successor = make_attempt(seeded["run"], attempt_number=2)
    operation = RuntimeOperation(
        id=OperationId.generate(),
        target=RuntimeOperationTarget(kind="training-attempt", id=str(successor.id)),
        type="submit",
        request_digest="test-submit-digest",
        caused_by_action_id=action.id,
    )
    executed = receipt(
        plan,
        action,
        RecoveryExecutionOutcome.EXECUTED,
        successor_attempt_id=successor.id,
        runtime_operation_id=operation.id,
    )
    with pytest.raises(sqlite3.IntegrityError), write_transaction(connection):
        repo.recovery_execution_receipts._insert(executed)
    with write_transaction(connection):
        repo.aggregates._insert_attempt(successor)
        repo.operations._insert(operation)
        repo.recovery_execution_receipts._insert(executed)
    assert repo.recovery_execution_receipts.executed_for_episode(str(plan.episode_id)) == executed


def test_executed_receipt_refuses_unrelated_operation(connection, seeded):
    repo, plan, action = prepared(connection, seeded)
    successor = make_attempt(seeded["run"], attempt_number=2)
    wrong_operation = RuntimeOperation(
        id=OperationId.generate(),
        target=RuntimeOperationTarget(kind="training-attempt", id=str(successor.id)),
        type="submit",
        request_digest="test-submit-digest",
    )
    with pytest.raises(sqlite3.IntegrityError, match="bound effect"), write_transaction(connection):
        repo.aggregates._insert_attempt(successor)
        repo.operations._insert(wrong_operation)
        repo.recovery_execution_receipts._insert(
            receipt(
                plan,
                action,
                RecoveryExecutionOutcome.EXECUTED,
                successor_attempt_id=successor.id,
                runtime_operation_id=wrong_operation.id,
            )
        )
    assert repo.recovery_execution_receipts.for_episode(str(plan.episode_id)) == ()
    assert repo.aggregates.get_attempt(str(successor.id)) is None


def test_executed_receipt_refuses_uncovered_incident(connection, seeded):
    repo, plan, action = prepared(connection, seeded)
    incident(repo, seeded["attempt"], sequence=10)
    assert not repo.recovery_plans.is_effective_and_fresh(str(plan.id))
    successor = make_attempt(seeded["run"], attempt_number=2)
    operation = RuntimeOperation(
        id=OperationId.generate(),
        target=RuntimeOperationTarget(kind="training-attempt", id=str(successor.id)),
        type="submit",
        request_digest="test-submit-digest",
        caused_by_action_id=action.id,
    )
    with (
        pytest.raises(sqlite3.IntegrityError, match="current decision"),
        write_transaction(connection),
    ):
        repo.aggregates._insert_attempt(successor)
        repo.operations._insert(operation)
        repo.recovery_execution_receipts._insert(
            receipt(
                plan,
                action,
                RecoveryExecutionOutcome.EXECUTED,
                successor_attempt_id=successor.id,
                runtime_operation_id=operation.id,
            )
        )
    assert repo.aggregates.get_attempt(str(successor.id)) is None
