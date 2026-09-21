"""Cancellation: Action + effect, atomically (ADR-005 §5, ADR-013 §4–§5).

PR-005 could not expose this. The `Action` owns the durable intent and the
`RuntimeOperation` carries the external effect, and §5 requires them to commit
together -- so until the Action aggregate existed, a public cancellation could
only have recorded an effect whose cause was unwritten.

These are ADR-005 §11 clauses 5, 5a and 9.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from xaytune.core import Actor, RunAttemptStatus
from xaytune.core.domain.action import Action, ActionOutcome, ActionStatus, ActionTarget
from xaytune.core.ids import ActionId, OperationId
from xaytune.storage import ControlPlaneRepository, connect, write_transaction
from xaytune.storage.journal import IdempotencyConflictError

from .conftest import make_attempt, make_experiment, make_node, make_run

ACTOR = Actor(type="system", id="controller")
DIGEST = "sha256:request-a"
CANCEL_DIGEST = "sha256:cancel-a"


@pytest.fixture
def repo(connection: sqlite3.Connection) -> ControlPlaneRepository:
    return ControlPlaneRepository(connection)


@pytest.fixture
def live_attempt(repo: ControlPlaneRepository) -> Any:
    """A running attempt with a confirmed submit operation behind it."""
    experiment = repo.create_experiment(make_experiment(), actor=ACTOR)
    node = repo.create_node(make_node(experiment), actor=ACTOR)
    run = repo.create_run(make_run(node), actor=ACTOR)
    attempt, _ = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest=DIGEST, actor=ACTOR
    )
    return attempt


def _target(attempt: Any) -> ActionTarget:
    return ActionTarget(kind="training-attempt", id=str(attempt.id))


def _drive_to(repo: ControlPlaneRepository, attempt: Any, final: RunAttemptStatus) -> Any:
    """Walk an attempt to *final* one committed transition at a time."""
    path = [RunAttemptStatus.QUEUED, RunAttemptStatus.STARTING, RunAttemptStatus.RUNNING]
    if final not in path:
        path.append(final)
    else:
        path = path[: path.index(final) + 1]

    current = attempt
    for status in path:
        current = current.with_status(status)
        repo.transition(current, actor=ACTOR)
    return current


# ---- §11.5 Action + caused operation are atomic --------------------------


def test_cancellation_commits_the_action_and_its_effect_together(
    repo: ControlPlaneRepository, live_attempt: Any
) -> None:
    action, operation = repo.request_cancellation(
        _target(live_attempt), reason="user asked", actor=ACTOR, request_digest=CANCEL_DIGEST
    )

    assert operation is not None
    assert action.type == "cancel-attempt"
    assert action.status is ActionStatus.EXECUTING
    assert operation.type == "cancel"
    assert operation.state == "intended"

    # The linkage, with a real foreign key behind it.
    assert operation.caused_by_action_id == action.id
    stored = repo.operations.get(str(operation.id))
    assert stored is not None and stored.caused_by_action_id == action.id


def test_no_operation_survives_a_failed_action_commit(
    repo: ControlPlaneRepository, live_attempt: Any
) -> None:
    """Intent that nothing will act on is the state §5 forbids."""
    original = repo.actions._insert

    def exploding(action: Action) -> None:
        original(action)
        raise RuntimeError("injected after the action insert")

    repo.actions._insert = exploding  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="injected"):
        repo.request_cancellation(
            _target(live_attempt),
            reason="user asked",
            actor=ACTOR,
            request_digest=CANCEL_DIGEST,
        )
    repo.actions._insert = original  # type: ignore[method-assign]

    assert repo.actions.for_target("training-attempt", str(live_attempt.id)) == ()
    cancels = [
        op
        for op in repo.operations.for_target("training-attempt", str(live_attempt.id))
        if op.type == "cancel"
    ]
    assert cancels == []


def test_no_action_survives_a_failed_operation_insert(
    repo: ControlPlaneRepository, live_attempt: Any
) -> None:
    """An effect with no recorded cause is the other half of §5."""
    original = repo.operations._insert

    def exploding(operation: Any) -> Any:
        original(operation)
        raise RuntimeError("injected after the operation insert")

    repo.operations._insert = exploding  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="injected"):
        repo.request_cancellation(
            _target(live_attempt),
            reason="user asked",
            actor=ACTOR,
            request_digest=CANCEL_DIGEST,
        )
    repo.operations._insert = original  # type: ignore[method-assign]

    assert repo.actions.for_target("training-attempt", str(live_attempt.id)) == ()


def test_the_foreign_key_forbids_an_orphan_cause(
    repo: ControlPlaneRepository, connection: sqlite3.Connection, live_attempt: Any
) -> None:
    """Migration 003 backs the linkage with REFERENCES actions(id).

    This is why the column waited for the table rather than shipping as an
    unenforced string in migration 002.
    """
    with pytest.raises(sqlite3.IntegrityError):
        with write_transaction(connection):
            connection.execute(
                "UPDATE runtime_operations SET caused_by_action_id = 'act_nonexistent'"
            )


# ---- §11.5a the cancellation race ----------------------------------------


def test_a_workload_that_finishes_first_supersedes_the_cancellation(
    repo: ControlPlaneRepository, live_attempt: Any
) -> None:
    """ADR-013 §5: the observed terminal state wins, and that is not a failure.

    The attempt stays SUCCEEDED, and the action records that it did what it was
    asked -- the answer being that there was nothing left to stop.
    """
    _drive_to(repo, live_attempt, RunAttemptStatus.SUCCEEDED)

    action, operation = repo.request_cancellation(
        _target(live_attempt), reason="too late", actor=ACTOR, request_digest=CANCEL_DIGEST
    )

    assert operation is None, "no runtime is asked to stop a workload that has ended"
    assert action.status is ActionStatus.SUCCEEDED
    assert action.outcome is ActionOutcome.SUPERSEDED
    assert repo.aggregates.load_attempt(str(live_attempt.id)).status is (RunAttemptStatus.SUCCEEDED)


def test_a_cancellation_that_takes_effect_is_applied(
    repo: ControlPlaneRepository, live_attempt: Any
) -> None:
    action, operation = repo.request_cancellation(
        _target(live_attempt), reason="user asked", actor=ACTOR, request_digest=CANCEL_DIGEST
    )
    assert operation is not None

    # Both halves of the evidence: the effect is confirmed and the attempt
    # observed CANCELLED. Neither alone is enough.
    repo.confirm_operation(operation, actor=ACTOR)
    cancelled = _drive_to(repo, live_attempt, RunAttemptStatus.CANCELLED)
    assert cancelled.status is RunAttemptStatus.CANCELLED

    settled = repo.reconcile_cancellation(action, actor=ACTOR)

    assert settled.status is ActionStatus.SUCCEEDED
    assert settled.outcome is ActionOutcome.APPLIED


def test_an_outcome_is_required_and_only_on_success() -> None:
    """The pairing the schema enforces, enforced in the domain too."""
    from xaytune.core.errors import DomainError

    action = Action(
        id=__import__("xaytune.core.ids", fromlist=["ActionId"]).ActionId.generate(),
        experiment_id=__import__(
            "xaytune.core.ids", fromlist=["ExperimentId"]
        ).ExperimentId.generate(),
        type="cancel-attempt",
        target=ActionTarget(kind="training-attempt", id="attempt_x"),
        proposed_by=ACTOR,
        reason="r",
    )
    validated = action.with_status(ActionStatus.VALIDATING).with_status(ActionStatus.VALIDATED)
    executing = validated.with_status(ActionStatus.EXECUTING)

    with pytest.raises(DomainError, match="must say how it resolved"):
        executing.with_status(ActionStatus.SUCCEEDED)

    with pytest.raises(DomainError, match="only a SUCCEEDED action"):
        executing.with_status(ActionStatus.FAILED, outcome=ActionOutcome.APPLIED)


def test_a_controller_owned_action_is_never_marked_approved(
    repo: ControlPlaneRepository, live_attempt: Any
) -> None:
    """VALIDATED -> EXECUTING, with no approval invented by nobody."""
    action, _ = repo.request_cancellation(
        _target(live_attempt), reason="user asked", actor=ACTOR, request_digest=CANCEL_DIGEST
    )

    statuses = [
        event.payload["status"] for event in repo.events.events_for_aggregate(str(action.id))
    ]
    assert "approved" not in statuses
    assert "approval_pending" not in statuses
    assert action.policy_decision_id is None


# ---- §11.9 cancellation intent survives a restart ------------------------


def test_cancellation_intent_survives_a_restart(
    repo: ControlPlaneRepository,
    connection: sqlite3.Connection,
    db_path: Path,
    live_attempt: Any,
) -> None:
    """Both halves: the Action still records the desire, the operation the effect."""
    action, operation = repo.request_cancellation(
        _target(live_attempt), reason="user asked", actor=ACTOR, request_digest=CANCEL_DIGEST
    )
    assert operation is not None
    connection.close()

    reopened = connect(db_path)
    try:
        recovered = ControlPlaneRepository(reopened)

        stored_action = recovered.actions.get(str(action.id))
        assert stored_action is not None
        assert stored_action.type == "cancel-attempt"
        assert not stored_action.is_terminal

        stored_op = recovered.operations.get(str(operation.id))
        assert stored_op is not None
        assert stored_op.is_unresolved
        assert stored_op.caused_by_action_id == action.id

        assert [a.id for a in recovered.actions.unresolved()] == [action.id]
    finally:
        reopened.close()


def test_an_unknown_action_type_is_refused() -> None:
    """The vocabulary lives in the registry, not in a schema CHECK."""
    from xaytune.core.domain.action import UnknownActionTypeError
    from xaytune.core.ids import ActionId, ExperimentId

    with pytest.raises(UnknownActionTypeError, match="change-learning-rate"):
        Action(
            id=ActionId.generate(),
            experiment_id=ExperimentId.generate(),
            type="change-learning-rate",
            target=ActionTarget(kind="run", id="run_x"),
            proposed_by=ACTOR,
            reason="phase 4 type, not registered in band B",
        )


def test_run_and_experiment_cancellation_are_refused_as_sagas(
    repo: ControlPlaneRepository, live_attempt: Any
) -> None:
    """ADR-013 §6 fans out over descendants; that coordination is not built.

    Refused rather than half-implemented: writing the Action with no operation
    would leave intent nothing carries out, and minting one operation against a
    run id would ask a runtime to cancel something it has no handle on.
    """
    from xaytune.storage.control_plane import CancellationSagaRequiredError

    run_id = str(repo.aggregates.load_attempt(str(live_attempt.id)).run_id)
    run = repo.aggregates.load_run(run_id)

    for target in (
        ActionTarget(kind="run", id=run_id),
        ActionTarget(kind="experiment", id=str(run.experiment_id)),
    ):
        with pytest.raises(CancellationSagaRequiredError, match="saga"):
            repo.request_cancellation(
                target, reason="stop everything", actor=ACTOR, request_digest=CANCEL_DIGEST
            )

    assert repo.actions.unresolved() == ()


# ---- retry safety at the public boundary ---------------------------------


def test_request_cancellation_is_idempotent(
    repo: ControlPlaneRepository, live_attempt: Any
) -> None:
    """The lost-response retry, which submission already handled and this did not."""
    action_id = ActionId.generate()
    operation_id = OperationId.generate()
    target = _target(live_attempt)

    first = repo.request_cancellation(
        target,
        reason="user asked",
        actor=ACTOR,
        request_digest=CANCEL_DIGEST,
        action_id=action_id,
        operation_id=operation_id,
    )
    second = repo.request_cancellation(
        target,
        reason="user asked",
        actor=ACTOR,
        request_digest=CANCEL_DIGEST,
        action_id=action_id,
        operation_id=operation_id,
    )

    assert second[0].id == first[0].id
    assert second[1] is not None and first[1] is not None
    assert second[1].id == first[1].id

    assert len(repo.actions.for_target("training-attempt", str(live_attempt.id))) == 1
    cancels = [
        op
        for op in repo.operations.for_target("training-attempt", str(live_attempt.id))
        if op.type == "cancel"
    ]
    assert len(cancels) == 1

    requested = [
        event
        for event in repo.events.events_for_aggregate(str(action_id))
        if event.event_type == "CancellationRequested"
    ]
    assert len(requested) == 1


def test_a_second_action_cannot_adopt_an_existing_operation(
    repo: ControlPlaneRepository, live_attempt: Any
) -> None:
    """The state ADR-005 §5 says must never exist.

    Before the cause joined the identity check, a second Action could reuse an
    operation id, see an "identical" request, and commit -- while the operation
    still pointed at the first. The result was one Action with an effect and
    another with none.
    """
    target = _target(live_attempt)
    operation_id = OperationId.generate()

    first_action, _ = repo.request_cancellation(
        target,
        reason="first",
        actor=ACTOR,
        request_digest=CANCEL_DIGEST,
        operation_id=operation_id,
    )

    second_action_id = ActionId.generate()
    with pytest.raises(IdempotencyConflictError, match="caused_by_action_id"):
        repo.request_cancellation(
            target,
            reason="second",
            actor=ACTOR,
            request_digest=CANCEL_DIGEST,
            action_id=second_action_id,
            operation_id=operation_id,
        )

    # The losing action rolled back entirely; the operation still names the first.
    assert repo.actions.get(str(second_action_id)) is None
    stored = repo.operations.get(str(operation_id))
    assert stored is not None and stored.caused_by_action_id == first_action.id


# ---- the outcome is evidence, not an argument ----------------------------


def test_an_unconfirmed_cancellation_stays_in_flight(
    repo: ControlPlaneRepository, live_attempt: Any
) -> None:
    """A caller must not be able to manufacture provenance.

    Before this, `outcome=APPLIED` could be persisted while the operation was
    still INTENDED and the workload still running.
    """
    action, operation = repo.request_cancellation(
        _target(live_attempt), reason="user asked", actor=ACTOR, request_digest=CANCEL_DIGEST
    )
    assert operation is not None and operation.state == "intended"

    unchanged = repo.reconcile_cancellation(action, actor=ACTOR)

    assert unchanged.status is ActionStatus.EXECUTING
    assert unchanged.outcome is None


def test_a_target_cancelled_by_something_else_is_not_claimed(
    repo: ControlPlaneRepository, live_attempt: Any
) -> None:
    """CANCELLED with an unresolved effect may not be our doing.

    Claiming APPLIED here would be a guess, so the action stays in flight.
    """
    action, operation = repo.request_cancellation(
        _target(live_attempt), reason="user asked", actor=ACTOR, request_digest=CANCEL_DIGEST
    )
    assert operation is not None
    _drive_to(repo, live_attempt, RunAttemptStatus.CANCELLED)

    unchanged = repo.reconcile_cancellation(action, actor=ACTOR)

    assert unchanged.status is ActionStatus.EXECUTING


def test_a_target_that_ends_otherwise_supersedes_an_in_flight_cancellation(
    repo: ControlPlaneRepository, live_attempt: Any
) -> None:
    action, _ = repo.request_cancellation(
        _target(live_attempt), reason="user asked", actor=ACTOR, request_digest=CANCEL_DIGEST
    )
    _drive_to(repo, live_attempt, RunAttemptStatus.SUCCEEDED)

    settled = repo.reconcile_cancellation(action, actor=ACTOR)

    assert settled.status is ActionStatus.SUCCEEDED
    assert settled.outcome is ActionOutcome.SUPERSEDED


# ---- already cancelled is NOOP, not SUPERSEDED ---------------------------


def test_an_already_cancelled_target_is_a_noop(
    repo: ControlPlaneRepository, live_attempt: Any
) -> None:
    """Already in the requested state is a different fact from being overtaken."""
    _drive_to(repo, live_attempt, RunAttemptStatus.CANCELLED)

    action, operation = repo.request_cancellation(
        _target(live_attempt), reason="again", actor=ACTOR, request_digest=CANCEL_DIGEST
    )

    assert operation is None
    assert action.status is ActionStatus.SUCCEEDED
    assert action.outcome is ActionOutcome.NOOP


def test_a_target_that_ended_otherwise_is_superseded(
    repo: ControlPlaneRepository, live_attempt: Any
) -> None:
    _drive_to(repo, live_attempt, RunAttemptStatus.FAILED)

    action, operation = repo.request_cancellation(
        _target(live_attempt), reason="too late", actor=ACTOR, request_digest=CANCEL_DIGEST
    )

    assert operation is None
    assert action.outcome is ActionOutcome.SUPERSEDED


# ---- a cancel effect can no longer be causeless --------------------------


def test_a_cancel_operation_must_name_its_cause() -> None:
    """The model no longer permits the state the public API avoids."""
    from xaytune.core.domain.operation import RuntimeOperation, RuntimeOperationTarget
    from xaytune.core.errors import DomainError
    from xaytune.core.ids import OperationId as OpId

    with pytest.raises(DomainError, match="must name the Action"):
        RuntimeOperation(
            id=OpId.generate(),
            target=RuntimeOperationTarget(kind="training-attempt", id="attempt_x"),
            type="cancel",
            request_digest="d",
        )
