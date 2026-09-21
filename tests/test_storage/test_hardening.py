"""Band B hardening: durable state is loaded, never supplied.

Every test here corresponds to a hole a caller could drive through before:
handing the repository a fabricated aggregate, retrying a creation with
different identity, confirming a submission that forgets where its workload is,
or leaving an action holding intent against an effect known to have failed.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from xaytune.core import (
    Actor,
    ControllerHostRef,
    Experiment,
    ExperimentId,
    ExperimentStatus,
    Objective,
    ObjectiveMetric,
    RunAttemptStatus,
    RuntimeRef,
)
from xaytune.core.domain.action import ActionStatus, ActionTarget
from xaytune.core.errors import ConcurrentModificationError
from xaytune.storage import AggregateNotFoundError, ControlPlaneRepository, StorageError
from xaytune.storage.journal import IdempotencyConflictError

from .conftest import make_attempt, make_experiment, make_node, make_run

ACTOR = Actor(type="system", id="controller")
DIGEST = "sha256:request-a"
REF = RuntimeRef(backend="local", external_id="pid-1")


@pytest.fixture
def repo(connection: sqlite3.Connection) -> ControlPlaneRepository:
    return ControlPlaneRepository(connection)


@pytest.fixture
def run(repo: ControlPlaneRepository) -> Any:
    experiment = repo.create_experiment(make_experiment(), actor=ACTOR)
    node = repo.create_node(make_node(experiment), actor=ACTOR)
    return repo.create_run(make_run(node), actor=ACTOR)


# ---- 1. lifecycle writes load durable state ------------------------------


def test_a_transition_names_the_aggregate_by_id(repo: ControlPlaneRepository, run: Any) -> None:
    attempt, _ = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest=DIGEST, actor=ACTOR
    )

    moved = repo.transition_attempt(
        attempt.id,
        expected_revision=attempt.revision,
        new_status=RunAttemptStatus.QUEUED,
        actor=ACTOR,
    )

    assert moved.status is RunAttemptStatus.QUEUED
    assert repo.aggregates.load_attempt(str(attempt.id)).status is RunAttemptStatus.QUEUED


def test_a_caller_cannot_substitute_a_fabricated_aggregate(
    repo: ControlPlaneRepository, connection: sqlite3.Connection, run: Any
) -> None:
    """The hole this change closes.

    A caller used to hand over a whole post-transition aggregate. One carrying
    a real id and revision but a different ``run_id`` would leave the indexed
    column, the payload and the event's experiment each saying something
    different, and nothing would raise.
    """
    attempt, _ = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest=DIGEST, actor=ACTOR
    )

    # There is no longer a method that accepts one.
    assert not hasattr(repo, "transition")

    repo.transition_attempt(
        attempt.id,
        expected_revision=attempt.revision,
        new_status=RunAttemptStatus.QUEUED,
        actor=ACTOR,
    )
    row = connection.execute(
        "SELECT run_id, payload_json FROM run_attempts WHERE id = ?", (str(attempt.id),)
    ).fetchone()
    assert row["run_id"] == str(run.id)
    assert str(run.id) in row["payload_json"]


def test_a_stale_expected_revision_is_refused(repo: ControlPlaneRepository, run: Any) -> None:
    """Checked before the state machine, so a lost race reads as contention."""
    attempt, _ = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest=DIGEST, actor=ACTOR
    )
    repo.transition_attempt(
        attempt.id,
        expected_revision=attempt.revision,
        new_status=RunAttemptStatus.QUEUED,
        actor=ACTOR,
    )

    with pytest.raises(ConcurrentModificationError):
        repo.transition_attempt(
            attempt.id,
            expected_revision=attempt.revision,
            new_status=RunAttemptStatus.QUEUED,
            actor=ACTOR,
        )


def test_transitioning_something_that_does_not_exist_is_an_error(
    repo: ControlPlaneRepository,
) -> None:
    with pytest.raises(AggregateNotFoundError):
        repo.transition_experiment(
            ExperimentId.generate(),
            expected_revision=0,
            new_status=ExperimentStatus.ACTIVE,
            actor=ACTOR,
        )


def test_creation_refuses_an_aggregate_that_is_not_new(
    repo: ControlPlaneRepository,
) -> None:
    """A creation event cannot record a state nothing transitioned into."""
    started = Experiment(
        id=ExperimentId.generate(),
        name="already running",
        objective=Objective(primary=ObjectiveMetric(name="task_success", direction="maximize")),
        controller_host=ControllerHostRef(kind="embedded", id="local"),
        status=ExperimentStatus.ACTIVE,
    )

    with pytest.raises(ValueError, match="must start in"):
        repo.create_experiment(started, actor=ACTOR)


# ---- 2. submit-intent replay checks both halves --------------------------


def test_a_replay_with_a_different_attempt_is_refused(
    repo: ControlPlaneRepository, run: Any
) -> None:
    """The operation half was checked and the attempt half was not.

    A retry naming a different run or attempt number returned the original
    silently -- the same omission as the cancellation replay, on the creation
    path.
    """
    from xaytune.core.ids import OperationId

    operation_id = OperationId.generate()
    first = make_attempt(run, attempt_number=1)
    repo.create_attempt_with_submit_intent(
        first, request_digest=DIGEST, actor=ACTOR, operation_id=operation_id
    )

    different = make_attempt(run, attempt_number=9)
    with pytest.raises(IdempotencyConflictError, match="id"):
        repo.create_attempt_with_submit_intent(
            different, request_digest=DIGEST, actor=ACTOR, operation_id=operation_id
        )


def test_a_true_replay_still_returns_the_original(repo: ControlPlaneRepository, run: Any) -> None:
    from xaytune.core.ids import OperationId

    operation_id = OperationId.generate()
    attempt = make_attempt(run)
    first = repo.create_attempt_with_submit_intent(
        attempt, request_digest=DIGEST, actor=ACTOR, operation_id=operation_id
    )
    second = repo.create_attempt_with_submit_intent(
        attempt, request_digest=DIGEST, actor=ACTOR, operation_id=operation_id
    )

    assert second[0].id == first[0].id
    assert len(repo.aggregates.attempts_for_run(str(run.id))) == 1


# ---- 3. a confirmed submit must say where the workload is ----------------


def test_confirming_a_submit_without_a_runtime_ref_is_refused(
    repo: ControlPlaneRepository, run: Any
) -> None:
    """It would leave the unresolved queue with nowhere to look.

    The record would say a workload exists and not say where -- the orphan
    ADR-013 exists to prevent, reached through confirmation rather than
    through a crash.
    """
    _, operation = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest=DIGEST, actor=ACTOR
    )

    with pytest.raises(StorageError, match="without a RuntimeRef"):
        repo.confirm_operation(operation.id, expected_revision=operation.revision, actor=ACTOR)

    assert [op.id for op in repo.operations.unresolved()] == [operation.id]


def test_confirming_a_submit_with_a_runtime_ref_succeeds(
    repo: ControlPlaneRepository, run: Any
) -> None:
    _, operation = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest=DIGEST, actor=ACTOR
    )

    confirmed = repo.confirm_operation(
        operation.id, expected_revision=operation.revision, actor=ACTOR, runtime_ref=REF
    )

    assert confirmed.runtime_ref is not None
    assert confirmed.runtime_ref.external_id == "pid-1"
    assert repo.operations.unresolved() == ()


def test_a_cancel_needs_no_runtime_ref(repo: ControlPlaneRepository, run: Any) -> None:
    """It stops something already identified."""
    attempt, _ = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest=DIGEST, actor=ACTOR
    )
    _, cancel = repo.request_cancellation(
        ActionTarget(kind="training-attempt", id=str(attempt.id)),
        reason="user asked",
        actor=ACTOR,
        request_digest="sha256:cancel",
    )
    assert cancel is not None

    confirmed = repo.confirm_operation(cancel.id, expected_revision=cancel.revision, actor=ACTOR)
    assert confirmed.state == "confirmed"


# ---- 4. a failed cancel effect settles its action ------------------------


def test_a_failed_cancel_effect_fails_its_action(repo: ControlPlaneRepository, run: Any) -> None:
    """Otherwise the action holds intent against an effect known not to happen.

    A failed operation is not unresolved, so nothing would revisit it and the
    action would stay EXECUTING for the life of the record.
    """
    attempt, _ = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest=DIGEST, actor=ACTOR
    )
    action, cancel = repo.request_cancellation(
        ActionTarget(kind="training-attempt", id=str(attempt.id)),
        reason="user asked",
        actor=ACTOR,
        request_digest="sha256:cancel",
    )
    assert cancel is not None

    repo.fail_operation(cancel.id, expected_revision=cancel.revision, actor=ACTOR)
    settled = repo.reconcile_cancellation(action.id, actor=ACTOR)

    assert settled.status is ActionStatus.FAILED
    assert settled.outcome is None
    assert repo.actions.unresolved() == ()
    # The attempt is untouched: a failed cancel stopped nothing.
    assert not repo.aggregates.load_attempt(str(attempt.id)).is_terminal


def test_an_unresolved_cancel_effect_leaves_its_action_in_flight(
    repo: ControlPlaneRepository, run: Any
) -> None:
    attempt, _ = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest=DIGEST, actor=ACTOR
    )
    action, cancel = repo.request_cancellation(
        ActionTarget(kind="training-attempt", id=str(attempt.id)),
        reason="user asked",
        actor=ACTOR,
        request_digest="sha256:cancel",
    )
    assert cancel is not None
    repo.mark_operation_sent(cancel.id, expected_revision=cancel.revision, actor=ACTOR)

    unchanged = repo.reconcile_cancellation(action.id, actor=ACTOR)
    assert unchanged.status is ActionStatus.EXECUTING


# ---- 5. every action transition carries its own event --------------------


def test_each_action_lifecycle_step_is_recorded(repo: ControlPlaneRepository, run: Any) -> None:
    """Batching four transitions into one write would leave three unrecorded.

    The rule every other aggregate already enforces, applied to the aggregate
    whose whole purpose is recording intent.
    """
    attempt, _ = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest=DIGEST, actor=ACTOR
    )
    action, _ = repo.request_cancellation(
        ActionTarget(kind="training-attempt", id=str(attempt.id)),
        reason="user asked",
        actor=ACTOR,
        request_digest="sha256:cancel",
    )

    events = repo.events.events_for_aggregate(str(action.id))
    assert [e.event_type for e in events] == [
        "ActionProposed",
        "ActionValidating",
        "ActionValidated",
        "ActionExecuting",
        "CancellationRequested",
    ]
    # One revision per transition, and the latest matches the stored aggregate.
    assert [e.aggregate_revision for e in events] == [0, 1, 2, 3, 3]
    assert repo.actions.get(str(action.id)).revision == action.revision
