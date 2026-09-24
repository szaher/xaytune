"""The experiment cancellation saga, at the repository (ADR-013 §6).

The controller-level tests run it against real workloads. These pin the cases
a real workload rarely produces on demand: an effect that definitively failed
(AC-8), a reconcile that runs while an attempt is still live (AC-9), a replay,
and an experiment that had already ended.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from xaytune.core import Actor
from xaytune.core.domain.action import ActionOutcome
from xaytune.core.ids import ActionId
from xaytune.core.state.status import (
    ActionStatus,
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)
from xaytune.storage import ControlPlaneRepository
from xaytune.storage.journal import IdempotencyConflictError

from .conftest import make_attempt, make_experiment, make_node, make_run

ACTOR = Actor(type="system", id="test")


@pytest.fixture
def repo(connection: sqlite3.Connection) -> ControlPlaneRepository:
    return ControlPlaneRepository(connection)


def _live_experiment(repo: ControlPlaneRepository, attempts: int = 1) -> dict[str, Any]:
    """An ACTIVE experiment with *attempts* RUNNING attempts, one run each."""
    experiment = repo.create_experiment(make_experiment(), actor=ACTOR)
    experiment = repo.transition_experiment(
        experiment.id, expected_revision=0, new_status=ExperimentStatus.ACTIVE, actor=ACTOR
    )
    node = repo.create_node(make_node(experiment), actor=ACTOR)
    live = []
    for _ in range(attempts):
        run = repo.create_run(make_run(node), actor=ACTOR)
        attempt, _ = repo.create_attempt_with_submit_intent(
            make_attempt(run), request_digest="sha256:x", actor=ACTOR
        )
        for status in (
            RunAttemptStatus.QUEUED,
            RunAttemptStatus.STARTING,
            RunAttemptStatus.RUNNING,
        ):
            attempt = repo.transition_attempt(
                attempt.id, expected_revision=attempt.revision, new_status=status, actor=ACTOR
            )
        live.append((run, attempt))
    return {"experiment": experiment, "node": node, "live": live}


def _stop(repo: ControlPlaneRepository, attempt: Any, status: RunAttemptStatus) -> None:
    current = repo.aggregates.load_attempt(str(attempt.id))
    repo.transition_attempt(
        current.id, expected_revision=current.revision, new_status=status, actor=ACTOR
    )


def _confirm(repo: ControlPlaneRepository, operation: Any) -> None:
    repo.confirm_operation(operation.id, expected_revision=operation.revision, actor=ACTOR)


def test_the_request_records_intent_and_one_effect_per_live_attempt(
    repo: ControlPlaneRepository,
) -> None:
    world = _live_experiment(repo, attempts=2)

    parent, children = repo.request_experiment_cancellation(
        world["experiment"].id, reason="stop", actor=ACTOR
    )

    assert parent.type == "cancel-experiment"
    assert parent.status is ActionStatus.EXECUTING
    assert len(children) == 2
    for child, operation in children:
        assert child.parent_action_id == parent.id
        assert child.type == "cancel-attempt"
        assert operation is not None and operation.state == "intended"
        assert operation.caused_by_action_id == child.id
    assert repo.aggregates.load_experiment(str(world["experiment"].id)).status is (
        ExperimentStatus.ACTIVE
    ), "intent lives in the Action; the experiment does not move yet"


def test_the_experiment_is_not_cancelled_while_an_attempt_is_live(
    repo: ControlPlaneRepository,
) -> None:
    """AC-9: CANCELLED means nothing Xaytune owns is executing."""
    world = _live_experiment(repo, attempts=2)
    parent, children = repo.request_experiment_cancellation(
        world["experiment"].id, reason="stop", actor=ACTOR
    )
    for _child, operation in children:
        _confirm(repo, operation)
    (first_run, first), (_second_run, _second) = world["live"]
    _stop(repo, first, RunAttemptStatus.CANCELLED)

    reconciled = repo.reconcile_experiment_cancellation(parent.id, actor=ACTOR)

    assert reconciled.status is ActionStatus.EXECUTING
    assert repo.aggregates.load_experiment(str(world["experiment"].id)).status is (
        ExperimentStatus.ACTIVE
    )


def test_once_nothing_is_live_everything_is_cancelled_together(
    repo: ControlPlaneRepository,
) -> None:
    world = _live_experiment(repo, attempts=2)
    parent, children = repo.request_experiment_cancellation(
        world["experiment"].id, reason="stop", actor=ACTOR
    )
    for _child, operation in children:
        _confirm(repo, operation)
    (run_a, a), (run_b, b) = world["live"]
    _stop(repo, a, RunAttemptStatus.CANCELLED)
    # The second finished on its own before the cancel took effect (AC-7).
    _stop(repo, b, RunAttemptStatus.SUCCEEDED)
    repo.transition_run(
        run_b.id,
        expected_revision=repo.aggregates.load_run(str(run_b.id)).revision,
        new_status=RunStatus.ACTIVE,
        actor=ACTOR,
    )
    repo.transition_run(
        run_b.id,
        expected_revision=repo.aggregates.load_run(str(run_b.id)).revision,
        new_status=RunStatus.SUCCEEDED,
        actor=ACTOR,
    )

    settled = repo.reconcile_experiment_cancellation(parent.id, actor=ACTOR)

    assert (settled.status, settled.outcome) == (ActionStatus.SUCCEEDED, ActionOutcome.APPLIED)
    outcomes = {child.target.id: child.outcome for child in repo.actions.children(str(parent.id))}
    assert outcomes == {str(a.id): ActionOutcome.APPLIED, str(b.id): ActionOutcome.SUPERSEDED}
    assert repo.aggregates.load_experiment(str(world["experiment"].id)).status is (
        ExperimentStatus.CANCELLED
    )
    assert repo.aggregates.load_node(str(world["node"].id)).status is (
        ExperimentNodeStatus.CANCELLED
    )
    assert repo.aggregates.load_run(str(run_a.id)).status is RunStatus.CANCELLED
    assert repo.aggregates.load_run(str(run_b.id)).status is RunStatus.SUCCEEDED, (
        "a run that finished keeps its outcome"
    )


def test_a_cancellation_that_failed_leaves_the_experiment_active(
    repo: ControlPlaneRepository,
) -> None:
    """AC-8: an effect known not to have happened, over a workload still running.

    The experiment must not reach CANCELLED, and the saga must not sit in
    flight forever either: it fails, and the experiment stays ACTIVE.
    """
    world = _live_experiment(repo)
    parent, ((_child, operation),) = repo.request_experiment_cancellation(
        world["experiment"].id, reason="stop", actor=ACTOR
    )
    repo.fail_operation(operation.id, expected_revision=operation.revision, actor=ACTOR)

    settled = repo.reconcile_experiment_cancellation(parent.id, actor=ACTOR)

    assert settled.status is ActionStatus.FAILED
    assert repo.aggregates.load_experiment(str(world["experiment"].id)).status is (
        ExperimentStatus.ACTIVE
    )


def test_a_second_request_in_flight_returns_the_first(repo: ControlPlaneRepository) -> None:
    world = _live_experiment(repo)
    parent, children = repo.request_experiment_cancellation(
        world["experiment"].id, reason="stop", actor=ACTOR
    )

    again, again_children = repo.request_experiment_cancellation(
        world["experiment"].id, reason="stop again", actor=ACTOR
    )

    assert again.id == parent.id
    assert [c.id for c, _ in again_children] == [c.id for c, _ in children]
    assert len(repo.actions.for_target("experiment", str(world["experiment"].id))) == 1


def test_a_replay_with_a_different_request_is_refused(repo: ControlPlaneRepository) -> None:
    world = _live_experiment(repo)
    action_id = ActionId.generate()
    repo.request_experiment_cancellation(
        world["experiment"].id, reason="stop", actor=ACTOR, action_id=action_id
    )

    with pytest.raises(IdempotencyConflictError):
        repo.request_experiment_cancellation(
            world["experiment"].id, reason="different", actor=ACTOR, action_id=action_id
        )


def test_an_experiment_that_already_ended_needs_no_effect(repo: ControlPlaneRepository) -> None:
    world = _live_experiment(repo, attempts=0)
    experiment = world["experiment"]
    repo.transition_experiment(
        experiment.id,
        expected_revision=experiment.revision,
        new_status=ExperimentStatus.FAILED,
        actor=ACTOR,
    )

    parent, children = repo.request_experiment_cancellation(
        experiment.id, reason="stop", actor=ACTOR
    )

    assert children == ()
    assert (parent.status, parent.outcome) == (ActionStatus.SUCCEEDED, ActionOutcome.SUPERSEDED)


def test_an_attempt_that_appeared_after_the_request_blocks_cancellation(
    repo: ControlPlaneRepository,
) -> None:
    """AC-9 where no child covers it: work submitted after the saga was recorded.

    The children are the attempts that were live when cancellation was
    requested. An attempt created after that has no child -- so "every child
    has settled" is not enough, and the experiment must still not be called
    cancelled while that attempt runs.
    """
    world = _live_experiment(repo, attempts=0)
    parent, children = repo.request_experiment_cancellation(
        world["experiment"].id, reason="stop", actor=ACTOR
    )
    assert children == ()

    run = repo.create_run(make_run(world["node"]), actor=ACTOR)
    late, _ = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest="sha256:late", actor=ACTOR
    )
    repo.transition_attempt(
        late.id, expected_revision=late.revision, new_status=RunAttemptStatus.QUEUED, actor=ACTOR
    )

    reconciled = repo.reconcile_experiment_cancellation(parent.id, actor=ACTOR)

    assert reconciled.status is ActionStatus.EXECUTING
    assert repo.aggregates.load_experiment(str(world["experiment"].id)).status is (
        ExperimentStatus.ACTIVE
    )
