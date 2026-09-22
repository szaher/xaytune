"""Band B hardening: durable state is loaded, never supplied.

Every test here corresponds to a hole a caller could drive through before:
handing the repository a fabricated aggregate, retrying a creation with
different identity, confirming a submission that forgets where its workload is,
or leaving an action holding intent against an effect known to have failed.
"""

from __future__ import annotations

import json
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
from xaytune.storage import (
    AggregateNotFoundError,
    ControlPlaneRepository,
    StorageError,
    write_transaction,
)
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


# ---- creation guards, across every aggregate -----------------------------


def _pristine_cases(repo: ControlPlaneRepository) -> list[tuple[str, Any, Any, Any]]:
    """One case per creation path, so a sibling cannot be missed again."""
    experiment = repo.create_experiment(make_experiment(), actor=ACTOR)
    node = repo.create_node(make_node(experiment), actor=ACTOR)
    created_run = repo.create_run(make_run(node), actor=ACTOR)

    from xaytune.core import ExperimentNodeStatus, RunStatus

    return [
        ("Experiment", make_experiment("fresh"), ExperimentStatus.ACTIVE, repo.create_experiment),
        ("ExperimentNode", make_node(experiment), ExperimentNodeStatus.ACTIVE, repo.create_node),
        ("Run", make_run(node), RunStatus.ACTIVE, repo.create_run),
        (
            "RunAttempt",
            make_attempt(created_run),
            RunAttemptStatus.RUNNING,
            lambda a, *, actor: repo.create_attempt_with_submit_intent(
                a, request_digest=DIGEST, actor=actor
            ),
        ),
    ]


def test_no_creation_path_accepts_a_non_pristine_aggregate(
    repo: ControlPlaneRepository,
) -> None:
    """Table-driven on purpose.

    ``create_node`` was the one path this guard did not reach -- added to three
    siblings and silently skipped on the fourth. Enumerating them means the
    next creation method is either in this list or visibly absent from it.
    """
    for name, fresh, started, create in _pristine_cases(repo):
        advanced = type(fresh).model_validate(
            {**fresh.model_dump(mode="python"), "status": started, "revision": 7}
        )
        with pytest.raises(ValueError, match="must start"):
            create(advanced, actor=ACTOR)
        assert name  # names the failing case in the assertion output


# ---- a run belongs to its node's experiment ------------------------------


def test_a_run_cannot_claim_a_different_experiment_than_its_node(
    repo: ControlPlaneRepository,
) -> None:
    """Both foreign keys pass; neither ties the node to that experiment.

    The run would sit under one candidate's node while its payload and events
    named another experiment.
    """
    owning = repo.create_experiment(make_experiment("owning"), actor=ACTOR)
    other = repo.create_experiment(make_experiment("other"), actor=ACTOR)
    node = repo.create_node(make_node(owning), actor=ACTOR)

    run = make_run(node)
    mismatched = type(run).model_validate(
        {**run.model_dump(mode="python"), "experiment_id": other.id}
    )

    with pytest.raises(StorageError, match="belongs to"):
        repo.create_run(mismatched, actor=ACTOR)

    assert repo.aggregates.get_run(str(run.id)) is None


def test_a_run_must_realize_its_node_s_candidate(
    repo: ControlPlaneRepository,
) -> None:
    """A run claiming another fingerprint realizes something never proposed."""
    experiment = repo.create_experiment(make_experiment(), actor=ACTOR)
    node = repo.create_node(make_node(experiment), actor=ACTOR)

    run = make_run(node)
    divergent = type(run).model_validate(
        {**run.model_dump(mode="python"), "candidate_fingerprint": "sha256:something-else"}
    )

    with pytest.raises(StorageError, match="realizes its node"):
        repo.create_run(divergent, actor=ACTOR)


# ---- every event of a compound operation reaches the outbox --------------


def test_every_event_of_a_cancellation_gets_its_outbox_records(
    repo: ControlPlaneRepository, run: Any
) -> None:
    """ADR-005 §3 pairs each event with its outbox records.

    The intermediate lifecycle steps are the ones most easily dropped, and
    making Action transitions explicit is what exposed that they were.
    """
    attempt, _ = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest=DIGEST, actor=ACTOR
    )
    action, _ = repo.request_cancellation(
        ActionTarget(kind="training-attempt", id=str(attempt.id)),
        reason="user asked",
        actor=ACTOR,
        request_digest="sha256:cancel",
        destinations=("mlflow",),
    )

    action_events = repo.events.events_for_aggregate(str(action.id))
    assert len(action_events) == 5

    delivered = {record.event_id for record in repo.events.pending_outbox()}
    for event in action_events:
        assert event.id in delivered, f"{event.event_type} has no outbox record"


def test_operation_settlement_reaches_the_outbox(repo: ControlPlaneRepository, run: Any) -> None:
    """ADR-013 puts operation transitions in the log atomically with the outbox."""
    _, operation = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest=DIGEST, actor=ACTOR
    )
    confirmed = repo.confirm_operation(
        operation.id,
        expected_revision=operation.revision,
        actor=ACTOR,
        runtime_ref=REF,
        destinations=("webhook",),
    )

    events = repo.events.events_for_aggregate(str(confirmed.id))
    settled = [e for e in events if e.event_type == "RuntimeOperationConfirmed"]
    assert len(settled) == 1

    delivered = {record.event_id for record in repo.events.pending_outbox()}
    assert settled[0].id in delivered


# ---- a node's fingerprint must describe its candidate --------------------


def test_a_node_whose_fingerprint_does_not_match_its_candidate_is_refused(
    repo: ControlPlaneRepository,
) -> None:
    """The stored identity has to describe the body it indexes.

    Every consumer downstream trusts it: graph comparison would call two
    nodes the same candidate, reuse would match the wrong hypothesis, and the
    run consistency check would compare against a value describing nothing.
    """
    experiment = repo.create_experiment(make_experiment(), actor=ACTOR)
    node = make_node(experiment)
    mismatched = type(node).model_validate(
        {
            **node.model_dump(mode="python"),
            "candidate_fingerprint": "sha256:describes-something-else",
        }
    )

    with pytest.raises(StorageError, match="would not describe"):
        repo.create_node(mismatched, actor=ACTOR)

    assert repo.aggregates.get_node(str(node.id)) is None


def test_a_legacy_node_payload_reports_the_format_change(
    repo: ControlPlaneRepository, connection: sqlite3.Connection
) -> None:
    """A band B row is unreadable, and says so.

    PR-007 restructured the node body -- an opaque ``training_spec`` became a
    typed ``candidate`` -- which is a deliberate format change rather than a
    compatibility bug. What it must not be is a ValidationError about missing
    fields, which tells the reader nothing about why.
    """
    from xaytune.storage.errors import IncompatiblePayloadError

    experiment = repo.create_experiment(make_experiment(), actor=ACTOR)
    node = repo.create_node(make_node(experiment), actor=ACTOR)
    # A child, so the traversals that walk *upwards* actually decode the
    # legacy row. Asking for a root's parents reads no rows at all, which is
    # correct behaviour and would prove nothing here.
    child = repo.create_node(
        make_node(experiment, fingerprint="child", parents=(node.id,)), actor=ACTOR
    )

    legacy = json.loads(
        connection.execute(
            "SELECT payload_json FROM experiment_nodes WHERE id = ?", (str(node.id),)
        ).fetchone()["payload_json"]
    )
    legacy["training_spec"] = {"kind": "sft", "payload": {}}
    del legacy["candidate"]

    with write_transaction(connection):
        connection.execute(
            "UPDATE experiment_nodes SET payload_json = ? WHERE id = ?",
            (json.dumps(legacy), str(node.id)),
        )

    # Every path that reads a node, not just the one that happens to be
    # centralised. Before this, get_node() reported the format change while
    # nodes_for_experiment() and every graph traversal raised a Pydantic
    # ValidationError about missing fields -- same database, same cause,
    # three different stories.
    readers = (
        ("get_node", lambda: repo.aggregates.get_node(str(node.id))),
        (
            "nodes_for_experiment",
            lambda: repo.aggregates.nodes_for_experiment(str(experiment.id)),
        ),
        ("graph.parents", lambda: repo.graph.parents(str(child.id))),
        ("graph.ancestors", lambda: repo.graph.ancestors(str(child.id))),
        ("graph.roots", lambda: repo.graph.roots(str(experiment.id))),
        ("graph.lineage", lambda: repo.graph.lineage(str(child.id))),
    )
    for name, read in readers:
        with pytest.raises(IncompatiblePayloadError, match="format"):
            read()
        assert name


def test_a_genuinely_invalid_payload_is_not_blamed_on_the_format_change(
    repo: ControlPlaneRepository, connection: sqlite3.Connection
) -> None:
    """A schema error must still read as one.

    Reporting every unreadable payload as a version problem would send the
    reader to recreate a database over what is actually a bug.
    """
    from pydantic import ValidationError

    experiment = repo.create_experiment(make_experiment(), actor=ACTOR)
    node = repo.create_node(make_node(experiment), actor=ACTOR)

    broken = json.loads(
        connection.execute(
            "SELECT payload_json FROM experiment_nodes WHERE id = ?", (str(node.id),)
        ).fetchone()["payload_json"]
    )
    broken["revision"] = "not-a-number"

    with write_transaction(connection):
        connection.execute(
            "UPDATE experiment_nodes SET payload_json = ? WHERE id = ?",
            (json.dumps(broken), str(node.id)),
        )

    with pytest.raises(ValidationError):
        repo.aggregates.get_node(str(node.id))
