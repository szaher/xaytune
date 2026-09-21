"""The public atomic write surface (ADR-005 §3–§5), against the §11 test list.

§11 names twelve required tests. PR-004 covered 1–4 partially and 8 in part;
this module covers the rest, and each test below names the clause it discharges.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from xaytune.core import Actor, ExperimentStatus, RunAttemptStatus, RuntimeRef
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.errors import ConcurrentModificationError, InvalidTransitionError
from xaytune.core.ids import OperationId
from xaytune.storage import ControlPlaneRepository, UnknownOperationTargetError
from xaytune.storage.journal import IdempotencyConflictError

from .conftest import make_attempt, make_experiment, make_node, make_run

ACTOR = Actor(type="system", id="controller")
DIGEST = "sha256:request-a"


@pytest.fixture
def repo(connection: sqlite3.Connection) -> ControlPlaneRepository:
    return ControlPlaneRepository(connection)


@pytest.fixture
def experiment(repo: ControlPlaneRepository) -> Any:
    return repo.create_experiment(make_experiment(), actor=ACTOR)


@pytest.fixture
def run(repo: ControlPlaneRepository, experiment: Any) -> Any:
    node = repo.create_node(make_node(experiment), actor=ACTOR)
    return repo.create_run(make_run(node), actor=ACTOR)


# ---- §11.1 state, event and outbox commit together -----------------------


def test_a_transition_writes_its_event_in_the_same_commit(
    repo: ControlPlaneRepository, experiment: Any
) -> None:
    activated = experiment.with_status(ExperimentStatus.ACTIVE)
    repo.transition(
        activated, experiment_id=str(experiment.id), actor=ACTOR, event_type="ExperimentActivated"
    )

    events = repo.events.events_for_aggregate(str(experiment.id))
    assert [event.event_type for event in events] == [
        "ExperimentCreated",
        "ExperimentActivated",
    ]
    assert repo.aggregates.load_experiment(str(experiment.id)).status is ExperimentStatus.ACTIVE


def test_an_outbox_row_is_written_with_the_event(
    repo: ControlPlaneRepository, experiment: Any
) -> None:
    """§11.6."""
    activated = experiment.with_status(ExperimentStatus.ACTIVE)
    repo.transition(
        activated,
        experiment_id=str(experiment.id),
        actor=ACTOR,
        destinations=("mlflow", "webhook"),
    )

    pending = repo.events.pending_outbox()
    assert sorted(record.destination for record in pending) == ["mlflow", "webhook"]
    assert {record.event_id for record in pending} == {
        repo.events.events_for_aggregate(str(experiment.id))[-1].id
    }


def test_a_failure_rolls_back_state_event_and_outbox_together(
    repo: ControlPlaneRepository, experiment: Any
) -> None:
    """§11.1: the whole unit, or none of it."""
    activated = experiment.with_status(ExperimentStatus.ACTIVE)

    # Fail after the state update and the event insert have both been issued,
    # so the rollback has something of each to discard.
    original_append = repo.events.append

    def exploding(event: Any) -> int:
        original_append(event)
        raise RuntimeError("injected after the event insert")

    repo.events.append = exploding  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="injected"):
        repo.transition(activated, experiment_id=str(experiment.id), actor=ACTOR)

    repo.events.append = original_append  # type: ignore[method-assign]

    # Neither half survived.
    assert repo.aggregates.load_experiment(str(experiment.id)).status is ExperimentStatus.CREATED
    assert [e.event_type for e in repo.events.events_for_aggregate(str(experiment.id))] == [
        "ExperimentCreated"
    ]


def test_every_aggregate_revision_has_exactly_one_event(
    repo: ControlPlaneRepository, experiment: Any
) -> None:
    """§10.2: an aggregate's revision equals its latest event's revision."""
    activated = experiment.with_status(ExperimentStatus.ACTIVE)
    repo.transition(activated, experiment_id=str(experiment.id), actor=ACTOR)

    stored = repo.aggregates.load_experiment(str(experiment.id))
    events = repo.events.events_for_aggregate(str(experiment.id))

    assert events[-1].aggregate_revision == stored.revision
    assert [e.aggregate_revision for e in events] == [0, 1]


def test_events_carry_a_total_order(repo: ControlPlaneRepository, experiment: Any) -> None:
    """§6 of chapter 07: consumers order on sequence, not on wall clock."""
    activated = experiment.with_status(ExperimentStatus.ACTIVE)
    repo.transition(activated, experiment_id=str(experiment.id), actor=ACTOR)

    sequences = [e.sequence for e in repo.events.events_for_experiment(str(experiment.id))]
    assert sequences == sorted(sequences)
    assert all(s is not None for s in sequences)


# ---- §11.4 attempt + INTENDED operation are atomic -----------------------


def test_attempt_and_submit_intent_commit_together(
    repo: ControlPlaneRepository, experiment: Any, run: Any
) -> None:
    """§11.4 and §4: intent is durable before the runtime is called."""
    attempt, operation = repo.create_attempt_with_submit_intent(
        make_attempt(run),
        experiment_id=str(experiment.id),
        request_digest=DIGEST,
        actor=ACTOR,
    )

    assert repo.aggregates.get_attempt(str(attempt.id)) is not None
    stored = repo.operations.get(str(operation.id))
    assert stored is not None
    assert stored.state == "intended"
    assert stored.target.kind == "training-attempt"
    assert stored.is_unresolved


def test_a_failure_rolls_back_attempt_and_intent_together(
    repo: ControlPlaneRepository, experiment: Any, run: Any
) -> None:
    """§11.4: neither can exist alone."""
    attempt = make_attempt(run)
    operation_id = OperationId.generate()

    original = repo.operations._insert

    def exploding(operation: Any) -> Any:
        original(operation)
        raise RuntimeError("injected after the operation insert")

    repo.operations._insert = exploding  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="injected"):
        repo.create_attempt_with_submit_intent(
            attempt,
            experiment_id=str(experiment.id),
            request_digest=DIGEST,
            actor=ACTOR,
            operation_id=operation_id,
        )
    repo.operations._insert = original  # type: ignore[method-assign]

    assert repo.aggregates.get_attempt(str(attempt.id)) is None
    assert repo.operations.get(str(operation_id)) is None


# ---- §11.6 committed intent survives a reopen ----------------------------


def test_committed_intent_and_digest_survive_a_reopen(
    repo: ControlPlaneRepository,
    connection: sqlite3.Connection,
    db_path: Any,
    experiment: Any,
    run: Any,
) -> None:
    """§11.6."""
    from xaytune.storage import connect

    _, operation = repo.create_attempt_with_submit_intent(
        make_attempt(run),
        experiment_id=str(experiment.id),
        request_digest=DIGEST,
        actor=ACTOR,
    )
    connection.close()

    reopened = connect(db_path)
    try:
        recovered = ControlPlaneRepository(reopened).operations.get(str(operation.id))
        assert recovered is not None
        assert recovered.request_digest == DIGEST
        assert recovered.state == "intended"
    finally:
        reopened.close()


# ---- §11.7 idempotency ---------------------------------------------------


def test_reusing_an_operation_id_with_the_same_request_is_a_no_op(
    repo: ControlPlaneRepository, experiment: Any, run: Any
) -> None:
    """§11.7: a retried create returns what is already recorded."""
    operation_id = OperationId.generate()
    attempt = make_attempt(run)

    _, first = repo.create_attempt_with_submit_intent(
        attempt,
        experiment_id=str(experiment.id),
        request_digest=DIGEST,
        actor=ACTOR,
        operation_id=operation_id,
    )

    with write_once(repo):
        second = repo.operations._insert(first)

    assert second.id == first.id
    assert len(repo.operations.for_target("training-attempt", str(attempt.id))) == 1


def test_reusing_an_operation_id_with_a_different_request_is_refused(
    repo: ControlPlaneRepository, connection: sqlite3.Connection, experiment: Any, run: Any
) -> None:
    """§11.7: guessing which request the caller meant would start a second workload."""
    operation_id = OperationId.generate()
    attempt = make_attempt(run)
    _, first = repo.create_attempt_with_submit_intent(
        attempt,
        experiment_id=str(experiment.id),
        request_digest=DIGEST,
        actor=ACTOR,
        operation_id=operation_id,
    )

    conflicting = first.model_validate(
        {**first.model_dump(mode="python"), "request_digest": "sha256:different"}
    )

    with pytest.raises(IdempotencyConflictError, match="request_digest"):
        with write_once(repo):
            repo.operations._insert(conflicting)


# ---- §11.8 illegal transitions -------------------------------------------


def test_illegal_operation_transitions_are_refused(
    repo: ControlPlaneRepository, experiment: Any, run: Any
) -> None:
    """§11.8, and §10.3: terminal records are immutable."""
    _, operation = repo.create_attempt_with_submit_intent(
        make_attempt(run),
        experiment_id=str(experiment.id),
        request_digest=DIGEST,
        actor=ACTOR,
    )
    confirmed = repo.confirm_operation(operation, experiment_id=str(experiment.id), actor=ACTOR)

    with pytest.raises(InvalidTransitionError):
        confirmed.with_state("failed")


def test_a_stale_operation_revision_is_refused(
    repo: ControlPlaneRepository, experiment: Any, run: Any
) -> None:
    """§11.8: two writers, one read; the second loses."""
    _, operation = repo.create_attempt_with_submit_intent(
        make_attempt(run),
        experiment_id=str(experiment.id),
        request_digest=DIGEST,
        actor=ACTOR,
    )
    repo.confirm_operation(operation, experiment_id=str(experiment.id), actor=ACTOR)

    with pytest.raises(ConcurrentModificationError):
        repo.fail_operation(operation, experiment_id=str(experiment.id), actor=ACTOR)


# ---- §11.12 a lost response stays unresolved -----------------------------


def test_an_unresolved_operation_is_never_recorded_as_failed(
    repo: ControlPlaneRepository, experiment: Any, run: Any
) -> None:
    """§11.12: "we do not know" must not collapse into "it did not happen"."""
    _, operation = repo.create_attempt_with_submit_intent(
        make_attempt(run),
        experiment_id=str(experiment.id),
        request_digest=DIGEST,
        actor=ACTOR,
    )
    sent = repo.mark_operation_sent(operation, experiment_id=str(experiment.id), actor=ACTOR)

    # The response was lost. Nothing transitions it; it stays queryable.
    assert sent.is_unresolved
    assert [op.id for op in repo.operations.unresolved()] == [operation.id]


def test_confirmed_operations_leave_the_unresolved_queue(
    repo: ControlPlaneRepository, experiment: Any, run: Any
) -> None:
    _, operation = repo.create_attempt_with_submit_intent(
        make_attempt(run),
        experiment_id=str(experiment.id),
        request_digest=DIGEST,
        actor=ACTOR,
    )
    repo.confirm_operation(
        operation,
        experiment_id=str(experiment.id),
        actor=ACTOR,
        runtime_ref=RuntimeRef(backend="local", external_id="pid-1"),
    )

    assert repo.operations.unresolved() == ()
    stored = repo.operations.get(str(operation.id))
    assert stored is not None and stored.runtime_ref is not None
    assert stored.runtime_ref.external_id == "pid-1"


# ---- §11.9 cancellation intent -------------------------------------------


def test_cancellation_intent_survives_a_restart(
    repo: ControlPlaneRepository,
    connection: sqlite3.Connection,
    db_path: Any,
    experiment: Any,
    run: Any,
) -> None:
    """§11.9: durable independently of the attempt's observed status."""
    from xaytune.storage import connect

    attempt, _ = repo.create_attempt_with_submit_intent(
        make_attempt(run),
        experiment_id=str(experiment.id),
        request_digest=DIGEST,
        actor=ACTOR,
    )
    cancel = repo.record_cancellation_intent(
        RuntimeOperationTarget(kind="training-attempt", id=str(attempt.id)),
        experiment_id=str(experiment.id),
        request_digest="sha256:cancel",
        actor=ACTOR,
    )
    connection.close()

    reopened = connect(db_path)
    try:
        recovered = ControlPlaneRepository(reopened).operations.get(str(cancel.id))
        assert recovered is not None
        assert recovered.type == "cancel"
        assert recovered.is_unresolved
    finally:
        reopened.close()


# ---- §11.10 evaluation targets behave identically ------------------------


def test_an_operation_against_a_missing_target_is_refused(
    repo: ControlPlaneRepository, experiment: Any
) -> None:
    """§10.1: SQLite cannot enforce a typed target, so the repository does."""
    with pytest.raises(UnknownOperationTargetError):
        repo.record_cancellation_intent(
            RuntimeOperationTarget(kind="training-attempt", id="attempt_does_not_exist"),
            experiment_id=str(experiment.id),
            request_digest="sha256:cancel",
            actor=ACTOR,
        )


def test_an_evaluation_target_is_refused_until_its_table_exists(
    repo: ControlPlaneRepository, experiment: Any
) -> None:
    """ADR-015's tables have not landed, so the target cannot be resolved.

    Refused rather than accepted unchecked: an operation naming a target that
    cannot be verified is exactly the dangling reference §10.1 forbids.
    """
    with pytest.raises(UnknownOperationTargetError, match="evaluation-attempt"):
        repo.record_cancellation_intent(
            RuntimeOperationTarget(kind="evaluation-attempt", id="eval_attempt_1"),
            experiment_id=str(experiment.id),
            request_digest="sha256:cancel",
            actor=ACTOR,
        )


# ---- the write boundary still holds --------------------------------------


def test_there_is_still_no_public_bare_state_write(repo: ControlPlaneRepository) -> None:
    """The escape hatch ADR-005 forbids must not have appeared with the events."""
    public = {name for name in dir(repo) if not name.startswith("_")}
    assert not {name for name in public if name.startswith("save_")}
    for forbidden in ("insert_experiment", "update_experiment", "append_event"):
        assert forbidden not in public


def test_attempt_transitions_carry_their_events(
    repo: ControlPlaneRepository, experiment: Any, run: Any
) -> None:
    attempt, _ = repo.create_attempt_with_submit_intent(
        make_attempt(run),
        experiment_id=str(experiment.id),
        request_digest=DIGEST,
        actor=ACTOR,
    )
    queued = attempt.with_status(RunAttemptStatus.QUEUED)
    repo.transition(
        queued, experiment_id=str(experiment.id), actor=ACTOR, event_type="RuntimeQueued"
    )

    types = [e.event_type for e in repo.events.events_for_aggregate(str(attempt.id))]
    assert types == ["RunAttemptCreated", "RuntimeQueued"]


class write_once:  # noqa: N801 - reads as a statement at the call site
    """Run a single journal write inside a transaction, for tests only."""

    def __init__(self, repo: ControlPlaneRepository) -> None:
        self._repo = repo

    def __enter__(self) -> None:
        self._repo._connection.execute("BEGIN IMMEDIATE")

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self._repo._connection.commit()
        else:
            self._repo._connection.rollback()
