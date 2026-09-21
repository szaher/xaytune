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
from xaytune.storage import ControlPlaneRepository
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
    repo.transition(activated, actor=ACTOR, event_type="ExperimentActivated")

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
    repo.transition(activated, actor=ACTOR, destinations=("mlflow", "webhook"))

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
    original_append = repo.events._append

    def exploding(event: Any) -> int:
        original_append(event)
        raise RuntimeError("injected after the event insert")

    repo.events._append = exploding  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="injected"):
        repo.transition(activated, actor=ACTOR)

    repo.events._append = original_append  # type: ignore[method-assign]

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
    repo.transition(activated, actor=ACTOR)

    stored = repo.aggregates.load_experiment(str(experiment.id))
    events = repo.events.events_for_aggregate(str(experiment.id))

    assert events[-1].aggregate_revision == stored.revision
    assert [e.aggregate_revision for e in events] == [0, 1]


def test_events_carry_a_total_order(repo: ControlPlaneRepository, experiment: Any) -> None:
    """§6 of chapter 07: consumers order on sequence, not on wall clock."""
    activated = experiment.with_status(ExperimentStatus.ACTIVE)
    repo.transition(activated, actor=ACTOR)

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
        request_digest=DIGEST,
        actor=ACTOR,
    )
    confirmed = repo.confirm_operation(operation, actor=ACTOR)

    with pytest.raises(InvalidTransitionError):
        confirmed.with_state("failed")


def test_a_stale_operation_revision_is_refused(
    repo: ControlPlaneRepository, experiment: Any, run: Any
) -> None:
    """§11.8: two writers, one read; the second loses."""
    _, operation = repo.create_attempt_with_submit_intent(
        make_attempt(run),
        request_digest=DIGEST,
        actor=ACTOR,
    )
    repo.confirm_operation(operation, actor=ACTOR)

    with pytest.raises(ConcurrentModificationError):
        repo.fail_operation(operation, actor=ACTOR)


# ---- §11.12 a lost response stays unresolved -----------------------------


def test_an_unresolved_operation_is_never_recorded_as_failed(
    repo: ControlPlaneRepository, experiment: Any, run: Any
) -> None:
    """§11.12: "we do not know" must not collapse into "it did not happen"."""
    _, operation = repo.create_attempt_with_submit_intent(
        make_attempt(run),
        request_digest=DIGEST,
        actor=ACTOR,
    )
    sent = repo.mark_operation_sent(operation, actor=ACTOR)

    # The response was lost. Nothing transitions it; it stays queryable.
    assert sent.is_unresolved
    assert [op.id for op in repo.operations.unresolved()] == [operation.id]


def test_confirmed_operations_leave_the_unresolved_queue(
    repo: ControlPlaneRepository, experiment: Any, run: Any
) -> None:
    _, operation = repo.create_attempt_with_submit_intent(
        make_attempt(run),
        request_digest=DIGEST,
        actor=ACTOR,
    )
    repo.confirm_operation(
        operation, actor=ACTOR, runtime_ref=RuntimeRef(backend="local", external_id="pid-1")
    )

    assert repo.operations.unresolved() == ()
    stored = repo.operations.get(str(operation.id))
    assert stored is not None and stored.runtime_ref is not None
    assert stored.runtime_ref.external_id == "pid-1"


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
        request_digest=DIGEST,
        actor=ACTOR,
    )
    queued = attempt.with_status(RunAttemptStatus.QUEUED)
    repo.transition(queued, actor=ACTOR, event_type="RuntimeQueued")

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


# ---- the escape hatches that had to be closed ----------------------------


def test_a_bare_event_write_is_refused(repo: ControlPlaneRepository) -> None:
    """The violation this module's docstring used to claim was impossible.

    ``connect()`` sets ``isolation_level=None``, so before the journal writers
    were made private and transaction-bound, this committed a fabricated event
    with no aggregate behind it -- exactly what ADR-005 §3 forbids.
    """
    from xaytune.core.domain.event import DomainEvent
    from xaytune.core.ids import EventId

    fabricated = DomainEvent(
        id=EventId.generate(),
        experiment_id="exp_nonexistent",
        aggregate_type="Experiment",
        aggregate_id="exp_nonexistent",
        aggregate_revision=0,
        event_type="Fabricated",
        actor=ACTOR,
    )

    assert not hasattr(repo.events, "append")
    with pytest.raises(sqlite3.ProgrammingError, match="write_transaction"):
        repo.events._append(fabricated)

    assert repo.events.events_for_aggregate("exp_nonexistent") == ()


def test_a_bare_operation_write_is_refused(repo: ControlPlaneRepository, run: Any) -> None:
    from xaytune.core.domain.operation import RuntimeOperation

    orphan = RuntimeOperation(
        id=OperationId.generate(),
        target=RuntimeOperationTarget(kind="training-attempt", id="attempt_nonexistent"),
        type="submit",
        request_digest=DIGEST,
    )

    with pytest.raises(sqlite3.ProgrammingError, match="write_transaction"):
        repo.operations._insert(orphan)

    assert repo.operations.get(str(orphan.id)) is None


def test_the_journal_classes_are_not_part_of_the_public_api() -> None:
    """Exporting them invited exactly the bare write above."""
    import xaytune.storage as storage

    assert "EventJournal" not in storage.__all__
    assert "OperationJournal" not in storage.__all__


# ---- §11.7 at the layer where it matters ---------------------------------


def test_the_public_compound_write_is_idempotent(
    repo: ControlPlaneRepository, experiment: Any, run: Any
) -> None:
    """The crash-and-retry case ADR-013 exists to make safe.

    Journal-level idempotency alone is not enough: the attempt insert runs
    first, so a naive retry hits the attempt's primary key before the operation
    id is ever consulted.
    """
    attempt = make_attempt(run)
    operation_id = OperationId.generate()

    first = repo.create_attempt_with_submit_intent(
        attempt, request_digest=DIGEST, actor=ACTOR, operation_id=operation_id
    )
    second = repo.create_attempt_with_submit_intent(
        attempt, request_digest=DIGEST, actor=ACTOR, operation_id=operation_id
    )

    assert second[0].id == first[0].id
    assert second[1].id == first[1].id

    assert len(repo.aggregates.attempts_for_run(str(run.id))) == 1
    assert len(repo.operations.for_target("training-attempt", str(attempt.id))) == 1
    created = [
        event
        for event in repo.events.events_for_aggregate(str(attempt.id))
        if event.event_type == "RunAttemptCreated"
    ]
    assert len(created) == 1


def test_a_retry_with_a_changed_request_is_refused(repo: ControlPlaneRepository, run: Any) -> None:
    operation_id = OperationId.generate()
    attempt = make_attempt(run)
    repo.create_attempt_with_submit_intent(
        attempt, request_digest=DIGEST, actor=ACTOR, operation_id=operation_id
    )

    with pytest.raises(IdempotencyConflictError, match="request_digest"):
        repo.create_attempt_with_submit_intent(
            attempt,
            request_digest="sha256:changed",
            actor=ACTOR,
            operation_id=operation_id,
        )


# ---- provenance: the owning experiment is derived, never supplied --------


def test_an_event_cannot_be_filed_under_the_wrong_experiment(
    repo: ControlPlaneRepository, experiment: Any, run: Any
) -> None:
    """A caller-supplied experiment id could disagree with actual ownership.

    The state change would land on one experiment and its event under another's
    history, with nothing raising and no later query able to detect it.
    """
    other = repo.create_experiment(make_experiment("other"), actor=ACTOR)
    attempt, _ = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest=DIGEST, actor=ACTOR
    )

    queued = attempt.with_status(RunAttemptStatus.QUEUED)
    repo.transition(queued, actor=ACTOR)

    # `other` has only its own creation event; nothing from this run leaked in.
    assert [e.aggregate_type for e in repo.events.events_for_experiment(str(other.id))] == [
        "Experiment"
    ]
    filed = {e.experiment_id for e in repo.events.events_for_aggregate(str(attempt.id))}
    assert filed == {str(experiment.id)}


# ---- observational events share an aggregate revision --------------------


def test_many_events_may_share_one_aggregate_revision(
    repo: ControlPlaneRepository, connection: sqlite3.Connection, experiment: Any, run: Any
) -> None:
    """ADR-014 emits thousands of observations per attempt.

    §10.2 requires the aggregate's revision to equal that of its latest
    *state-transition* event -- not that a revision carries exactly one event.
    A unique index here would have made the second observation impossible and
    forced a revision bump per metric.
    """
    attempt, _ = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest=DIGEST, actor=ACTOR
    )

    from xaytune.core.domain.event import DomainEvent
    from xaytune.core.ids import EventId
    from xaytune.storage import write_transaction

    with write_transaction(connection):
        for observation in ("MetricObserved", "MetricObserved", "Heartbeat"):
            repo.events._append(
                DomainEvent(
                    id=EventId.generate(),
                    experiment_id=str(experiment.id),
                    aggregate_type="RunAttempt",
                    aggregate_id=str(attempt.id),
                    aggregate_revision=attempt.revision,
                    event_type=observation,
                    actor=ACTOR,
                )
            )

    events = repo.events.events_for_aggregate(str(attempt.id))
    at_revision = [e for e in events if e.aggregate_revision == attempt.revision]
    assert len(at_revision) == 4
    assert repo.aggregates.load_attempt(str(attempt.id)).revision == attempt.revision
