"""Aggregate persistence: round-trips, revision CAS, and the write boundary."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from xaytune.core import ExperimentStatus, RunAttemptStatus
from xaytune.core.errors import ConcurrentModificationError
from xaytune.storage import AggregateStore, connect, write_transaction
from xaytune.storage.errors import AggregateNotFoundError

from .conftest import make_attempt, make_experiment, make_node


def test_aggregates_round_trip_through_the_database(
    store: AggregateStore, seeded: dict[str, Any]
) -> None:
    assert store.get_experiment(str(seeded["experiment"].id)) == seeded["experiment"]
    assert store.get_node(str(seeded["node"].id)) == seeded["node"]
    assert store.get_run(str(seeded["run"].id)) == seeded["run"]
    assert store.get_attempt(str(seeded["attempt"].id)) == seeded["attempt"]


def test_committed_state_survives_reopening_the_database(
    db_path: Path, connection: sqlite3.Connection, store: AggregateStore
) -> None:
    """The property an in-memory database cannot demonstrate."""
    experiment = make_experiment()
    with write_transaction(connection):
        store._insert_experiment(experiment)
    connection.close()

    reopened = connect(db_path)
    try:
        assert AggregateStore(reopened).get_experiment(str(experiment.id)) == experiment
    finally:
        reopened.close()


def test_missing_aggregate_reads_as_none_and_loads_as_an_error(
    store: AggregateStore,
) -> None:
    assert store.get_experiment("exp_00000000000000000000000000") is None

    with pytest.raises(AggregateNotFoundError, match="does not exist"):
        store.load_experiment("exp_00000000000000000000000000")


def test_update_advances_the_stored_revision(
    store: AggregateStore, connection: sqlite3.Connection, seeded: dict[str, Any]
) -> None:
    activated = seeded["experiment"].with_status(ExperimentStatus.ACTIVE)
    with write_transaction(connection):
        store._update_experiment(activated)

    stored = store.load_experiment(str(activated.id))
    assert stored.status is ExperimentStatus.ACTIVE
    assert stored.revision == seeded["experiment"].revision + 1


def test_stale_revision_is_refused(
    store: AggregateStore, connection: sqlite3.Connection, seeded: dict[str, Any]
) -> None:
    """Two writers derive a transition from the same read; the second loses."""
    experiment = seeded["experiment"]
    first = experiment.with_status(ExperimentStatus.ACTIVE)
    second = experiment.with_status(ExperimentStatus.CANCELLED)

    with write_transaction(connection):
        store._update_experiment(first)

    with pytest.raises(ConcurrentModificationError) as caught:
        with write_transaction(connection):
            store._update_experiment(second)

    assert caught.value.aggregate == "Experiment"
    assert caught.value.expected_revision == experiment.revision

    # The loser changed nothing.
    assert store.load_experiment(str(experiment.id)).status is ExperimentStatus.ACTIVE


def test_a_failed_transaction_rolls_the_whole_unit_back(
    store: AggregateStore, connection: sqlite3.Connection
) -> None:
    experiment = make_experiment()
    node = make_node(experiment)

    with pytest.raises(RuntimeError, match="injected"):
        with write_transaction(connection):
            store._insert_experiment(experiment)
            store._insert_node(node)
            raise RuntimeError("injected failure after both inserts")

    assert store.get_experiment(str(experiment.id)) is None
    assert store.get_node(str(node.id)) is None


def test_writes_outside_a_transaction_are_refused(store: AggregateStore) -> None:
    """An autocommitted transition would commit state without its event."""
    with pytest.raises(sqlite3.ProgrammingError, match="write_transaction"):
        store._insert_experiment(make_experiment())


def test_nested_transactions_are_refused(connection: sqlite3.Connection) -> None:
    """SQLite has no nested transactions; flattening one silently is worse."""
    with pytest.raises(sqlite3.OperationalError, match="already open"):
        with write_transaction(connection):
            with write_transaction(connection):
                pass


def test_foreign_keys_are_enforced(store: AggregateStore, connection: sqlite3.Connection) -> None:
    """PRAGMA foreign_keys is per-connection and off by default."""
    orphan = make_node(make_experiment())

    with pytest.raises(sqlite3.IntegrityError):
        with write_transaction(connection):
            store._insert_node(orphan)


def test_lineage_edges_are_written_for_each_parent(
    store: AggregateStore, connection: sqlite3.Connection, seeded: dict[str, Any]
) -> None:
    parent = seeded["node"]
    child = make_node(seeded["experiment"], fingerprint="sha256:cand-b", parents=(parent.id,))

    with write_transaction(connection):
        store._insert_node(child)

    rows = connection.execute(
        "SELECT parent_id, child_id FROM experiment_edges WHERE child_id = ?",
        (str(child.id),),
    ).fetchall()
    assert [(row["parent_id"], row["child_id"]) for row in rows] == [
        (str(parent.id), str(child.id))
    ]


def test_a_node_cannot_be_its_own_parent(
    store: AggregateStore, connection: sqlite3.Connection, seeded: dict[str, Any]
) -> None:
    node = seeded["node"]

    with pytest.raises(sqlite3.IntegrityError):
        with write_transaction(connection):
            connection.execute(
                "INSERT INTO experiment_edges (parent_id, child_id, reason, payload_json) "
                "VALUES (?, ?, NULL, '{}')",
                (str(node.id), str(node.id)),
            )


def test_two_attempts_cannot_claim_the_same_number(
    store: AggregateStore, connection: sqlite3.Connection, seeded: dict[str, Any]
) -> None:
    duplicate = make_attempt(seeded["run"], attempt_number=1)

    with pytest.raises(sqlite3.IntegrityError):
        with write_transaction(connection):
            store._insert_attempt(duplicate)


def test_unresolved_attempts_excludes_terminal_ones(
    store: AggregateStore, connection: sqlite3.Connection, seeded: dict[str, Any]
) -> None:
    run = seeded["run"]
    live = seeded["attempt"]
    finished = make_attempt(run, attempt_number=2)

    with write_transaction(connection):
        store._insert_attempt(finished)

    # One transition per commit: each carries its own event (ADR-005 §3).
    queued = finished.with_status(RunAttemptStatus.QUEUED)
    with write_transaction(connection):
        store._update_attempt(queued)

    cancelled = queued.with_status(RunAttemptStatus.CANCELLED)
    with write_transaction(connection):
        store._update_attempt(cancelled)

    unresolved = {attempt.id for attempt in store.unresolved_attempts()}
    assert unresolved == {live.id}


def test_batching_transitions_is_refused_with_a_distinct_error(
    store: AggregateStore, connection: sqlite3.Connection, seeded: dict[str, Any]
) -> None:
    """Two transitions, one write, would leave the middle one with no event."""
    attempt = seeded["attempt"]
    two_ahead = attempt.with_status(RunAttemptStatus.QUEUED).with_status(RunAttemptStatus.STARTING)

    with pytest.raises(ValueError, match="transitions ahead"):
        with write_transaction(connection):
            store._update_attempt(two_ahead)


def test_children_are_listed_in_order(
    store: AggregateStore, connection: sqlite3.Connection, seeded: dict[str, Any]
) -> None:
    run = seeded["run"]
    second = make_attempt(run, attempt_number=2)
    third = make_attempt(run, attempt_number=3)

    with write_transaction(connection):
        store._insert_attempt(third)
        store._insert_attempt(second)

    numbers = [attempt.attempt_number for attempt in store.attempts_for_run(str(run.id))]
    assert numbers == [1, 2, 3]


def test_node_fingerprint_is_queryable_as_a_column(
    connection: sqlite3.Connection, seeded: dict[str, Any]
) -> None:
    """ "Has this hypothesis been explored?" must not require parsing payloads."""
    row = connection.execute(
        "SELECT id FROM experiment_nodes WHERE candidate_fingerprint = ?",
        (seeded["node"].candidate_fingerprint,),
    ).fetchone()
    assert row["id"] == str(seeded["node"].id)


def test_run_fingerprint_is_queryable_as_a_column(
    connection: sqlite3.Connection, seeded: dict[str, Any]
) -> None:
    row = connection.execute(
        "SELECT id FROM runs WHERE candidate_fingerprint = ?",
        (seeded["run"].candidate_fingerprint,),
    ).fetchone()
    assert row["id"] == str(seeded["run"].id)


def test_separate_connections_contend_without_losing_an_update(
    db_path: Path, connection: sqlite3.Connection, store: AggregateStore
) -> None:
    """Correctness does not depend on a single writer (ADR-005 §8).

    Two connections, each with its own transaction, both derive a transition
    from the same read. SQLite serializes the writes and revision CAS rejects
    the loser -- no external lease involved.
    """
    experiment = make_experiment()
    with write_transaction(connection):
        store._insert_experiment(experiment)

    other = connect(db_path)
    try:
        other_store = AggregateStore(other)
        read_by_a = store.load_experiment(str(experiment.id))
        read_by_b = other_store.load_experiment(str(experiment.id))

        with write_transaction(connection):
            store._update_experiment(read_by_a.with_status(ExperimentStatus.ACTIVE))

        with pytest.raises(ConcurrentModificationError):
            with write_transaction(other):
                other_store._update_experiment(read_by_b.with_status(ExperimentStatus.CANCELLED))

        assert other_store.load_experiment(str(experiment.id)).status is ExperimentStatus.ACTIVE
    finally:
        other.close()
