"""Schema migrations: ordering, idempotency, and atomicity with the ledger."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from xaytune.storage import applied_versions, available_migrations, connect, migrate
from xaytune.storage.errors import MigrationError


def _tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row["name"] for row in rows}


def test_migrate_creates_every_shipped_table(db_path: Path) -> None:
    connection = connect(db_path)
    applied = migrate(connection)

    assert applied == tuple(m.version for m in available_migrations())
    assert {
        "experiments",
        "experiment_nodes",
        "experiment_edges",
        "runs",
        "run_attempts",
        "events",
        "outbox",
        "runtime_operations",
    } <= _tables(connection)


def test_migrate_is_idempotent(db_path: Path) -> None:
    connection = connect(db_path)
    migrate(connection)

    assert migrate(connection) == ()
    assert applied_versions(connection) == tuple(m.version for m in available_migrations())


def test_migrate_resumes_on_a_fresh_connection(db_path: Path) -> None:
    """The ledger is durable, so a later process does not re-run applied work."""
    first = connect(db_path)
    migrate(first)
    first.close()

    second = connect(db_path)
    assert migrate(second) == ()
    assert applied_versions(second) == tuple(m.version for m in available_migrations())


def test_every_shipped_migration_is_well_named() -> None:
    migrations = available_migrations()

    assert migrations, "expected at least migration 001"
    assert [m.version for m in migrations] == sorted(m.version for m in migrations)


def test_malformed_filename_is_refused(tmp_path: Path) -> None:
    (tmp_path / "add-actions.sql").write_text("CREATE TABLE t (id TEXT);", encoding="utf-8")

    with pytest.raises(MigrationError, match="must look like"):
        available_migrations(tmp_path)


def test_duplicate_version_is_refused(tmp_path: Path) -> None:
    (tmp_path / "001_first.sql").write_text("CREATE TABLE a (id TEXT);", encoding="utf-8")
    (tmp_path / "001_second.sql").write_text("CREATE TABLE b (id TEXT);", encoding="utf-8")

    with pytest.raises(MigrationError, match="claim version 1"):
        available_migrations(tmp_path)


def test_migration_inserted_behind_an_applied_one_is_refused(tmp_path: Path, db_path: Path) -> None:
    """A file cannot be slipped in behind a database that has moved past it."""
    (tmp_path / "002_later.sql").write_text("CREATE TABLE later (id TEXT);", encoding="utf-8")
    connection = connect(db_path)
    migrate(connection, tmp_path)

    (tmp_path / "001_earlier.sql").write_text("CREATE TABLE earlier (id TEXT);", encoding="utf-8")

    with pytest.raises(MigrationError, match="has already been applied"):
        migrate(connection, tmp_path)


def test_a_failing_migration_leaves_no_partial_schema(tmp_path: Path, db_path: Path) -> None:
    """The migration and its ledger row are one unit, so neither half survives.

    This is the case ``executescript`` would silently break: it commits any
    pending transaction before running, so an outer transaction would have
    committed the successful half of the script.
    """
    (tmp_path / "001_broken.sql").write_text(
        "CREATE TABLE good (id TEXT);\nCREATE TABLE bad (this is not sql);",
        encoding="utf-8",
    )
    connection = connect(db_path)

    with pytest.raises(MigrationError, match="failed"):
        migrate(connection, tmp_path)

    assert "good" not in _tables(connection)
    assert applied_versions(connection) == ()
