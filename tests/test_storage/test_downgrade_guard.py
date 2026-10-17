"""The downgrade guard (ADR-019 M6, step 1 of the durable harness path).

A build refuses a state database that a newer build migrated: it would
otherwise read and write rows of a shape it does not know. The guard must
ship before any migration that needs it, so these tests pin it from every
entry point that migrates: the runner, the embedded controller host and the
daemon client.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from xaytune.storage import SchemaTooNewError, available_migrations, connect, migrate
from xaytune.storage.errors import MigrationError
from xaytune.storage.migrations import MIGRATIONS_DIR, recorded_versions, require_known_schema


def _schema(connection: sqlite3.Connection) -> list[tuple[str, str]]:
    rows = connection.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
    return [(row["name"], row["sql"]) for row in rows]


def _upgraded(path: Path, version: int = 999) -> None:
    """A database a newer build migrated: every shipped migration, then one this build lacks."""
    connection = connect(path)
    migrate(connection)
    connection.execute(
        "INSERT INTO schema_migrations (version, name, applied_at) "
        "VALUES (?, 'from_a_newer_build', datetime('now'))",
        (version,),
    )
    connection.close()


def _older_build(tmp_path: Path, count: int) -> Path:
    """The migration set of an older build: only its first *count* migrations."""
    directory = tmp_path / "older-build"
    directory.mkdir()
    for migration in available_migrations()[:count]:
        shutil.copy(migration.path, directory / migration.path.name)
    return directory


def test_a_database_a_newer_build_migrated_is_refused_and_left_untouched(db_path: Path) -> None:
    _upgraded(db_path)
    connection = connect(db_path)
    before = (_schema(connection), recorded_versions(connection))
    with pytest.raises(SchemaTooNewError) as refused:
        migrate(connection)
    assert refused.value.unknown == (999,)
    assert refused.value.supported == available_migrations()[-1].version
    assert isinstance(refused.value, MigrationError), "callers catching MigrationError still do"
    assert "Upgrade Xaytune" in str(refused.value)
    assert (_schema(connection), recorded_versions(connection)) == before


def test_an_older_build_refuses_a_database_a_newer_build_migrated(
    tmp_path: Path, db_path: Path
) -> None:
    """What the guard is for: a deployment that has it, meeting a database migrated past it."""
    connection = connect(db_path)
    migrate(connection)
    older = _older_build(tmp_path, count=len(available_migrations()) - 2)
    with pytest.raises(SchemaTooNewError) as refused:
        migrate(connection, older)
    assert refused.value.unknown == tuple(m.version for m in available_migrations()[-2:])


def test_an_older_database_is_still_migrated_forward(tmp_path: Path, db_path: Path) -> None:
    connection = connect(db_path)
    older = _older_build(tmp_path, count=3)
    assert migrate(connection, older) == (1, 2, 3)
    assert migrate(connection) == tuple(m.version for m in available_migrations()[3:])


def test_a_fresh_database_is_migrated() -> None:
    connection = connect(":memory:")
    assert migrate(connection) == tuple(m.version for m in available_migrations())


def test_the_guard_compares_versions_not_counts() -> None:
    known = available_migrations()
    require_known_schema([m.version for m in known], known)
    require_known_schema([], known)
    with pytest.raises(SchemaTooNewError):
        require_known_schema([1, 2, 4000], known)


def test_every_entry_point_that_migrates_refuses_a_newer_database(db_path: Path) -> None:
    from xaytune.daemon.client import DaemonClient
    from xaytune.experiment import EmbeddedControllerHost

    _upgraded(db_path)
    with pytest.raises(SchemaTooNewError):
        EmbeddedControllerHost(db_path)
    with pytest.raises(SchemaTooNewError):
        DaemonClient(db_path)


def test_the_daemon_refuses_a_newer_database_at_startup_and_says_so(db_path: Path) -> None:
    """ADR-019 M6: the daemon's startup check names the refusal; no traceback, its own exit."""
    import os
    import subprocess
    import sys

    from xaytune.daemon.__main__ import EXIT_SCHEMA_TOO_NEW

    _upgraded(db_path)
    before = recorded_versions(connect(db_path))
    repository = Path(__file__).resolve().parents[2]
    finished = subprocess.run(
        [
            sys.executable,
            "-m",
            "xaytune.daemon",
            "--state",
            str(db_path),
            "--config",
            "tests.test_daemon.daemon_config:create_config",
        ],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert finished.returncode == EXIT_SCHEMA_TOO_NEW
    last = finished.stderr.strip().splitlines()[-1]
    assert last.startswith("xaytune daemon: the state database records migration(s) 999")
    assert "Traceback" not in finished.stderr
    assert recorded_versions(connect(db_path)) == before


def test_the_shipped_migrations_are_the_ones_the_guard_knows() -> None:
    assert MIGRATIONS_DIR.is_dir()
    assert {m.version for m in available_migrations()} == set(
        range(1, available_migrations()[-1].version + 1)
    )
