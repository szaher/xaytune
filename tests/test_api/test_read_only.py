"""The API never writes, reads one snapshot, and refuses a database it cannot read (ADR-023 §S3)."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from tests.test_api.conftest import Recorded, client
from tests.test_api.test_real_data import _every_path
from xaytune.api import StoreUnavailableError, StudioStore, create_app
from xaytune.core import Actor
from xaytune.core.sqlite import connect
from xaytune.storage import ControlPlaneRepository
from xaytune.storage.migrations import available_migrations, migrate


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_route_leaves_a_read_only_file_byte_identical(
    recorded: Recorded, state_copy: Path
) -> None:
    """S-F1: the file is read-only on disk, and unchanged after every route has read it."""
    before = _digest(state_copy)
    state_copy.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    try:
        api = client(state_copy)
        for path in _every_path(api, recorded):
            assert api.get(path).status_code == 200, path
            assert api.head(path).status_code == 200, path
    finally:
        state_copy.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert _digest(state_copy) == before


def test_the_api_never_migrates(recorded: Recorded, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the Studio API must never migrate")

    monkeypatch.setattr("xaytune.storage.migrations.migrate", refuse)
    monkeypatch.setattr("xaytune.storage.migrate", refuse)
    api = client(recorded.state)
    for path in _every_path(api, recorded):
        assert api.get(path).status_code == 200, path


def test_one_response_is_one_snapshot(state_copy: Path) -> None:
    """S-F2: a commit landing between two reads of one snapshot is not seen by the second."""
    store = StudioStore(state_copy)
    writer = connect(state_copy)
    try:
        with store.snapshot() as repository:
            before = repository.aggregates.list_experiments()
            from tests.test_storage.conftest import make_experiment

            ControlPlaneRepository(writer).create_experiment(
                make_experiment("committed-meanwhile"), actor=Actor(type="system", id="test")
            )
            during = repository.aggregates.list_experiments()
        with store.snapshot() as repository:
            after = repository.aggregates.list_experiments()
    finally:
        writer.close()
    assert [e.id for e in during] == [e.id for e in before]
    assert len(after) == len(before) + 1


# ---- the schema check --------------------------------------------------------------------------


def _older(tmp_path: Path) -> Path:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    for migration in available_migrations()[:-1]:
        (migrations / migration.path.name).write_text(migration.sql)
    path = tmp_path / "older.db"
    connection = connect(path)
    migrate(connection, migrations)
    connection.close()
    return path


def _newer(state_copy: Path) -> Path:
    connection = sqlite3.connect(state_copy)
    connection.execute(
        "INSERT INTO schema_migrations (version, name, applied_at) "
        "VALUES (999, 'from_the_future', datetime('now'))"
    )
    connection.commit()
    connection.close()
    return state_copy


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda tmp, copy: _older(tmp), "older migration"),
        (lambda tmp, copy: _newer(copy), "newer migration"),
    ],
    ids=["older", "newer"],
)
def test_a_database_at_another_migration_is_refused_per_request(
    tmp_path: Path, state_copy: Path, make: object, reason: str
) -> None:
    path = make(tmp_path, state_copy)  # type: ignore[operator]
    api = client(path)
    meta = api.get("/api/v1/meta")
    assert meta.status_code == 200
    assert meta.json()["store"]["readable"] is False
    assert reason in meta.json()["store"]["reason"]
    refused = api.get("/api/v1/experiments")
    assert refused.status_code == 503
    assert refused.headers["content-type"] == "application/problem+json"
    assert reason in refused.json()["detail"]


def test_a_database_migrated_while_the_api_runs_is_read_from_then_on(tmp_path: Path) -> None:
    path = _older(tmp_path)
    api = client(path)
    assert api.get("/api/v1/experiments").status_code == 503
    connection = connect(path)
    migrate(connection)
    connection.close()
    assert api.get("/api/v1/experiments").status_code == 200


def test_a_missing_file_is_refused_and_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"
    with pytest.raises(StoreUnavailableError, match="does not exist"):
        create_app(missing)
    assert not missing.exists()


def test_a_file_that_is_not_a_state_database_is_refused(tmp_path: Path) -> None:
    plain = tmp_path / "plain.db"
    sqlite3.connect(plain).close()
    with pytest.raises(StoreUnavailableError, match="not a Xaytune state database"):
        create_app(plain)
    garbage = tmp_path / "garbage.db"
    garbage.write_bytes(os.urandom(4096))
    with pytest.raises(StoreUnavailableError):
        create_app(garbage)
