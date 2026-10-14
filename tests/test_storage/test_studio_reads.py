"""The three reads the Studio API adds to the stores (ADR-023 §S4)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from xaytune.core import Actor, Experiment, ExperimentId, ExperimentStatus
from xaytune.core.clock import utc_now
from xaytune.core.sqlite import connect, connect_read_only
from xaytune.storage import ControlPlaneRepository
from xaytune.storage.migrations import applied_versions, available_migrations, recorded_versions

from .conftest import make_experiment

ACTOR = Actor(type="system", id="controller")


@pytest.fixture
def repo(connection: sqlite3.Connection) -> ControlPlaneRepository:
    return ControlPlaneRepository(connection)


def _at(name: str, created_at: datetime) -> Experiment:
    """An experiment created at *created_at*: built whole, since model_copy refuses timestamps."""
    record = make_experiment(name).model_dump()
    return Experiment.model_validate({**record, "created_at": created_at, "updated_at": created_at})


def _experiments(
    repo: ControlPlaneRepository, count: int, start: datetime | None = None
) -> list[str]:
    start = start or utc_now()
    ids = []
    for index in range(count):
        experiment = _at(f"e{index}", start + timedelta(seconds=index))
        ids.append(str(repo.create_experiment(experiment, actor=ACTOR).id))
    return ids


# ---- list_experiments --------------------------------------------------------------------------


def test_experiments_are_listed_newest_first(repo: ControlPlaneRepository) -> None:
    ids = _experiments(repo, 3)
    listed = repo.aggregates.list_experiments()
    assert [str(experiment.id) for experiment in listed] == ids[::-1]


def test_pages_follow_one_another_by_key_without_gaps_or_repeats(
    repo: ControlPlaneRepository,
) -> None:
    ids = _experiments(repo, 5)
    first = repo.aggregates.list_experiments(limit=2)
    last = first[-1]
    second = repo.aggregates.list_experiments(before=(last.created_at, str(last.id)), limit=2)
    last = second[-1]
    third = repo.aggregates.list_experiments(before=(last.created_at, str(last.id)), limit=2)
    seen = [str(experiment.id) for page in (first, second, third) for experiment in page]
    assert seen == ids[::-1]


def test_a_page_is_stable_while_newer_experiments_are_created(
    repo: ControlPlaneRepository,
) -> None:
    ids = _experiments(repo, 3)
    first = repo.aggregates.list_experiments(limit=2)
    _experiments(repo, 2, start=utc_now() + timedelta(hours=1))  # after the first page
    last = first[-1]
    rest = repo.aggregates.list_experiments(before=(last.created_at, str(last.id)))
    assert [str(experiment.id) for experiment in rest] == [ids[0]]


def test_experiments_with_the_same_timestamp_are_ordered_by_id(
    repo: ControlPlaneRepository,
) -> None:
    moment = utc_now()
    ids = sorted(
        str(
            repo.create_experiment(
                _at(f"tie{index}", moment),
                actor=ACTOR,
            ).id
        )
        for index in range(3)
    )
    first = repo.aggregates.list_experiments(limit=1)
    rest = repo.aggregates.list_experiments(before=(moment, str(first[0].id)))
    assert [str(e.id) for e in (*first, *rest)] == ids[::-1]


def test_experiments_can_be_filtered_by_status(repo: ControlPlaneRepository) -> None:
    ids = _experiments(repo, 2)
    repo.transition_experiment(
        ExperimentId(ids[0]), expected_revision=0, new_status=ExperimentStatus.ACTIVE, actor=ACTOR
    )
    active = repo.aggregates.list_experiments(status="active")
    created = repo.aggregates.list_experiments(status="created")
    assert [str(e.id) for e in active] == [ids[0]]
    assert [str(e.id) for e in created] == [ids[1]]


def test_a_limit_below_one_is_refused(repo: ControlPlaneRepository) -> None:
    with pytest.raises(ValueError):
        repo.aggregates.list_experiments(limit=0)


# ---- latest_sequence ---------------------------------------------------------------------------


def test_the_latest_sequence_spans_experiments(repo: ControlPlaneRepository) -> None:
    assert repo.events.latest_sequence() == 0
    ids = _experiments(repo, 2)
    latest = repo.events.latest_sequence()
    assert latest > 0
    assert latest == max(repo.events.latest_sequence_for_experiment(i) for i in ids)


# ---- recorded_versions -------------------------------------------------------------------------


def test_recorded_versions_reads_the_ledger(connection: sqlite3.Connection) -> None:
    assert recorded_versions(connection) == applied_versions(connection)
    assert recorded_versions(connection) == tuple(m.version for m in available_migrations())


def test_recorded_versions_never_creates_a_ledger(tmp_path: Path) -> None:
    path = tmp_path / "plain.sqlite3"
    connect(path).close()
    reader = connect_read_only(path)
    try:
        assert recorded_versions(reader) is None
    finally:
        reader.close()
    check = sqlite3.connect(path)
    try:
        tables = check.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    finally:
        check.close()
    assert tables == []


def test_a_read_only_connection_refuses_writes(
    db_path: Path, connection: sqlite3.Connection
) -> None:
    reader = connect_read_only(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            reader.execute("DELETE FROM experiments")
    finally:
        reader.close()


def test_a_read_only_connection_never_creates_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        connect_read_only(missing)
    assert not missing.exists()
