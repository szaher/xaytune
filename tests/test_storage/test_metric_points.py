"""The metric-point store (migration 020): append-only, idempotent on the stream position."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.test_storage.conftest import make_attempt, make_experiment, make_node, make_run
from xaytune.core.errors import IdempotencyConflictError
from xaytune.core.metric_points import metric_points
from xaytune.core.telemetry import MetricObservedPayload, TrainingMetricObserved
from xaytune.storage import write_transaction
from xaytune.storage.control_plane import ControlPlaneRepository
from xaytune.storage.database import connect
from xaytune.storage.errors import AggregateNotFoundError
from xaytune.storage.metrics import TelemetryPosition
from xaytune.storage.migrations import migrate


def _repository(path: Path) -> ControlPlaneRepository:
    connection = connect(path)
    migrate(connection)
    return ControlPlaneRepository(connection)


def _seeded(path: Path):
    repo = _repository(path)
    experiment = make_experiment()
    node = make_node(experiment)
    run = make_run(node)
    attempt = make_attempt(run)
    with write_transaction(repo._connection):
        repo.aggregates._insert_experiment(experiment)
        repo.aggregates._insert_node(node)
        repo.aggregates._insert_run(run)
        repo.aggregates._insert_attempt(attempt)
    return repo, run, attempt


def _at(sequence: int, event_id: str | None = None, generation: int = 0) -> TelemetryPosition:
    return TelemetryPosition(
        stream_generation=generation,
        sequence=sequence,
        event_id=event_id or f"event-{generation}-{sequence}",
        protocol_version="xaytune.telemetry/v1alpha3",
        emitted_at=datetime(2026, 10, 15, 12, 0, sequence, tzinfo=timezone.utc),
    )


def _loss(step: int, loss: float, **fields: float):
    return metric_points(TrainingMetricObserved(optimizer_step=step, loss=loss, **fields))


def test_points_are_recorded_with_their_provenance(tmp_path: Path) -> None:
    repo, run, attempt = _seeded(tmp_path / "state.db")
    assert repo.record_metric_points(attempt.id, _at(0), _loss(1, 2.5, learning_rate=1e-3)) == 2
    assert repo.record_metric_points(attempt.id, _at(1), _loss(2, 2.0)) == 1
    (series_lr, series_loss) = repo.metric_points.series(str(attempt.id))
    assert (series_loss.name, series_loss.count) == ("training.loss", 2)
    assert (series_lr.name, series_lr.count) == ("training.learning_rate", 1)
    points = repo.metric_points.points(str(attempt.id), "training.loss", unit=None)
    assert [(p.optimizer_step, p.value, p.sequence) for p in points] == [(1, 2.5, 0), (2, 2.0, 1)]
    row = repo._connection.execute(
        "SELECT run_id, experiment_id, event_id, protocol_version, observation, emitted_at "
        "FROM metric_points WHERE sequence = 0 AND name = 'training.loss'"
    ).fetchone()
    assert dict(row) == {
        "run_id": str(run.id),
        "experiment_id": str(run.experiment_id),
        "event_id": "event-0-0",
        "protocol_version": "xaytune.telemetry/v1alpha3",
        "observation": "TrainingMetricObserved",
        "emitted_at": "2026-10-15T12:00:00+00:00",
    }


def test_a_replayed_event_is_recorded_once(tmp_path: Path) -> None:
    repo, _, attempt = _seeded(tmp_path / "state.db")
    repo.record_metric_points(attempt.id, _at(4), _loss(5, 1.5))
    assert repo.record_metric_points(attempt.id, _at(4), _loss(5, 1.5)) == 0
    assert repo.metric_points.series(str(attempt.id))[0].count == 1


@pytest.mark.parametrize(
    ("position", "points", "differing"),
    [
        (_at(4), _loss(5, 9.0), "value"),
        (_at(4, event_id="another-event"), _loss(5, 1.5), "event_id"),
        (_at(4), _loss(6, 1.5), "optimizer_step"),
        (_at(4), _loss(5, 1.5, learning_rate=1e-3), "name"),
        (
            replace(_at(4), protocol_version="xaytune.telemetry/v9"),
            _loss(5, 1.5),
            "protocol_version",
        ),
        (replace(_at(4), emitted_at=_at(5).emitted_at), _loss(5, 1.5), "emitted_at"),
        (replace(_at(4), emitted_at=None), _loss(5, 1.5), "emitted_at"),
    ],
    ids=["value", "event", "step", "points", "protocol", "emitted", "emitted-absent"],
)
def test_a_different_event_at_a_recorded_position_is_a_conflict(
    tmp_path: Path, position, points, differing
) -> None:
    repo, _, attempt = _seeded(tmp_path / "state.db")
    repo.record_metric_points(attempt.id, _at(4), _loss(5, 1.5))
    with pytest.raises(IdempotencyConflictError, match="metric observation") as raised:
        repo.record_metric_points(attempt.id, position, points)
    assert differing in raised.value.differing
    (only,) = repo.metric_points.points(str(attempt.id), "training.loss", unit=None)
    assert (only.value, only.event_id) == (1.5, "event-0-4")


def test_a_new_stream_generation_is_a_new_position(tmp_path: Path) -> None:
    repo, _, attempt = _seeded(tmp_path / "state.db")
    repo.record_metric_points(attempt.id, _at(0), _loss(1, 3.0))
    repo.record_metric_points(attempt.id, _at(0, generation=1), _loss(2, 2.0))
    points = repo.metric_points.points(str(attempt.id), "training.loss", unit=None)
    assert [(p.stream_generation, p.sequence, p.value) for p in points] == [
        (0, 0, 3.0),
        (1, 0, 2.0),
    ]


def test_points_survive_a_restart(tmp_path: Path) -> None:
    repo, _, attempt = _seeded(tmp_path / "state.db")
    repo.record_metric_points(attempt.id, _at(0), _loss(1, 3.0))
    repo._connection.close()
    reopened = _repository(tmp_path / "state.db")
    (point,) = reopened.metric_points.points(str(attempt.id), "training.loss", unit=None)
    assert (point.optimizer_step, point.value) == (1, 3.0)
    assert reopened.record_metric_points(attempt.id, _at(0), _loss(1, 3.0)) == 0


def test_an_unknown_attempt_records_nothing(tmp_path: Path) -> None:
    repo, _, _ = _seeded(tmp_path / "state.db")
    with pytest.raises(AggregateNotFoundError):
        repo.record_metric_points("att_unknown", _at(0), _loss(1, 3.0))
    assert repo._connection.execute("SELECT COUNT(*) FROM metric_points").fetchone()[0] == 0


def test_points_are_append_only_and_belong_to_their_attempt(tmp_path: Path) -> None:
    repo, run, attempt = _seeded(tmp_path / "state.db")
    repo.record_metric_points(attempt.id, _at(0), _loss(1, 3.0))
    connection = repo._connection
    for statement in (
        "UPDATE metric_points SET value = 0",
        "DELETE FROM metric_points",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(statement)
    with pytest.raises(sqlite3.IntegrityError, match="does not belong"):
        connection.execute(
            "INSERT INTO metric_points (attempt_id, run_id, experiment_id, stream_generation, "
            "sequence, name, value, observation, event_id, protocol_version, recorded_at) "
            "VALUES (?, ?, 'exp_other', 0, 9, 'training.loss', 1.0, 'TrainingMetricObserved', "
            "'e', 'p', 'now')",
            (str(attempt.id), str(run.id)),
        )


def test_infinities_are_refused_by_the_schema(tmp_path: Path) -> None:
    repo, run, attempt = _seeded(tmp_path / "state.db")
    with pytest.raises(sqlite3.IntegrityError):
        repo._connection.execute(
            "INSERT INTO metric_points (attempt_id, run_id, experiment_id, stream_generation, "
            "sequence, name, value, observation, event_id, protocol_version, recorded_at) "
            "VALUES (?, ?, ?, 0, 9, 'training.loss', ?, 'TrainingMetricObserved', 'e', 'p', 'now')",
            (str(attempt.id), str(run.id), str(run.experiment_id), float("inf")),
        )


def test_a_series_is_a_name_and_the_rank_that_reported_it(tmp_path: Path) -> None:
    from xaytune.core.telemetry import DistributedMetricObserved

    repo, _, attempt = _seeded(tmp_path / "state.db")
    for sequence, rank in ((0, 0), (1, 1), (2, 0)):
        observation = DistributedMetricObserved(
            rank=rank, world_size=2, step_duration_seconds=0.1 * (sequence + 1)
        )
        repo.record_metric_points(attempt.id, _at(sequence), metric_points(observation))
    summaries = repo.metric_points.series(str(attempt.id))
    assert [(s.name, s.unit, s.rank, s.count) for s in summaries] == [
        ("distributed.step_duration_seconds", "s", 0, 2),
        ("distributed.step_duration_seconds", "s", 1, 1),
    ]
    rank_zero = repo.metric_points.points(
        str(attempt.id), "distributed.step_duration_seconds", unit="s", rank=0
    )
    assert [p.sequence for p in rank_zero] == [0, 2]


def test_one_name_in_two_units_is_two_series_never_one(tmp_path: Path) -> None:
    """A custom metric reported in ms, then in s: neither series holds the other's values."""
    repo, _, attempt = _seeded(tmp_path / "state.db")
    reported = [(0, 1000.0, "ms"), (1, 1.0, "s"), (2, 1500.0, "ms")]
    for sequence, value, unit in reported:
        observation = MetricObservedPayload(name="duration", value=value, unit=unit)
        repo.record_metric_points(attempt.id, _at(sequence), metric_points(observation))
    summaries = repo.metric_points.series(str(attempt.id))
    assert [(s.name, s.unit, s.count) for s in summaries] == [
        ("custom.duration", "ms", 2),
        ("custom.duration", "s", 1),
    ]
    by_unit = {
        s.unit: [p.value for p in repo.metric_points.points(str(attempt.id), s.name, unit=s.unit)]
        for s in summaries
    }
    assert by_unit == {"ms": [1000.0, 1500.0], "s": [1.0]}
    assert repo.metric_points.points(str(attempt.id), "custom.duration", unit=None) == ()
