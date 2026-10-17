"""The metric-point store (migration 020): append-only scalars from training telemetry.

Rows are keyed by the event's position in its attempt's stream, so a
replayed event (ADR-014 §4: at-least-once delivery) is recognised, never
stored twice. Writes go through
:meth:`~xaytune.storage.control_plane.ControlPlaneRepository.record_metric_points`;
this store only inserts inside that transaction and reads.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from xaytune.core.clock import utc_now
from xaytune.core.errors import IdempotencyConflictError
from xaytune.core.metric_points import MetricPoint
from xaytune.storage.journal import _require_transaction

__all__ = ["MetricPointStore", "MetricSeriesSummary", "StoredMetricPoint", "TelemetryPosition"]


@dataclass(frozen=True)
class TelemetryPosition:
    """Where one telemetry event sits, and who wrote it: a point's provenance."""

    stream_generation: int
    sequence: int
    event_id: str
    protocol_version: str
    emitted_at: datetime | None


@dataclass(frozen=True)
class StoredMetricPoint:
    """One recorded point, in stream order."""

    stream_generation: int
    sequence: int
    optimizer_step: int | None
    value: float
    worker: str | None
    rank: int | None
    event_id: str
    emitted_at: str | None


@dataclass(frozen=True)
class MetricSeriesSummary:
    """One series of one attempt: a name and unit, and the worker and rank that reported it."""

    name: str
    unit: str | None
    worker: str | None
    rank: int | None
    count: int


_COMPARED = (
    "name",
    "value",
    "unit",
    "optimizer_step",
    "worker",
    "rank",
    "observation",
    "event_id",
    "protocol_version",
    "emitted_at",
)
"""Everything a recorded point says: a replay must repeat all of it, provenance included."""


class MetricPointStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def series(self, attempt_id: str) -> tuple[MetricSeriesSummary, ...]:
        """Every series *attempt_id* reported, by name, then worker and rank."""
        rows = self._connection.execute(
            "SELECT name, unit, worker, rank, COUNT(*) AS count FROM metric_points "
            "WHERE attempt_id = ? GROUP BY name, unit, worker, rank "
            "ORDER BY name, worker IS NOT NULL, worker, rank IS NOT NULL, rank",
            (attempt_id,),
        ).fetchall()
        return tuple(
            MetricSeriesSummary(
                name=row["name"],
                unit=row["unit"],
                worker=row["worker"],
                rank=row["rank"],
                count=int(row["count"]),
            )
            for row in rows
        )

    def points(
        self,
        attempt_id: str,
        name: str,
        *,
        unit: str | None,
        worker: str | None = None,
        rank: int | None = None,
    ) -> tuple[StoredMetricPoint, ...]:
        """Every point of one series, in stream order (generation, then sequence).

        A series is a name and unit *and* the worker and rank that reported
        it: two ranks' throughput are two series, never one interleaved line,
        and a value in milliseconds is never drawn on a line of seconds. *unit*
        is required: ``None`` means a unitless series, not any unit.
        """
        rows = self._connection.execute(
            "SELECT stream_generation, sequence, optimizer_step, value, worker, rank, event_id, "
            "emitted_at FROM metric_points WHERE attempt_id = ? AND name = ? AND unit IS ? "
            "AND worker IS ? AND rank IS ? ORDER BY stream_generation, sequence",
            (attempt_id, name, unit, worker, rank),
        ).fetchall()
        return tuple(
            StoredMetricPoint(
                stream_generation=int(row["stream_generation"]),
                sequence=int(row["sequence"]),
                optimizer_step=row["optimizer_step"],
                value=float(row["value"]),
                worker=row["worker"],
                rank=row["rank"],
                event_id=row["event_id"],
                emitted_at=row["emitted_at"],
            )
            for row in rows
        )

    def _record(
        self,
        *,
        attempt_id: str,
        run_id: str,
        experiment_id: str,
        position: TelemetryPosition,
        points: Sequence[MetricPoint],
    ) -> int:
        """Insert one event's points; the number inserted (0 for a replay).

        Raises:
            IdempotencyConflictError: If the stream position already holds
                points from a different event, or different points.
        """
        _require_transaction(self._connection, "metric points")
        existing = self._connection.execute(
            f"SELECT {', '.join(_COMPARED)} FROM metric_points "
            "WHERE attempt_id = ? AND stream_generation = ? AND sequence = ? ORDER BY name",
            (attempt_id, position.stream_generation, position.sequence),
        ).fetchall()
        emitted_at = None if position.emitted_at is None else position.emitted_at.isoformat()
        offered = sorted(
            (
                point.name,
                point.value,
                point.unit,
                point.optimizer_step,
                point.worker,
                point.rank,
                point.observation,
                position.event_id,
                position.protocol_version,
                emitted_at,
            )
            for point in points
        )
        if existing:
            recorded = [tuple(row[column] for column in _COMPARED) for row in existing]
            if recorded != offered:
                differing = tuple(
                    column
                    for index, column in enumerate(_COMPARED)
                    if len(recorded) != len(offered)
                    or any(r[index] != o[index] for r, o in zip(recorded, offered, strict=True))
                )
                raise IdempotencyConflictError(
                    f"{attempt_id}@{position.stream_generation}:{position.sequence}",
                    differing or ("points",),
                    kind="metric observation",
                )
            return 0
        recorded_at = utc_now().isoformat()
        self._connection.executemany(
            "INSERT INTO metric_points (attempt_id, run_id, experiment_id, stream_generation, "
            "sequence, name, value, unit, optimizer_step, worker, rank, observation, event_id, "
            "protocol_version, emitted_at, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    attempt_id,
                    run_id,
                    experiment_id,
                    position.stream_generation,
                    position.sequence,
                    point.name,
                    point.value,
                    point.unit,
                    point.optimizer_step,
                    point.worker,
                    point.rank,
                    point.observation,
                    position.event_id,
                    position.protocol_version,
                    emitted_at,
                    recorded_at,
                )
                for point in points
            ],
        )
        return len(points)
