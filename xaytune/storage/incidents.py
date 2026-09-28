"""Append-only incident records; writes are composed by the repository."""

from __future__ import annotations

import sqlite3

from xaytune.core.domain.incident import Incident
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.storage.journal import _require_transaction


class IncidentStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, incident_id: str) -> Incident | None:
        row = self._connection.execute(
            "SELECT payload_json FROM incidents WHERE id = ?", (incident_id,)
        ).fetchone()
        return None if row is None else Incident.model_validate_json(row["payload_json"])

    def for_observation(self, observation_key: str) -> Incident | None:
        row = self._connection.execute(
            "SELECT payload_json FROM incidents WHERE observation_key = ?", (observation_key,)
        ).fetchone()
        return None if row is None else Incident.model_validate_json(row["payload_json"])

    def for_attempt(self, target: RuntimeOperationTarget) -> tuple[Incident, ...]:
        rows = self._connection.execute(
            "SELECT payload_json FROM incidents WHERE target_kind = ? AND target_id = ? "
            "ORDER BY stream_generation, sequence",
            (target.kind, target.id),
        ).fetchall()
        return tuple(Incident.model_validate_json(row["payload_json"]) for row in rows)

    def for_experiment(self, experiment_id: str) -> tuple[Incident, ...]:
        rows = self._connection.execute(
            "SELECT payload_json FROM incidents WHERE experiment_id = ? ORDER BY created_at, id",
            (experiment_id,),
        ).fetchall()
        return tuple(Incident.model_validate_json(row["payload_json"]) for row in rows)

    def _insert(self, incident: Incident) -> None:
        _require_transaction(self._connection, "incidents")
        context = incident.context
        self._connection.execute(
            "INSERT INTO incidents (id, observation_key, experiment_id, run_id, target_kind, "
            "target_id, stream_generation, sequence, category, evidence_fingerprint, "
            "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(incident.id),
                incident.observation_key,
                str(context.experiment_id),
                context.run_id,
                context.target.kind,
                context.target.id,
                incident.stream_generation,
                incident.sequence,
                incident.category.value,
                incident.evidence_fingerprint,
                incident.model_dump_json(),
                incident.created_at.isoformat(),
            ),
        )
