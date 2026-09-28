"""Append-only commit reports and replay receipts; no checkpoint file I/O."""

from __future__ import annotations

import sqlite3
from typing import cast

from xaytune.core.checkpoint import RecordedCheckpoint
from xaytune.core.fingerprint import fingerprint
from xaytune.storage.journal import _require_transaction


class CheckpointRecordStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, checkpoint_id: str) -> RecordedCheckpoint | None:
        row = self._connection.execute(
            "SELECT payload_json FROM checkpoints WHERE id = ?", (checkpoint_id,)
        ).fetchone()
        return None if row is None else RecordedCheckpoint.model_validate_json(row["payload_json"])

    def for_attempt(self, attempt_id: str) -> tuple[RecordedCheckpoint, ...]:
        rows = self._connection.execute(
            "SELECT payload_json FROM checkpoints WHERE attempt_id = ? ORDER BY optimizer_step, id",
            (attempt_id,),
        ).fetchall()
        return tuple(RecordedCheckpoint.model_validate_json(row["payload_json"]) for row in rows)

    def _receipt(self, attempt_id: str, generation: int, sequence: int) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            self._connection.execute(
                "SELECT checkpoint_id, evidence_digest FROM checkpoint_receipts "
                "WHERE attempt_id = ? AND stream_generation = ? AND sequence = ?",
                (attempt_id, generation, sequence),
            ).fetchone(),
        )

    def _insert(self, record: RecordedCheckpoint) -> None:
        _require_transaction(self._connection, "checkpoints")
        context = record.context
        self._connection.execute(
            "INSERT INTO checkpoints (id, attempt_id, run_id, experiment_id, optimizer_step, "
            "payload_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(record.payload.checkpoint_ref.id),
                context.target.id,
                context.run_id,
                str(context.experiment_id),
                record.payload.optimizer_step,
                record.model_dump_json(),
            ),
        )

    def _insert_receipt(self, record: RecordedCheckpoint) -> None:
        _require_transaction(self._connection, "checkpoint receipts")
        self._connection.execute(
            "INSERT INTO checkpoint_receipts (attempt_id, stream_generation, sequence, "
            "checkpoint_id, evidence_digest) VALUES (?, ?, ?, ?, ?)",
            (
                record.context.target.id,
                record.evidence["stream_generation"],
                record.evidence["sequence"],
                str(record.payload.checkpoint_ref.id),
                fingerprint(record.evidence),
            ),
        )
