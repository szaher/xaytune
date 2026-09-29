"""Read-only receipt queries and a transaction-internal append operation."""

from __future__ import annotations

import sqlite3

from xaytune.core.domain.recovery_execution import RecoveryExecutionReceipt
from xaytune.storage.journal import _require_transaction


class RecoveryExecutionReceiptStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, receipt_id: str) -> RecoveryExecutionReceipt | None:
        row = self._connection.execute(
            "SELECT payload_json FROM recovery_execution_receipts WHERE id = ?", (receipt_id,)
        ).fetchone()
        return (
            None
            if row is None
            else RecoveryExecutionReceipt.model_validate_json(row["payload_json"])
        )

    def for_episode(self, episode_id: str) -> tuple[RecoveryExecutionReceipt, ...]:
        rows = self._connection.execute(
            "SELECT payload_json FROM recovery_execution_receipts WHERE episode_id = ? "
            "ORDER BY created_at, id",
            (episode_id,),
        ).fetchall()
        return tuple(
            RecoveryExecutionReceipt.model_validate_json(row["payload_json"]) for row in rows
        )

    def executed_for_episode(self, episode_id: str) -> RecoveryExecutionReceipt | None:
        row = self._connection.execute(
            "SELECT payload_json FROM recovery_execution_receipts "
            "WHERE episode_id = ? AND outcome = 'EXECUTED'",
            (episode_id,),
        ).fetchone()
        return (
            None
            if row is None
            else RecoveryExecutionReceipt.model_validate_json(row["payload_json"])
        )

    def _insert(self, receipt: RecoveryExecutionReceipt) -> None:
        """Called only by a higher-level writer that also commits the effects."""
        _require_transaction(self._connection, "recovery execution receipts")
        self._connection.execute(
            "INSERT INTO recovery_execution_receipts (id, episode_id, plan_id, plan_sequence, "
            "action_id, outcome, successor_attempt_id, runtime_operation_id, "
            "checkpoint_id, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(receipt.id),
                str(receipt.episode_id),
                str(receipt.plan_id),
                receipt.plan_sequence,
                str(receipt.action_id),
                receipt.outcome.value,
                None if receipt.successor_attempt_id is None else str(receipt.successor_attempt_id),
                None if receipt.runtime_operation_id is None else str(receipt.runtime_operation_id),
                None if receipt.checkpoint_ref is None else str(receipt.checkpoint_ref.id),
                receipt.model_dump_json(),
                receipt.created_at.isoformat(),
            ),
        )
