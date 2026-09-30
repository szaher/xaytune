"""Append-only binding from a recovery plan revision to its governed Action."""

from __future__ import annotations

import sqlite3

from xaytune.core.domain.recovery_action import RecoveryActionBinding
from xaytune.storage.journal import _require_transaction


class RecoveryActionBindingStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def for_action(self, action_id: str) -> RecoveryActionBinding | None:
        row = self._connection.execute(
            "SELECT payload_json FROM recovery_action_bindings WHERE action_id = ?", (action_id,)
        ).fetchone()
        return (
            None if row is None else RecoveryActionBinding.model_validate_json(row["payload_json"])
        )

    def for_plan(self, plan_id: str) -> RecoveryActionBinding | None:
        row = self._connection.execute(
            "SELECT payload_json FROM recovery_action_bindings WHERE plan_id = ?", (plan_id,)
        ).fetchone()
        return (
            None if row is None else RecoveryActionBinding.model_validate_json(row["payload_json"])
        )

    def for_episode(self, episode_id: str) -> tuple[RecoveryActionBinding, ...]:
        rows = self._connection.execute(
            "SELECT payload_json FROM recovery_action_bindings WHERE episode_id = ? "
            "ORDER BY plan_sequence, action_id",
            (episode_id,),
        ).fetchall()
        return tuple(RecoveryActionBinding.model_validate_json(row["payload_json"]) for row in rows)

    def _insert(self, binding: RecoveryActionBinding) -> None:
        _require_transaction(self._connection, "recovery Action bindings")
        self._connection.execute(
            "INSERT INTO recovery_action_bindings (action_id, episode_id, plan_id, "
            "plan_sequence, proposal_fingerprint, input_fingerprint, "
            "source_execution_state_fingerprint, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(binding.action_id),
                str(binding.episode_id),
                str(binding.plan_id),
                binding.plan_sequence,
                binding.proposal_fingerprint,
                binding.input_fingerprint,
                binding.source_execution_state_fingerprint,
                binding.model_dump_json(),
                binding.created_at.isoformat(),
            ),
        )
