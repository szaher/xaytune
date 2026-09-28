"""Append-only recovery decisions, written with their event by the repository."""

from __future__ import annotations

import sqlite3

from xaytune.core.domain.recovery import RecoveryPlan
from xaytune.storage.journal import _require_transaction


class RecoveryPlanStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, plan_id: str) -> RecoveryPlan | None:
        row = self._connection.execute(
            "SELECT payload_json FROM recovery_plans WHERE id = ?", (plan_id,)
        ).fetchone()
        return None if row is None else RecoveryPlan.model_validate_json(row["payload_json"])

    def for_incident(self, incident_id: str) -> RecoveryPlan | None:
        row = self._connection.execute(
            "SELECT payload_json FROM recovery_plans WHERE incident_id = ?", (incident_id,)
        ).fetchone()
        return None if row is None else RecoveryPlan.model_validate_json(row["payload_json"])

    def for_experiment(self, experiment_id: str) -> tuple[RecoveryPlan, ...]:
        rows = self._connection.execute(
            "SELECT payload_json FROM recovery_plans WHERE experiment_id = ? ORDER BY rowid",
            (experiment_id,),
        ).fetchall()
        return tuple(RecoveryPlan.model_validate_json(row["payload_json"]) for row in rows)

    def _insert(self, plan: RecoveryPlan) -> None:
        _require_transaction(self._connection, "recovery plans")
        self._connection.execute(
            "INSERT INTO recovery_plans (id, incident_id, experiment_id, run_id, "
            "target_kind, target_id, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(plan.id),
                str(plan.incident_id),
                str(plan.context.experiment_id),
                plan.context.run_id,
                plan.context.target.kind,
                plan.context.target.id,
                plan.model_dump_json(),
            ),
        )
