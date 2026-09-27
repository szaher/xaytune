"""Row-level persistence for policy decisions (PR-023).

Private writes, composed into atomic units by
:class:`~xaytune.storage.control_plane.ControlPlaneRepository`: a decision is
written in the transaction that records the action it governs, and never
changed after (migration 009 enforces both).
"""

from __future__ import annotations

import json
import sqlite3

from xaytune.core.domain.policy import PolicyDecision
from xaytune.storage.journal import _require_transaction

__all__ = ["PolicyDecisionStore"]


class PolicyDecisionStore:
    """Reads, and append-only writes, of policy decisions."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def for_action(self, action_id: str) -> PolicyDecision | None:
        """The decision about *action_id*, or ``None`` if policy never judged it."""
        row = self._connection.execute(
            "SELECT payload_json FROM policy_decisions WHERE action_id = ?", (action_id,)
        ).fetchone()
        return None if row is None else PolicyDecision.model_validate_json(row["payload_json"])

    def for_experiment(self, experiment_id: str) -> tuple[PolicyDecision, ...]:
        """Every decision in an experiment, oldest first."""
        rows = self._connection.execute(
            "SELECT payload_json FROM policy_decisions WHERE experiment_id = ? "
            "ORDER BY created_at, id",
            (experiment_id,),
        ).fetchall()
        return tuple(PolicyDecision.model_validate_json(row["payload_json"]) for row in rows)

    def _insert(self, decision: PolicyDecision) -> None:
        _require_transaction(self._connection, "policy_decisions")
        self._connection.execute(
            "INSERT INTO policy_decisions (id, action_id, experiment_id, engine_name, "
            "engine_version, verdict, input_fingerprint, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(decision.id),
                str(decision.action_id),
                str(decision.experiment_id),
                decision.engine_name,
                decision.engine_version,
                decision.verdict.value,
                decision.input_fingerprint,
                json.dumps(decision.model_dump(mode="json"), sort_keys=True),
                decision.created_at.isoformat(),
            ),
        )
