"""Row-level persistence for actions.

Private for the same reason as every other writer here: an `Action` is durable
*intent*, and ADR-005 §5 requires it to commit with the effect it causes.
Writing one alone produces an intent nothing will act on and nothing will
retry, because nothing about it is unresolved.

Composed into atomic units by
:class:`~xaytune.storage.control_plane.ControlPlaneRepository`.
"""

from __future__ import annotations

import json
import sqlite3

from xaytune.core.domain.action import Action
from xaytune.core.errors import ConcurrentModificationError
from xaytune.storage.journal import _require_transaction

__all__ = ["ActionStore"]


class ActionStore:
    """Reads and revision-guarded writes for the `Action` aggregate."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, action_id: str) -> Action | None:
        """Return the action, or ``None`` if it does not exist."""
        row = self._connection.execute(
            "SELECT payload_json FROM actions WHERE id = ?", (action_id,)
        ).fetchone()
        return None if row is None else Action.model_validate_json(row["payload_json"])

    def for_target(self, kind: str, target_id: str) -> tuple[Action, ...]:
        """Return every action against one target, oldest first."""
        rows = self._connection.execute(
            "SELECT payload_json FROM actions WHERE target_kind = ? AND target_id = ? "
            "ORDER BY created_at, id",
            (kind, target_id),
        ).fetchall()
        return tuple(Action.model_validate_json(row["payload_json"]) for row in rows)

    def unresolved(self) -> tuple[Action, ...]:
        """Return actions that have not reached a terminal state, oldest first.

        What a restarting controller asks: which intents did we record and not
        finish carrying out?
        """
        rows = self._connection.execute(
            "SELECT payload_json FROM actions "
            "WHERE status NOT IN ('succeeded', 'failed', 'rejected') "
            "ORDER BY updated_at, id"
        ).fetchall()
        return tuple(Action.model_validate_json(row["payload_json"]) for row in rows)

    def _insert(self, action: Action) -> None:
        _require_transaction(self._connection, "actions")
        self._connection.execute(
            "INSERT INTO actions (id, experiment_id, type, status, outcome, "
            "target_kind, target_id, proposed_by_json, reason, payload_json, "
            "policy_decision_id, revision, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(action.id),
                str(action.experiment_id),
                action.type,
                action.status.value,
                action.outcome.value if action.outcome else None,
                action.target.kind,
                action.target.id,
                json.dumps(action.proposed_by.model_dump(mode="json"), sort_keys=True),
                action.reason,
                json.dumps(action.model_dump(mode="json"), sort_keys=True),
                action.policy_decision_id,
                action.revision,
                action.created_at.isoformat(),
                action.updated_at.isoformat(),
            ),
        )

    def _update(self, action: Action) -> None:
        """Write a transitioned action back, guarded on its revision.

        Raises:
            ConcurrentModificationError: If another writer moved it first.
        """
        _require_transaction(self._connection, "actions")
        expected = action.revision - 1
        cursor = self._connection.execute(
            "UPDATE actions SET status = ?, outcome = ?, payload_json = ?, "
            "policy_decision_id = ?, revision = ?, updated_at = ? "
            "WHERE id = ? AND revision = ?",
            (
                action.status.value,
                action.outcome.value if action.outcome else None,
                json.dumps(action.model_dump(mode="json"), sort_keys=True),
                action.policy_decision_id,
                action.revision,
                action.updated_at.isoformat(),
                str(action.id),
                expected,
            ),
        )
        if cursor.rowcount == 0:
            raise ConcurrentModificationError("Action", str(action.id), expected)
