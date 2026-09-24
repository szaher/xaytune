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
from xaytune.storage.journal import IdempotencyConflictError, _require_transaction

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

    def caused_operation_ids(self, action_id: str) -> tuple[str, ...]:
        """Return the ids of the operations an action caused, oldest first."""
        rows = self._connection.execute(
            "SELECT id FROM runtime_operations WHERE caused_by_action_id = ? "
            "ORDER BY created_at, id",
            (action_id,),
        ).fetchall()
        return tuple(str(row["id"]) for row in rows)

    def children(self, action_id: str) -> tuple[Action, ...]:
        """Return the actions carrying out part of *action_id*, oldest first."""
        rows = self._connection.execute(
            "SELECT payload_json FROM actions WHERE parent_action_id = ? ORDER BY created_at, id",
            (action_id,),
        ).fetchall()
        return tuple(Action.model_validate_json(row["payload_json"]) for row in rows)

    @staticmethod
    def _assert_same_request(existing: Action, requested: Action) -> None:
        """Refuse a reused action id whose request differs.

        The same contract the operation journal enforces, on the other half of
        the compound write: same id and same request returns the existing
        record; same id and a different request is refused. Without it a retry
        carrying a new reason, actor or target would silently return the
        original, and the record would describe a decision nobody made.

        ``status``, ``outcome`` and ``revision`` are excluded: those are what
        the action's lifecycle changes, and the stored copy is expected to have
        moved past the candidate the caller just built.

        Raises:
            IdempotencyConflictError: Naming every request field that differs.
        """
        differing = tuple(
            field
            for field in (
                "type",
                "target",
                "experiment_id",
                "proposed_by",
                "reason",
                "payload",
                "parent_action_id",
            )
            if getattr(existing, field) != getattr(requested, field)
        )
        if differing:
            raise IdempotencyConflictError(str(existing.id), differing, kind="action")

    def _insert(self, action: Action) -> None:
        _require_transaction(self._connection, "actions")
        self._connection.execute(
            "INSERT INTO actions (id, experiment_id, type, status, outcome, "
            "target_kind, target_id, proposed_by_json, reason, payload_json, "
            "policy_decision_id, revision, created_at, updated_at, parent_action_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                str(action.parent_action_id) if action.parent_action_id else None,
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
            # The indexed identity columns are rewritten alongside the payload,
            # not left behind. They cannot change through a legitimate
            # transition, but writing only payload_json would let the JSON and
            # the columns disagree if one ever did -- and a row whose index
            # says one thing and whose body says another is worse than either.
            "UPDATE actions SET status = ?, outcome = ?, type = ?, "
            "target_kind = ?, target_id = ?, payload_json = ?, "
            "policy_decision_id = ?, revision = ?, updated_at = ? "
            "WHERE id = ? AND revision = ?",
            (
                action.status.value,
                action.outcome.value if action.outcome else None,
                action.type,
                action.target.kind,
                action.target.id,
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
