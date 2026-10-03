"""The local daemon's request mailbox (ADR-004 §2-§3; migration 016).

Rows are read here and written only through
:class:`~xaytune.storage.ControlPlaneRepository`, inside its transactions: a
submission's ``ACCEPTED`` commits with the experiment it admits.
"""

from __future__ import annotations

import json
import sqlite3

from xaytune.core.domain.controller_request import ControllerRequest, ControllerRequestState
from xaytune.core.errors import ConcurrentModificationError
from xaytune.core.immutable import thaw
from xaytune.storage.journal import _require_transaction

__all__ = ["ControllerRequestStore"]

_COLUMNS = (
    "id, kind, state, revision, experiment_id, payload_json, payload_digest, error_json, "
    "created_at, updated_at"
)


def _load(row: sqlite3.Row) -> ControllerRequest:
    return ControllerRequest(
        id=row["id"],
        kind=row["kind"],
        state=ControllerRequestState(row["state"]),
        revision=row["revision"],
        experiment_id=row["experiment_id"],
        payload=json.loads(row["payload_json"]),
        payload_digest=row["payload_digest"],
        error=None if row["error_json"] is None else json.loads(row["error_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class ControllerRequestStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, request_id: str) -> ControllerRequest | None:
        row = self._connection.execute(
            f"SELECT {_COLUMNS} FROM controller_requests WHERE id = ?", (request_id,)
        ).fetchone()
        return None if row is None else _load(row)

    def unfinished(self) -> tuple[ControllerRequest, ...]:
        """Every ``PENDING`` or ``ACCEPTED`` request, oldest first: the daemon's work."""
        rows = self._connection.execute(
            f"SELECT {_COLUMNS} FROM controller_requests "
            "WHERE state IN ('pending', 'accepted') ORDER BY created_at, id"
        ).fetchall()
        return tuple(_load(row) for row in rows)

    def for_experiment(self, experiment_id: str) -> tuple[ControllerRequest, ...]:
        rows = self._connection.execute(
            f"SELECT {_COLUMNS} FROM controller_requests WHERE experiment_id = ? "
            "ORDER BY created_at, id",
            (experiment_id,),
        ).fetchall()
        return tuple(_load(row) for row in rows)

    def _insert(self, request: ControllerRequest) -> None:
        _require_transaction(self._connection, "controller requests")
        self._connection.execute(
            f"INSERT INTO controller_requests ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(request.id),
                request.kind,
                request.state.value,
                request.revision,
                str(request.experiment_id),
                json.dumps(thaw(request.payload), sort_keys=True),
                request.payload_digest,
                None if request.error is None else json.dumps(thaw(request.error), sort_keys=True),
                request.created_at.isoformat(),
                request.updated_at.isoformat(),
            ),
        )

    def _update(self, request: ControllerRequest) -> None:
        """Write a moved request back, guarded on the revision it moved from."""
        _require_transaction(self._connection, "controller requests")
        cursor = self._connection.execute(
            "UPDATE controller_requests SET state = ?, revision = ?, error_json = ?, "
            "updated_at = ? WHERE id = ? AND revision = ?",
            (
                request.state.value,
                request.revision,
                None if request.error is None else json.dumps(thaw(request.error), sort_keys=True),
                request.updated_at.isoformat(),
                str(request.id),
                request.revision - 1,
            ),
        )
        if cursor.rowcount != 1:
            raise ConcurrentModificationError(
                "ControllerRequest", str(request.id), request.revision - 1
            )
