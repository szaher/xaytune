"""The local daemon's request mailbox, and where its controller came to rest (ADR-004 §2-§3).

Migrations 016 and 018. Rows are read here and written only through
:class:`~xaytune.storage.ControlPlaneRepository`, inside its transactions: a
submission's ``ACCEPTED`` commits with the experiment it admits, and a rest
is fenced like every other controller write.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from xaytune.core.domain.controller_request import ControllerRequest, ControllerRequestState
from xaytune.core.errors import ConcurrentModificationError
from xaytune.core.ids import ExperimentId
from xaytune.core.immutable import thaw
from xaytune.storage.journal import _require_transaction

__all__ = ["ControllerRest", "ControllerRequestStore"]


@dataclass(frozen=True)
class ControllerRest:
    """The daemon's controller had nothing left it could do for an experiment (PR-029).

    Attributes:
        sequence: The experiment's latest event when it came to rest. While
            that is still its latest event, the experiment is at rest; any
            later event means the controller has moved it since.
        escalation: Why it stopped short, if it did: ``{"type", "message"}``
            of the error its ``wait()`` raised.
    """

    experiment_id: ExperimentId
    controller_id: str
    sequence: int
    escalation: dict[str, Any] | None
    recorded_at: datetime


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

    def rest(self, experiment_id: str) -> ControllerRest | None:
        """Where the daemon's controller last came to rest on the experiment, if it has."""
        row = self._connection.execute(
            "SELECT experiment_id, controller_id, sequence, escalation_json, recorded_at "
            "FROM controller_rests WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return None
        return ControllerRest(
            experiment_id=ExperimentId(row["experiment_id"]),
            controller_id=row["controller_id"],
            sequence=row["sequence"],
            escalation=(
                None if row["escalation_json"] is None else json.loads(row["escalation_json"])
            ),
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
        )

    def _put_rest(self, rest: ControllerRest) -> None:
        _require_transaction(self._connection, "controller rests")
        self._connection.execute(
            "INSERT INTO controller_rests "
            "(experiment_id, controller_id, sequence, escalation_json, recorded_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT (experiment_id) DO UPDATE SET "
            "controller_id = excluded.controller_id, sequence = excluded.sequence, "
            "escalation_json = excluded.escalation_json, recorded_at = excluded.recorded_at",
            (
                str(rest.experiment_id),
                rest.controller_id,
                rest.sequence,
                None if rest.escalation is None else json.dumps(rest.escalation, sort_keys=True),
                rest.recorded_at.isoformat(),
            ),
        )

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
