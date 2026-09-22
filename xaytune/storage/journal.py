"""Row-level persistence for events, outbox records and runtime operations.

Private, like the aggregate writers in :mod:`xaytune.storage.repository`, and
for the same reason: none of these rows is meaningful on its own. An event
without its transition is a claim about something that did not happen; an
operation without its attempt is an effect with no owner. They are composed into
atomic units by :class:`~xaytune.storage.repository.ControlPlaneRepository`,
which is the only public write surface.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from xaytune.core.domain.event import DomainEvent, OutboxRecord
from xaytune.core.domain.operation import RuntimeOperation
from xaytune.core.errors import ConcurrentModificationError, IdempotencyConflictError

__all__ = ["IdempotencyConflictError"]
"""Re-exported from :mod:`xaytune.core.errors`, where it moved once a runtime
backend needed to raise the same condition from its own registry. The name
stays here because this is where the control plane's callers import it."""


class EventJournal:
    """Appends domain events and their outbox records."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def _append(self, event: DomainEvent) -> int:
        """Insert *event* and return the sequence the database assigned.

        Private, and it refuses to run outside a transaction. An event written
        on its own is a claim that a transition happened when none did, which
        ADR-005 §3 forbids -- and with ``isolation_level=None`` a bare call
        would autocommit exactly that.

        The sequence is assigned here rather than by the caller: it is a total
        order over the database, and a caller-chosen value could collide or
        leave a hole that a consumer's cursor would read as a lost event.
        """
        _require_transaction(self._connection, "events")
        cursor = self._connection.execute(
            "INSERT INTO events (id, experiment_id, aggregate_type, aggregate_id, "
            "aggregate_revision, event_type, schema_version, occurred_at, "
            "actor_json, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(event.id),
                event.experiment_id,
                event.aggregate_type,
                event.aggregate_id,
                event.aggregate_revision,
                event.event_type,
                event.schema_version,
                event.occurred_at.isoformat(),
                json.dumps(event.actor.model_dump(mode="json"), sort_keys=True),
                json.dumps(dict(event.payload), sort_keys=True),
            ),
        )
        return int(cursor.lastrowid or 0)

    def _enqueue(self, record: OutboxRecord) -> None:
        """Insert an outbox record for an already-inserted event.

        Private for the same reason as :meth:`_append`: an outbox row without
        its event is a delivery of something that was never recorded.
        """
        _require_transaction(self._connection, "outbox records")
        self._connection.execute(
            "INSERT INTO outbox (id, event_id, destination, state, attempts, "
            "next_attempt_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                str(record.event_id),
                record.destination,
                record.state,
                record.attempts,
                record.next_attempt_at.isoformat() if record.next_attempt_at else None,
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
            ),
        )

    def events_for_aggregate(self, aggregate_id: str) -> tuple[DomainEvent, ...]:
        """Return an aggregate's events in commit order."""
        rows = self._connection.execute(
            "SELECT * FROM events WHERE aggregate_id = ? ORDER BY sequence",
            (aggregate_id,),
        ).fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def events_for_experiment(self, experiment_id: str) -> tuple[DomainEvent, ...]:
        """Return an experiment's events in commit order."""
        rows = self._connection.execute(
            "SELECT * FROM events WHERE experiment_id = ? ORDER BY sequence",
            (experiment_id,),
        ).fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def pending_outbox(self) -> tuple[OutboxRecord, ...]:
        """Return undelivered outbox records, oldest first."""
        rows = self._connection.execute(
            "SELECT * FROM outbox WHERE state IN ('pending', 'sending') ORDER BY created_at, id"
        ).fetchall()
        return tuple(_outbox_from_row(row) for row in rows)


class OperationJournal:
    """Durable records of external side effects (ADR-013)."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, operation_id: str) -> RuntimeOperation | None:
        """Return the operation, or ``None`` if it was never recorded."""
        row = self._connection.execute(
            "SELECT * FROM runtime_operations WHERE id = ?", (operation_id,)
        ).fetchone()
        return None if row is None else _operation_from_row(row)

    def for_target(self, kind: str, target_id: str) -> tuple[RuntimeOperation, ...]:
        """Return every operation against one attempt, oldest first."""
        rows = self._connection.execute(
            "SELECT * FROM runtime_operations WHERE target_kind = ? AND target_id = ? "
            "ORDER BY created_at, id",
            (kind, target_id),
        ).fetchall()
        return tuple(_operation_from_row(row) for row in rows)

    def unresolved(self) -> tuple[RuntimeOperation, ...]:
        """Return operations whose outcome is not yet known, oldest first.

        Reconciliation's query after a restart. Each of these means an effect
        may or may not exist, which is exactly when ``lookup_operation()`` is
        consulted rather than the request re-issued.
        """
        rows = self._connection.execute(
            "SELECT * FROM runtime_operations WHERE state IN ('intended', 'sent') "
            "ORDER BY updated_at, id"
        ).fetchall()
        return tuple(_operation_from_row(row) for row in rows)

    def _insert(self, operation: RuntimeOperation) -> RuntimeOperation:
        """Insert *operation*, or return the identical existing one.

        Idempotent by operation id (ADR-013 §2). Reusing an id with the same
        target, type and digest returns what is already recorded, so a retried
        create is harmless.

        Raises:
            IdempotencyConflictError: If the id exists with a different target,
                type or request digest.
        """
        _require_transaction(self._connection, "runtime operations")
        existing = self.get(str(operation.id))
        if existing is not None:
            self._assert_same_request(existing, operation)
            return existing

        self._connection.execute(
            "INSERT INTO runtime_operations (id, target_kind, target_id, type, "
            "request_digest, state, runtime_ref_json, caused_by_action_id, "
            "revision, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(operation.id),
                operation.target.kind,
                operation.target.id,
                operation.type,
                operation.request_digest,
                operation.state,
                _ref_json(operation),
                str(operation.caused_by_action_id) if operation.caused_by_action_id else None,
                operation.revision,
                operation.created_at.isoformat(),
                operation.updated_at.isoformat(),
            ),
        )
        return operation

    @staticmethod
    def _assert_same_request(existing: RuntimeOperation, requested: RuntimeOperation) -> None:
        """Refuse a reused operation id whose request differs (ADR-013 §2).

        Raises:
            IdempotencyConflictError: Naming every field that differs, because
                the caller needs to know *what* it changed -- guessing which
                request was meant would start a second workload.
        """
        differing = tuple(
            field
            # caused_by_action_id is compared even though it is not part of the
            # external request. Once an effect has an Action behind it, that
            # cause is part of its control-plane identity: without this check a
            # second Action could reuse the operation id, see an "identical"
            # request, and commit while the operation still pointed at the
            # first -- leaving an Action with no effect and an effect whose
            # recorded cause is not the one that asked for it (ADR-005 §5).
            for field in ("target", "type", "request_digest", "caused_by_action_id")
            if getattr(existing, field) != getattr(requested, field)
        )
        if differing:
            raise IdempotencyConflictError(str(existing.id), differing)

    def _update(self, operation: RuntimeOperation) -> None:
        """Write a transitioned operation back, guarded on its revision.

        Raises:
            ConcurrentModificationError: If another writer moved it first.
        """
        _require_transaction(self._connection, "runtime operations")
        expected = operation.revision - 1
        cursor = self._connection.execute(
            # caused_by_action_id is deliberately not updated: an effect's cause
            # is fixed when it is created, and rewriting it would let a later
            # transition reassign responsibility for a side effect that already
            # happened.
            "UPDATE runtime_operations SET state = ?, runtime_ref_json = ?, "
            "revision = ?, updated_at = ? WHERE id = ? AND revision = ?",
            (
                operation.state,
                _ref_json(operation),
                operation.revision,
                operation.updated_at.isoformat(),
                str(operation.id),
                expected,
            ),
        )
        if cursor.rowcount == 0:
            raise ConcurrentModificationError("RuntimeOperation", str(operation.id), expected)


def _require_transaction(connection: sqlite3.Connection, what: str) -> None:
    """Refuse a journal write outside a transaction.

    ``connect()`` sets ``isolation_level=None``, so a write issued outside
    ``write_transaction()`` autocommits on its own. For an aggregate that would
    commit state without its event; for these tables it is the mirror image --
    an event or an operation with nothing it belongs to. Both are the split
    ADR-005 §3 exists to prevent, so both fail loudly here.
    """
    if not connection.in_transaction:
        raise sqlite3.ProgrammingError(
            f"{what} must be written inside write_transaction(), composed with "
            f"the transition they describe (ADR-005 section 3). A bare write "
            f"would autocommit a record with nothing it belongs to."
        )


def _ref_json(operation: RuntimeOperation) -> str | None:
    if operation.runtime_ref is None:
        return None
    return json.dumps(operation.runtime_ref.model_dump(mode="json"), sort_keys=True)


def _event_from_row(row: sqlite3.Row) -> DomainEvent:
    return DomainEvent.model_validate(
        {
            "id": row["id"],
            "sequence": row["sequence"],
            "experiment_id": row["experiment_id"],
            "aggregate_type": row["aggregate_type"],
            "aggregate_id": row["aggregate_id"],
            "aggregate_revision": row["aggregate_revision"],
            "event_type": row["event_type"],
            "schema_version": row["schema_version"],
            "occurred_at": row["occurred_at"],
            "actor": json.loads(row["actor_json"]),
            "payload": json.loads(row["payload_json"]),
        }
    )


def _outbox_from_row(row: sqlite3.Row) -> OutboxRecord:
    return OutboxRecord.model_validate(
        {
            "id": row["id"],
            "event_id": row["event_id"],
            "destination": row["destination"],
            "state": row["state"],
            "attempts": row["attempts"],
            "next_attempt_at": row["next_attempt_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
    )


def _operation_from_row(row: sqlite3.Row) -> RuntimeOperation:
    payload: dict[str, Any] = {
        "id": row["id"],
        "target": {"kind": row["target_kind"], "id": row["target_id"]},
        "type": row["type"],
        "request_digest": row["request_digest"],
        "state": row["state"],
        "runtime_ref": json.loads(row["runtime_ref_json"]) if row["runtime_ref_json"] else None,
        "caused_by_action_id": row["caused_by_action_id"],
        "revision": row["revision"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    return RuntimeOperation.model_validate(payload)
