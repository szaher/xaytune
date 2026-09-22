"""Durable identity for local operations and workloads (ADR-013 §2).

An in-memory dictionary would pass every test that submits twice in one
process, and fail the only property this exists for: after the controller
restarts, ``lookup_operation`` must still be able to say whether a request was
received and what became of it. So the registry is a SQLite database, opened
with the same pragmas as the control plane, and every answer survives the
process that wrote it.

Two tables, and one invariant spanning them: an operation either started a
workload or was refused, never both and never neither. That is the same rule
:class:`~xaytune.runtimes.OperationOutcome` enforces in memory, written here as
a ``CHECK`` so a bug cannot persist a contradiction and hand it to a controller
after a restart, when nothing is left to contradict it with.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from xaytune.core.clock import utc_now
from xaytune.core.domain.operation import OperationTargetKind, OperationType
from xaytune.core.errors import IdempotencyConflictError
from xaytune.core.ids import OperationId
from xaytune.core.sqlite import connect, write_transaction

__all__ = ["LocalOperationRecord", "LocalWorkloadRecord", "LocalWorkloadRegistry"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workloads (
    external_id         TEXT PRIMARY KEY,
    target_kind         TEXT NOT NULL,
    target_id           TEXT NOT NULL,
    directory           TEXT NOT NULL,
    spawned_pid         INTEGER,
    launcher_pid        INTEGER,
    cancel_requested_at TEXT,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS operations (
    operation_id    TEXT PRIMARY KEY,
    operation_type  TEXT NOT NULL CHECK (operation_type IN ('submit', 'cancel')),
    request_digest  TEXT NOT NULL,
    external_id     TEXT REFERENCES workloads (external_id),
    rejected_detail TEXT,
    recorded_at     TEXT NOT NULL,

    -- An effect with no workload, or a refusal that still started one, are the
    -- two records a controller cannot recover from. Neither can be written.
    CHECK ((external_id IS NULL) = (rejected_detail IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS operations_by_workload ON operations (external_id);
"""


@dataclass(frozen=True)
class LocalWorkloadRecord:
    """What the registry knows about a workload, independent of its process."""

    external_id: str
    target_kind: OperationTargetKind
    target_id: str
    directory: Path
    spawned_pid: int | None
    """The pid the runtime saw ``Popen`` return, recorded best-effort.

    A hint for reporting only, never for ownership. Ownership is
    ``launcher_pid``, which only a launcher may set, and which is what keeps
    two of them from both producing a worker.
    """

    launcher_pid: int | None
    cancel_requested_at: str | None


@dataclass(frozen=True)
class LocalOperationRecord:
    """What the registry knows about one request it received."""

    operation_id: OperationId
    operation_type: OperationType
    request_digest: str
    external_id: str | None
    rejected_detail: str | None


class LocalWorkloadRegistry:
    """The local runtime's durable memory of what it was asked to do."""

    def __init__(self, path: Path) -> None:
        self._connection = connect(path)
        self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        self._connection.close()

    # -- reads ------------------------------------------------------------

    def operation(self, operation_id: OperationId) -> LocalOperationRecord | None:
        row = self._connection.execute(
            "SELECT * FROM operations WHERE operation_id = ?", (str(operation_id),)
        ).fetchone()
        return None if row is None else _operation(row)

    def workload(self, external_id: str) -> LocalWorkloadRecord | None:
        row = self._connection.execute(
            "SELECT * FROM workloads WHERE external_id = ?", (external_id,)
        ).fetchone()
        return None if row is None else _workload(row)

    # -- writes -----------------------------------------------------------

    def claim_submission(
        self,
        *,
        operation_id: OperationId,
        request_digest: str,
        external_id: str,
        target_kind: OperationTargetKind,
        target_id: str,
        directory: Path,
    ) -> LocalOperationRecord:
        """Record the intent to run a workload, before anything is spawned.

        Returns the existing record when this operation has been claimed
        before, which is what makes ``submit_or_get`` a get rather than a
        second create. The durable write happens first and the process is
        spawned afterwards, because it is always recoverable to hold intent
        with no effect and never recoverable to have an effect with no intent
        (ADR-005 §9).

        Raises:
            IdempotencyConflictError: If this id was used for a materially
                different request.
        """
        with write_transaction(self._connection):
            existing = self.operation(operation_id)
            if existing is not None:
                _require_same_request(existing, "submit", request_digest)
                return existing

            now = utc_now().isoformat()
            self._connection.execute(
                "INSERT INTO workloads (external_id, target_kind, target_id, "
                "directory, created_at) VALUES (?, ?, ?, ?, ?)",
                (external_id, target_kind, target_id, str(directory), now),
            )
            self._connection.execute(
                "INSERT INTO operations (operation_id, operation_type, "
                "request_digest, external_id, recorded_at) VALUES (?, 'submit', ?, ?, ?)",
                (str(operation_id), request_digest, external_id, now),
            )

        claimed = self.operation(operation_id)
        assert claimed is not None
        return claimed

    def record_rejection(
        self, *, operation_id: OperationId, request_digest: str, detail: str
    ) -> LocalOperationRecord:
        """Record that a request was refused and nothing was started.

        Durable on purpose. A refusal the runtime forgot would be indexed as
        "never received" after a restart, and never-received is the one answer
        that makes re-issuing the identical request look safe.
        """
        with write_transaction(self._connection):
            existing = self.operation(operation_id)
            if existing is not None:
                _require_same_request(existing, "submit", request_digest)
                return existing

            self._connection.execute(
                "INSERT INTO operations (operation_id, operation_type, "
                "request_digest, rejected_detail, recorded_at) VALUES (?, 'submit', ?, ?, ?)",
                (str(operation_id), request_digest, detail, utc_now().isoformat()),
            )

        recorded = self.operation(operation_id)
        assert recorded is not None
        return recorded

    def claim_cancellation(
        self, *, operation_id: OperationId, request_digest: str, external_id: str
    ) -> bool:
        """Claim a cancellation, returning whether this call is the first.

        The return value is for the record, **not** a licence to skip the
        effect. A caller that treated ``False`` as "already done" would lose
        every cancellation interrupted between this commit and the request
        reaching the workload: the claim would be durable, the cancellation
        would never have happened, and the retry would decline to try again.
        Recording the claim and making the request durable are two steps, and
        the second one runs every time.
        """
        with write_transaction(self._connection):
            existing = self.operation(operation_id)
            if existing is not None:
                _require_same_request(existing, "cancel", request_digest)
                return False

            self._connection.execute(
                "INSERT INTO operations (operation_id, operation_type, "
                "request_digest, external_id, recorded_at) VALUES (?, 'cancel', ?, ?, ?)",
                (str(operation_id), request_digest, external_id, utc_now().isoformat()),
            )
        return True

    def claim_launcher_ownership(self, external_id: str, pid: int) -> bool:
        """Take ownership of a workload, returning whether this caller got it.

        A compare-and-set, claimed by the launcher itself rather than recorded
        by whoever spawned it. Between ``Popen`` returning and any write the
        spawner could make there is a window, and a controller that died inside
        it would restart, see no owner, and spawn a second launcher for the
        same operation -- two workers for one ``operation_id``, which is the
        duplicate external effect ADR-013 exists to prevent.

        Moving the write to the launcher closes it. Two launchers may briefly
        exist; only one can win the CAS, and the loser exits before spawning
        anything.

        A dead owner is **not** displaced. Its absence is ambiguous -- it may
        have spawned a worker before it died -- and stealing the workload on
        that basis is how the duplicate comes back.
        """
        with write_transaction(self._connection):
            cursor = self._connection.execute(
                "UPDATE workloads SET launcher_pid = ? "
                "WHERE external_id = ? AND launcher_pid IS NULL",
                (pid, external_id),
            )
            return cursor.rowcount == 1

    def record_spawned_pid(self, external_id: str, pid: int) -> None:
        """Note the process the runtime just started, for reporting only.

        Deliberately a different column from ``launcher_pid``. Writing the
        ownership column here would make the launcher's claim fail against a
        pid the runtime had already filled in, and the launcher would exit
        without ever starting the worker.

        Losing this write to a crash costs a little precision and no
        correctness: the workload reports ``unknown`` until a launcher claims
        it, which is what an unobserved spawn actually is.
        """
        with write_transaction(self._connection):
            self._connection.execute(
                "UPDATE workloads SET spawned_pid = ? WHERE external_id = ?",
                (pid, external_id),
            )

    def record_cancellation_request(self, external_id: str) -> None:
        """Note when a cancellation was first asked for."""
        with write_transaction(self._connection):
            self._connection.execute(
                "UPDATE workloads SET cancel_requested_at = COALESCE(cancel_requested_at, ?) "
                "WHERE external_id = ?",
                (utc_now().isoformat(), external_id),
            )


def _require_same_request(
    existing: LocalOperationRecord, operation_type: OperationType, request_digest: str
) -> None:
    differing = tuple(
        field
        for field, is_same in (
            ("operation_type", existing.operation_type == operation_type),
            ("request_digest", existing.request_digest == request_digest),
        )
        if not is_same
    )
    if differing:
        raise IdempotencyConflictError(str(existing.operation_id), differing)


def _operation(row: sqlite3.Row) -> LocalOperationRecord:
    return LocalOperationRecord(
        operation_id=OperationId.validate(row["operation_id"]),
        operation_type=row["operation_type"],
        request_digest=row["request_digest"],
        external_id=row["external_id"],
        rejected_detail=row["rejected_detail"],
    )


def _workload(row: sqlite3.Row) -> LocalWorkloadRecord:
    return LocalWorkloadRecord(
        external_id=row["external_id"],
        target_kind=row["target_kind"],
        target_id=row["target_id"],
        directory=Path(row["directory"]),
        spawned_pid=row["spawned_pid"],
        launcher_pid=row["launcher_pid"],
        cancel_requested_at=row["cancel_requested_at"],
    )
