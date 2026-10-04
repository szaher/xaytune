"""Durable controller ownership: the lease and the write fences (ADR-004 §8; migration 017).

One lease per state database says which daemon may write controller state
now, and in which generation::

    controller_id      the daemon process instance holding it
    epoch              the fencing generation; +1 on every new owner
    heartbeat_at       when the owner last renewed it
    lease_expires_at   after this, it is nobody's

The kernel ``flock`` keeps a second daemon on the same machine out; the lease
is what the *record* says, and what every controller write proves. Two fences
check it inside the write transaction, after SQLite's write lock is taken, so
a write and a takeover serialize -- one commits wholly before the other:

``ControllerLeaseFence``  the daemon's: this controller, this epoch, unexpired,
                          or :class:`LeaseLostError` and nothing written.
``NoLiveLeaseFence``      an embedded host's: no daemon holds a live lease, or
                          :class:`ControllerLeaseHeldError`.

Expiry is the only authority for takeover. Nothing checks whether the old
owner's process, host or container still exists: an owner that outlived its
lease finds its next write refused, and any runtime effect it retries carries
an operation id recorded while it still owned the database (ADR-013), so it is
the same effect, not a second one.

The state database must be on a local filesystem; SQLite over NFS, SMB or any
shared or distributed filesystem is unsupported, and nothing here detects it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

from xaytune.core.clock import utc_now
from xaytune.core.sqlite import write_transaction
from xaytune.storage.errors import StorageError

__all__ = [
    "ControllerLease",
    "ControllerLeaseFence",
    "ControllerLeaseHeldError",
    "ControllerLeaseStore",
    "LeaseLostError",
    "NoLiveLeaseFence",
    "WriteFence",
]

Clock = Callable[[], datetime]


@dataclass(frozen=True)
class ControllerLease:
    """The lease row as read: who owns the database, in which generation, until when."""

    controller_id: str
    epoch: int
    heartbeat_at: datetime
    lease_expires_at: datetime

    def is_live(self, now: datetime) -> bool:
        return self.lease_expires_at > now


class LeaseLostError(StorageError):
    """This controller no longer owns the database: another epoch, or its own expired.

    Not a retryable failure. The process has lost the authority to control
    the database, and must stop without writing anything more.
    """

    def __init__(self, controller_id: str, epoch: int, detail: str) -> None:
        self.controller_id = controller_id
        self.epoch = epoch
        super().__init__(f"controller {controller_id} lost its lease (epoch {epoch}): {detail}")


class ControllerLeaseHeldError(StorageError):
    """A live daemon lease owns the database; an embedded host may read it, not control it."""

    def __init__(self, lease: ControllerLease) -> None:
        self.lease = lease
        super().__init__(
            f"daemon {lease.controller_id} owns this database (epoch {lease.epoch}, until "
            f"{lease.lease_expires_at.isoformat()}): an embedded host may read it but not "
            f"write controller state; hand work to the daemon instead"
        )


class WriteFence(Protocol):
    """What a write transaction proves before its first mutation."""

    def check(self, connection: sqlite3.Connection) -> None:
        """Raise if this writer may not write now. Called inside the transaction."""


@dataclass(frozen=True)
class ControllerLeaseFence:
    """A daemon controller's fence: only this owner, at this epoch, before expiry.

    *on_lost* is told of every refusal before it is raised, so the process
    learns it lost the database even where a caller further up swallows the
    error.
    """

    controller_id: str
    epoch: int
    on_lost: Callable[[LeaseLostError], None] | None = field(default=None, compare=False)
    clock: Clock = field(default=utc_now, compare=False)

    def check(self, connection: sqlite3.Connection) -> None:
        problem = _ownership_problem(
            _read(connection), self.controller_id, self.epoch, self.clock()
        )
        if problem is None:
            return
        error = LeaseLostError(self.controller_id, self.epoch, problem)
        if self.on_lost is not None:
            self.on_lost(error)
        raise error


@dataclass(frozen=True)
class NoLiveLeaseFence:
    """An embedded host's fence: no daemon holds a live lease on the database."""

    clock: Clock = field(default=utc_now, compare=False)

    def check(self, connection: sqlite3.Connection) -> None:
        lease = _read(connection)
        if lease is not None and lease.is_live(self.clock()):
            raise ControllerLeaseHeldError(lease)


class ControllerLeaseStore:
    """Acquire, renew and release the database's controller lease, each one transaction.

    Lease writes are not fenced: they are how ownership is established, so
    they check it themselves, under the same write lock.
    """

    def __init__(self, connection: sqlite3.Connection, *, clock: Clock = utc_now) -> None:
        self._connection = connection
        self._clock = clock

    def current(self) -> ControllerLease | None:
        return _read(self._connection)

    def acquire(self, controller_id: str, ttl: timedelta) -> ControllerLease | None:
        """Take the lease as a new owner, unless another holds it live.

        A new acquisition is always a new epoch -- after a clean release as
        after a crash -- so no earlier generation, even one under the same
        controller id, is ever current again.

        Returns:
            The lease now held, or ``None`` while a live lease is held: wait
            for its expiry (:meth:`current` says when) and try again.

        Raises:
            ValueError: If *ttl* is not positive.
        """
        if ttl <= timedelta(0):
            raise ValueError("a lease TTL must be positive")
        with write_transaction(self._connection):
            now = self._clock()
            held = _read(self._connection)
            if held is None:
                lease = ControllerLease(controller_id, 1, now, now + ttl)
                self._connection.execute(
                    "INSERT INTO controller_leases (singleton_key, controller_id, epoch, "
                    "heartbeat_at, lease_expires_at) VALUES (1, ?, ?, ?, ?)",
                    (controller_id, lease.epoch, _stamp(now), _stamp(lease.lease_expires_at)),
                )
                return lease
            if held.is_live(now):
                return None
            lease = ControllerLease(controller_id, held.epoch + 1, now, now + ttl)
            cursor = self._connection.execute(
                "UPDATE controller_leases SET controller_id = ?, epoch = ?, heartbeat_at = ?, "
                "lease_expires_at = ? WHERE singleton_key = 1 AND controller_id = ? AND epoch = ?",
                (
                    controller_id,
                    lease.epoch,
                    _stamp(now),
                    _stamp(lease.lease_expires_at),
                    held.controller_id,
                    held.epoch,
                ),
            )
            # Under the write lock nothing can move the row between the read
            # and this compare-and-set; if it did, report it held, not taken.
            return lease if cursor.rowcount == 1 else None

    def renew(self, lease: ControllerLease, ttl: timedelta) -> ControllerLease:
        """Extend a lease its owner still holds live. The epoch never changes.

        Raises:
            LeaseLostError: If another controller or epoch holds it, or it has
                already expired -- an expired lease is not revived; only a new
                acquisition, as a new epoch, can follow it.
        """
        with write_transaction(self._connection):
            now = self._clock()
            problem = _ownership_problem(
                _read(self._connection), lease.controller_id, lease.epoch, now
            )
            if problem is not None:
                raise LeaseLostError(lease.controller_id, lease.epoch, problem)
            renewed = ControllerLease(lease.controller_id, lease.epoch, now, now + ttl)
            cursor = self._connection.execute(
                "UPDATE controller_leases SET heartbeat_at = ?, lease_expires_at = ? "
                "WHERE singleton_key = 1 AND controller_id = ? AND epoch = ?",
                (_stamp(now), _stamp(renewed.lease_expires_at), lease.controller_id, lease.epoch),
            )
            if cursor.rowcount != 1:
                raise LeaseLostError(lease.controller_id, lease.epoch, "the lease row moved")
            return renewed

    def release(self, lease: ControllerLease) -> bool:
        """Expire a lease now, if it is still this owner's; whether it was.

        The row stays, with its epoch: the next owner takes it at once, as the
        next generation. A lease another owner holds is left untouched.
        """
        with write_transaction(self._connection):
            now = self._clock()
            cursor = self._connection.execute(
                "UPDATE controller_leases SET heartbeat_at = ?, lease_expires_at = ? "
                "WHERE singleton_key = 1 AND controller_id = ? AND epoch = ?",
                (_stamp(now), _stamp(now), lease.controller_id, lease.epoch),
            )
            return cursor.rowcount == 1


def _ownership_problem(
    lease: ControllerLease | None, controller_id: str, epoch: int, now: datetime
) -> str | None:
    """Why *controller_id* at *epoch* does not own the database at *now*, or ``None``."""
    if lease is None:
        return "no lease is recorded"
    if lease.controller_id != controller_id or lease.epoch != epoch:
        return f"the database is owned by {lease.controller_id} at epoch {lease.epoch}"
    if not lease.is_live(now):
        return f"the lease expired at {lease.lease_expires_at.isoformat()}"
    return None


def _read(connection: sqlite3.Connection) -> ControllerLease | None:
    row = connection.execute(
        "SELECT controller_id, epoch, heartbeat_at, lease_expires_at FROM controller_leases "
        "WHERE singleton_key = 1"
    ).fetchone()
    if row is None:
        return None
    return ControllerLease(
        controller_id=row["controller_id"],
        epoch=int(row["epoch"]),
        heartbeat_at=datetime.fromisoformat(row["heartbeat_at"]),
        lease_expires_at=datetime.fromisoformat(row["lease_expires_at"]),
    )


def _stamp(value: datetime) -> str:
    """A fixed-width UTC timestamp, so stored values also compare in order as text."""
    return value.isoformat(timespec="microseconds")
