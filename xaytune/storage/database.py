"""Connection setup and the write-transaction boundary (ADR-005 §8).

Two decisions here carry the weight of the whole persistence contract.

**Every write uses ``BEGIN IMMEDIATE``.** SQLite's default deferred transaction
takes the write lock lazily, on the first write statement. A transaction that
reads an aggregate, decides a transition from what it read, and then writes can
therefore fail with ``SQLITE_BUSY`` *at the write*, after its logic has already
run against a snapshot another writer has since replaced. Taking the lock up
front converts that into a fast, obvious contention failure that the caller's
retry path handles, at the cost of serializing writers -- which is what the
contract wants anyway.

**Correctness does not assume a single writer.** Multiple processes may contend.
SQLite serializes write transactions and revision CAS (:mod:`xaytune.storage.repository`)
protects aggregate-level concurrency; together those are the guarantee. ADR-004's
controller lease is a different concern -- it stops two controllers making
duplicate *control decisions*, which no amount of database locking can detect,
because both writes would be individually valid.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = ["connect", "write_transaction"]

# Long enough to ride out a contended writer, short enough that a genuine
# deadlock surfaces as an error rather than a hang.
_BUSY_TIMEOUT_MS = 5_000


def connect(path: Path | str) -> sqlite3.Connection:
    """Open *path* with the pragmas ADR-005 §8 requires.

    ``:memory:`` is accepted for tests. Note that an in-memory database cannot
    demonstrate crash durability, so the crash tests use a real file.

    Pragmas, and why each one:

    ``journal_mode=WAL``
        Readers do not block the writer, so reconciliation and inspection can
        run against a live controller.
    ``synchronous=FULL``
        A commit is durable when it returns. ``NORMAL`` can lose the tail of the
        WAL on power loss, which would mean losing a committed intent while the
        external effect it describes already happened -- the one inconsistency
        ADR-005 §9 is arranged to prevent.
    ``foreign_keys=ON``
        Off by default in SQLite, per-connection, and silently ignored if left
        unset. The schema's references are only real if this is set on every
        connection.
    ``busy_timeout``
        Contended writers wait rather than failing immediately.
    """
    connection = sqlite3.connect(
        str(path),
        # Transactions are managed explicitly by write_transaction(); the
        # default isolation level would emit its own BEGIN and defer the lock.
        isolation_level=None,
        detect_types=0,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    return connection


@contextmanager
def write_transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a write transaction, taking the write lock up front.

    Commits on clean exit and rolls back on any exception, including
    :class:`KeyboardInterrupt` and :class:`SystemExit` -- a control-plane process
    is expected to be killed, and a partially applied transition is exactly what
    must not survive that.

    Nesting is refused rather than silently flattened. SQLite has no nested
    transactions, so an inner block that appeared to commit would in fact be
    committed by the outer one, and an inner rollback would discard the outer
    work too. Callers that need one atomic unit must express it as one block --
    which is the point of ADR-005 §3-§5.

    Raises:
        sqlite3.OperationalError: If a transaction is already open on this
            connection, or if the write lock cannot be acquired within the busy
            timeout.
    """
    if connection.in_transaction:
        raise sqlite3.OperationalError(
            "a transaction is already open on this connection: SQLite has no "
            "nested transactions, so the inner block could neither commit nor "
            "roll back independently. Express the whole unit as one "
            "write_transaction()."
        )

    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
    except BaseException:
        connection.rollback()
        raise
    connection.commit()
