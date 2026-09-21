"""Durable persistence for the control plane (ADR-005).

PR-004 scope: the schema, the migration runner, the write-transaction boundary
and revision-based optimistic concurrency.

The aggregate *write* API is intentionally absent from this namespace. ADR-005
§3 requires a state transition to commit with its domain event and outbox
record, and its consequences say callers must not be able to write state
separately. PR-005 adds the events and the outbox and exposes the single
combined operation; until then :class:`~xaytune.storage.repository.AggregateStore`
offers reads, and its writers are private and refuse to run outside a
transaction.
"""

from __future__ import annotations

from xaytune.core.errors import ConcurrentModificationError
from xaytune.storage.database import connect, write_transaction
from xaytune.storage.errors import AggregateNotFoundError, MigrationError, StorageError
from xaytune.storage.migrations import applied_versions, available_migrations, migrate
from xaytune.storage.repository import AggregateStore

__all__ = [
    "AggregateNotFoundError",
    "AggregateStore",
    "ConcurrentModificationError",
    "MigrationError",
    "StorageError",
    "applied_versions",
    "available_migrations",
    "connect",
    "migrate",
    "write_transaction",
]
