"""Durable persistence for the control plane (ADR-005).

The public write surface is :class:`ControlPlaneRepository`, and it is the only
one. Every method there is a single transaction over one of the units ADR-005
names -- transition plus event plus outbox (§3), attempt plus INTENDED operation
(§4), intent plus the effect it causes (§5).

There is deliberately no ``save_experiment()``. The row-level writers on
:class:`~xaytune.storage.repository.AggregateStore` and in
:mod:`xaytune.storage.journal` stay private and refuse to run outside a
transaction, so a caller cannot write state without its event or request an
external effect without durable intent: the API does not offer those operations
separately.
"""

from __future__ import annotations

from xaytune.core.errors import ConcurrentModificationError
from xaytune.storage.control_plane import (
    ControlPlaneRepository,
    UnknownOperationTargetError,
)
from xaytune.storage.database import connect, write_transaction
from xaytune.storage.errors import AggregateNotFoundError, MigrationError, StorageError
from xaytune.storage.graph import CandidateComparison, ExperimentGraph, LineageError
from xaytune.storage.journal import IdempotencyConflictError
from xaytune.storage.migrations import applied_versions, available_migrations, migrate
from xaytune.storage.repository import AggregateStore

__all__ = [
    "AggregateNotFoundError",
    "AggregateStore",
    "ConcurrentModificationError",
    "CandidateComparison",
    "ControlPlaneRepository",
    "ExperimentGraph",
    "LineageError",
    "IdempotencyConflictError",
    "UnknownOperationTargetError",
    "MigrationError",
    "StorageError",
    "applied_versions",
    "available_migrations",
    "connect",
    "migrate",
    "write_transaction",
]
