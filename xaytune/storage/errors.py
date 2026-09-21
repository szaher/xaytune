"""Storage-layer errors.

:class:`~xaytune.core.errors.ConcurrentModificationError` is deliberately not
here. It lives in :mod:`xaytune.core.errors` so that callers can catch a lost
revision race without importing the persistence layer -- the retry that follows
it is control-plane logic, not storage logic.
"""

from __future__ import annotations

from xaytune.core.errors import XaytuneError

__all__ = ["AggregateNotFoundError", "MigrationError", "StorageError"]


class StorageError(XaytuneError):
    """A persistence operation failed."""


class MigrationError(StorageError):
    """The schema could not be migrated, or the migration set is inconsistent."""


class AggregateNotFoundError(StorageError):
    """An aggregate was requested by id and does not exist.

    Distinct from returning ``None``: the ``get_*`` readers return ``None`` for
    a caller that is checking, while the loaders raise for a caller that has an
    id it believes is valid. A missing aggregate at that point means a dangling
    reference, which is worth a traceback rather than an ``AttributeError`` two
    frames later.
    """

    def __init__(self, aggregate: str, aggregate_id: str) -> None:
        self.aggregate = aggregate
        self.aggregate_id = aggregate_id
        super().__init__(f"{aggregate} {aggregate_id} does not exist")
