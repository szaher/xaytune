"""Storage-layer errors.

:class:`~xaytune.core.errors.ConcurrentModificationError` is deliberately not
here. It lives in :mod:`xaytune.core.errors` so that callers can catch a lost
revision race without importing the persistence layer -- the retry that follows
it is control-plane logic, not storage logic.
"""

from __future__ import annotations

from xaytune.core.errors import XaytuneError

__all__ = [
    "AggregateNotFoundError",
    "IncompatiblePayloadError",
    "MigrationError",
    "StorageError",
]


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


class IncompatiblePayloadError(StorageError):
    """A stored payload predates a control-plane format change.

    PR-007 restructured the node body: ``training_spec`` (an opaque payload)
    became ``candidate`` (a typed :class:`CandidateSpec`). The two are not
    the same shape, so a node written by band B cannot be read by this code.

    **This is a deliberate format change, not a compatibility bug.** The
    control-plane storage format is experimental and has no released users;
    writing a translation for a shape that existed for a day would mean
    carrying it forever. The field *rename* was worth a compatibility alias
    because it cost one line and the old key still meant the same thing;
    restructuring the body does not, because the old payload does not contain
    a candidate to translate into.

    What it must not be is an incidental ``ValidationError`` from inside
    ``load_node()``, which tells the reader that some field is missing rather
    than that their database predates a format change. Hence this error, and
    the instruction it carries.
    """

    def __init__(self, aggregate: str, aggregate_id: str, detail: str) -> None:
        self.aggregate = aggregate
        self.aggregate_id = aggregate_id
        super().__init__(
            f"{aggregate} {aggregate_id} was written in a control-plane format "
            f"this code cannot read ({detail}). The storage format changed in "
            f"PR-007: node bodies now hold a typed CandidateSpec rather than an "
            f"opaque training_spec payload. Recreate the database; the format "
            f"is experimental and there is no migration."
        )
