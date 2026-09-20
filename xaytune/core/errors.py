"""Error taxonomy for the Xaytune control-plane core.

Errors raised at architectural boundaries are typed so that callers can
distinguish a domain-rule violation from an infrastructure failure without
inspecting messages.
"""

from __future__ import annotations

__all__ = [
    "ConcurrentModificationError",
    "DomainError",
    "InvalidIdError",
    "InvalidTransitionError",
    "XaytuneError",
]


class XaytuneError(Exception):
    """Base class for every error raised by Xaytune."""


class DomainError(XaytuneError):
    """A domain rule was violated."""


class InvalidIdError(DomainError, ValueError):
    """An identifier is malformed or carries the wrong type prefix.

    Also a :class:`ValueError` so that Pydantic reports it as a validation
    error when it surfaces during model construction.
    """


class InvalidTransitionError(DomainError):
    """A state transition is not permitted by the aggregate's state machine."""

    def __init__(self, aggregate: str, current: object, requested: object) -> None:
        self.aggregate = aggregate
        self.current = current
        self.requested = requested
        super().__init__(
            f"{aggregate} cannot transition from "
            f"{getattr(current, 'value', current)!r} to "
            f"{getattr(requested, 'value', requested)!r}"
        )


class ConcurrentModificationError(XaytuneError):
    """An aggregate was modified by someone else since it was read.

    Raised when a revision-guarded write affects no rows. Defined here so the
    persistence layer and its callers share one error type.
    """

    def __init__(self, aggregate: str, aggregate_id: str, expected_revision: int) -> None:
        self.aggregate = aggregate
        self.aggregate_id = aggregate_id
        self.expected_revision = expected_revision
        super().__init__(
            f"{aggregate} {aggregate_id} was modified concurrently "
            f"(expected revision {expected_revision})"
        )
