"""Error taxonomy for the Xaytune control-plane core.

Errors raised at architectural boundaries are typed so that callers can
distinguish a domain-rule violation from an infrastructure failure without
inspecting messages.
"""

from __future__ import annotations

__all__ = [
    "ConcurrentModificationError",
    "DomainError",
    "IdempotencyConflictError",
    "InvalidDomainValueError",
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


class InvalidDomainValueError(DomainError, ValueError):
    """A value cannot be stored in a domain record.

    Domain payloads must be canonically persistable: JSON-shaped, with string
    keys, no sets (which have no stable order), and no NaN or infinity. Values
    that are not are rejected at construction rather than at serialization,
    because a record that cannot round-trip cannot be fingerprinted.

    Also a :class:`ValueError` so Pydantic reports it as a validation error.
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


class IdempotencyConflictError(XaytuneError):
    """An operation id was reused for a materially different request.

    ADR-013 §2: same id and same request returns the original record; same id
    and anything else is refused. Guessing which one the caller meant would be
    worse than stopping, because one of the two answers starts a second
    workload.

    In the core alongside :class:`ConcurrentModificationError`, and for the
    same reason: the condition arises wherever get-or-create is implemented,
    which is now the control plane's journal *and* a runtime backend's own
    registry. A controller should catch one type for one condition rather than
    learn which layer refused it.

    Raised for both halves of a compound write, so *kind* names which record
    conflicted -- an error reading "operation act_..." would send the reader
    looking in the wrong table.
    """

    def __init__(
        self, record_id: str, differing: tuple[str, ...], *, kind: str = "operation"
    ) -> None:
        self.record_id = record_id
        self.operation_id = record_id  # retained for callers that predate `kind`
        self.differing = differing
        self.kind = kind
        super().__init__(
            f"{kind} {record_id} already exists with a different "
            f"{', '.join(differing)}; reusing a{'n' if kind[0] in 'aeiou' else ''} "
            f"{kind} id requires an identical request"
        )
