"""The runtime operation journal (ADR-013).

Every external side effect gets a durable record written **before** the call and
updated after it. That ordering is the whole design: it is always safe to hold
durable intent with no effect, and never safe to have an effect with no durable
intent (ADR-005 §9), so every boundary is arranged so only the first can happen.

```text
BEGIN IMMEDIATE
  create attempt
  create RuntimeOperation(state=INTENDED, request_digest=...)
  insert events and outbox records
COMMIT

runtime.submit_or_get(operation_id, plan)   <- the only step outside a transaction

BEGIN IMMEDIATE
  operation -> CONFIRMED, persist RuntimeRef
  insert event
COMMIT
```

A crash anywhere in that sequence leaves a state the controller can act on
rather than guess about: an operation still `INTENDED` or `SENT` means
*reconcile*, never *re-issue blindly*.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from xaytune.core.clock import utc_now
from xaytune.core.errors import InvalidTransitionError
from xaytune.core.ids import OperationId
from xaytune.core.immutable import AggregateModel, FrozenDomainModel
from xaytune.core.refs import RuntimeRef

__all__ = [
    "OperationState",
    "OperationTargetKind",
    "OperationType",
    "RuntimeOperation",
    "RuntimeOperationTarget",
]

OperationTargetKind = Literal["training-attempt", "evaluation-attempt"]
"""What an operation acts on.

Typed rather than a ``RunAttemptId``, because evaluation attempts use this same
journal (ADR-015 §4) and live in a different table. This is deliberately **not**
a generic ``Execution`` aggregate: what training and evaluation share is the
external side effect -- one submission, one cancellation, one idempotency key --
not the domain meaning of the workload. A runtime does not care which aggregate
asked.
"""

OperationType = Literal["submit", "cancel"]

OperationState = Literal["intended", "sent", "confirmed", "failed"]

_ALLOWED: dict[OperationState, frozenset[OperationState]] = {
    "intended": frozenset({"sent", "confirmed", "failed"}),
    "sent": frozenset({"confirmed", "failed"}),
    "confirmed": frozenset(),
    "failed": frozenset(),
}
"""Transitions, per ADR-013.

``INTENDED`` may go straight to ``CONFIRMED``: a fast local runtime can return
before the controller ever records that it sent the request.

There is no edge out of ``CONFIRMED`` or ``FAILED``. Terminal records are
immutable (ADR-005 §10.3).
"""


class RuntimeOperationTarget(FrozenDomainModel):
    """The subject of an operation: which attempt, of which kind."""

    kind: OperationTargetKind
    id: str


class RuntimeOperation(AggregateModel):
    """One external side effect, with its intent recorded before it happens.

    Attributes:
        request_digest: A canonical hash of the **full external request** --
            operation type plus the resolved execution plan -- and deliberately
            *not* the ``ExecutionFingerprint``. Those answer different questions:
            the fingerprint asks whether two executions are scientifically or
            operationally equivalent, while this asks whether this is literally
            the same side-effect request. Two submissions can agree on compiler,
            runtime, GPU type and topology while differing in entrypoint,
            arguments or dataset, so deriving idempotency from the fingerprint
            would return the original workload for a request that was not the
            same request (ADR-013 §2).
    ``caused_by_action_id`` is deliberately absent until PR-006a. ADR-005 §5
    requires an effect to carry the ``Action`` that caused it, and that column
    arrives with its foreign key in migration 003 -- SQLite cannot attach one to
    an existing column afterwards. A field with nowhere to persist it would be a
    declaration the storage layer silently drops, which is the same
    declared-but-unenforced split this contract exists to prevent.
    """

    id: OperationId
    target: RuntimeOperationTarget

    type: OperationType
    request_digest: str

    state: OperationState = "intended"
    runtime_ref: RuntimeRef | None = None

    revision: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def is_terminal(self) -> bool:
        """Whether the outcome is known and settled."""
        return not _ALLOWED[self.state]

    @property
    def is_unresolved(self) -> bool:
        """Whether reconciliation still has to determine what happened.

        The reconciler's query after a restart. An unresolved operation means
        the effect may or may not exist, which is exactly when
        ``lookup_operation()`` must be consulted rather than the request
        re-issued.
        """
        return not self.is_terminal

    def with_state(
        self,
        new_state: OperationState,
        *,
        runtime_ref: RuntimeRef | None = None,
    ) -> RuntimeOperation:
        """Return a copy in *new_state*, with the revision bumped.

        A ``runtime_ref`` may be attached on the way to ``CONFIRMED``; it is
        never cleared, because the reference to a workload that ran is part of
        the record even once the operation settles.

        Raises:
            InvalidTransitionError: If the transition is not permitted --
                including any transition out of a terminal state.
        """
        if new_state not in _ALLOWED[self.state]:
            raise InvalidTransitionError("RuntimeOperation", self.state, new_state)

        return self._validated_copy(
            {
                "state": new_state,
                "runtime_ref": runtime_ref if runtime_ref is not None else self.runtime_ref,
                "revision": self.revision + 1,
                "updated_at": utc_now(),
            }
        )
