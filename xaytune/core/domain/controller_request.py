"""A request handed to a local daemon controller through the state database (ADR-004 §2-§3).

The daemon's mailbox is the database it controls. A client commits a request;
the daemon polls for it and carries it out. The request is as durable as the
state it changes: a client that dies after committing it has handed it off,
and a daemon that dies part-way finds it again.

```text
submit   PENDING ─► ACCEPTED ─► COMPLETED      ACCEPTED commits with the
            └─────► FAILED                     experiment it admits
attach   PENDING ─► COMPLETED
            └─────► FAILED
```

``COMPLETED`` means the handoff reached the point at which
:meth:`~xaytune.experiment.EmbeddedControllerHost.submit` or ``attach()``
returns -- not that the experiment has finished. ``FAILED`` is only for a
definitive failure before anything was admitted. There is no ``PROCESSING``:
one daemon consumes the database, and a daemon that died holding one would
need a lease to release it (PR-028).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field

from xaytune.core.clock import utc_now
from xaytune.core.errors import DomainError, InvalidTransitionError
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import ControllerRequestId, ExperimentId
from xaytune.core.immutable import AggregateModel, FrozenDict

__all__ = [
    "ControllerRequest",
    "ControllerRequestKind",
    "ControllerRequestState",
]

ControllerRequestKind = Literal["submit", "attach"]
"""What a request asks of the daemon. Not a generic command bus: a later
mutating operation becomes an explicit kind of its own."""


class ControllerRequestState(str, Enum):
    """Where a request's handoff stands."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    FAILED = "failed"


_ALLOWED: dict[str, dict[ControllerRequestState, frozenset[ControllerRequestState]]] = {
    "submit": {
        ControllerRequestState.PENDING: frozenset(
            {ControllerRequestState.ACCEPTED, ControllerRequestState.FAILED}
        ),
        ControllerRequestState.ACCEPTED: frozenset({ControllerRequestState.COMPLETED}),
        ControllerRequestState.COMPLETED: frozenset(),
        ControllerRequestState.FAILED: frozenset(),
    },
    "attach": {
        ControllerRequestState.PENDING: frozenset(
            {ControllerRequestState.COMPLETED, ControllerRequestState.FAILED}
        ),
        ControllerRequestState.ACCEPTED: frozenset(),
        ControllerRequestState.COMPLETED: frozenset(),
        ControllerRequestState.FAILED: frozenset(),
    },
}


class ControllerRequest(AggregateModel):
    """One durable request to the daemon.

    Attributes:
        id: Client-generated, and the request's idempotency key: the same id
            with the same kind, experiment and payload is the same request;
            with anything else, an idempotency conflict.
        experiment_id: For ``submit``, pre-minted by the client, so the client
            knows which experiment its request becomes before the daemon has
            touched it; for ``attach``, the experiment to adopt.
        payload: Canonical JSON. For ``submit``, the ``ExperimentSpec``; for
            ``attach``, empty.
        payload_digest: The payload's canonical fingerprint, set from it.
        error: Why a ``FAILED`` request failed.
    """

    id: ControllerRequestId
    kind: ControllerRequestKind
    experiment_id: ExperimentId
    payload: FrozenDict = Field(default_factory=FrozenDict)
    payload_digest: str
    state: ControllerRequestState = ControllerRequestState.PENDING
    error: FrozenDict | None = None

    revision: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def submit(
        cls,
        payload: FrozenDict,
        *,
        experiment_id: ExperimentId | None = None,
        request_id: ControllerRequestId | None = None,
    ) -> ControllerRequest:
        """A new ``submit`` request for the spec in *payload*, its experiment id pre-minted."""
        return cls(
            id=request_id or ControllerRequestId.generate(),
            kind="submit",
            experiment_id=experiment_id or ExperimentId.generate(),
            payload=payload,
            payload_digest=fingerprint(payload),
        )

    @classmethod
    def attach(
        cls, experiment_id: ExperimentId, *, request_id: ControllerRequestId | None = None
    ) -> ControllerRequest:
        """A new ``attach`` request for an experiment already in the record."""
        empty = FrozenDict()
        return cls(
            id=request_id or ControllerRequestId.generate(),
            kind="attach",
            experiment_id=experiment_id,
            payload=empty,
            payload_digest=fingerprint(empty),
        )

    def model_post_init(self, _context: object) -> None:
        if self.payload_digest != fingerprint(self.payload):
            raise DomainError(
                f"controller request {self.id}: payload_digest does not match its payload"
            )
        if self.kind == "attach" and self.payload:
            raise DomainError("an attach request carries no payload; it names its experiment")
        if (self.state is ControllerRequestState.FAILED) != (self.error is not None):
            raise DomainError("a request carries an error exactly when it FAILED")

    @property
    def is_unfinished(self) -> bool:
        """Whether its handoff has yet to complete or fail: the daemon's work."""
        return self.state in (ControllerRequestState.PENDING, ControllerRequestState.ACCEPTED)

    def same_request(self, other: ControllerRequest) -> tuple[str, ...]:
        """The fields in which *other*, under this id, asks for something else."""
        return tuple(
            field
            for field in ("kind", "experiment_id", "payload_digest")
            if getattr(self, field) != getattr(other, field)
        )

    def with_state(
        self, new_state: ControllerRequestState, *, error: FrozenDict | None = None
    ) -> ControllerRequest:
        """A copy in *new_state*, with the revision bumped.

        Raises:
            InvalidTransitionError: If the request's kind does not allow the edge.
        """
        if new_state not in _ALLOWED[self.kind][self.state]:
            raise InvalidTransitionError("ControllerRequest", self.state.value, new_state.value)
        return self._validated_copy(
            {
                "state": new_state,
                "error": error,
                "revision": self.revision + 1,
                "updated_at": utc_now(),
            }
        )
