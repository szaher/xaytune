"""A request handed to a local daemon controller through the state database (ADR-004 §2-§3).

The daemon's mailbox is the database it controls. A client commits a request;
the daemon polls for it and carries it out. The request is as durable as the
state it changes: a client that dies after committing it has handed it off,
and a daemon that dies part-way finds it again.

```text
submit           PENDING ─► ACCEPTED ─► COMPLETED   ACCEPTED commits with the
                    └─────► FAILED                  experiment it admits
attach           PENDING ─► COMPLETED
cancel              └─────► FAILED
propose-action
approve-action
reject-action
```

``COMPLETED`` means the handoff reached the point at which the matching
:class:`~xaytune.experiment.EmbeddedControllerHost` call returns --
``submit()``, ``attach()``, ``ExperimentHandle.cancel()`` once its effects are
issued, ``propose()`` once the action is judged, ``approve_action()`` -- not
that the experiment has finished. ``FAILED`` is only for a definitive refusal
before the request changed anything. There is no ``PROCESSING``: one daemon
consumes the database, under a lease (PR-028).

**Every mutation is its own kind** (PR-029), never a generic command. The
four after ``attach`` carry the identity of what they create or resolve --
the ``cancel-experiment`` or proposed action's id, minted by the client, or
the action being approved -- so carrying one out again finds what the first
attempt recorded instead of recording it twice. That is why they need no
``ACCEPTED``: unlike a submission, whose admission is what names its
experiment, each is idempotent by an identity it carries from the start.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from xaytune.core.clock import utc_now
from xaytune.core.errors import DomainError, InvalidTransitionError
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import ActionId, ControllerRequestId, ExperimentId
from xaytune.core.immutable import AggregateModel, FrozenDict
from xaytune.core.refs import Actor

if TYPE_CHECKING:
    from xaytune.core.domain.action import Action

__all__ = [
    "ControllerRequest",
    "ControllerRequestKind",
    "ControllerRequestState",
    "MisdirectedRequestError",
]

ControllerRequestKind = Literal[
    "submit", "attach", "cancel", "propose-action", "approve-action", "reject-action"
]
"""What a request asks of the daemon. Not a generic command bus: every
mutating operation is an explicit kind of its own, with its own payload."""

_ACTION_KINDS = ("cancel", "propose-action", "approve-action", "reject-action")

_PAYLOAD_KEYS: dict[str, frozenset[str]] = {
    "attach": frozenset(),
    "cancel": frozenset({"action_id", "reason"}),
    "propose-action": frozenset(
        {"action_id", "type", "target", "payload", "reason", "proposed_by"}
    ),
    "approve-action": frozenset({"action_id", "approver", "reason"}),
    "reject-action": frozenset({"action_id", "approver", "reason"}),
}
"""Exactly what each kind's payload holds. ``submit`` carries an
``ExperimentSpec``, which validates itself when the daemon reads it."""


class MisdirectedRequestError(DomainError):
    """A request names an action under an experiment the action does not belong to."""


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
    **{
        kind: {
            ControllerRequestState.PENDING: frozenset(
                {ControllerRequestState.COMPLETED, ControllerRequestState.FAILED}
            ),
            ControllerRequestState.ACCEPTED: frozenset(),
            ControllerRequestState.COMPLETED: frozenset(),
            ControllerRequestState.FAILED: frozenset(),
        }
        for kind in ("attach", *_ACTION_KINDS)
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
            touched it; for every other kind, the experiment it acts on --
            for an approval, the action's.
        payload: Canonical JSON. For ``submit``, the ``ExperimentSpec``; for
            ``attach``, empty; for the action kinds, the action's id and what
            the matching host call takes (see the constructors).
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

    @classmethod
    def cancel(
        cls,
        experiment_id: ExperimentId,
        *,
        reason: str,
        action_id: ActionId | None = None,
        request_id: ControllerRequestId | None = None,
    ) -> ControllerRequest:
        """A ``cancel`` request; *action_id* is the ``cancel-experiment`` Action it records."""
        return cls._carrying(
            "cancel",
            experiment_id,
            {"action_id": str(action_id or ActionId.generate()), "reason": reason},
            request_id,
        )

    @classmethod
    def propose(
        cls, action: Action, *, request_id: ControllerRequestId | None = None
    ) -> ControllerRequest:
        """A ``propose-action`` request for *action*, built by ``action_from_spec``.

        Carries the action's id, type, target, typed payload envelope, reason
        and proposer: everything the daemon needs to rebuild the spec, and
        nothing it decides -- status and policy are the daemon's.
        """
        return cls._carrying(
            "propose-action",
            action.experiment_id,
            {
                "action_id": str(action.id),
                "type": action.type,
                "target": action.target.model_dump(mode="json"),
                "payload": action.payload,
                "reason": action.reason,
                "proposed_by": action.proposed_by.model_dump(mode="json"),
            },
            request_id,
        )

    @classmethod
    def resolve(
        cls,
        kind: Literal["approve-action", "reject-action"],
        action_id: ActionId,
        experiment_id: ExperimentId,
        *,
        approver: Actor,
        reason: str,
        request_id: ControllerRequestId | None = None,
    ) -> ControllerRequest:
        """An ``approve-action`` or ``reject-action`` request for an action awaiting approval."""
        return cls._carrying(
            kind,
            experiment_id,
            {
                "action_id": str(action_id),
                "approver": approver.model_dump(mode="json"),
                "reason": reason,
            },
            request_id,
        )

    @classmethod
    def _carrying(
        cls,
        kind: ControllerRequestKind,
        experiment_id: ExperimentId,
        payload: dict[str, Any],
        request_id: ControllerRequestId | None,
    ) -> ControllerRequest:
        frozen = FrozenDict(payload)
        return cls(
            id=request_id or ControllerRequestId.generate(),
            kind=kind,
            experiment_id=experiment_id,
            payload=frozen,
            payload_digest=fingerprint(frozen),
        )

    def model_post_init(self, _context: object) -> None:
        if self.payload_digest != fingerprint(self.payload):
            raise DomainError(
                f"controller request {self.id}: payload_digest does not match its payload"
            )
        if self.kind == "attach" and self.payload:
            raise DomainError("an attach request carries no payload; it names its experiment")
        expected = _PAYLOAD_KEYS.get(self.kind)
        if expected is not None and set(self.payload) != expected:
            raise DomainError(
                f"a {self.kind} request carries {', '.join(sorted(expected))}; "
                f"got {', '.join(sorted(self.payload)) or 'nothing'}"
            )
        if (self.state is ControllerRequestState.FAILED) != (self.error is not None):
            raise DomainError("a request carries an error exactly when it FAILED")

    @property
    def action_id(self) -> ActionId | None:
        """The action an action kind creates or resolves; ``None`` for ``submit`` and ``attach``."""
        if self.kind not in _ACTION_KINDS:
            return None
        return ActionId(self.payload["action_id"])

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
