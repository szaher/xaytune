"""What a model-backed planner asked a model, and what came of it -- durably (PR-032).

```text
              begin()                 the model is asked
   ─────────▶ INTENDED ──────────────────────────────────▶ ANSWERED ──▶ COMPLETED
   written      │  │  │                                     │            (proposal or none)
   before       │  │  └──▶ FAILED      no answer, or the    └──────────▶ REFUSED
   the call     │  │                   model changed while               (the answer named
                │  └─────▶ REFUSED     it answered                        something it may not)
                │                      (schema-invalid answer)
                └────────▶ OUTCOME_UNKNOWN   found still INTENDED by the next begin():
                                             the process stopped during the call
```

**Intent before the call.** An invocation is recorded ``INTENDED`` before the
model is asked, so a crash after an expensive, unrepeatable call cannot make
the history pretend it never happened. The answer is recorded (``ANSWERED``)
before anything is derived from it, and a proposal is derived only from the
recorded answer.

**One logical round, one record at a time.** A planning round is the
experiment, the bound planner (its spec fingerprint) and the planning
context it saw (its fingerprint). :meth:`AgentInvocationJournal.begin` for a
round that already has an answer returns that record instead of asking again,
so a restarted controller replays the round rather than manufacturing a
second, contradictory invocation. A round whose last attempt failed or ended
unknown gets a new attempt, numbered.

**What is never recorded.** No exception object, message, cause, context or
traceback from an adapter or SDK; a failure is a classification
(:class:`AgentInvocationFailure`): its kind, the exception's type name, and
reasons xaytune itself wrote. No hidden reasoning: the response is the
structured answer and the metadata the provider reported, nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field, model_validator
from typing_extensions import Self

from xaytune.core.clock import utc_now
from xaytune.core.domain.planning import ActionProposal, action_proposal_identity_v1
from xaytune.core.errors import (
    ConcurrentModificationError,
    DomainError,
    InvalidTransitionError,
)
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import AgentInvocationId, ExperimentId
from xaytune.core.immutable import FrozenDict, FrozenDomainModel

__all__ = [
    "AgentInvocation",
    "AgentInvocationConflictError",
    "AgentInvocationFailure",
    "AgentInvocationIntent",
    "AgentInvocationJournal",
    "AgentInvocationMismatchError",
    "AgentInvocationStatus",
    "InMemoryAgentInvocationJournal",
    "RecordsAgentInvocations",
    "StampingAgentInvocationJournal",
    "next_invocation",
    "require_derived_from",
]


class AgentInvocationStatus(str, Enum):
    """Where one invocation stands. Four of them are final."""

    INTENDED = "intended"
    ANSWERED = "answered"
    COMPLETED = "completed"
    REFUSED = "refused"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL


_TERMINAL = frozenset(
    {
        AgentInvocationStatus.COMPLETED,
        AgentInvocationStatus.REFUSED,
        AgentInvocationStatus.FAILED,
        AgentInvocationStatus.OUTCOME_UNKNOWN,
    }
)

_TRANSITIONS: Mapping[AgentInvocationStatus, frozenset[AgentInvocationStatus]] = {
    AgentInvocationStatus.INTENDED: frozenset(
        {
            AgentInvocationStatus.ANSWERED,
            AgentInvocationStatus.REFUSED,
            AgentInvocationStatus.FAILED,
            AgentInvocationStatus.OUTCOME_UNKNOWN,
        }
    ),
    AgentInvocationStatus.ANSWERED: frozenset(
        {
            AgentInvocationStatus.COMPLETED,
            AgentInvocationStatus.REFUSED,
        }
    ),
}
"""``ANSWERED`` never becomes ``FAILED``: once the answer is recorded, the call
happened and is known, and ``FAILED`` would let the round ask again. A local
error deriving from the answer leaves it ``ANSWERED``, to be derived again."""

_REPLAYED = frozenset(
    {
        AgentInvocationStatus.ANSWERED,
        AgentInvocationStatus.COMPLETED,
        AgentInvocationStatus.REFUSED,
    }
)
"""A round whose last attempt got an answer is replayed from it, never asked again."""


class AgentInvocationConflictError(DomainError):
    """A round's recorded request is not the request now being made for it."""


class AgentInvocationMismatchError(DomainError):
    """A proposal claims an invocation that did not produce it."""


class AgentInvocationFailure(FrozenDomainModel):
    """How an invocation failed, classified -- never the exception itself.

    ``error_type`` is the exception's type name. ``reasons`` are written by
    xaytune (schema violations, refusals of what an answer named), never an
    adapter's or SDK's message, which can carry URLs, headers or keys.
    """

    kind: Literal[
        "invocation-failed",
        "output-refused",
        "model-identity-changed",
        "plugin-incompatible",
        "internal-error",
    ]
    error_type: str = Field(min_length=1)
    reasons: tuple[str, ...] = ()


class AgentInvocationIntent(FrozenDomainModel):
    """Everything fixed before the model is asked: who asks, about what, with which request.

    ``request`` is the whole :class:`~xaytune.agent.AgentModelRequest` and
    ``agent_model`` the whole :class:`~xaytune.agent.AgentModelDescriptor` --
    adapter and model -- as canonical JSON: the request fingerprint names the
    logical request, the descriptor says what carried it.
    """

    experiment_id: ExperimentId
    planner_kind: str = Field(min_length=1)
    planner_version: str = Field(min_length=1)
    planner_spec_fingerprint: str = Field(min_length=1)
    context_identity_version: int = Field(ge=1)
    context_fingerprint: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_fingerprint: str = Field(min_length=1)
    request_identity_version: int = Field(ge=1)
    request_fingerprint: str = Field(min_length=1)
    request: FrozenDict
    agent_model: FrozenDict


class AgentInvocation(FrozenDomainModel):
    """One invocation of an agent model by a planner, from intent to outcome."""

    id: AgentInvocationId
    attempt: int = Field(ge=1)
    intent: AgentInvocationIntent
    status: AgentInvocationStatus = AgentInvocationStatus.INTENDED
    response: FrozenDict | None = None
    """The :class:`~xaytune.agent.AgentModelResponse` as recorded: structured
    content, served model and revision, finish reason, usage, provider request
    id and latency, where reported."""
    failure: AgentInvocationFailure | None = None
    proposal: FrozenDict | None = None
    """The proposal derived from the answer, if it proposed one."""
    proposal_fingerprint: str | None = None
    created_at: datetime
    settled_at: datetime | None = None
    revision: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _consistent_with_its_status(self) -> AgentInvocation:
        status = self.status
        problems = []
        if (self.settled_at is not None) != status.is_terminal:
            problems.append("settled_at is set exactly when the invocation is final")
        if (self.failure is not None) != (
            status in (AgentInvocationStatus.REFUSED, AgentInvocationStatus.FAILED)
        ):
            problems.append(
                "a failure is recorded exactly when the invocation was refused or failed"
            )
        if self.response is None and status in (
            AgentInvocationStatus.ANSWERED,
            AgentInvocationStatus.COMPLETED,
        ):
            problems.append(f"a {status.value} invocation records its answer")
        if self.response is not None and status in (
            AgentInvocationStatus.INTENDED,
            AgentInvocationStatus.OUTCOME_UNKNOWN,
        ):
            problems.append(f"a {status.value} invocation has no answer on record")
        if (self.proposal is None) != (self.proposal_fingerprint is None):
            problems.append("a proposal and its fingerprint are recorded together")
        if self.proposal is not None and status is not AgentInvocationStatus.COMPLETED:
            problems.append("only a completed invocation records a proposal")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @property
    def round_key(self) -> tuple[str, str, str]:
        """The planning round: experiment, bound planner, planning context."""
        intent = self.intent
        return (
            str(intent.experiment_id),
            intent.planner_spec_fingerprint,
            intent.context_fingerprint,
        )

    # ---- transitions ------------------------------------------------------------------

    def answered(self, response: Mapping[str, Any], *, at: datetime) -> Self:
        return self._to(AgentInvocationStatus.ANSWERED, at, response=FrozenDict(response))

    def completed(self, proposal: ActionProposal | None, *, at: datetime) -> Self:
        if proposal is not None and proposal.agent_invocation_id != self.id:
            raise AgentInvocationMismatchError(
                f"a proposal naming invocation {proposal.agent_invocation_id} cannot complete "
                f"invocation {self.id}"
            )
        return self._to(
            AgentInvocationStatus.COMPLETED,
            at,
            proposal=None if proposal is None else FrozenDict(proposal.model_dump(mode="json")),
            proposal_fingerprint=None
            if proposal is None
            else fingerprint(action_proposal_identity_v1(proposal)),
        )

    def refused(
        self,
        failure: AgentInvocationFailure,
        *,
        at: datetime,
        response: Mapping[str, Any] | None = None,
    ) -> Self:
        return self._to(
            AgentInvocationStatus.REFUSED,
            at,
            failure=failure,
            response=self.response if response is None else FrozenDict(response),
        )

    def failed(
        self,
        failure: AgentInvocationFailure,
        *,
        at: datetime,
        response: Mapping[str, Any] | None = None,
    ) -> Self:
        return self._to(
            AgentInvocationStatus.FAILED,
            at,
            failure=failure,
            response=self.response if response is None else FrozenDict(response),
        )

    def outcome_unknown(self, *, at: datetime) -> Self:
        return self._to(AgentInvocationStatus.OUTCOME_UNKNOWN, at)

    def _to(self, status: AgentInvocationStatus, at: datetime, **changes: Any) -> Self:
        if status not in _TRANSITIONS.get(self.status, frozenset()):
            raise InvalidTransitionError(f"agent invocation {self.id}", self.status, status)
        return self._validated_copy(
            {
                **changes,
                "status": status,
                "settled_at": at if status.is_terminal else None,
                "revision": self.revision + 1,
            }
        )


def next_invocation(
    latest: AgentInvocation | None, intent: AgentInvocationIntent, *, at: datetime
) -> tuple[AgentInvocation | None, AgentInvocation, bool]:
    """What :meth:`AgentInvocationJournal.begin` does, given the round's latest attempt.

    Returns ``(closed, invocation, replayed)``: an attempt found still
    ``INTENDED`` and now closed as ``OUTCOME_UNKNOWN`` (to be written), the
    invocation to go on with, and whether that is a recorded one to replay
    rather than a new one to ask.

    Raises:
        AgentInvocationConflictError: The round's recorded request is another
            request. A bound planner builds one request per context, so this
            means the record and the planner disagree, and neither is chosen.
    """
    if latest is not None and latest.intent.request_fingerprint != intent.request_fingerprint:
        raise AgentInvocationConflictError(
            f"planning round {latest.round_key} recorded request "
            f"{latest.intent.request_fingerprint} (invocation {latest.id}); the planner now "
            f"makes {intent.request_fingerprint}"
        )
    if latest is not None and latest.status in _REPLAYED:
        return None, latest, True
    closed = None
    if latest is not None and latest.status is AgentInvocationStatus.INTENDED:
        closed = latest.outcome_unknown(at=at)
    fresh = AgentInvocation(
        id=AgentInvocationId.generate(),
        attempt=1 if latest is None else latest.attempt + 1,
        intent=intent,
        created_at=at,
    )
    return closed, fresh, False


@runtime_checkable
class AgentInvocationJournal(Protocol):
    """Where a model-backed planner records its invocations. The host provides it.

    The journal stamps every transition with its own clock: a planner reads
    none.
    """

    def begin(self, intent: AgentInvocationIntent) -> AgentInvocation:
        """The invocation to go on with for *intent*'s round (see :func:`next_invocation`).

        A new one is written ``INTENDED`` -- with any attempt found still
        ``INTENDED`` closed as ``OUTCOME_UNKNOWN`` -- before this returns.
        """
        ...

    def answered(
        self, invocation: AgentInvocation, response: Mapping[str, Any]
    ) -> AgentInvocation: ...

    def completed(
        self, invocation: AgentInvocation, proposal: ActionProposal | None
    ) -> AgentInvocation: ...

    def refused(
        self,
        invocation: AgentInvocation,
        failure: AgentInvocationFailure,
        response: Mapping[str, Any] | None = None,
    ) -> AgentInvocation: ...

    def failed(
        self,
        invocation: AgentInvocation,
        failure: AgentInvocationFailure,
        response: Mapping[str, Any] | None = None,
    ) -> AgentInvocation: ...

    def get(self, invocation_id: AgentInvocationId | str) -> AgentInvocation | None:
        """The recorded invocation, or ``None``."""
        ...


class StampingAgentInvocationJournal(ABC):
    """The transitions every journal shares, stamped with the journal's clock.

    A journal implements :meth:`begin`, :meth:`get` and :meth:`_write`, which
    writes one transition guarded on the revision it moved from.
    """

    def __init__(self, clock: Callable[[], datetime] = utc_now) -> None:
        self._clock = clock

    @abstractmethod
    def begin(self, intent: AgentInvocationIntent) -> AgentInvocation: ...

    @abstractmethod
    def get(self, invocation_id: AgentInvocationId | str) -> AgentInvocation | None: ...

    @abstractmethod
    def _write(self, moved: AgentInvocation) -> AgentInvocation: ...

    def answered(self, invocation: AgentInvocation, response: Mapping[str, Any]) -> AgentInvocation:
        return self._write(invocation.answered(response, at=self._clock()))

    def completed(
        self, invocation: AgentInvocation, proposal: ActionProposal | None
    ) -> AgentInvocation:
        return self._write(invocation.completed(proposal, at=self._clock()))

    def refused(
        self,
        invocation: AgentInvocation,
        failure: AgentInvocationFailure,
        response: Mapping[str, Any] | None = None,
    ) -> AgentInvocation:
        return self._write(invocation.refused(failure, at=self._clock(), response=response))

    def failed(
        self,
        invocation: AgentInvocation,
        failure: AgentInvocationFailure,
        response: Mapping[str, Any] | None = None,
    ) -> AgentInvocation:
        return self._write(invocation.failed(failure, at=self._clock(), response=response))


class InMemoryAgentInvocationJournal(StampingAgentInvocationJournal):
    """A journal that keeps invocations in this process: for tests and planners used directly.

    Nothing survives the process; a host gives its planners the durable one.
    """

    def __init__(self, clock: Callable[[], datetime] = utc_now) -> None:
        super().__init__(clock)
        self._records: dict[str, AgentInvocation] = {}

    @property
    def invocations(self) -> tuple[AgentInvocation, ...]:
        """Every invocation, in the order they were begun."""
        return tuple(self._records.values())

    def begin(self, intent: AgentInvocationIntent) -> AgentInvocation:
        latest = max(
            (
                record
                for record in self._records.values()
                if record.round_key
                == (
                    str(intent.experiment_id),
                    intent.planner_spec_fingerprint,
                    intent.context_fingerprint,
                )
            ),
            key=lambda record: record.attempt,
            default=None,
        )
        closed, invocation, replayed = next_invocation(latest, intent, at=self._clock())
        if closed is not None:
            self._write(closed)
        if not replayed:
            self._records[str(invocation.id)] = invocation
        return invocation

    def get(self, invocation_id: AgentInvocationId | str) -> AgentInvocation | None:
        return self._records.get(str(invocation_id))

    def _write(self, moved: AgentInvocation) -> AgentInvocation:
        current = self._records.get(str(moved.id))
        if current is None or current.revision != moved.revision - 1:
            raise ConcurrentModificationError("AgentInvocation", str(moved.id), moved.revision - 1)
        self._records[str(moved.id)] = moved
        return moved


@runtime_checkable
class RecordsAgentInvocations(Protocol):
    """A planner that asks a model, and so must be given a journal before it plans."""

    def with_journal(self, journal: AgentInvocationJournal) -> Any:
        """This planner, bound identically, recording into *journal*."""
        ...


def require_derived_from(proposal: ActionProposal, invocation: AgentInvocation | None) -> None:
    """Refuse *proposal* unless *invocation* is the recorded call that produced it.

    The invocation must be the one the proposal names, ``COMPLETED``, for the
    same bound planner and planning context as the proposal's provenance, and
    must have recorded exactly this proposal.

    Raises:
        AgentInvocationMismatchError: Naming what disagrees.
    """
    claimed = proposal.agent_invocation_id
    if claimed is None:
        raise AgentInvocationMismatchError("the proposal names no agent invocation")
    if invocation is None:
        raise AgentInvocationMismatchError(f"agent invocation {claimed} is not on record")
    wrong = []
    if invocation.id != claimed:
        wrong.append(f"it names invocation {claimed}, not {invocation.id}")
    if invocation.status is not AgentInvocationStatus.COMPLETED:
        wrong.append(f"invocation {invocation.id} is {invocation.status.value}, not completed")
    provenance = proposal.provenance
    if provenance.planner_spec_fingerprint != invocation.intent.planner_spec_fingerprint:
        wrong.append("its planner is not the one that made the invocation")
    if provenance.context_fingerprint != invocation.intent.context_fingerprint:
        wrong.append("its planning context is not the one the invocation was made for")
    if invocation.proposal_fingerprint != fingerprint(action_proposal_identity_v1(proposal)):
        wrong.append("it is not the proposal the invocation recorded")
    if wrong:
        raise AgentInvocationMismatchError(
            f"the proposal was not derived from agent invocation {claimed}: " + "; ".join(wrong)
        )
