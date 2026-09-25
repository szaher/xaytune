"""The execute half of the compile/execute boundary.

A :class:`RuntimeBackend` runs a :class:`ResolvedExecutionPlan` and reports
what happens. **It never interprets the candidate.** A runtime that understood
what SFT or GRPO meant would be a second place where scientific intent lived,
and the two would drift; it sees entrypoints, arguments, resources and
artifacts, which is everything it needs and nothing it can misread.

No implementations here — PR-009 brings ``LocalRuntime``. This module is the
contract those implement, declared in full so the first implementation writes
against a stable protocol rather than growing one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Literal, Protocol, get_args, runtime_checkable

from pydantic import Field, model_validator

from xaytune.core.capabilities import CapabilityDocument, PluginDescriptor
from xaytune.core.domain.operation import OperationTargetKind, RuntimeOperationTarget
from xaytune.core.execution import ResolvedExecutionPlan
from xaytune.core.ids import OperationId
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.observability import CorrelationContext, Counter, TraceContext
from xaytune.core.refs import RuntimeRef
from xaytune.core.telemetry import (
    TELEMETRY_V1ALPHA3,
    EvaluationCompletedPayload,
    EvaluationObservation,
    TelemetryProtocolVersion,
    TrainingObservation,
)

__all__ = [
    "EvaluationEventPayload",
    "OperationDisposition",
    "OperationOutcome",
    "RuntimeBackend",
    "RuntimeEventEnvelope",
    "RuntimeLog",
    "RuntimeState",
    "RuntimeStatus",
    "RuntimeEventPayload",
    "StreamCursor",
    "TrainingEventPayload",
]

RuntimeState = Literal[
    "pending",
    "queued",
    "starting",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "preempted",
    "unknown",
]
"""What a runtime observes about a workload.

Deliberately **not** ``RunAttemptStatus``. These are the backend's
observations; the attempt's status is the controller's conclusion, reached by
reconciling observations with its own record. Collapsing them would let a
runtime's opinion overwrite a decision the controller had already made and
recorded -- a cancelled attempt reported ``running`` by a lagging backend, for
instance.

``unknown`` is a real answer, not a failure to answer. A backend that has lost
track of a workload must say so rather than guess, because the controller's
response to "I cannot tell" is different from its response to "it failed".
"""


class RuntimeStatus(FrozenDomainModel):
    """What a backend currently observes about one workload."""

    state: RuntimeState
    detail: str | None = None
    observed_at: datetime | None = None
    exit_code: int | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


_LIVE_STATES: frozenset[RuntimeState] = frozenset({"pending", "queued", "starting", "running"})
"""States a workload can still leave. ``unknown`` is deliberately absent:
it is neither an ending nor a reason to keep waiting."""

OperationDisposition = Literal["accepted", "completed", "rejected"]
"""What became of an operation the runtime *did* receive.

``None`` from :meth:`RuntimeBackend.lookup_operation` is the fourth answer --
never received -- and is deliberately not a member here: the absence of a
record and a record of absence are different facts, and only one of them makes
re-issuing safe.
"""


class OperationOutcome(FrozenDomainModel):
    """What became of an operation (ADR-013 §3).

    A controller reads this after a restart to decide whether to adopt a
    workload, re-issue a request, or stop and escalate. Each disposition leads
    somewhere different:

    ```text
    accepted    the runtime has it, still running   -> adopt it
    completed   it ran and already finished         -> adopt the result
    rejected    it was refused; nothing is running  -> safe to re-issue
    None        never received                      -> safe to re-issue
    ```

    A lookup that could only say "running" or "not found" would collapse
    *completed* into *never received*, and the controller would cheerfully
    re-run a finished workload.

    The invariants are enforced rather than documented, because this type
    exists to carry evidence during recovery and contradictory evidence is
    worse than none: a rejected operation cannot also have completed or carry
    a workload reference, and anything the runtime accepted must say which
    workload it accepted.
    """

    operation_id: OperationId
    disposition: OperationDisposition

    runtime_ref: RuntimeRef | None = None
    status: RuntimeStatus | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def _states_are_coherent(self) -> OperationOutcome:
        if self.disposition == "rejected":
            if self.runtime_ref is not None:
                raise ValueError(
                    "a rejected operation cannot carry a RuntimeRef: nothing "
                    "was started, so there is no workload to reference"
                )
        elif self.runtime_ref is None:
            raise ValueError(
                f"an {self.disposition} operation must name the workload it "
                f"started: without a RuntimeRef the controller is told a "
                f"workload exists and not where, which is the orphan ADR-013 "
                f"exists to prevent"
            )
        return self

    @property
    def is_running(self) -> bool | None:
        """Whether the workload is still going, or ``None`` if nobody can say.

        Three-valued because :data:`RuntimeState` has an ``unknown`` and it
        would otherwise be thrown away here. A backend that accepted an
        operation and later lost track of it reports ``accepted`` with a status
        of ``unknown``; reading that as "running" would hand the controller a
        certainty the runtime explicitly refused to give, and reading it as
        "not running" would retire a workload that may still be producing
        artifacts.

        An observation outranks the disposition when there is one, because the
        disposition records what the runtime was asked and the status records
        what it can currently see.
        """
        if self.status is not None:
            if self.status.state == "unknown":
                return None
            return self.status.state in _LIVE_STATES
        return self.disposition == "accepted"

    @property
    def may_reissue(self) -> bool:
        """Whether re-issuing this request is safe.

        Only after an explicit rejection. An accepted or completed operation
        must be adopted, not repeated.
        """
        return self.disposition == "rejected"


class StreamCursor(FrozenDomainModel):
    """A position in one target's telemetry stream (ADR-014 §4).

    Two integers with defined meaning, never an opaque provider token: a
    backend cannot smuggle its own pagination state through it, which is what
    a string cursor invited.

    It carries the generation because the sequence alone stopped being unique
    within a target once a supervisor could be replaced mid-attempt.
    """

    generation: int = Field(default=0, ge=0)
    sequence: int = Field(default=-1, ge=-1)
    """The last sequence the controller **durably recorded**.

    Not the last it received: an event received and then lost in a crash must
    be redelivered. ``-1`` is the only value below zero, and it means nothing
    has been recorded yet -- sequences themselves start at 0.
    """


class TrainingEventPayload(FrozenDomainModel):
    """A training-only discriminated observation body."""

    workload: Literal["training"] = "training"
    data: TrainingObservation

    @property
    def type(self) -> str:
        return self.data.type


class EvaluationEventPayload(FrozenDomainModel):
    """Evaluation observations cannot contain checkpoints or training lifecycle."""

    workload: Literal["evaluation"] = "evaluation"
    data: EvaluationObservation

    @property
    def type(self) -> str:
        return self.data.type


RuntimeEventPayload = Annotated[
    TrainingEventPayload | EvaluationEventPayload, Field(discriminator="workload")
]


_PAYLOAD_WORKLOAD_FOR_TARGET: dict[OperationTargetKind, Literal["training", "evaluation"]] = {
    "training-attempt": "training",
    "evaluation-attempt": "evaluation",
}
"""Which payload family each target kind may carry (ADR-014 §1).

A mapping checked for coverage rather than a chain of ``if`` branches: a third
target kind added without a payload family fails at import here, whereas an
unmatched branch would silently stop enforcing the pairing for exactly the new
kind nobody had thought about yet.
"""

if set(_PAYLOAD_WORKLOAD_FOR_TARGET) != set(get_args(OperationTargetKind)):  # pragma: no cover
    raise RuntimeError(
        "every operation target kind must declare which telemetry payload "
        "family it carries; unpaired kinds: "
        f"{set(get_args(OperationTargetKind)) ^ set(_PAYLOAD_WORKLOAD_FOR_TARGET)}"
    )


class RuntimeEventEnvelope(FrozenDomainModel):
    """One telemetry event (ADR-014 §1).

    **The envelope is workload-neutral; the payload is not.** Delivery,
    ordering, deduplication, cursors, generations and gap detection are
    properties of a stream and are identical for training and evaluation. What
    differs is what the events *say*, and that lives in the payload.

    That split is what lets one protocol serve both without introducing a
    generic ``Execution`` aggregate: the transport is shared, the domain
    meaning is not (ADR-015 §2).

    ``target`` is the same typed reference the operation journal uses, so an
    event stream and the operation that started it name their subject the same
    way -- and the pairing between the two is **enforced here**, not left to
    the caller. Splitting the payload by workload only makes a bad event
    unrepresentable once it is already being built as the right family;
    without this validator an evaluation attempt could still carry a
    ``CheckpointCommitted``, which is the exact confusion ADR-014 §1 pairs the
    target with the payload to prevent.
    """

    protocol_version: TelemetryProtocolVersion = "xaytune.telemetry/v1alpha2"
    """Which contract the payload was written against; both are read.

    Set from the plan's ``TelemetryContract``, never assumed: a controller
    adopting a workload started before an upgrade must still read what it
    writes.
    """

    event_id: str

    target: RuntimeOperationTarget

    stream_generation: int = Field(default=0, ge=0)
    sequence: int = Field(ge=0)
    """Monotonic and gapless within a generation, starting at 0.

    Never negative: a sequence is a counter, and ``-1`` belongs only to a
    cursor that has recorded nothing.
    """

    emitted_at: datetime | None = None
    payload: RuntimeEventPayload
    context: CorrelationContext | None = None
    trace_context: TraceContext | None = None

    @model_validator(mode="after")
    def _payload_family_matches_target(self) -> RuntimeEventEnvelope:
        expected = _PAYLOAD_WORKLOAD_FOR_TARGET[self.target.kind]
        if self.payload.workload != expected:
            raise ValueError(
                f"a {self.target.kind} carries {expected} telemetry, not "
                f"{self.payload.workload}: the target and the payload family "
                f"are two statements about the same workload, and a consumer "
                f"that trusted either one would be wrong about the other"
            )
        self._completion_matches_protocol()
        if self.context is not None:
            context = self.context
            if (
                context.stream_generation is not None
                and context.stream_generation != self.stream_generation
            ):
                raise ValueError("context and envelope generations disagree")
            if self.target.kind == "training-attempt":
                if (
                    context.evaluation_attempt_id is not None
                    or context.evaluation_run_id is not None
                ):
                    raise ValueError("training telemetry cannot claim evaluation context")
                context_id: str | None = context.attempt_id
            else:
                if context.attempt_id is not None or context.run_id is not None:
                    raise ValueError("evaluation telemetry cannot claim training context")
                context_id = context.evaluation_attempt_id
            if context_id is not None and context_id != self.target.id:
                raise ValueError("context attempt and envelope target disagree")
        return self

    def _completion_matches_protocol(self) -> None:
        """``EvaluationCompleted`` carries its metrics exactly when v1alpha3 says so."""
        data = self.payload.data
        if not isinstance(data, EvaluationCompletedPayload):
            return
        if self.protocol_version == TELEMETRY_V1ALPHA3 and data.metrics is None:
            raise ValueError(
                "an EvaluationCompleted under xaytune.telemetry/v1alpha3 must carry its "
                "metrics: they are the evaluation's result, and a completion without "
                "them would leave a successful evaluation that measured nothing"
            )
        if self.protocol_version != TELEMETRY_V1ALPHA3 and data.metrics is not None:
            raise ValueError(
                f"an EvaluationCompleted under {self.protocol_version} carries no metrics; "
                f"inline results are xaytune.telemetry/v1alpha3, and the same version "
                f"with a different payload would not be the same protocol"
            )


class RuntimeLog(FrozenDomainModel):
    """Human/debugging output. Never infer state transitions from log lines."""

    stream: Literal["stdout", "stderr"] = "stdout"
    line: str
    emitted_at: datetime | None = None
    worker: str | None = None
    level: Literal["trace", "debug", "info", "warning", "error", "critical"] | None = None
    logger: str | None = None
    rank: Counter | None = None
    trace_context: TraceContext | None = None
    context: CorrelationContext | None = None
    attributes: FrozenDict = Field(default_factory=FrozenDict)


@runtime_checkable
class RuntimeBackend(Protocol):
    """Executes plans and reports what happens. Interprets nothing.

    Every mutating method is keyed by ``operation_id``, from the first
    implementation onwards. Retrofitting idempotency onto a runtime API is not
    a refactor -- until it exists, a controller that crashes between submitting
    and recording the reference cannot tell a lost submission from a running
    workload, and either answer can be catastrophic.
    """

    descriptor: PluginDescriptor

    def capabilities(self) -> CapabilityDocument:
        """What this backend can execute, and what it can report.

        Includes the honest negatives: whether it can replay telemetry from a
        cursor, and whether it can report completed operations. Both change
        what the controller is allowed to conclude.
        """
        ...

    async def submit_or_get(
        self, operation_id: OperationId, plan: ResolvedExecutionPlan
    ) -> RuntimeRef:
        """Start the plan, or return the workload this operation already started.

        **Get-or-create, never create** (ADR-013 §2). Re-submitting the same
        ``operation_id`` returns the original workload and never starts a
        second one; the same id with a materially different request is a
        conflict rather than a new submission.

        ``operation_id`` is the first parameter because it is the identity of
        the operation, not a tag on it. The same method submits an evaluation
        attempt -- the operation's target is typed, so a backend needs no
        training-specific knowledge.
        """
        ...

    async def lookup_operation(self, operation_id: OperationId) -> OperationOutcome | None:
        """Answer what became of an operation. ``None`` means never received."""
        ...

    async def get_status(self, runtime_ref: RuntimeRef) -> RuntimeStatus:
        """Observe a workload. Authoritative in a way the event stream is not."""
        ...

    def watch(
        self, runtime_ref: RuntimeRef, cursor: StreamCursor | None = None
    ) -> AsyncIterator[RuntimeEventEnvelope]:
        """Stream telemetry after *cursor*, in increasing order.

        Ordered by ``(generation, sequence)``. Without that guarantee a
        reordered arrival is indistinguishable from a real gap, so the adapter
        buffers out-of-order transport rather than passing it through.

        Delivery is at-least-once; handlers must be idempotent on
        ``(target, generation, sequence)``. A backend that cannot replay
        declares ``supports_event_replay: false``, and reconnects are then
        treated as gaps rather than assumed to be quiet.
        """
        ...

    async def cancel(self, runtime_ref: RuntimeRef, operation_id: OperationId) -> None:
        """Ask for a workload to stop.

        Keyed by ``operation_id`` like every other effect, because a
        cancellation is itself an external effect that can be retried, lost or
        raced. Asking twice must not mean two cancellations.

        Cancellation is a request, not a state: whether the workload actually
        stops is observed through :meth:`get_status`, and one that finishes
        first simply means the cancel arrived too late (ADR-013 §5).
        """
        ...

    def get_logs(self, runtime_ref: RuntimeRef) -> AsyncIterator[RuntimeLog]:
        """Stream worker output.

        Separate from :meth:`watch`: logs are for humans and carry no ordering
        contract, while telemetry is the record the controller acts on.
        """
        ...
