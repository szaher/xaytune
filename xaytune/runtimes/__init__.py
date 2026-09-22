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
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field

from xaytune.core.capabilities import CapabilityDocument, PluginDescriptor
from xaytune.core.execution import ResolvedExecutionPlan
from xaytune.core.ids import OperationId
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.refs import RuntimeRef

__all__ = [
    "OperationOutcome",
    "RuntimeBackend",
    "RuntimeEventEnvelope",
    "RuntimeLog",
    "RuntimeState",
    "RuntimeStatus",
    "StreamCursor",
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


class OperationOutcome(FrozenDomainModel):
    """What became of an operation (ADR-013 §3).

    The answer a controller needs after a restart, and it must be able to
    express three different truths:

    ```text
    the runtime has the workload, running       -> adopt it
    the runtime never received it               -> safe to re-issue
    it ran and already finished                 -> adopt the result
    ```

    A lookup that can only say "running" or "not found" collapses the last two
    into one, and the controller will cheerfully re-run a completed workload.
    So ``completed`` is distinct from a missing outcome, and a backend that
    cannot retain finished operations declares
    ``reports_completed_operations: false`` rather than implying it can.
    """

    operation_id: OperationId
    accepted: bool
    completed: bool = False
    runtime_ref: RuntimeRef | None = None
    status: RuntimeStatus | None = None
    detail: str | None = None


class StreamCursor(FrozenDomainModel):
    """A position in one target's telemetry stream (ADR-014 §4).

    Two integers with defined meaning, never an opaque provider token: a
    backend cannot smuggle its own pagination state through it, which is what
    a string cursor invited.

    It carries the generation because the sequence alone stopped being unique
    within a target once a supervisor could be replaced mid-attempt.
    """

    generation: int = 0
    sequence: int = -1
    """The last sequence the controller **durably recorded**.

    Not the last it received. An event received and then lost in a crash must
    be redelivered, and ``-1`` means nothing has been recorded yet.
    """


class RuntimeEventEnvelope(FrozenDomainModel):
    """One telemetry event, workload-neutral (ADR-014 §1).

    The envelope carries identity and ordering; the payload carries meaning.
    Training and evaluation share the transport and not the vocabulary, which
    is what lets one protocol serve both without either pretending to be the
    other.
    """

    protocol_version: str = "xaytune.telemetry/v1alpha1"
    event_id: str

    target_kind: str
    target_id: str

    stream_generation: int = 0
    sequence: int

    emitted_at: datetime | None = None
    type: str
    payload: FrozenDict = Field(default_factory=FrozenDict)


class RuntimeLog(FrozenDomainModel):
    """A line of worker output."""

    stream: Literal["stdout", "stderr"] = "stdout"
    line: str
    emitted_at: datetime | None = None
    worker: str | None = None


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
