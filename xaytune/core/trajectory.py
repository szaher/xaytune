"""What one harness execution did, as evidence (H05, ADR-021).

```text
TrajectoryRecordingSpec   xaytune.agent-trajectory-recording/v1alpha1   in the spec's config
TrajectoryOpenRecord      xaytune.agent-trajectory-open/v1alpha1        persisted first
TrajectoryEvent           a closed union, one JSON line each             in bounded chunks
TrajectoryManifest        xaytune.agent-trajectory/v1alpha1             sealed once
```

A trajectory is **evidence of an execution, not proof that it can be
reproduced** (ADR-021). Everything here is pure: the records, their
identities, the canonical bytes of a chunk, the agreement between a
trajectory and its :class:`~xaytune.core.agent_execution.HarnessRunResult`,
and the replay comparison projection. Writing, storing, verifying bytes and
reconciling live in :mod:`xaytune.trajectory`.

**Content is never inline in an event.** Every message, reply, argument and
result is a :data:`ContentSlot`: what happened to it when it was captured
(captured, captured in the restricted tier, redacted, or never provided).
That disposition is immutable and fingerprinted. Whether the bytes are still
there today is a separate, observed fact that is never written here (T6).

**One trajectory per attempt instance.** ``trajectory_id`` is a minted
:class:`~xaytune.core.ids.AttemptInstanceId`, kept apart from the request
digest, the replicate and any durable attempt ordinal (T3). Sequences are
gapless from 0; event ids are derived from ``(trajectory_id, sequence)``;
every parent is earlier.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Final, Literal

from pydantic import ConfigDict, Field, StrictBool, TypeAdapter, model_validator

from xaytune.core.agent_execution import COMPLETIONS, HarnessRunResult, TerminationReason
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import AttemptInstanceId
from xaytune.core.immutable import FrozenDomainModel

__all__ = [
    "ABANDONMENTS",
    "REPLAY_PROJECTION_V1",
    "REPLAY_VERIFICATION_SCHEMA",
    "TRAJECTORY_EVENT_KINDS",
    "TRAJECTORY_OPEN_SCHEMA",
    "TRAJECTORY_RECORDING_SCHEMA",
    "TRAJECTORY_SCHEMA",
    "Abandonment",
    "CapturedContent",
    "ChunkRef",
    "ContentReference",
    "ContentSlot",
    "ContextChanged",
    "ErrorObserved",
    "ModelCallFinished",
    "ModelCallStarted",
    "NotProvidedContent",
    "RedactedContent",
    "ReplayDivergence",
    "ReplayLevel",
    "ReplayVerification",
    "RunStarted",
    "RunTerminated",
    "TierPolicy",
    "ToolCallFinished",
    "ToolCallStarted",
    "TrajectoryAttribution",
    "TrajectoryEvent",
    "TrajectoryManifest",
    "TrajectoryManifestBody",
    "TrajectoryOpenRecord",
    "TrajectoryRecordingSpec",
    "TrajectoryTask",
    "WriterIdentity",
    "chunk_bytes",
    "content_references",
    "event_id_for",
    "first_divergence",
    "parse_chunk",
    "parse_event",
    "replay_projection_v1",
    "result_disagreements",
    "seal",
    "structural_refusals",
    "trajectory_identity_v1",
    "validate_manifest",
]

TRAJECTORY_SCHEMA: Final = "xaytune.agent-trajectory/v1alpha1"
"""The manifest's schema, and the ``kind`` of a spec's trajectory output."""
TRAJECTORY_RECORDING_SCHEMA: Final = "xaytune.agent-trajectory-recording/v1alpha1"
TRAJECTORY_OPEN_SCHEMA: Final = "xaytune.agent-trajectory-open/v1alpha1"
REPLAY_PROJECTION_V1: Final = "xaytune.trajectory-replay-projection/v1"

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_CLASS = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,199}$")
_MEDIA = re.compile(r"^[a-z]+/[a-z0-9.+-]+(;\s*charset=utf-8)?$")

Digest = Annotated[str, Field(pattern=_DIGEST.pattern)]
Count = Annotated[int, Field(strict=True, ge=0)]
Sequence_ = Annotated[int, Field(strict=True, ge=0)]

ReplayLevel = Literal["none", "recorded-responses"]

Abandonment = Literal[
    "writer-lost",
    "recording-failed",
    "secret-detected",
    "result-unpublished",
    "result-published-by-another-attempt",
    "unverifiable-tail",
    "post-commit-evidence-lost",
]
ABANDONMENTS: Final = frozenset(Abandonment.__args__)  # type: ignore[attr-defined]


class _Closed(FrozenDomainModel):
    """Frozen, closed to unknown fields, and silent about refused input."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


# ---- the recording spec (T1, T9) --------------------------------------------------------------

_MIN_EVENTS, _MAX_EVENTS = 1, 10_000
_MIN_BYTES, _MAX_BYTES = 4 * 1024, 16 * 1024 * 1024


class WriterIdentity(_Closed):
    """Which trajectory writer recorded an attempt, and at what version."""

    name: str = Field(pattern=_NAME.pattern)
    version: str = Field(min_length=1, max_length=64)


class TierPolicy(_Closed):
    """Who may read a storage tier, and how long its blobs are kept.

    ``retention_seconds=None`` is ``indefinite``. Retention removes blobs only,
    through an authorized expiry record (T6).
    """

    retention_seconds: int | None = Field(default=None, strict=True, gt=0)
    readers: Literal["experiment", "owner"]


class TrajectoryRecordingSpec(_Closed):
    """How an attempt is recorded: operational, so it enters execution identity only (T9).

    Carried in a spec's ``config["trajectory"]``. Bounds are validated against
    fixed ranges: ``max_events`` 1–10,000, ``max_bytes`` 4 KiB–16 MiB (T1).
    """

    schema_version: Literal["xaytune.agent-trajectory-recording/v1alpha1"] = (
        "xaytune.agent-trajectory-recording/v1alpha1"
    )
    max_events: int = Field(default=1000, strict=True, ge=_MIN_EVENTS, le=_MAX_EVENTS)
    max_bytes: int = Field(default=1024 * 1024, strict=True, ge=_MIN_BYTES, le=_MAX_BYTES)
    replay: ReplayLevel = "recorded-responses"
    redaction_policy_digest: str | None = Field(default=None, pattern=_DIGEST.pattern)
    ordinary: TierPolicy = TierPolicy(readers="experiment")
    restricted: TierPolicy = TierPolicy(readers="owner")
    writer: WriterIdentity

    def digest(self) -> str:
        """The digest every open record and manifest names this spec by."""
        return fingerprint(
            {"type": "agent-trajectory-recording", "spec": self.model_dump(mode="json")}
        )


# ---- content: capture disposition (T6) --------------------------------------------------------


class CapturedContent(_Closed):
    """Stored at capture time, in the ordinary or the restricted tier, with its digest."""

    disposition: Literal["captured", "captured-restricted"]
    digest: Digest
    size: Count
    media_type: str = Field(pattern=_MEDIA.pattern)

    @property
    def tier(self) -> Literal["ordinary", "restricted"]:
        return "ordinary" if self.disposition == "captured" else "restricted"


class RedactedContent(_Closed):
    """Removed by a policy rule before it was written. A secret leaves no digest (T8)."""

    disposition: Literal["redacted"] = "redacted"
    rule: str = Field(min_length=1, max_length=200)
    secret: StrictBool
    removed_digest: str | None = Field(default=None, pattern=_DIGEST.pattern)

    @model_validator(mode="after")
    def _no_secret_digest(self) -> RedactedContent:
        if self.secret and self.removed_digest is not None:
            raise ValueError("a redacted secret leaves no digest: one can be guessed offline")
        return self


class NotProvidedContent(_Closed):
    """The producer never had it."""

    disposition: Literal["not-provided"] = "not-provided"


ContentSlot = Annotated[
    CapturedContent | RedactedContent | NotProvidedContent, Field(discriminator="disposition")
]


# ---- attribution (T3) -------------------------------------------------------------------------


class TrajectoryTask(_Closed):
    """The task an attempt ran: H03's task id, and H04's pin when the spec carries one."""

    task_id: str = Field(min_length=1, max_length=200)
    revision: str | None = None
    task_fingerprint: str | None = Field(default=None, pattern=_DIGEST.pattern)


class TrajectoryAttribution(_Closed):
    """What an attempt instance executed. Recorded beside its identity, never part of it.

    The request digest is shared by every execution of a spec; the seed and
    replicate by every attempt at one sample; a durable attempt ordinal exists
    only once the durable path does. None of them is the attempt.
    """

    request_digest: Digest
    execution_fingerprint: Digest
    harness_fingerprint: Digest
    task: TrajectoryTask
    seed: int = Field(strict=True)
    replicate: Count
    assignment_cohort: str | None = Field(default=None, pattern=_DIGEST.pattern)
    experiment_id: str | None = None
    node_id: str | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    attempt_ordinal: int | None = Field(default=None, strict=True, ge=0)


class TrajectoryOpenRecord(_Closed):
    """The first thing persisted for an attempt instance, before any call (T4).

    It records how many secret values were registered, and by which sources,
    and never the values or their digests (T8).
    """

    schema_version: Literal["xaytune.agent-trajectory-open/v1alpha1"] = (
        "xaytune.agent-trajectory-open/v1alpha1"
    )
    trajectory_id: AttemptInstanceId
    attribution: TrajectoryAttribution
    writer: WriterIdentity
    runner: WriterIdentity
    recording: TrajectoryRecordingSpec
    secrets_registered: Count
    secret_sources: tuple[str, ...] = ()
    opened_at: str | None = None


# ---- events (T2) ------------------------------------------------------------------------------


def event_id_for(trajectory_id: str, sequence: int) -> str:
    """The stable id of the event at *sequence*: a redelivery has the same id (T3)."""
    digest = hashlib.sha256(f"{trajectory_id}\x00{sequence}".encode()).hexdigest()
    return f"tev_{digest[:40]}"


class _Event(_Closed):
    trajectory_id: AttemptInstanceId
    sequence: Sequence_
    event_id: str
    parent: int | None = Field(default=None, strict=True, ge=0)
    emitted_at: str | None = None
    """Advisory only: ``sequence`` is the order."""

    @model_validator(mode="after")
    def _identity(self) -> _Event:
        if self.event_id != event_id_for(self.trajectory_id, self.sequence):
            raise ValueError("event_id is not derived from (trajectory_id, sequence)")
        if self.parent is not None and self.parent >= self.sequence:
            raise ValueError("a parent is always an earlier event")
        return self


class RunStarted(_Event):
    kind: Literal["run-started"] = "run-started"
    attribution: TrajectoryAttribution
    runner: WriterIdentity
    writer: WriterIdentity
    effective_tools: tuple[str, ...]
    configuration_digest: Digest
    """The harness's context, stopping and generation settings, by digest."""

    @model_validator(mode="after")
    def _first(self) -> RunStarted:
        if self.sequence != 0 or self.parent is not None:
            raise ValueError("run-started is sequence 0, with no parent")
        return self


class ModelCallStarted(_Event):
    """Durable before the call it announces is made (T4)."""

    kind: Literal["model-call-started"] = "model-call-started"
    turn: Count
    request: ContentSlot
    messages_digest: Digest
    tools_digest: Digest
    context_rule: Literal["full-history", "recent-turns"]
    messages_shown: Count


class ModelCallFinished(_Event):
    kind: Literal["model-call-finished"] = "model-call-finished"
    outcome: Literal["replied", "failed", "cancelled"]
    reply: ContentSlot | None = None
    error_class: str | None = Field(default=None, pattern=_CLASS.pattern)
    prompt_tokens: Count = 0
    output_tokens: Count = 0
    tool_calls_requested: tuple[str, ...] = ()
    output_truncated: StrictBool = False
    latency_ms: int | None = Field(default=None, strict=True, ge=0)
    """Advisory only; never compared."""

    @model_validator(mode="after")
    def _outcome(self) -> ModelCallFinished:
        if (self.reply is not None) != (self.outcome == "replied"):
            raise ValueError("a reply is recorded exactly when the model replied")
        if (self.error_class is not None) != (self.outcome == "failed"):
            raise ValueError("an error class is recorded exactly when the call failed")
        if self.outcome != "replied" and (
            self.prompt_tokens or self.output_tokens or self.tool_calls_requested
        ):
            raise ValueError("only a reply counts tokens or requests tool calls")
        return self


class ToolCallStarted(_Event):
    """Durable before the call it announces is made, refused calls included (T4)."""

    kind: Literal["tool-call-started"] = "tool-call-started"
    call_id: str = Field(min_length=1, max_length=200)
    tool: str = Field(min_length=1, max_length=200)
    version: str | None = None
    arguments: ContentSlot


ToolRefusal = Literal["not-enabled", "invalid-arguments"]
ToolFailure = Literal["tool-raised", "non-text-result"]


class ToolCallFinished(_Event):
    kind: Literal["tool-call-finished"] = "tool-call-finished"
    outcome: Literal["executed", "refused", "failed"]
    reason: ToolRefusal | ToolFailure | None = None
    output: ContentSlot | None = None
    """The tool's own output: what a recorded-responses re-drive serves."""
    shown: ContentSlot
    """What the model was shown."""
    truncated: StrictBool = False
    error_class: str | None = Field(default=None, pattern=_CLASS.pattern)
    """The exception class a tool raised: a class name, never its message."""

    @model_validator(mode="after")
    def _outcome(self) -> ToolCallFinished:
        if (self.output is not None) != (self.outcome == "executed"):
            raise ValueError("a tool output is recorded exactly when the tool executed")
        refusal = self.reason in ("not-enabled", "invalid-arguments")
        failure = self.reason in ("tool-raised", "non-text-result")
        if self.outcome == "executed" and self.reason is not None:
            raise ValueError("an executed tool call has no refusal or failure reason")
        if self.outcome == "refused" and not refusal:
            raise ValueError("a refused tool call names its refusal")
        if self.outcome == "failed" and not failure:
            raise ValueError("a failed tool call names its failure")
        if (self.error_class is not None) != (self.reason == "tool-raised"):
            raise ValueError("an error class is recorded exactly when the tool raised")
        return self


class ContextChanged(_Event):
    """Turns left out of what the next model call is shown, and by which rule."""

    kind: Literal["context-changed"] = "context-changed"
    rule: Literal["recent-turns", "truncate-oldest"]
    turns_dropped: int = Field(strict=True, ge=1)
    before_digest: Digest
    after_digest: Digest


class ErrorObserved(_Event):
    """A sanitized error: a code and a field role, never a body or an input value."""

    kind: Literal["error-observed"] = "error-observed"
    code: Literal["secret-detected", "recording-failed"]
    role: str | None = Field(default=None, max_length=200)


class RunTerminated(_Event):
    """How the run ended, and the digest of the result it is about to publish (T4)."""

    kind: Literal["run-terminated"] = "run-terminated"
    termination: TerminationReason
    completed: StrictBool
    answer: ContentSlot | None = None
    error_class: str | None = Field(default=None, pattern=_CLASS.pattern)
    result_digest: Digest

    @model_validator(mode="after")
    def _ending(self) -> RunTerminated:
        if self.completed != (self.termination in COMPLETIONS):
            raise ValueError("completed holds exactly for a completing termination")
        if (self.answer is not None) != self.completed:
            raise ValueError("only a completed run has an answer, and every completed run has one")
        if (self.error_class is not None) != (self.termination == "model-error"):
            raise ValueError("only a model-error run records an error class, and it always does")
        return self


TrajectoryEvent = Annotated[
    RunStarted
    | ModelCallStarted
    | ModelCallFinished
    | ToolCallStarted
    | ToolCallFinished
    | ContextChanged
    | ErrorObserved
    | RunTerminated,
    Field(discriminator="kind"),
]
TRAJECTORY_EVENT_KINDS: Final = frozenset(
    {
        "run-started",
        "model-call-started",
        "model-call-finished",
        "tool-call-started",
        "tool-call-finished",
        "context-changed",
        "error-observed",
        "run-terminated",
    }
)

_EVENT: TypeAdapter[Any] = TypeAdapter(TrajectoryEvent)


def parse_event(data: Mapping[str, Any] | str | bytes) -> Any:
    """One event, from its JSON or its mapping. Unknown kinds and fields are refused."""
    if isinstance(data, (str, bytes)):
        return _EVENT.validate_json(data)
    return _EVENT.validate_python(data)


# ---- chunks: canonical bytes and the chain (T1, T5) ----------------------------------------


def _line(record: Mapping[str, Any]) -> bytes:
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def chunk_bytes(
    trajectory_id: str, index: int, previous: str | None, events: Sequence[Any]
) -> bytes:
    """The exact bytes of a chunk: a header naming the chunk before it, then one event per line.

    The header is what chains chunks: a chunk's digest covers the previous
    chunk's digest, so the last digest commits to the whole ordered history.
    """
    header = {"chunk": index, "previous": previous, "trajectory_id": trajectory_id}
    return b"".join([_line(header), *(_line(e.model_dump(mode="json")) for e in events)])


def parse_chunk(data: bytes) -> tuple[Mapping[str, Any], tuple[Any, ...]]:
    """A chunk's header and events. Raises ``ValueError`` if any line is not what it claims."""
    lines = data.split(b"\n")
    if not lines or lines[-1] != b"":
        raise ValueError("a chunk ends with a newline")
    header = json.loads(lines[0])
    if not isinstance(header, dict) or set(header) != {"chunk", "previous", "trajectory_id"}:
        raise ValueError("a chunk starts with its header")
    return header, tuple(parse_event(line) for line in lines[1:-1])


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class ChunkRef(_Closed):
    """One immutable chunk, by digest, and where it sits in the chain."""

    index: Count
    first_sequence: Sequence_
    last_sequence: Sequence_
    digest: Digest
    previous: str | None = Field(default=None, pattern=_DIGEST.pattern)
    bytes: int = Field(strict=True, gt=0)
    events: int = Field(strict=True, gt=0)

    @model_validator(mode="after")
    def _range(self) -> ChunkRef:
        if self.last_sequence < self.first_sequence:
            raise ValueError("a chunk's range runs forward")
        if self.events != self.last_sequence - self.first_sequence + 1:
            raise ValueError("a chunk holds exactly the events of its range")
        return self


# ---- the manifest (T4, T5, T6) ---------------------------------------------------------------


class ContentReference(_Closed):
    """One content slot an event names: the manifest's disposition table, row by row."""

    sequence: Sequence_
    field: str = Field(pattern=r"^[a-z_]{1,40}$")
    slot: ContentSlot


_CONTENT_FIELDS: Final = {
    "model-call-started": ("request",),
    "model-call-finished": ("reply",),
    "tool-call-started": ("arguments",),
    "tool-call-finished": ("output", "shown"),
    "run-terminated": ("answer",),
}


def content_references(events: Iterable[Any]) -> tuple[ContentReference, ...]:
    """Every content slot *events* name, in (sequence, field) order."""
    found: list[ContentReference] = []
    for event in events:
        for name in _CONTENT_FIELDS.get(event.kind, ()):
            slot = getattr(event, name)
            if slot is not None:
                found.append(ContentReference(sequence=event.sequence, field=name, slot=slot))
    return tuple(found)


class TrajectoryManifestBody(_Closed):
    """Everything a sealed manifest says except its fingerprint. Build one, then :func:`seal`."""

    schema_version: Literal["xaytune.agent-trajectory/v1alpha1"] = (
        "xaytune.agent-trajectory/v1alpha1"
    )
    trajectory_id: AttemptInstanceId
    attribution: TrajectoryAttribution
    writer: WriterIdentity
    runner: WriterIdentity
    recording: TrajectoryRecordingSpec
    chunks: tuple[ChunkRef, ...]
    content: tuple[ContentReference, ...]
    completeness: Literal["complete", "partial"]
    abandonment: Abandonment | None = None
    termination: TerminationReason | None = None
    result_digest: str | None = Field(default=None, pattern=_DIGEST.pattern)
    """The attested result: set exactly when complete."""
    unattested_result: str | None = Field(default=None, pattern=_DIGEST.pattern)
    """A published result this attempt can no longer attest (``post-commit-evidence-lost``)."""
    torn_tail_bytes: Count = 0
    sealed_at: str
    """When the manifest was sealed: the start of every retention period. Not identity."""

    @model_validator(mode="after")
    def _consistent(self) -> TrajectoryManifestBody:
        problems = list(_manifest_problems(self))
        if problems:
            raise ValueError("invalid trajectory manifest: " + "; ".join(problems))
        return self


class TrajectoryManifest(TrajectoryManifestBody):
    """A sealed trajectory: immutable, fingerprinted, and the only entry point to its chunks.

    Validation of a manifest proves its own structure only (``validate_manifest``,
    T5): that its bindings agree with each other, that its chunk ranges are
    contiguous and its declared chain links, and that its fingerprint is its
    own. It proves nothing about chunk bytes, events or the result.
    """

    trajectory_fingerprint: Digest

    @model_validator(mode="after")
    def _fingerprinted(self) -> TrajectoryManifest:
        if self.trajectory_fingerprint != fingerprint(trajectory_identity_v1(self)):
            raise ValueError("invalid trajectory manifest: the fingerprint is not its own")
        return self


def seal(body: TrajectoryManifestBody) -> TrajectoryManifest:
    """*body*, with the fingerprint of its v1 identity: a sealed manifest."""
    fields = {name: getattr(body, name) for name in TrajectoryManifestBody.model_fields}
    return TrajectoryManifest(
        **fields, trajectory_fingerprint=fingerprint(trajectory_identity_v1(body))
    )


def _manifest_problems(m: TrajectoryManifestBody) -> Iterable[str]:
    if m.completeness == "complete":
        if m.abandonment is not None:
            yield "a complete trajectory has no abandonment reason"
        if m.result_digest is None or m.termination is None:
            yield "a complete trajectory attests a result and its termination"
        if not m.chunks:
            yield "a complete trajectory has chunks"
    else:
        if m.abandonment is None:
            yield "a partial trajectory names its abandonment reason"
        if m.result_digest is not None:
            yield "a partial trajectory attests no result"
    if (m.unattested_result is not None) != (m.abandonment == "post-commit-evidence-lost"):
        yield "an unattested result is named exactly when post-commit evidence was lost"
    expected_next, previous = 0, None
    for position, chunk in enumerate(m.chunks):
        if chunk.index != position:
            yield f"chunk {position} is indexed {chunk.index}"
        if chunk.first_sequence != expected_next:
            yield f"chunk {position} does not continue the previous chunk's range"
        if chunk.previous != previous:
            yield f"chunk {position} does not name the previous chunk's digest"
        expected_next, previous = chunk.last_sequence + 1, chunk.digest
    seen: set[tuple[int, str]] = set()
    for reference in m.content:
        key = (reference.sequence, reference.field)
        if key in seen:
            yield f"content {reference.field} at sequence {reference.sequence} is listed twice"
        seen.add(key)
        if reference.sequence >= expected_next:
            yield f"content names sequence {reference.sequence}, beyond the recorded events"
    if [(r.sequence, r.field) for r in m.content] != sorted(seen):
        yield "the content table is in (sequence, field) order"


def trajectory_identity_v1(manifest: TrajectoryManifestBody) -> Mapping[str, Any]:
    """The frozen v1 projection a sealed trajectory is fingerprinted by (T5).

    Includes the attempt and its attribution, the recording spec by digest, the
    ordered chunk digests, every capture disposition, the completeness and the
    result bindings. Excludes locators, timestamps and current availability:
    retention never changes a sealed fingerprint (T6).
    """
    return {
        "type": "agent-trajectory",
        "version": 1,
        "trajectory_id": str(manifest.trajectory_id),
        "attribution": manifest.attribution.model_dump(mode="json"),
        "writer": manifest.writer.model_dump(mode="json"),
        "runner": manifest.runner.model_dump(mode="json"),
        "recording": manifest.recording.digest(),
        "chunks": [chunk.digest for chunk in manifest.chunks],
        "content": [reference.model_dump(mode="json") for reference in manifest.content],
        "completeness": manifest.completeness,
        "abandonment": manifest.abandonment,
        "termination": manifest.termination,
        "result_digest": manifest.result_digest,
        "unattested_result": manifest.unattested_result,
        "torn_tail_bytes": manifest.torn_tail_bytes,
    }


_MANIFEST: TypeAdapter[TrajectoryManifest] = TypeAdapter(TrajectoryManifest)


def validate_manifest(data: Mapping[str, Any] | str | bytes) -> TrajectoryManifest:
    """The manifest alone, validated: structure, bindings, declared chain, fingerprint (T5).

    Proves nothing about chunk bytes, events, blobs or the result: those need
    ``verify_trajectory`` and ``verify_publication`` against a store.

    Raises:
        pydantic.ValidationError: If the manifest is not well formed.
    """
    if isinstance(data, (str, bytes)):
        return _MANIFEST.validate_json(data)
    return _MANIFEST.validate_python(data)


# ---- structure of an event sequence (T3) ---------------------------------------------------

_PARENT_KINDS: Final[Mapping[str, frozenset[str]]] = {
    "model-call-started": frozenset(
        {"run-started", "model-call-finished", "tool-call-finished", "context-changed"}
    ),
    "model-call-finished": frozenset({"model-call-started"}),
    "tool-call-started": frozenset({"model-call-finished"}),
    "tool-call-finished": frozenset({"tool-call-started"}),
    "context-changed": frozenset(
        {"run-started", "model-call-finished", "tool-call-finished", "context-changed"}
    ),
    "error-observed": TRAJECTORY_EVENT_KINDS - {"run-terminated"},
    "run-terminated": TRAJECTORY_EVENT_KINDS - {"run-terminated"},
}


def structural_refusals(trajectory_id: str, events: Sequence[Any]) -> tuple[str, ...]:
    """Every way *events* fail to be one attempt's ordered, causal history. ``()`` if none.

    Checks the trajectory id; gapless sequences from 0; that ``run-started``
    comes first and only first; that each parent exists, is earlier and is of
    an allowed kind; that each start is finished at most once; that a tool call
    is one its model turn requested; and that nothing follows
    ``run-terminated``.
    """
    problems: list[str] = []
    finished: set[int] = set()
    for position, event in enumerate(events):
        if event.trajectory_id != trajectory_id:
            problems.append(f"event {position} belongs to another trajectory")
        if event.sequence != position:
            problems.append(f"sequence gap: expected {position}, found {event.sequence}")
            break
        if position == 0:
            if event.kind != "run-started":
                problems.append("the first event is run-started")
            continue
        if event.kind == "run-started":
            problems.append(f"run-started appears again at {position}")
            continue
        if events[position - 1].kind == "run-terminated":
            problems.append(f"event {position} follows run-terminated")
        if event.parent is None or event.parent >= position:
            problems.append(f"event {position} names no earlier parent")
            continue
        parent = events[event.parent]
        if parent.kind not in _PARENT_KINDS[event.kind]:
            problems.append(f"event {position} ({event.kind}) cannot follow a {parent.kind}")
            continue
        if event.kind in ("model-call-finished", "tool-call-finished"):
            if event.parent in finished:
                problems.append(f"event {position} finishes a call that was already finished")
            finished.add(event.parent)
        if event.kind == "tool-call-started" and event.call_id not in parent.tool_calls_requested:
            problems.append(f"event {position} starts a tool call its model turn did not request")
    return tuple(problems)


# ---- agreement with the result (T5) --------------------------------------------------------


def _counts(events: Sequence[Any]) -> dict[str, Any]:
    replies = [e for e in events if e.kind == "model-call-finished" and e.outcome == "replied"]
    finishes = [e for e in events if e.kind == "tool-call-finished"]
    terminated = next((e for e in events if e.kind == "run-terminated"), None)
    return {
        "turns": len(replies),
        "tool_calls": sum(1 for e in events if e.kind == "tool-call-started"),
        "refused_tool_calls": sum(1 for e in finishes if e.outcome == "refused"),
        "failed_tool_calls": sum(1 for e in finishes if e.outcome == "failed"),
        "context_truncations": sum(
            e.turns_dropped
            for e in events
            if e.kind == "context-changed" and e.rule == "truncate-oldest"
        ),
        "prompt_tokens": sum(e.prompt_tokens for e in replies),
        "output_tokens": sum(e.output_tokens for e in replies),
        "termination": terminated.termination if terminated else None,
        "completed": terminated.completed if terminated else None,
        "error": terminated.error_class if terminated else None,
    }


def result_disagreements(events: Sequence[Any], result: HarnessRunResult) -> tuple[str, ...]:
    """Every field on which *events* and *result* disagree. ``()`` when they are one run's.

    The answer is compared by digest: the events hold its slot, the result its
    text.
    """
    counts = _counts(events)
    problems = [
        f"{name}: the events say {value!r}, the result {getattr(result, name)!r}"
        for name, value in counts.items()
        if getattr(result, name) != value
    ]
    if result.total_tokens != counts["prompt_tokens"] + counts["output_tokens"]:
        problems.append("total_tokens: not the events' prompt plus output tokens")
    terminated = next((e for e in events if e.kind == "run-terminated"), None)
    answer = terminated.answer if terminated else None
    if result.answer is None:
        if answer is not None:
            problems.append("answer: the events record one, the result has none")
    elif not isinstance(answer, CapturedContent):
        problems.append("answer: the result has one the events did not capture")
    elif answer.digest != _digest(result.answer.encode("utf-8")):
        problems.append("answer: its digest differs from the captured answer")
    return tuple(problems)


# ---- the replay comparison projection (T7) -------------------------------------------------


def _slot_digest(slot: Any) -> str | None:
    return slot.digest if isinstance(slot, CapturedContent) else None


def replay_projection_v1(events: Iterable[Any]) -> tuple[Mapping[str, Any], ...]:
    """The fields of each event that state a harness decision (``REPLAY_PROJECTION_V1``).

    Compared: sequence, kind and parent; what each model call was shown; each
    tool call's id, version, arguments and outcome; each context change; how the
    run ended; and tokens, under one counter. Not compared: event and attempt
    ids, timestamps and latency, writer identity, chunk boundaries, tiers,
    retention, current availability -- and the recorded responses themselves,
    which a re-drive is served rather than produces.
    """
    projected: list[Mapping[str, Any]] = []
    for e in events:
        row: dict[str, Any] = {"sequence": e.sequence, "kind": e.kind, "parent": e.parent}
        if e.kind == "run-started":
            row |= {
                "effective_tools": list(e.effective_tools),
                "configuration_digest": e.configuration_digest,
            }
        elif e.kind == "model-call-started":
            row |= {
                "turn": e.turn,
                "messages_digest": e.messages_digest,
                "tools_digest": e.tools_digest,
                "context_rule": e.context_rule,
                "messages_shown": e.messages_shown,
            }
        elif e.kind == "model-call-finished":
            row |= {
                "outcome": e.outcome,
                "error_class": e.error_class,
                "prompt_tokens": e.prompt_tokens,
                "output_tokens": e.output_tokens,
                "tool_calls_requested": list(e.tool_calls_requested),
                "output_truncated": e.output_truncated,
            }
        elif e.kind == "tool-call-started":
            row |= {
                "call_id": e.call_id,
                "tool": e.tool,
                "version": e.version,
                "arguments": _slot_digest(e.arguments),
            }
        elif e.kind == "tool-call-finished":
            row |= {
                "outcome": e.outcome,
                "reason": e.reason,
                "error_class": e.error_class,
                "shown": _slot_digest(e.shown),
                "truncated": e.truncated,
            }
        elif e.kind == "context-changed":
            row |= {
                "rule": e.rule,
                "turns_dropped": e.turns_dropped,
                "before_digest": e.before_digest,
                "after_digest": e.after_digest,
            }
        elif e.kind == "error-observed":
            row |= {"code": e.code, "role": e.role}
        elif e.kind == "run-terminated":
            row |= {
                "termination": e.termination,
                "completed": e.completed,
                "answer": _slot_digest(e.answer),
                "error_class": e.error_class,
            }
        projected.append(row)
    return tuple(projected)


REPLAY_VERIFICATION_SCHEMA: Final = "xaytune.trajectory-replay-verification/v1alpha1"


class ReplayDivergence(_Closed):
    """Where a re-drive first stated a different decision: an event and a projected field."""

    sequence: Count
    field: str = Field(min_length=1, max_length=60)


class ReplayVerification(_Closed):
    """An independent verifier's account of re-driving a sealed trajectory (T7).

    A separate artifact: a manifest has no verification field, the writer has
    no API that makes one, and a verifier that is the trajectory's own writer is
    refused when it is made.
    """

    schema_version: Literal["xaytune.trajectory-replay-verification/v1alpha1"] = (
        "xaytune.trajectory-replay-verification/v1alpha1"
    )
    verifier: WriterIdentity
    projection: Literal["xaytune.trajectory-replay-projection/v1"] = (
        "xaytune.trajectory-replay-projection/v1"
    )
    trajectory_fingerprint: Digest
    runner: WriterIdentity
    outcome: Literal["equivalent", "diverged", "prefix-equivalent", "ineligible"]
    divergence: ReplayDivergence | None = None
    unmet: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _outcome(self) -> ReplayVerification:
        if (self.divergence is not None) != (self.outcome == "diverged"):
            raise ValueError("a divergence is named exactly when the re-drive diverged")
        if bool(self.unmet) != (self.outcome == "ineligible"):
            raise ValueError("unmet prerequisites are named exactly when ineligible")
        return self


def first_divergence(
    recorded: Sequence[Mapping[str, Any]], redriven: Sequence[Mapping[str, Any]]
) -> ReplayDivergence | None:
    """The first sequence and field where two replay projections differ, or ``None``."""
    for position in range(max(len(recorded), len(redriven))):
        if position >= len(recorded) or position >= len(redriven):
            return ReplayDivergence(sequence=position, field="kind")
        left, right = recorded[position], redriven[position]
        for name in ("kind", *sorted((set(left) | set(right)) - {"kind"})):
            if left.get(name) != right.get(name):
                return ReplayDivergence(sequence=position, field=name)
    return None
