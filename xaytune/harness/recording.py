"""Turning a harness run's decisions into trajectory events (ADR-021 T2).

The runner calls an :class:`EventRecorder` at every decision point. The
recorder builds the closed event union, chooses each parent, and captures
every piece of content as a slot, through a sink:

- the persisting :class:`~xaytune.trajectory.writer.TrajectoryWriter`, during a
  recorded run, where every start is durable before its call;
- a :class:`MemorySink`, during a re-drive, which keeps events in memory and
  captures content by digest only.

Both produce events the same way, which is what makes the replay comparison
meaningful.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import AttemptInstanceId
from xaytune.core.trajectory import (
    CapturedContent,
    ContentSlot,
    ContextChanged,
    ModelCallFinished,
    ModelCallStarted,
    RunStarted,
    RunTerminated,
    ToolCallFinished,
    ToolCallStarted,
    TrajectoryAttribution,
    WriterIdentity,
    event_id_for,
)
from xaytune.trajectory._files import digest_of

__all__ = ["EventRecorder", "EventSink", "MemorySink", "canonical_json", "messages_digest"]

JSON = "application/json"
TEXT = "text/plain; charset=utf-8"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def messages_digest(messages: Sequence[Any]) -> str:
    """The digest of a message list, as a context change and a model call name it."""
    return fingerprint(
        {"type": "harness-messages", "messages": [m.model_dump(mode="json") for m in messages]}
    )


class EventSink(Protocol):
    """Where events and content go: durably (the writer) or in memory (a re-drive)."""

    def content(
        self, text: str, *, role: str, media_type: str = ..., sensitive: bool = ...
    ) -> ContentSlot: ...

    def record(self, event_type: type[Any], *, parent: int | None, **fields: Any) -> Any: ...


class MemorySink:
    """Events in memory, content by digest only: what a re-drive produces."""

    def __init__(self, trajectory_id: str | None = None) -> None:
        self.trajectory_id = AttemptInstanceId(trajectory_id or AttemptInstanceId.generate())
        self.events: list[Any] = []

    def content(
        self, text: str, *, role: str, media_type: str = TEXT, sensitive: bool = False
    ) -> ContentSlot:
        data = text.encode("utf-8")
        return CapturedContent(
            disposition="captured-restricted" if sensitive else "captured",
            digest=digest_of(data),
            size=len(data),
            media_type=media_type,
        )

    def record(self, event_type: type[Any], *, parent: int | None, **fields: Any) -> Any:
        sequence = len(self.events)
        event = event_type(
            trajectory_id=self.trajectory_id,
            sequence=sequence,
            event_id=event_id_for(self.trajectory_id, sequence),
            parent=parent,
            **fields,
        )
        self.events.append(event)
        return event


class EventRecorder:
    """Builds one run's events in order, choosing each parent, through a sink."""

    def __init__(self, sink: EventSink) -> None:
        self.sink = sink
        self._last: int | None = None

    def _record(self, event_type: type[Any], *, parent: int | None, **fields: Any) -> int:
        event = self.sink.record(event_type, parent=parent, **fields)
        self._last = event.sequence
        return int(event.sequence)

    def run_started(
        self,
        *,
        attribution: TrajectoryAttribution,
        runner: WriterIdentity,
        writer: WriterIdentity,
        effective_tools: Sequence[str],
        configuration: Any,
    ) -> int:
        return self._record(
            RunStarted,
            parent=None,
            attribution=attribution,
            runner=runner,
            writer=writer,
            effective_tools=tuple(effective_tools),
            configuration_digest=fingerprint(
                {"type": "harness-run-configuration", "configuration": configuration}
            ),
        )

    def context_changed(self, *, rule: str, dropped: int, before: str, after: str) -> int:
        return self._record(
            ContextChanged,
            parent=self._last,
            rule=rule,
            turns_dropped=dropped,
            before_digest=before,
            after_digest=after,
        )

    def model_call_started(
        self, *, turn: int, request: Any, messages: Sequence[Any], context_rule: str
    ) -> int:
        slot = self.sink.content(
            canonical_json(request.model_dump(mode="json")),
            role="model-call.request",
            media_type=JSON,
        )
        return self._record(
            ModelCallStarted,
            parent=self._last,
            turn=turn,
            request=slot,
            messages_digest=messages_digest(messages),
            tools_digest=fingerprint(
                {
                    "type": "harness-tools",
                    "tools": [t.model_dump(mode="json") for t in request.tools],
                }
            ),
            context_rule=context_rule,
            messages_shown=len(messages),
        )

    def model_call_failed(self, start: int, *, error_class: str, latency_ms: int | None) -> int:
        return self._record(
            ModelCallFinished,
            parent=start,
            outcome="failed",
            error_class=error_class,
            latency_ms=latency_ms,
        )

    def model_call_replied(
        self,
        start: int,
        *,
        reply: Any,
        prompt_tokens: int,
        output_tokens: int,
        output_truncated: bool,
        latency_ms: int | None,
    ) -> int:
        slot = self.sink.content(
            canonical_json(reply.model_dump(mode="json")), role="model-call.reply", media_type=JSON
        )
        return self._record(
            ModelCallFinished,
            parent=start,
            outcome="replied",
            reply=slot,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            tool_calls_requested=tuple(call.call_id for call in reply.tool_calls),
            output_truncated=output_truncated,
            latency_ms=latency_ms,
        )

    def tool_call_started(
        self, requested_by: int, *, call: Any, version: str | None, arguments: Any
    ) -> int:
        slot = self.sink.content(
            canonical_json(arguments), role="tool-call.arguments", media_type=JSON
        )
        return self._record(
            ToolCallStarted,
            parent=requested_by,
            call_id=call.call_id,
            tool=call.tool,
            version=version,
            arguments=slot,
        )

    def tool_call_finished(
        self,
        start: int,
        *,
        outcome: str,
        reason: str | None,
        output: str | None,
        shown: str,
        truncated: bool,
        error_class: str | None,
    ) -> int:
        output_slot = (
            self.sink.content(output, role="tool-call.output") if output is not None else None
        )
        return self._record(
            ToolCallFinished,
            parent=start,
            outcome=outcome,
            reason=reason,
            output=output_slot,
            shown=self.sink.content(shown, role="tool-call.shown"),
            truncated=truncated,
            error_class=error_class,
        )

    def run_terminated(
        self,
        *,
        termination: str,
        completed: bool,
        answer: str | None,
        error_class: str | None,
        result_digest: str,
    ) -> int:
        slot = self.sink.content(answer, role="answer") if answer is not None else None
        return self._record(
            RunTerminated,
            parent=self._last,
            termination=termination,
            completed=completed,
            answer=slot,
            error_class=error_class,
            result_digest=result_digest,
        )
