"""Re-driving a recorded trajectory against its recorded responses (ADR-021 T7).

A re-drive runs the same spec's harness logic, but every model reply and
every tool result is **served from the trajectory**: no model and no host tool
is called. The re-driven events are compared with the recorded ones under the
versioned projection :data:`~xaytune.core.trajectory.REPLAY_PROJECTION_V1`.

What it shows, and only that: the harness logic reproduces its decisions
given the same observations. It is not evidence that a provider, a tool or
the world would answer the same again. Host tools carry no purity evidence,
so re-executing against live tools is never a replay level.

The result is a separate :class:`~xaytune.core.trajectory.ReplayVerification`
made by an independent verifier. The trajectory's writer cannot make one.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xaytune.core.agent_execution import AgentExecutionSpec, agent_run_request_digest
from xaytune.core.trajectory import (
    CapturedContent,
    ReplayVerification,
    TrajectoryManifest,
    WriterIdentity,
    first_divergence,
    replay_projection_v1,
)
from xaytune.harness.recording import EventRecorder, MemorySink
from xaytune.harness.runner import (
    RECORDING_RUNNER,
    HarnessModelReply,
    HarnessModelRequest,
    HarnessToolbox,
    _Run,
)
from xaytune.trajectory.sealing import read_manifest
from xaytune.trajectory.store import AttemptPaths, LocalTrajectoryStore
from xaytune.trajectory.verify import (
    check_availability,
    load_events,
    verify_publication,
    verify_trajectory,
)

__all__ = ["replay_prerequisites", "verify_replay"]


def replay_prerequisites(
    store: LocalTrajectoryStore,
    paths: AttemptPaths,
    manifest: TrajectoryManifest,
    spec: AgentExecutionSpec,
) -> tuple[str, ...]:
    """Every unmet prerequisite for re-driving *manifest* (T7). ``()`` if it may be re-driven.

    *manifest* must be the one sealed at the attempt's ``manifest.json``, never
    a staged one, and its evidence must verify. A complete trajectory's result
    must also be published at the spec's declared location by this attempt
    (``verify_publication``). A partial one attests no result, so it is
    re-driven as a prefix check without that.
    """
    unmet: list[str] = []
    if manifest.recording.replay != "recorded-responses":
        unmet.append("the trajectory was recorded at replay level 'none'")
    attributed = agent_run_request_digest(spec) == manifest.attribution.request_digest
    if not attributed:
        unmet.append("the spec is not the one the trajectory attributes")
    if manifest.runner != RECORDING_RUNNER:
        unmet.append("no runner here matches the recorded runner")
    if read_manifest(paths.manifest) != manifest:
        # Staging builds a complete manifest before the commit point; only the
        # one sealed at manifest.json is a trajectory's lifecycle completed (T4).
        unmet.append("the manifest is not the attempt's sealed manifest")
        return tuple(unmet)
    problems = verify_trajectory(store, paths, manifest)
    if problems:
        unmet.append("the trajectory does not verify")
        return tuple(unmet)
    if (
        attributed
        and manifest.completeness == "complete"
        and verify_publication(paths, manifest, Path(spec.outputs[0].uri))
    ):
        unmet.append("the attested result is not published by this attempt")
        return tuple(unmet)
    events, _ = load_events(store, paths, manifest)
    availability = check_availability(store, paths, manifest)
    for event in events:
        consumed = None
        if event.kind == "model-call-finished" and event.outcome == "replied":
            consumed = ("reply", event.reply)
        elif event.kind == "tool-call-finished" and event.outcome == "executed":
            consumed = ("output", event.output)
        if consumed is None:
            continue
        field, slot = consumed
        if not isinstance(slot, CapturedContent):
            unmet.append(f"sequence {event.sequence} {field} was {slot.disposition}")
            continue
        status = availability.get((slot.tier, slot.digest), "missing")
        if status != "available":
            unmet.append(f"sequence {event.sequence} {field} is {status}")
        elif slot.tier == "restricted" and not os.access(
            paths.blob(slot.tier, slot.digest), os.R_OK
        ):
            unmet.append(
                f"sequence {event.sequence} {field} is in a tier this verifier cannot read"
            )
    return tuple(unmet)


class _RecordedError(Exception):
    """Raised where the recording raised: its class name is the recorded one."""


def _raising(class_name: str) -> Exception:
    error: Exception = type(class_name, (_RecordedError,), {})()
    return error


class _RecordedModel:
    """Serves each recorded model outcome, by turn. Calls nothing."""

    def __init__(self, outcomes: Sequence[HarnessModelReply | str], digest: str) -> None:
        self._outcomes = tuple(outcomes)
        self._digest = digest

    @property
    def artifact_digest(self) -> str:
        """The recorded harness's pinned model digest: the re-drive stands in for it."""
        return self._digest

    def respond(self, request: HarnessModelRequest, *, turn: int) -> HarnessModelReply:
        if turn >= len(self._outcomes):
            raise _raising("ReplayExhausted")
        outcome = self._outcomes[turn]
        if isinstance(outcome, str):
            raise _raising(outcome)
        return outcome


class _RecordedTools:
    """Serves each recorded tool outcome in call order, whatever the tool. Calls nothing."""

    def __init__(self, outcomes: Sequence[tuple[str, Any]]) -> None:
        self._outcomes = list(outcomes)

    def toolbox(self, spec: AgentExecutionSpec) -> HarnessToolbox:
        enabled = set(spec.harness.tools.enabled)
        return HarnessToolbox(
            {
                (tool.id, tool.version): self._serve
                for tool in spec.harness.tools.definitions
                if tool.id in enabled
            }
        )

    def _serve(self, arguments: Mapping[str, Any]) -> Any:
        if not self._outcomes:
            raise _raising("ReplayExhausted")
        kind, value = self._outcomes.pop(0)
        if kind == "raise":
            raise _raising(value)
        return value


class _CancelAt:
    """Set exactly where the recording was cancelled: just before its last event."""

    def __init__(self, sink: MemorySink, recorded: Sequence[Any]) -> None:
        self._sink = sink
        last = recorded[-1] if recorded else None
        self._at = (
            len(recorded) - 1
            if last is not None
            and last.kind == "run-terminated"
            and last.termination == "cancelled"
            else None
        )

    def is_set(self) -> bool:
        return self._at is not None and len(self._sink.events) >= self._at


def _served(
    store: LocalTrajectoryStore, paths: AttemptPaths, events: Sequence[Any]
) -> tuple[list[HarnessModelReply | str], list[tuple[str, Any]]]:
    def blob(slot: CapturedContent) -> str:
        data = store.read_blob(paths, slot.tier, slot.digest)
        assert data is not None  # a prerequisite
        return data.decode("utf-8")

    model: list[HarnessModelReply | str] = []
    tools: list[tuple[str, Any]] = []
    for event in events:
        if event.kind == "model-call-finished":
            if event.outcome == "replied":
                model.append(HarnessModelReply.model_validate(json.loads(blob(event.reply))))
            elif event.outcome == "failed":
                model.append(event.error_class)
        elif event.kind == "tool-call-finished":
            if event.outcome == "executed":
                tools.append(("value", blob(event.output)))
            elif event.reason == "tool-raised":
                tools.append(("raise", event.error_class))
            elif event.reason == "non-text-result":
                tools.append(("value", None))
    return model, tools


def verify_replay(
    store: LocalTrajectoryStore,
    paths: AttemptPaths,
    manifest: TrajectoryManifest,
    spec: AgentExecutionSpec,
    *,
    verifier: WriterIdentity,
) -> ReplayVerification:
    """Re-drive *manifest* against its recorded responses, and say what that showed.

    Raises:
        ValueError: If *verifier* is the trajectory's own writer: verification
            is independent of the writer (T7).
    """
    if verifier == manifest.writer:
        raise ValueError("a trajectory's writer cannot verify its own replay")

    def verdict(outcome: Any, **detail: Any) -> ReplayVerification:
        return ReplayVerification(
            verifier=verifier,
            trajectory_fingerprint=manifest.trajectory_fingerprint,
            runner=RECORDING_RUNNER,
            outcome=outcome,
            **detail,
        )

    unmet = replay_prerequisites(store, paths, manifest, spec)
    if unmet:
        return verdict(outcome="ineligible", unmet=unmet)
    recorded, _ = load_events(store, paths, manifest)
    model, tools = _served(store, paths, recorded)
    sink = MemorySink(str(manifest.trajectory_id))
    pinned = spec.harness.model.artifact_digest or ""
    run = _Run(spec, _RecordedModel(model, pinned), _RecordedTools(tools).toolbox(spec), None)
    run.cancel = _CancelAt(sink, recorded)
    run.recorder = EventRecorder(sink)
    run.result(manifest.attribution.request_digest)
    right = replay_projection_v1(sink.events)
    if manifest.completeness == "partial":
        # A partial trajectory may end with the writer's own sanitized error:
        # a note about recording, not a harness decision a re-drive makes.
        decided = list(recorded)
        while decided and decided[-1].kind == "error-observed":
            decided.pop()
        left = replay_projection_v1(decided)
        divergence = first_divergence(left, right[: len(left)])
        if divergence is None:
            return verdict(outcome="prefix-equivalent")
        return verdict(outcome="diverged", divergence=divergence)
    divergence = first_divergence(replay_projection_v1(recorded), right)
    if divergence is None:
        return verdict(outcome="equivalent")
    return verdict(outcome="diverged", divergence=divergence)
