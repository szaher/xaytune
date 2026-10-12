"""Recorded scripted runs, and ways to break them at a chosen point."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.test_harness.fixtures import SCRIPT, RecordingTools, context, harness, harness_record
from xaytune.core.agent_execution import AgentExecutionSpec, HarnessRunResult
from xaytune.core.domain.harness import AgentHarnessSpec
from xaytune.core.trajectory import TrajectoryManifest, TrajectoryRecordingSpec
from xaytune.harness import ReferenceHarnessCompiler, ScriptedHarnessModel, ScriptedHarnessRunner
from xaytune.trajectory import WRITER, AttemptPaths, LocalTrajectoryStore
from xaytune.trajectory.sealing import read_manifest
from xaytune.trajectory.verify import load_events

AUTHORIZED = frozenset({"kb.search", "tickets.close"})


class SimulatedCrash(BaseException):
    """The process dies here: nothing after this point runs, not even an abandon.

    A real exit releases every lock the process held; :func:`die` does the same
    for the writers a test opened.
    """


OPENED: list[Any] = []


def track_writers(monkeypatch: Any) -> None:
    """Remember every writer opened, so a simulated crash can release their locks."""
    from xaytune.trajectory.writer import TrajectoryWriter

    original = TrajectoryWriter.open.__func__  # type: ignore[attr-defined]

    def opened(cls: Any, *args: Any, **kwargs: Any) -> Any:
        writer = original(cls, *args, **kwargs)
        OPENED.append(writer)
        return writer

    monkeypatch.setattr(TrajectoryWriter, "open", classmethod(opened))


def die() -> None:
    """What the operating system does when a process exits: every lock is released."""
    while OPENED:
        OPENED.pop().close()


def recording(**changes: Any) -> TrajectoryRecordingSpec:
    return TrajectoryRecordingSpec(writer=WRITER, **changes)


def recorded_spec(
    output: Path,
    script: Iterable[Mapping[str, Any]] = SCRIPT,
    *,
    record: Mapping[str, Any] | None = None,
    **changes: Any,
) -> AgentExecutionSpec:
    spec = AgentHarnessSpec.model_validate(record) if record is not None else harness(script)
    return ReferenceHarnessCompiler().compile(
        spec, context(spec, output, trajectory=recording(**changes))
    )


def harness_with(script: Iterable[Mapping[str, Any]] = SCRIPT, **sections: Any) -> dict[str, Any]:
    return harness_record(script, **sections)


def run(
    spec: AgentExecutionSpec,
    script: Iterable[Mapping[str, Any]] = SCRIPT,
    tools: RecordingTools | None = None,
    **options: Any,
) -> HarnessRunResult:
    options.setdefault("authorized_tools", AUTHORIZED)
    tools = tools or RecordingTools()
    return ScriptedHarnessRunner().run(
        spec, model=ScriptedHarnessModel(script), toolbox=tools.toolbox(), **options
    )


@dataclass
class Recorded:
    """One recorded attempt, opened from disk."""

    spec: AgentExecutionSpec
    store: LocalTrajectoryStore
    paths: AttemptPaths

    @property
    def location(self) -> Path:
        return Path(self.spec.outputs[0].uri)

    @property
    def manifest(self) -> TrajectoryManifest:
        manifest = read_manifest(self.paths.manifest)
        assert manifest is not None, "the attempt is not sealed"
        return manifest

    @property
    def events(self) -> tuple[Any, ...]:
        events, problems = load_events(self.store, self.paths, self.manifest)
        assert not problems, problems
        return events

    def tail_kinds(self) -> list[str]:
        data = self.store.read_tail(self.paths)
        return [json.loads(line)["kind"] for line in data.splitlines() if line.strip()]


def store_of(spec: AgentExecutionSpec) -> LocalTrajectoryStore:
    return LocalTrajectoryStore(Path(spec.outputs[1].uri))


def attempts(spec: AgentExecutionSpec) -> list[Recorded]:
    store = store_of(spec)
    return [Recorded(spec, store, paths) for paths in store.attempts()]


def only_attempt(spec: AgentExecutionSpec) -> Recorded:
    (attempt,) = attempts(spec)
    return attempt
