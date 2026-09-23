"""The worker → supervisor → event stream path, with a real worker process.

```text
worker            ObservationWriter        observations.jsonl   (transport)
launcher          AppendOnlyJsonlReader    one sequencer
                  RuntimeEventEnvelope     events.jsonl         (protocol)
```

Tested with a plain Python worker rather than NativeWorker, so this proves the
launcher's half on its own: whatever the trainer does later, the path its
observations travel is already pinned.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from xaytune.core.capabilities import PLUGIN_API_VERSIONS, PluginDescriptor
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.execution import (
    CommandEntrypoint,
    CompilerIdentity,
    ResolvedExecutionPlan,
    TrainingExecutionSpec,
)
from xaytune.core.ids import OperationId
from xaytune.runtimes.local import LocalRuntime
from xaytune.runtimes.local.paths import WorkloadPaths

_TERMINAL = frozenset({"succeeded", "failed", "cancelled", "unknown"})

_DESCRIPTOR = PluginDescriptor(
    api_version=PLUGIN_API_VERSIONS[0],
    name="test",
    plugin_version="0",
    provider="tests",
    xaytune_version="0.6.0",
)

_PRELUDE = (
    "import sys, time\n"
    "from xaytune.core import telemetry as t\n"
    "from xaytune.runtimes.worker import ObservationWriter\n"
    "writer = ObservationWriter.from_environment()\n"
)


def _plan(body: str, *, config: dict | None = None) -> ResolvedExecutionPlan:
    spec = TrainingExecutionSpec(
        compiler=CompilerIdentity(name="test", version="0", descriptor=_DESCRIPTOR),
        candidate_fingerprint="sha256:" + "0" * 64,
        entrypoint=CommandEntrypoint(argv=(sys.executable, "-c", _PRELUDE + body)),
        config=config or {},
    )
    return ResolvedExecutionPlan(
        spec=spec,
        runtime="local",
        target=RuntimeOperationTarget(kind="training-attempt", id="ra_transport"),
    )


async def _settle(runtime: LocalRuntime, ref: object) -> object:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 30.0
    status = await runtime.get_status(ref)  # type: ignore[arg-type]
    while status.state not in _TERMINAL and loop.time() < deadline:
        await asyncio.sleep(0.02)
        status = await runtime.get_status(ref)  # type: ignore[arg-type]
    return status


def _run(tmp_path: Path, body: str, **plan_kwargs: object) -> tuple[object, list, Path]:
    runtime = LocalRuntime(tmp_path / "runtime")

    async def scenario() -> tuple[object, list, Path]:
        ref = await runtime.submit_or_get(OperationId.generate(), _plan(body, **plan_kwargs))
        status = await _settle(runtime, ref)
        events = [event async for event in runtime.watch(ref)]
        workload = runtime._registry.workload(ref.external_id)
        assert workload is not None
        return status, events, workload.directory

    try:
        return asyncio.run(scenario())
    finally:
        runtime.close()


def _types(events: list) -> list[str]:
    return [event.payload.data.type for event in events]


def _incidents(events: list) -> list[str]:
    return [e.payload.data.reason for e in events if e.payload.data.type == "IncidentObserved"]


# ---- the path works, in order --------------------------------------------


def test_worker_observations_become_sequenced_envelopes(tmp_path: Path) -> None:
    """One sequencer, gapless, and WorkerReady first however it is scheduled."""
    status, events, _ = _run(
        tmp_path,
        "writer.write(t.TrainingStartedPayload())\n"
        "writer.write(t.TrainingMetricObserved(optimizer_step=1, loss=0.5))\n"
        "writer.write(t.TrainingCompletedPayload())\n",
    )

    assert status.state == "succeeded"
    assert _types(events) == [
        "WorkerReady",
        "TrainingStarted",
        "TrainingMetricObserved",
        "TrainingCompleted",
    ]
    assert [e.sequence for e in events] == list(range(len(events)))
    assert all(e.target.id == "ra_transport" for e in events)


def test_an_envelope_records_when_it_happened_not_when_it_was_read(tmp_path: Path) -> None:
    """``emitted_at`` is the worker's clock at the event, not the poll."""
    _, events, directory = _run(
        tmp_path,
        "writer.write(t.TrainingStartedPayload())\ntime.sleep(0.5)\n",
    )

    observed = [
        json.loads(line)["observed_at"]
        for line in WorkloadPaths(directory).observations.read_text().splitlines()
    ]
    started = next(e for e in events if e.payload.data.type == "TrainingStarted")

    assert started.emitted_at.isoformat().replace("+00:00", "Z") == observed[0]


def test_the_last_observation_before_exit_is_not_lost(tmp_path: Path) -> None:
    """The final drain, pinned.

    The worker reports completion and exits at once, after the supervisor has
    already polled and found nothing. Without a drain after exit, that last
    record is still in the file when ``finished.json`` is written -- a
    completed run recorded without its completion.
    """
    for _ in range(5):
        _, events, _ = _run(
            tmp_path / f"attempt-{_}",
            "time.sleep(0.3)\nwriter.write(t.TrainingCompletedPayload())\nimport os; os._exit(0)\n",
        )
        assert "TrainingCompleted" in _types(events), "the final observation was dropped"


def test_the_worker_receives_its_config(tmp_path: Path) -> None:
    """Nested, because every real config is.

    This used a flat dict and passed while the launcher crashed on any nested
    config -- ``dict()`` unfreezes only the top level. The first real compiler
    found it; this is the test that should have.
    """
    config = {"api_version": "anything", "optimization": {"lr": 0.001, "epochs": 1}}
    status, _, directory = _run(
        tmp_path,
        "import json, os\n"
        "cfg = json.load(open(os.environ['XAYTUNE_WORKER_CONFIG_PATH']))\n"
        "json.dump(cfg, open('received.json', 'w'))\n",
        config=config,
    )

    assert status.state == "succeeded"
    assert json.loads((directory / "received.json").read_text()) == config


# ---- process outcome and scientific outcome stay separate ----------------


def test_a_clean_exit_does_not_become_a_completed_training(tmp_path: Path) -> None:
    """Exit zero is an operating-system fact; completion is the worker's claim.

    A worker that exits cleanly without saying it finished training leaves a
    run that *succeeded as a process* and has *no evidence of completion*.
    Both have to remain representable, so the supervisor never supplies the
    second from the first.
    """
    status, events, _ = _run(tmp_path, "writer.write(t.TrainingStartedPayload())\n")

    assert status.state == "succeeded"
    assert "TrainingCompleted" not in _types(events)


# ---- the supervisor survives bad worker output ---------------------------


def test_a_corrupt_observation_is_reported_and_its_neighbours_survive(tmp_path: Path) -> None:
    """The worker is user code, and the supervisor must outlive its mistakes.

    If a garbage line killed the launcher, nothing would write
    ``finished.json`` and a finished run would report ``unknown`` forever.
    """
    status, events, _ = _run(
        tmp_path,
        "import os\n"
        "writer.write(t.TrainingStartedPayload())\n"
        "open(os.environ['XAYTUNE_OBSERVATIONS_PATH'], 'a').write('garbage\\n')\n"
        "writer.write(t.TrainingCompletedPayload())\n",
    )

    assert status.state == "succeeded", "the supervisor must survive to record the outcome"
    assert "TrainingStarted" in _types(events)
    assert "TrainingCompleted" in _types(events), "a neighbour of the bad line was lost"
    assert "corrupt-observation" in _incidents(events)


def test_an_evaluation_event_from_a_training_worker_is_refused(tmp_path: Path) -> None:
    """The family is pinned by the target, before anything is enveloped."""
    _, events, _ = _run(tmp_path, "writer.write(t.EvaluationStartedPayload())\n")

    assert "EvaluationStarted" not in _types(events)
    assert "corrupt-observation" in _incidents(events)


def test_an_observation_the_envelope_refuses_costs_no_sequence_number(tmp_path: Path) -> None:
    """A refused observation becomes an incident, and leaves no gap.

    Consuming a sequence number for an envelope that was never written would
    put a hole in the stream that a controller would read as lost telemetry.
    """
    _, events, _ = _run(
        tmp_path,
        "from xaytune.core.observability import CorrelationContext\n"
        "writer.write(t.TrainingStartedPayload(), "
        "context=CorrelationContext(stream_generation=99))\n"
        "writer.write(t.TrainingCompletedPayload())\n",
    )

    assert "invalid-observation" in _incidents(events)
    assert "TrainingCompleted" in _types(events)
    assert [e.sequence for e in events] == list(range(len(events))), "the stream has a gap"


def test_a_worker_that_dies_mid_write_is_reported(tmp_path: Path) -> None:
    """A partial last line is never delivered, and never silently discarded."""
    _, events, _ = _run(
        tmp_path,
        "import os\n"
        "open(os.environ['XAYTUNE_OBSERVATIONS_PATH'], 'a').write('{\"observed_at\": \"2026')\n"
        "os._exit(0)\n",
    )

    assert "truncated-observation" in _incidents(events)


@pytest.mark.parametrize("name", ["XAYTUNE_OBSERVATIONS_PATH", "XAYTUNE_WORKER_CONFIG_PATH"])
def test_the_plan_cannot_redirect_the_runtimes_channels(tmp_path: Path, name: str) -> None:
    """Where the config and observations live is the runtime's decision."""
    runtime = LocalRuntime(tmp_path / "runtime")
    out = tmp_path / "seen.txt"
    spec = TrainingExecutionSpec(
        compiler=CompilerIdentity(name="test", version="0", descriptor=_DESCRIPTOR),
        candidate_fingerprint="sha256:" + "0" * 64,
        entrypoint=CommandEntrypoint(
            argv=(
                sys.executable,
                "-c",
                f"import os, sys; open(sys.argv[1], 'w').write(os.environ[{name!r}])",
                str(out),
            )
        ),
        environment={name: "/tmp/elsewhere"},
    )
    plan = ResolvedExecutionPlan(
        spec=spec,
        runtime="local",
        target=RuntimeOperationTarget(kind="training-attempt", id="ra_redirect"),
    )

    async def scenario() -> None:
        ref = await runtime.submit_or_get(OperationId.generate(), plan)
        await _settle(runtime, ref)

    try:
        asyncio.run(scenario())
    finally:
        runtime.close()

    assert out.read_text() != "/tmp/elsewhere"
