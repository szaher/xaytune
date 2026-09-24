"""The public path: submit an experiment, get a handle, and ask it for the truth.

```text
ExperimentSpec
     │
     ▼
EmbeddedControllerHost.submit()
     ├── Experiment / Node / Run          durable, with events
     ├── compile CandidateSpec            the compiler the spec names
     ├── Attempt + INTENDED operation     one commit, before any effect
     ├── LocalRuntime.submit_or_get()
     ├── operation CONFIRMED + RuntimeRef
     └── observe telemetry → durable transitions
              │
              ▼
       ExperimentHandle  ── status() / wait() / events()
```

Real SQLite, real LocalRuntime, real trainer. Nothing on the load-bearing path
is a fake: the point is that a handle is a way of *asking the durable record*,
so a test that faked the record would prove nothing.

What training success means here was decided before this was written
(implementation plan, PR-012): the run succeeds, the node and the experiment
stay ACTIVE, and ``wait()`` returns because the controller is quiescent -- not
because anything is finished scientifically.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.training_fixtures import sft_candidate, tiny_dataset, tiny_model
from xaytune.core.domain.event import DomainEvent
from xaytune.core.domain.objective import Objective, ObjectiveMetric
from xaytune.core.state.status import (
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker inherits this; a test that reached the network is not offline."""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _spec(tmp_path: Path, *, compiler: str = "native"):
    from xaytune.experiment import CompilerSpec, ExperimentSpec, RuntimeSpec

    data_format = "text"
    candidate = sft_candidate(
        tiny_model(tmp_path / "model"),
        tiny_dataset(tmp_path / "data" / "train.jsonl", data_format),
        data_format,
    )
    return ExperimentSpec(
        name="tiny-sft",
        objective=Objective(primary=ObjectiveMetric(name="loss", direction="minimize")),
        candidate=candidate,
        seed=7,
        compiler=CompilerSpec(name=compiler),
        runtime=RuntimeSpec(kind="local", config={"root": str(tmp_path / "runtime")}),
        artifact_root=str(tmp_path / "artifacts"),
    )


async def _until(stream, predicate, *, timeout: float = 120.0) -> list[DomainEvent]:
    """Collect from *stream* up to and including the first event matching *predicate*."""
    seen: list[DomainEvent] = []

    async def collect() -> None:
        async for event in stream:
            seen.append(event)
            if predicate(event):
                return

    await asyncio.wait_for(collect(), timeout=timeout)
    return seen


def _run_status(status: RunStatus):
    return lambda e: e.aggregate_type == "Run" and e.payload.get("status") == status.value


def test_submit_returns_a_handle_that_reports_a_trained_but_unevaluated_experiment(
    tmp_path: Path,
) -> None:
    """The milestone: a spec in, a handle out, and the record tells the story."""
    from xaytune.experiment import EmbeddedControllerHost

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.submit(_spec(tmp_path))
            assert await handle.status() is ExperimentStatus.ACTIVE

            # Follow from the first event, concurrently with training, so
            # events committed after the iterator started must arrive too.
            following = asyncio.ensure_future(
                _until(handle.events(), _run_status(RunStatus.SUCCEEDED))
            )
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            followed = await following
            return handle.experiment_id, result, followed
        finally:
            await host.close()

    experiment_id, result, followed = asyncio.run(scenario())

    # Training success is not candidate success, nor experiment success.
    assert result.experiment_id == experiment_id
    assert result.status is ExperimentStatus.ACTIVE
    assert result.quiescent is True, "wait() returned because nothing is left to execute"
    assert result.next_stage == "evaluation"

    (node,) = result.nodes
    assert node.status is ExperimentNodeStatus.ACTIVE
    (run,) = node.runs
    assert run.status is RunStatus.SUCCEEDED
    assert run.attempt_status is RunAttemptStatus.SUCCEEDED
    (artifact,) = run.artifacts
    assert artifact.kind == "model"

    from transformers import AutoModelForCausalLM

    AutoModelForCausalLM.from_pretrained(artifact.uri, local_files_only=True)

    # events() is the durable control-plane history, in database order.
    assert all(isinstance(e, DomainEvent) for e in followed)
    sequences = [e.sequence for e in followed]
    assert sequences == sorted(sequences) and len(set(sequences)) == len(sequences)


def test_intent_is_durable_before_the_effect_and_confirmed_after(tmp_path: Path) -> None:
    """ADR-013: never an effect without a record that it was intended."""
    from xaytune.experiment import EmbeddedControllerHost

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.submit(_spec(tmp_path))
            await asyncio.wait_for(handle.wait(), timeout=180)
            return [e async for e in _replay(host, handle)]
        finally:
            await host.close()

    events = asyncio.run(scenario())
    shape = [
        (e.aggregate_type, e.event_type, e.payload.get("state") or e.payload.get("status"))
        for e in events
    ]

    intended = shape.index(("RuntimeOperation", "RuntimeOperationIntended", "intended"))
    attempt_created = next(
        i for i, s in enumerate(shape) if s[:2] == ("RunAttempt", "RunAttemptCreated")
    )
    confirmed = next(
        i for i, s in enumerate(shape) if s[0] == "RuntimeOperation" and s[2] == "confirmed"
    )
    running = next(i for i, s in enumerate(shape) if s[0] == "RunAttempt" and s[2] == "running")

    assert attempt_created < intended < confirmed < running, shape


def test_a_second_host_on_the_same_record_sees_the_same_history(tmp_path: Path) -> None:
    """A handle asks the record, so any host on the record gives the same answers.

    Not restart *reconciliation* -- nothing is in flight when the second host
    attaches. That is PR-012a. This is only that ``attach()`` reads rather than
    remembers.
    """
    from xaytune.experiment import EmbeddedControllerHost

    async def first():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.submit(_spec(tmp_path))
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            return handle.experiment_id, result, [e async for e in _replay(host, handle)]
        finally:
            await host.close()

    async def second(experiment_id):
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.attach(experiment_id)
            return (
                await handle.status(),
                await asyncio.wait_for(handle.wait(), timeout=30),
                [e async for e in _replay(host, handle)],
            )
        finally:
            await host.close()

    experiment_id, result, history = asyncio.run(first())
    status, attached_result, replayed = asyncio.run(second(experiment_id))

    assert status is ExperimentStatus.ACTIVE
    assert attached_result == result
    assert [e.id for e in replayed] == [e.id for e in history]


def test_the_record_holds_specs_not_implementations(tmp_path: Path) -> None:
    """ADR-016: after a restart the host rebuilds compiler and runtime from this alone."""
    from xaytune.experiment import EmbeddedControllerHost

    spec = _spec(tmp_path)

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.submit(spec)
            await asyncio.wait_for(handle.wait(), timeout=180)
            return host.repository.aggregates.load_experiment(str(handle.experiment_id))
        finally:
            await host.close()

    experiment = asyncio.run(scenario())

    # The request, with the implementation version resolved and recorded: the
    # record says which implementation ran, not only which name was asked for.
    assert experiment.runtime.model_copy(update={"version": None}) == spec.runtime
    assert experiment.compiler.model_copy(update={"version": None}) == spec.compiler
    assert experiment.runtime.version == "0.1.0"
    assert experiment.compiler.version == "0.1.0"
    assert experiment.artifact_root == spec.artifact_root
    assert experiment.controller_host.kind == "embedded"


async def _replay(host, handle):
    """Everything recorded so far, then stop -- ``events()`` alone would follow forever.

    The bound comes from the repository, not the handle: the handle's API is
    the five operations, and a test helper is no reason to widen it.
    """
    recorded = host.repository.events.events_for_experiment(str(handle.experiment_id))
    last = recorded[-1].sequence
    async for event in handle.events():
        yield event
        if event.sequence >= last:
            return
