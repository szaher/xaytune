"""Reconciling a PAUSED experiment adopts its existing work and starts nothing (PR-028).

A restarted controller sweeps paused experiments too: a workload already
running still needs an owner. But pause must keep meaning pause:

```text
may     adopt a running workload, settle it, settle the ledger
must not begin an evaluation cycle, give an evaluation run its first attempt,
         plan or realize a candidate
```

Each case stops a first host at a resting point, pauses the experiment, and
attaches a second; resuming and attaching a third shows the work was held,
not lost.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.test_experiment.adaptive_fixtures import adaptive_spec
from tests.test_experiment.test_adaptive_mvp import _TIMEOUT, _host, _never, _record, _world
from xaytune.core.state.status import (
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)
from xaytune.experiment.host import _ACTOR


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _move(host: Any, experiment_id: Any, status: ExperimentStatus) -> None:
    experiment = host.repository.aggregates.load_experiment(str(experiment_id))
    host.repository.transition_experiment(
        experiment.id, expected_revision=experiment.revision, new_status=status, actor=_ACTOR
    )


async def _until(predicate: Any) -> None:
    for _ in range(_TIMEOUT * 20):
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("timed out")


async def _observed(host: Any, experiment_id: Any) -> None:
    """Wait for every observer the host started for the experiment."""
    tasks = list(host._controllers.get(str(experiment_id), {}).values())
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=_TIMEOUT)


def _evaluation_runs(host: Any, experiment_id: Any) -> list[Any]:
    return [
        run
        for node in host.repository.aggregates.nodes_for_experiment(str(experiment_id))
        for run in host.repository.aggregates.evaluation_runs_for_node(str(node.id))
    ]


def test_a_paused_trained_node_begins_no_evaluation_cycle(tmp_path: Path) -> None:
    world = _world(tmp_path, oom_rank=None)
    runtime = world["runtime"]

    async def scenario() -> None:
        first = _host(tmp_path, world)
        try:
            first._continue_to_evaluation = _never  # type: ignore[method-assign]
            handle = await first.submit(adaptive_spec(tmp_path))
            resting = await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)
            assert resting.next_stage == "evaluation", "trained; the cycle never began"
            experiment_id = handle.experiment_id
            _move(first, experiment_id, ExperimentStatus.PAUSED)
        finally:
            await first.close()

        second = _host(tmp_path, world)
        try:
            await second.attach(experiment_id)
            await _observed(second, experiment_id)
            (node,) = _record(second, experiment_id)["nodes"]
            assert node.status is ExperimentNodeStatus.ACTIVE, "still trained, not evaluating"
            assert _evaluation_runs(second, experiment_id) == []
            assert runtime.evaluation_plans == []
            assert (
                second.repository.aggregates.load_experiment(str(experiment_id)).status
                is ExperimentStatus.PAUSED
            )
            _move(second, experiment_id, ExperimentStatus.ACTIVE)
        finally:
            await second.close()

        third = _host(tmp_path, world)
        try:
            await third.attach(experiment_id)
            await _until(lambda: runtime.evaluation_plans != [])
            await _observed(third, experiment_id)
        finally:
            await third.close()

    asyncio.run(scenario())


def test_a_paused_evaluation_run_with_no_attempt_does_not_get_one(tmp_path: Path) -> None:
    world = _world(tmp_path, oom_rank=None)
    runtime = world["runtime"]

    async def scenario() -> None:
        first = _host(tmp_path, world)
        try:
            first._start_evaluation_run = _never  # type: ignore[method-assign]
            handle = await first.submit(adaptive_spec(tmp_path))
            experiment_id = handle.experiment_id
            await _until(lambda: _evaluation_runs(first, experiment_id) != [])
            await _observed(first, experiment_id)
            _move(first, experiment_id, ExperimentStatus.PAUSED)
        finally:
            await first.close()
        (evaluation_run,) = _evaluation_runs(first_reader := _host(tmp_path, world), experiment_id)
        await first_reader.close()

        second = _host(tmp_path, world)
        try:
            await second.attach(experiment_id)
            await _observed(second, experiment_id)
            attempts = second.repository.aggregates.evaluation_attempts_for_run(
                str(evaluation_run.id)
            )
            assert attempts == (), "no first attempt while paused"
            assert runtime.evaluation_plans == []
            _move(second, experiment_id, ExperimentStatus.ACTIVE)
        finally:
            await second.close()

        third = _host(tmp_path, world)
        try:
            await third.attach(experiment_id)
            await _until(lambda: runtime.evaluation_plans != [])
            (attempt,) = third.repository.aggregates.evaluation_attempts_for_run(
                str(evaluation_run.id)
            )
            assert attempt.attempt_number == 1
            await _observed(third, experiment_id)
        finally:
            await third.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("stopped", "stage"),
    [("_continue_adaptive_experiment", "planning"), ("_realize_planned_candidate", "training")],
    ids=["paused-after-branch", "paused-after-planning"],
)
def test_a_paused_experiment_plans_and_realizes_nothing(
    tmp_path: Path, stopped: str, stage: str
) -> None:
    world = _world(tmp_path, oom_rank=None)
    runtime = world["runtime"]

    async def scenario() -> None:
        first = _host(tmp_path, world)
        try:
            setattr(first, stopped, _never)
            handle = await first.submit(adaptive_spec(tmp_path))
            resting = await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)
            assert resting.next_stage == stage
            experiment_id = handle.experiment_id
            _move(first, experiment_id, ExperimentStatus.PAUSED)
            before = _record(first, experiment_id)
        finally:
            await first.close()
        trained = len(runtime.training_plans)

        second = _host(tmp_path, world)
        try:
            await second.attach(experiment_id)
            await _observed(second, experiment_id)
            after = _record(second, experiment_id)
            assert [node.id for node in after["nodes"]] == [node.id for node in before["nodes"]]
            assert after["runs"] == before["runs"], "no run for a planned candidate"
            assert len(runtime.training_plans) == trained
        finally:
            await second.close()

    asyncio.run(scenario())


def test_a_paused_experiments_running_workload_is_adopted_and_settled(tmp_path: Path) -> None:
    world = _world(tmp_path, oom_rank=None)
    runtime = world["runtime"]

    async def scenario() -> None:
        first = _host(tmp_path, world)
        try:
            # Issued and confirmed, then the controller died before observing it.
            first._adopt = lambda *args, **kwargs: None  # type: ignore[method-assign]
            handle = await first.submit(adaptive_spec(tmp_path))
            experiment_id = handle.experiment_id
            _move(first, experiment_id, ExperimentStatus.PAUSED)
        finally:
            await first.close()

        second = _host(tmp_path, world)
        try:
            await second.attach(experiment_id)
            await _observed(second, experiment_id)
            record = _record(second, experiment_id)
            (node,) = record["nodes"]
            (run,) = record["runs"][node.id]
            (attempt,) = record["attempts"][run.id]
            assert attempt.status is RunAttemptStatus.SUCCEEDED, "adopted and observed"
            assert run.status is RunStatus.SUCCEEDED, "its existing work settled"
            assert len(runtime.training_plans) == 1, "adopted, not resubmitted"
            assert record["experiment"].status is ExperimentStatus.PAUSED
            assert node.status is ExperimentNodeStatus.ACTIVE
            assert _evaluation_runs(second, experiment_id) == [], "and nothing new begun"
        finally:
            await second.close()

    asyncio.run(scenario())
