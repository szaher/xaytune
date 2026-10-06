"""The controller does not care which runtime runs underneath it -- Ray included (PR-033).

```text
                 ┌─ RuntimeSpec(kind="local")     ─ LocalRuntime    ─ launcher   ─ worker
ExperimentHandle ┤
                 └─ RuntimeSpec(kind="ray-jobs")  ─ RayJobsRuntime  ─ Ray job ─ supervisor ─ worker
```

The same spec, differing only in the runtime it names, must produce the same
durable history and the same answers. The host has no Ray-specific code: it
is given a ``ray-jobs`` factory like any other. Ray's job manager is
``ProcessJobs`` here, so the matrix runs this without Ray; a host restarted
after Ray accepted the job, before the reference was recorded, adopts it --
even once Ray has forgotten the job it already finished.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from tests.test_experiment.test_cross_compiler_controller import _history, _shape
from tests.test_ray.ray_support import ProcessJobs
from tests.training_fixtures import sft_candidate, tiny_dataset, tiny_model
from xaytune.core.domain.objective import Objective, ObjectiveMetric
from xaytune.ray import RayJobsConfig, RayJobsRuntime


def _spec(root: Path, runtime: str) -> Any:
    from xaytune.experiment import CompilerSpec, ExperimentSpec, RuntimeSpec

    config = (
        {"root": str(root / "runtime")}
        if runtime == "local"
        else {
            "address": "http://ray.test:8265",
            "runtime_env": {},
            "shared_state_root": str(root / "ray-state"),
        }
    )
    return ExperimentSpec(
        name="tiny-sft",
        objective=Objective(primary=ObjectiveMetric(name="loss", direction="minimize")),
        candidate=sft_candidate(
            tiny_model(root / "model"), tiny_dataset(root / "data" / "train.jsonl", "text"), "text"
        ),
        seed=7,
        compiler=CompilerSpec(name="native"),
        runtime=RuntimeSpec(kind=runtime, config=config),
        artifact_root=str(root / "artifacts"),
    )


def _runtimes(jobs: ProcessJobs) -> dict[str, Any]:
    from xaytune.experiment.host import _local_runtime

    return {
        "local": _local_runtime,
        "ray-jobs": lambda config: RayJobsRuntime(RayJobsConfig(**config), submission=jobs),
    }


def _drive(root: Path, runtime: str, jobs: ProcessJobs) -> dict[str, Any]:
    from xaytune.experiment import EmbeddedControllerHost

    async def scenario() -> dict[str, Any]:
        host = EmbeddedControllerHost(root / "state.db", runtimes=_runtimes(jobs))
        try:
            handle = await host.submit(_spec(root, runtime))
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            return {
                "result": result,
                "history": host.repository.events.events_for_experiment(str(handle.experiment_id)),
            }
        finally:
            await host.close()

    return asyncio.run(scenario())


def test_the_same_spec_succeeds_identically_on_local_and_on_ray(tmp_path: Path) -> None:
    jobs = ProcessJobs()
    try:
        local = _drive(tmp_path / "local", "local", jobs)
        ray = _drive(tmp_path / "ray", "ray-jobs", jobs)
    finally:
        jobs.close()
    assert _shape(ray["result"]) == _shape(local["result"])
    assert _history(ray["history"]) == _history(local["history"])
    assert ("Run", "RunStatusChanged", "succeeded") in _history(ray["history"])
    assert len(jobs.jobs) == 1 and jobs.submissions == 1


def test_a_host_restarted_after_ray_accepted_the_job_adopts_it(tmp_path: Path) -> None:
    """The controller dies after Ray accepted the job, before recording its reference."""
    from xaytune.experiment import EmbeddedControllerHost

    jobs = ProcessJobs()

    async def scenario() -> None:
        first = EmbeddedControllerHost(tmp_path / "state.db", runtimes=_runtimes(jobs))
        issue = first._issue

        async def dies_after_ray_accepts(experiment_id, run_id, attempt_id, operation, plan, rt):
            await rt.submit_or_get(operation.id, plan)
            # ... and the process dies: the operation stays INTENDED, no
            # reference is recorded, nothing observes the workload.
            return None

        first._issue = dies_after_ray_accepts  # type: ignore[method-assign]
        try:
            handle = await first.submit(_spec(tmp_path, "ray-jobs"))
            experiment_id = handle.experiment_id
            while not jobs.jobs:
                await asyncio.sleep(0.05)
        finally:
            first._issue = issue  # type: ignore[method-assign]
            await first.close()

        second = EmbeddedControllerHost(tmp_path / "state.db", runtimes=_runtimes(jobs))
        try:
            handle = await second.attach(experiment_id)
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            assert result.quiescent
            assert ("Run", "RunStatusChanged", "succeeded") in _history(
                second.repository.events.events_for_experiment(str(experiment_id))
            )
            assert jobs.submissions == 1, "the restarted host adopted the job; it never resubmitted"
        finally:
            await second.close()

    try:
        asyncio.run(scenario())
    finally:
        jobs.close()


def test_a_restarted_host_never_reruns_a_finished_job_ray_forgot(tmp_path: Path) -> None:
    """Ray restarted without its job store; Xaytune's record of the ending still wins."""
    from xaytune.experiment import EmbeddedControllerHost

    jobs = ProcessJobs()

    async def scenario() -> None:
        first = EmbeddedControllerHost(tmp_path / "state.db", runtimes=_runtimes(jobs))
        issue = first._issue

        async def dies_after_ray_accepts(experiment_id, run_id, attempt_id, operation, plan, rt):
            await rt.submit_or_get(operation.id, plan)
            return None

        first._issue = dies_after_ray_accepts  # type: ignore[method-assign]
        try:
            handle = await first.submit(_spec(tmp_path, "ray-jobs"))
            experiment_id = handle.experiment_id
            while not jobs.jobs:
                await asyncio.sleep(0.05)
        finally:
            first._issue = issue  # type: ignore[method-assign]
            await first.close()

        (operation,) = list(jobs.jobs)
        finished = tmp_path / "ray-state" / operation / "finished.json"
        for _ in range(3600):
            if finished.exists():
                break
            await asyncio.sleep(0.05)
        assert finished.exists(), "the workload never finished"
        jobs.forget(operation)

        second = EmbeddedControllerHost(tmp_path / "state.db", runtimes=_runtimes(jobs))
        try:
            handle = await second.attach(experiment_id)
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            assert result.quiescent
            assert ("Run", "RunStatusChanged", "succeeded") in _history(
                second.repository.events.events_for_experiment(str(experiment_id))
            )
            assert jobs.submissions == 1, "a finished job Ray forgot was run again"
        finally:
            await second.close()

    try:
        asyncio.run(scenario())
    finally:
        jobs.close()
