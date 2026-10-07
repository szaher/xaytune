"""The controller does not care which runtime runs underneath it -- Ray included (PR-033).

```text
                 ┌─ RuntimeSpec(kind="local")     ─ LocalRuntime    ─ launcher   ─ worker
ExperimentHandle ┼─ RuntimeSpec(kind="ray-jobs")  ─ RayJobsRuntime  ─ Ray job ─ supervisor ─ worker
                 └─ RuntimeSpec(kind="ray-train") ─ RayTrainRuntime ─ Ray job ─ TorchTrainer ─ ranks
```

The same spec, differing only in the runtime it names, must produce the same
durable history and the same answers. The host has no Ray-specific code: it
is given ``ray-jobs`` and ``ray-train`` factories like any other. Ray's job manager is
``ProcessJobs`` here, so the matrix runs this without Ray -- submitted through the
Ray Jobs API or as KubeRay RayJobs (``FakeKubernetes``), which neither the host
nor the runtime can tell apart; a host restarted
after Ray accepted the job, before the reference was recorded, adopts it --
even once Ray has forgotten the job it already finished.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.test_experiment.test_cross_compiler_controller import _history, _shape
from tests.test_ray.kube_support import FakeKubernetes
from tests.test_ray.ray_support import ProcessJobs
from tests.training_fixtures import sft_candidate, tiny_dataset, tiny_model
from xaytune.core.domain.objective import Objective, ObjectiveMetric
from xaytune.ray import (
    KubeRayJobsBackend,
    RayJobsConfig,
    RayJobsRuntime,
    RayTrainConfig,
    RayTrainRuntime,
)

_KUBERAY = {
    "kind": "kuberay",
    "namespace": "ml",
    "context": "kind-xaytune",
    "cluster": {"kind": "existing", "selector": {"ray.io/cluster": "trainers"}},
}


def _spec(root: Path, runtime: str, submission: str = "ray-jobs") -> Any:
    from xaytune.experiment import CompilerSpec, ExperimentSpec, RuntimeSpec

    config = (
        {"root": str(root / "runtime")}
        if runtime == "local"
        else {
            **(
                {"address": "http://ray.test:8265"}
                if submission == "ray-jobs"
                else {"submission": _KUBERAY}
            ),
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


def _runtimes(jobs: ProcessJobs, kube: FakeKubernetes | None = None) -> dict[str, Any]:
    """The hosts' factories. Given *kube*, the configured KubeRay backend talks to it."""
    from xaytune.experiment.host import _local_runtime

    def backend(config: RayJobsConfig | RayTrainConfig) -> Any:
        if config.submission.kind == "kuberay":
            assert kube is not None
            return KubeRayJobsBackend(config.submission, api=kube)
        return jobs

    def ray_jobs(config: Any) -> RayJobsRuntime:
        parsed = RayJobsConfig(**config)
        return RayJobsRuntime(parsed, submission=backend(parsed))

    def ray_train(config: Any) -> RayTrainRuntime:
        parsed = RayTrainConfig(**config)
        return RayTrainRuntime(parsed, submission=backend(parsed))

    return {"local": _local_runtime, "ray-jobs": ray_jobs, "ray-train": ray_train}


def _drive(
    root: Path,
    runtime: str,
    jobs: ProcessJobs,
    submission: str = "ray-jobs",
    kube: FakeKubernetes | None = None,
) -> dict[str, Any]:
    from xaytune.experiment import EmbeddedControllerHost

    async def scenario() -> dict[str, Any]:
        host = EmbeddedControllerHost(root / "state.db", runtimes=_runtimes(jobs, kube))
        try:
            handle = await host.submit(_spec(root, runtime, submission))
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            return {
                "result": result,
                "history": host.repository.events.events_for_experiment(str(handle.experiment_id)),
            }
        finally:
            await host.close()

    return asyncio.run(scenario())


@pytest.mark.parametrize("submission", ["ray-jobs", "kuberay"])
@pytest.mark.parametrize("kind", ["ray-jobs", "ray-train"])
def test_the_same_spec_succeeds_identically_on_local_and_on_ray(
    tmp_path: Path, kind: str, submission: str
) -> None:
    jobs = ProcessJobs()
    kube = FakeKubernetes(jobs)
    try:
        local = _drive(tmp_path / "local", "local", jobs)
        ray = _drive(tmp_path / "ray", kind, jobs, submission, kube)
    finally:
        jobs.close()
    assert _shape(ray["result"]) == _shape(local["result"])
    assert _history(ray["history"]) == _history(local["history"])
    assert ("Run", "RunStatusChanged", "succeeded") in _history(ray["history"])
    assert len(jobs.jobs) == 1 and jobs.submissions == 1
    assert len(kube.objects) == (1 if submission == "kuberay" else 0)


@pytest.mark.parametrize("submission", ["ray-jobs", "kuberay"])
def test_a_host_restarted_after_ray_accepted_the_job_adopts_it(
    tmp_path: Path, submission: str
) -> None:
    """The controller dies after Ray accepted the job, before recording its reference."""
    from xaytune.experiment import EmbeddedControllerHost

    jobs = ProcessJobs()
    kube = FakeKubernetes(jobs)

    async def scenario() -> None:
        first = EmbeddedControllerHost(tmp_path / "state.db", runtimes=_runtimes(jobs, kube))
        issue = first._issue

        async def dies_after_ray_accepts(experiment_id, run_id, attempt_id, operation, plan, rt):
            await rt.submit_or_get(operation.id, plan)
            # ... and the process dies: the operation stays INTENDED, no
            # reference is recorded, nothing observes the workload.
            return None

        first._issue = dies_after_ray_accepts  # type: ignore[method-assign]
        try:
            handle = await first.submit(_spec(tmp_path, "ray-jobs", submission))
            experiment_id = handle.experiment_id
            while not jobs.jobs:
                await asyncio.sleep(0.05)
        finally:
            first._issue = issue  # type: ignore[method-assign]
            await first.close()

        second = EmbeddedControllerHost(tmp_path / "state.db", runtimes=_runtimes(jobs, kube))
        try:
            handle = await second.attach(experiment_id)
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            assert result.quiescent
            assert ("Run", "RunStatusChanged", "succeeded") in _history(
                second.repository.events.events_for_experiment(str(experiment_id))
            )
            assert jobs.submissions == 1, "the restarted host adopted the job; it never resubmitted"
            assert kube.creates == (1 if submission == "kuberay" else 0)
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


def test_an_experiment_on_ray_train_trains_as_a_group_and_still_evaluates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One recorded runtime serves both: training as Ray Train, evaluation as one Ray job."""
    from tests.evaluation_fixtures import native_evaluation
    from xaytune.core.state.status import EvaluationRunStatus, ExperimentNodeStatus
    from xaytune.experiment import EmbeddedControllerHost

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    spec = _spec(tmp_path, "ray-train").model_copy(
        update={"evaluation": native_evaluation(tmp_path / "eval")}
    )
    jobs = ProcessJobs()

    async def scenario() -> Any:
        host = EmbeddedControllerHost(tmp_path / "state.db", runtimes=_runtimes(jobs))
        try:
            handle = await host.submit(spec)
            return await asyncio.wait_for(handle.wait(), timeout=300)
        finally:
            await host.close()

    try:
        result = asyncio.run(scenario())
    finally:
        jobs.close()
    (node,) = result.nodes
    assert node.status is ExperimentNodeStatus.DECIDING
    (evaluated,) = node.evaluations
    assert evaluated.status is EvaluationRunStatus.SUCCEEDED
    entrypoints = sorted(job.entrypoint.split()[2] for job in jobs.jobs.values())
    assert entrypoints == ["xaytune.ray.runtime.supervisor", "xaytune.ray.runtime.train_driver"]
