"""RayTrainRuntime: training as a Ray Train worker group, evaluation as one job, one stream.

Every scenario runs the real driver and real ranks with real workers. In the
matrix ``ProcessJobs`` stands in for Ray's job manager and its "image" runs
:mod:`tests.test_ray.fake_train_driver`, whose group is threads in place of
``TorchTrainer``'s actors; the ``ray`` marked tests at the end run the actual
``TorchTrainer`` on a real head, with two workers forming a real process group.
"""

from __future__ import annotations

import ast
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.test_ray.fake_train_driver import FAIL_GROUP
from tests.test_ray.ray_support import ProcessJobs, RayHead
from tests.test_ray.test_ray_jobs_runtime import (
    _FAILS,
    _PRELUDE,
    _SLEEPS,
    _TRAINS,
    _await,
    _plan,
    _settle,
    _types,
)
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.errors import IdempotencyConflictError
from xaytune.core.execution import (
    CommandEntrypoint,
    ResolvedExecutionPlan,
    ResourceRequirements,
)
from xaytune.core.ids import OperationId
from xaytune.core.refs import RuntimeRef
from xaytune.ray import RayTrainConfig, RayTrainRuntime, RayUnavailableError, ray_train_runtime
from xaytune.ray.runtime._workloads import ACCEPTED, LAUNCH, REJECTED
from xaytune.ray.runtime.train import TRAIN_DRIVER, scaling_for
from xaytune.ray.runtime.train_group import ABORT, RANKS
from xaytune.runtimes import RuntimeBackend, StreamCursor, UnsupportedPlanError
from xaytune.runtimes.local.paths import WorkloadPaths, read_json

# The workers' own process group: each rank adds rank + 1, and rank 0 reports
# the sum -- 3 for two workers -- as its loss. Every rank also reports a loss of
# 100 + its rank, which must never reach the stream from any rank but 0.
_GROUP = (
    "import os, torch, torch.distributed as dist\n"
    "rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])\n"
    "writer.write(t.TrainingStartedPayload())\n"
    "writer.write(t.TrainingMetricObserved(optimizer_step=1, loss=100.0 + rank))\n"
    "dist.init_process_group('gloo', rank=rank, world_size=size)\n"
    "total = torch.tensor([rank + 1.0])\n"
    "dist.all_reduce(total)\n"
    "writer.write(t.TrainingMetricObserved(optimizer_step=2, loss=float(total)))\n"
    "dist.destroy_process_group()\n"
    "writer.write(t.TrainingCompletedPayload())\n"
)
# Rank 1 fails; rank 0 would otherwise sleep on.
_RANK_ONE_FAILS = (
    "import os\n"
    "writer.write(t.TrainingStartedPayload())\n"
    "if os.environ['RANK'] == '1':\n"
    "    time.sleep(0.5)\n"
    "    sys.exit(3)\n"
    "time.sleep(60)\n"
)


def _train(body: str = _TRAINS, workers: int = 1, **resources: Any) -> ResolvedExecutionPlan:
    return _plan(
        body,
        runtime="ray-train",
        resources=ResourceRequirements(workers=workers, **resources),
    )


def _config(root: Path, address: str = "http://ray.test:8265") -> RayTrainConfig:
    return RayTrainConfig(address=address, runtime_env={}, shared_state_root=str(root))


def _runtime(root: Path, jobs: Any) -> RayTrainRuntime:
    return RayTrainRuntime(_config(root), submission=jobs)


@pytest.fixture
def jobs() -> Any:
    fake = ProcessJobs()
    yield fake
    fake.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "ray-state"


def _metrics(events: list[Any]) -> list[tuple[int, float]]:
    return [
        (event.payload.data.optimizer_step, event.payload.data.loss)
        for event in events
        if event.payload.data.type == "TrainingMetricObserved"
    ]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - someone else's
        return True
    return True


# ---- the contract, and Ray Train at every size ---------------------------------------------


def test_the_ray_train_runtime_is_a_runtime_backend(root: Path, jobs: ProcessJobs) -> None:
    runtime = _runtime(root, jobs)
    assert isinstance(runtime, RuntimeBackend)
    assert runtime.descriptor.name == "ray-train"
    capabilities = runtime.capabilities()
    assert capabilities.distributed is not None
    assert (capabilities.distributed.min_workers, capabilities.distributed.max_workers) == (1, None)
    assert capabilities.resilience is not None
    assert capabilities.resilience.reports_completed_operations is True


def test_one_worker_is_still_a_ray_train_worker_group(root: Path, jobs: ProcessJobs) -> None:
    """workers=1 runs the train driver and one rank -- not the single-process supervisor."""
    runtime = _runtime(root, jobs)
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _train(_TRAINS, workers=1))
        assert ref == RuntimeRef(backend="ray-train", external_id=str(operation))
        assert (await _settle(runtime, ref)).state == "succeeded"
        job = jobs.jobs[str(operation)]
        assert job.entrypoint.split()[1:3] == ["-m", TRAIN_DRIVER]
        assert job.resources == {}  # the driver holds nothing; the group's workers do
        launch = read_json(root / str(operation) / LAUNCH)
        assert launch == {
            "scaling": {"num_workers": 1, "use_gpu": False, "resources_per_worker": None}
        }
        rank = read_json(root / str(operation) / RANKS / "0.finished.json")
        assert rank is not None and rank["exit_code"] == 0
        events = [event async for event in runtime.watch(ref)]
        assert _types(events)[0] == "WorkerReady"
        assert _types(events)[-1] == "TrainingCompleted"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("resources", "scaling"),
    [
        ({"workers": 1}, {"num_workers": 1, "use_gpu": False, "resources_per_worker": None}),
        ({"workers": 4}, {"num_workers": 4, "use_gpu": False, "resources_per_worker": None}),
        (
            {"workers": 4, "gpus": 4},
            {"num_workers": 4, "use_gpu": True, "resources_per_worker": {"GPU": 1}},
        ),
        (
            {"workers": 1, "gpus": 1, "cpus": 8.0, "memory_bytes": 2**34},
            {
                "num_workers": 1,
                "use_gpu": True,
                "resources_per_worker": {"CPU": 8.0, "memory": 2**34, "GPU": 1},
            },
        ),
        (
            {"workers": 2, "gpus": 0},
            {"num_workers": 2, "use_gpu": False, "resources_per_worker": None},
        ),
    ],
    ids=["one", "four", "one-gpu-each", "one-worker-sized", "zero-gpus"],
)
def test_the_plan_becomes_a_scaling_config_only_where_unambiguous(
    resources: dict[str, Any], scaling: dict[str, Any]
) -> None:
    assert scaling_for(_train(**_split(resources))) == scaling


def _split(resources: dict[str, Any]) -> dict[str, Any]:
    workers = resources.pop("workers")
    return {"workers": workers, **resources}


@pytest.mark.parametrize(
    ("plan", "reason"),
    [
        (
            _plan(runtime="ray-train", resources=ResourceRequirements()),
            "must state resources.workers",
        ),
        (_train(workers=2, gpus=3), "does not say how they divide"),
        (_train(workers=2, gpus=1), "does not say how they divide"),
        (_train(workers=2, cpus=4.0), "resources.cpus is the workload's total"),
        (_train(workers=2, memory_bytes=2**30), "resources.memory_bytes"),
        (_train(workers=1, gpu_type="a100"), "gpu_type"),
        (_train(workers=1, max_runtime_seconds=60), "max_runtime_seconds"),
        (
            _plan(
                runtime="ray-train",
                resources=ResourceRequirements(workers=2),
                environment={"WORLD_SIZE": "8"},
            ),
            "WORLD_SIZE",
        ),
        (_plan(runtime="ray-jobs", resources=ResourceRequirements(workers=1)), "resolved for"),
    ],
    ids=[
        "no-size",
        "gpus-uneven",
        "gpus-fewer",
        "cpus-total",
        "memory-total",
        "gpu-type",
        "deadline",
        "topology",
        "other-runtime",
    ],
)
def test_ambiguous_or_unplaceable_plans_are_refused_and_recorded(
    plan: ResolvedExecutionPlan, reason: str, root: Path, jobs: ProcessJobs
) -> None:
    runtime = _runtime(root, jobs)
    operation = OperationId.generate()

    async def scenario() -> None:
        with pytest.raises(UnsupportedPlanError, match=reason):
            await runtime.submit_or_get(operation, plan)
        outcome = await _runtime(root, jobs).lookup_operation(operation)
        assert outcome is not None and outcome.disposition == "rejected"

    asyncio.run(scenario())
    assert jobs.submissions == 0
    assert read_json(root / str(operation) / REJECTED) is not None


def test_managed_worker_requests_address_one_worker_not_a_group(
    root: Path, jobs: ProcessJobs
) -> None:
    from xaytune.core.execution import PythonModuleEntrypoint, TrainingExecutionSpec

    plan = _train(workers=2)
    spec = plan.spec
    assert isinstance(spec, TrainingExecutionSpec)
    managed = ResolvedExecutionPlan(
        spec=spec.model_copy(
            update={
                "entrypoint": PythonModuleEntrypoint(module="xaytune.workers.native"),
                "checkpoint": spec.checkpoint.model_copy(update={"format": "native-torch/v1"}),
            }
        ),
        runtime="ray-train",
        target=plan.target,
        runtime_options={"training_interventions": {}},
    )
    with pytest.raises(UnsupportedPlanError, match="single managed worker"):
        asyncio.run(_runtime(root, jobs).submit_or_get(OperationId.generate(), managed))


# ---- one ordered stream from a group --------------------------------------------------------


def test_a_two_worker_group_forms_its_own_process_group_and_streams_rank_zero_only(
    root: Path, jobs: ProcessJobs
) -> None:
    runtime = _runtime(root, jobs)
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _train(_GROUP, workers=2))
        status = await _settle(runtime, ref)
        assert status.state == "succeeded", status
        events = [event async for event in runtime.watch(ref)]
        # One sequencer: one generation, 0..n-1, no gaps, WorkerReady first.
        assert [event.sequence for event in events] == list(range(len(events)))
        assert {event.stream_generation for event in events} == {0}
        assert _types(events)[0] == "WorkerReady"
        assert _types(events).count("TrainingStarted") == 1
        # Rank 0's observations only; the all-reduce proves both ranks ran.
        assert _metrics(events) == [(1, 100.0), (2, 3.0)]
        diagnostics = root / str(operation) / RANKS / "1.observations.jsonl"
        assert "101.0" in diagnostics.read_text()
        started = read_json(WorkloadPaths(root / str(operation)).started)
        assert started is not None and [r["rank"] for r in started["ranks"]] == [0, 1]

    asyncio.run(scenario())


def test_watching_a_group_resumes_after_the_cursor_on_a_restarted_runtime(
    root: Path, jobs: ProcessJobs
) -> None:
    runtime = _runtime(root, jobs)

    async def scenario() -> None:
        ref = await runtime.submit_or_get(OperationId.generate(), _train(_GROUP, workers=2))
        await _settle(runtime, ref)
        everything = [event async for event in runtime.watch(ref)]
        rest = [
            event
            async for event in _runtime(root, jobs).watch(
                ref, StreamCursor(generation=0, sequence=1)
            )
        ]
        assert rest == everything[2:]

    asyncio.run(scenario())


# ---- exactly once ---------------------------------------------------------------------------


def test_the_same_operation_is_one_group_across_restarts(root: Path, jobs: ProcessJobs) -> None:
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await _runtime(root, jobs).submit_or_get(operation, _train(_SLEEPS, workers=2))
        restarted = _runtime(root, jobs)
        outcome = await restarted.lookup_operation(operation)
        assert outcome is not None and outcome.runtime_ref == ref
        assert await restarted.submit_or_get(operation, _train(_SLEEPS, workers=2)) == ref
        with pytest.raises(IdempotencyConflictError):
            await restarted.submit_or_get(operation, _train(_SLEEPS, workers=3))
        await restarted.cancel(ref, OperationId.generate())
        assert (await _settle(restarted, ref)).state == "cancelled"

    asyncio.run(scenario())
    assert jobs.submissions == 1


def test_a_lost_submit_answer_adopts_the_group_ray_accepted(root: Path, jobs: ProcessJobs) -> None:
    runtime = _runtime(root, jobs)
    original = jobs.submit

    def accepted_then_lost(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        raise RayUnavailableError("the response never arrived")

    jobs.submit = accepted_then_lost  # type: ignore[method-assign]
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _train(_TRAINS, workers=2))
        assert ref.external_id == str(operation) and len(jobs.jobs) == 1
        assert (await _settle(runtime, ref)).state == "succeeded"
        assert read_json(root / str(operation) / ACCEPTED) is not None

    asyncio.run(scenario())


def test_a_finished_group_ray_forgot_is_completed_and_never_rerun(
    root: Path, jobs: ProcessJobs
) -> None:
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await _runtime(root, jobs).submit_or_get(operation, _train(_TRAINS, workers=2))
        assert (await _settle(_runtime(root, jobs), ref)).state == "succeeded"
        jobs.forget(str(operation))
        restarted = _runtime(root, jobs)
        outcome = await restarted.lookup_operation(operation)
        assert outcome is not None and outcome.disposition == "completed"
        assert outcome.status is not None and outcome.status.state == "succeeded"
        assert await restarted.submit_or_get(operation, _train(_TRAINS, workers=2)) == ref

    asyncio.run(scenario())
    assert jobs.submissions == 1


# ---- failure and cancellation -----------------------------------------------------------------


def test_a_failing_rank_fails_the_group_and_stops_its_siblings(
    root: Path, jobs: ProcessJobs
) -> None:
    runtime = _runtime(root, jobs)
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _train(_RANK_ONE_FAILS, workers=2))
        status = await _settle(runtime, ref)
        assert (status.state, status.exit_code) == ("failed", 3)
        ranks = root / str(operation) / RANKS
        assert read_json(ranks / ABORT) is not None
        zero = read_json(ranks / "0.finished.json")
        assert zero is not None and zero["stopped_by"] == "group"
        finished = read_json(WorkloadPaths(root / str(operation)).finished)
        assert finished is not None and finished["failed_rank"] == 1
        events = [event async for event in runtime.watch(ref)]
        incidents = [e.payload.data for e in events if e.payload.data.type == "IncidentObserved"]
        assert [(i.reason, i.exit_code) for i in incidents] == [("nonzero-exit", 3)]

    asyncio.run(scenario())


def test_a_group_that_fails_as_a_group_is_failed(root: Path, jobs: ProcessJobs) -> None:
    """Ray Train itself reporting the group failed -- a worker actor lost, say -- is failure."""
    runtime = _runtime(root, jobs)
    operation = OperationId.generate()
    (root / str(operation)).mkdir(parents=True)
    (root / str(operation) / FAIL_GROUP).touch()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _train(_TRAINS, workers=2))
        status = await _settle(runtime, ref)
        assert (status.state, status.exit_code) == ("failed", 1)
        events = [event async for event in runtime.watch(ref)]
        assert "worker-group-failed" in [
            e.payload.data.reason for e in events if e.payload.data.type == "IncidentObserved"
        ]

    asyncio.run(scenario())


def test_cancelling_a_group_stops_every_rank_and_is_idempotent_by_its_id(
    root: Path, jobs: ProcessJobs
) -> None:
    runtime = _runtime(root, jobs)
    operation, other = OperationId.generate(), OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _train(_SLEEPS, workers=2))
        second = await runtime.submit_or_get(other, _train(_SLEEPS, workers=1))
        paths = WorkloadPaths(root / str(operation))
        await _await(lambda: paths.started.exists(), "the group started")
        pids = [rank["pid"] for rank in (read_json(paths.started) or {})["ranks"]]
        assert len(pids) == 2 and all(_alive(pid) for pid in pids)
        cancel = OperationId.generate()
        await runtime.cancel(ref, cancel)
        await _runtime(root, jobs).cancel(ref, cancel)
        with pytest.raises(IdempotencyConflictError):
            await runtime.cancel(second, cancel)
        assert (await _settle(runtime, ref)).state == "cancelled"
        assert not any(_alive(pid) for pid in pids), "a rank's worker outlived the cancellation"
        for rank in (0, 1):
            record = read_json(root / str(operation) / RANKS / f"{rank}.finished.json")
            assert record is not None and record["stopped_by"] == "cancel"
        await runtime.cancel(second, OperationId.generate())
        await _settle(runtime, second)

    asyncio.run(scenario())


def test_a_group_cancelled_before_it_was_placed_never_starts(root: Path) -> None:
    held = ProcessJobs(hold=True)
    runtime = _runtime(root, held)
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _train(_SLEEPS, workers=2))
        await runtime.cancel(ref, OperationId.generate())
        assert (await _settle(runtime, ref)).state == "cancelled"
        assert not (root / str(operation) / RANKS).exists()

    try:
        asyncio.run(scenario())
    finally:
        held.close()


def test_a_driver_that_finds_the_cancellation_first_starts_no_rank(
    root: Path, jobs: ProcessJobs
) -> None:
    """Ray stopped a pending job late and started its driver anyway: nothing runs."""
    held = ProcessJobs(hold=True)
    held.stop_ignores_pending = True
    runtime = _runtime(root, held)
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _train(_SLEEPS, workers=2))
        await runtime.cancel(ref, OperationId.generate())
        held.release(str(operation))
        status = await _settle(runtime, ref)
        assert status.state == "cancelled"
        finished = read_json(WorkloadPaths(root / str(operation)).finished)
        assert finished is not None and finished["never_started"] is True
        assert not (root / str(operation) / RANKS / "0.started.json").exists()

    try:
        asyncio.run(scenario())
    finally:
        held.close()


# ---- evaluation, configuration, boundaries ----------------------------------------------------


def test_an_evaluation_runs_as_one_supervised_job_under_ray_train(
    root: Path, jobs: ProcessJobs
) -> None:
    from tests.test_runtimes.test_evaluation_transport import _evaluation_spec

    body = (
        "from xaytune.core.telemetry import EvaluationStartedPayload\n"
        "writer.write(EvaluationStartedPayload())\n"
    )
    plan = ResolvedExecutionPlan(
        spec=_evaluation_spec(
            entrypoint=CommandEntrypoint(argv=(sys.executable, "-c", _PRELUDE + body))
        ),
        runtime="ray-train",
        target=RuntimeOperationTarget(kind="evaluation-attempt", id="ea_ray"),
    )
    runtime = _runtime(root, jobs)
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, plan)
        assert (await _settle(runtime, ref)).state == "succeeded"
        assert jobs.jobs[str(operation)].entrypoint.split()[2] == "xaytune.ray.runtime.supervisor"
        assert not (root / str(operation) / RANKS).exists()
        events = [event async for event in runtime.watch(ref)]
        assert "EvaluationStarted" in _types(events)

    asyncio.run(scenario())


def test_a_failed_worker_in_a_one_rank_group_reports_its_exit_code(
    root: Path, jobs: ProcessJobs
) -> None:
    runtime = _runtime(root, jobs)

    async def scenario() -> None:
        ref = await runtime.submit_or_get(OperationId.generate(), _train(_FAILS, workers=1))
        status = await _settle(runtime, ref)
        assert (status.state, status.exit_code) == ("failed", 3)

    asyncio.run(scenario())


def test_the_factory_and_its_configuration(tmp_path: Path) -> None:
    runtime = ray_train_runtime(
        {"address": "http://127.0.0.1:1", "runtime_env": {}, "shared_state_root": str(tmp_path)}
    )
    assert isinstance(runtime, RayTrainRuntime)
    with pytest.raises(ValueError):
        ray_train_runtime({"address": "http://h:8265", "runtime_env": {}})


def test_ray_train_is_imported_by_the_train_driver_alone() -> None:
    probe = (
        "import sys\n"
        "import xaytune.core, xaytune.planning, xaytune.agent, xaytune.storage\n"
        "import xaytune.experiment, xaytune.daemon, xaytune.runtimes\n"
        "import xaytune.ray, xaytune.ray.runtime.train, xaytune.ray.runtime.train_group\n"
        "print(sorted(m for m in sys.modules if m == 'ray' or m.startswith('ray.')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
    root = Path(__file__).resolve().parents[2] / "xaytune"
    train_importers = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if any(
            isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ray.train")
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
    )
    assert train_importers == ["ray/runtime/train_driver.py"]


# ---- a real Ray head ------------------------------------------------------------------------


def _real(root: Path, head: RayHead) -> RayTrainRuntime:
    return RayTrainRuntime(_config(root, head.address))


@pytest.mark.ray
def test_on_ray_the_plan_builds_a_torch_trainer_with_its_scaling(tmp_path: Path) -> None:
    pytest.importorskip("ray.train")
    from ray.train.torch import TorchTrainer

    from xaytune.ray.runtime.train_driver import build_trainer

    paths = WorkloadPaths(tmp_path / "op_x")
    trainer = build_trainer(paths, scaling_for(_train(workers=3)), lambda context: None)
    assert isinstance(trainer, TorchTrainer)
    assert trainer.scaling_config.num_workers == 3
    assert trainer.scaling_config.use_gpu is False
    assert trainer.run_config.failure_config.max_failures == 0
    assert trainer.run_config.storage_path == str(paths.directory / "ray-train")


@pytest.mark.ray
def test_on_ray_a_two_worker_torch_trainer_runs_one_process_group_and_one_stream(
    root: Path,
    ray_head: RayHead,
) -> None:
    operation = OperationId.generate()

    async def scenario() -> None:
        first = _real(root, ray_head)
        ref = await first.submit_or_get(operation, _train(_GROUP, workers=2))
        restarted = _real(root, ray_head)
        assert await restarted.submit_or_get(operation, _train(_GROUP, workers=2)) == ref
        status = await _settle(restarted, ref)
        assert status.state == "succeeded", (status, _logs(root, operation))
        events = [event async for event in restarted.watch(ref)]
        assert [event.sequence for event in events] == list(range(len(events)))
        assert _metrics(events) == [(1, 100.0), (2, 3.0)]
        assert (root / str(operation) / "ray-train").is_dir()  # Ray Train's own storage

    asyncio.run(scenario())


@pytest.mark.ray
def test_on_ray_a_failing_rank_fails_the_group_and_a_cancelled_group_leaves_nothing(
    root: Path,
    ray_head: RayHead,
) -> None:
    runtime = _real(root, ray_head)

    async def scenario() -> None:
        failing_op = OperationId.generate()
        failing = await runtime.submit_or_get(failing_op, _train(_RANK_ONE_FAILS, workers=2))
        status = await _settle(runtime, failing)
        assert (status.state, status.exit_code) == ("failed", 3), _logs(root, failing_op)

        operation = OperationId.generate()
        ref = await runtime.submit_or_get(operation, _train(_SLEEPS, workers=2))
        paths = WorkloadPaths(root / str(operation))
        await _await(lambda: paths.started.exists(), "the group started")
        pids = [rank["pid"] for rank in (read_json(paths.started) or {})["ranks"]]
        await runtime.cancel(ref, OperationId.generate())
        assert (await _settle(runtime, ref)).state == "cancelled"
        assert not any(_alive(pid) for pid in pids)

    asyncio.run(scenario())


def _logs(root: Path, operation: OperationId) -> str:
    directory = root / str(operation)
    return "\n".join(
        f"--- {path.name}\n{path.read_text()[-2000:]}"
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix in ("", ".log", ".jsonl") and path.stat().st_size
    )


# ---- review regressions: built-in workers, failure vs cancel, readiness ---------------------


def _built_in(module: str, workers: int, *, by_command: bool = False) -> ResolvedExecutionPlan:
    from xaytune.core.execution import PythonModuleEntrypoint, TrainingExecutionSpec

    plan = _train(workers=workers)
    assert isinstance(plan.spec, TrainingExecutionSpec)
    entrypoint = (
        CommandEntrypoint(argv=("python", "-m", module))
        if by_command
        else PythonModuleEntrypoint(module=module)
    )
    return ResolvedExecutionPlan(
        spec=plan.spec.model_copy(update={"entrypoint": entrypoint}),
        runtime="ray-train",
        target=plan.target,
    )


@pytest.mark.parametrize(
    "plan",
    [
        _built_in("xaytune.workers.native", 2),
        _built_in("xaytune.workers.trl", 4),
        _built_in("xaytune.workers.native", 2, by_command=True),
    ],
    ids=["native", "trl", "native-by-command"],
)
def test_built_in_workers_are_refused_as_a_group(
    plan: ResolvedExecutionPlan, root: Path, jobs: ProcessJobs
) -> None:
    """They publish the model from every rank; FSDP-wrapped, none of them holds it whole."""
    with pytest.raises(UnsupportedPlanError, match="publishes its model from every rank"):
        asyncio.run(_runtime(root, jobs).submit_or_get(OperationId.generate(), plan))
    assert jobs.submissions == 0


def test_a_built_in_worker_still_runs_as_a_group_of_one(root: Path) -> None:
    held = ProcessJobs(hold=True)
    try:
        ref = asyncio.run(
            _runtime(root, held).submit_or_get(
                OperationId.generate(), _built_in("xaytune.workers.native", 1)
            )
        )
        assert ref.backend == "ray-train" and held.submissions == 1
    finally:
        held.close()


class _ScriptedGroup:
    """A worker group whose ranks' records are written in a chosen order -- no timing."""

    world_size = 2

    def __init__(self, script: Any) -> None:
        self.script = script

    def run(self, rank_fn: Any) -> None:
        self.script()


def _drive_scripted(root: Path, script_for: Any) -> tuple[dict[str, Any], list[str]]:
    """Run the real driver over a scripted group; its ending and its stream's event types."""
    import json as _json

    from xaytune.ray.runtime.train_group import RankPaths, drive

    directory = root / "op_scripted"
    directory.mkdir(parents=True)
    paths = WorkloadPaths(directory)
    plan = _train(workers=2)
    paths.plan.write_text(plan.model_dump_json(), encoding="utf-8")
    ranks = [RankPaths(paths, rank) for rank in range(2)]
    ranks[0].directory.mkdir(parents=True, exist_ok=True)

    def write(path: Path, record: dict[str, Any]) -> None:
        path.write_text(_json.dumps(record), encoding="utf-8")

    code = drive(directory, lambda plan, paths: _ScriptedGroup(script_for(paths, ranks, write)))
    finished = read_json(paths.finished)
    assert finished is not None
    lines = paths.events.read_text().splitlines() if paths.events.exists() else []
    types = [_json.loads(line)["payload"]["data"]["type"] for line in lines]
    return {"exit": code, **finished}, types


def test_a_rank_that_failed_on_its_own_outranks_a_cancellation_of_its_sibling(
    root: Path,
) -> None:
    """Rank 1 exits 3 by itself; rank 0 is stopped for a cancellation. The group failed."""

    def script(paths: WorkloadPaths, ranks: list[Any], write: Any) -> Any:
        def run() -> None:
            for rank in ranks:
                rank.spawning.touch()
                write(rank.started, {"pid": 1, "host": "h"})
            write(paths.cancel, {"requested_at": "now"})
            write(ranks[1].finished, {"exit_code": 3, "signal": None, "stopped_by": None})
            write(
                ranks[0].finished,
                {"exit_code": -15, "signal": 15, "cancelled": True, "stopped_by": "cancel"},
            )

        return run

    finished, types = _drive_scripted(root, script)
    assert (finished["exit_code"], finished["cancelled"], finished["failed_rank"]) == (3, False, 1)
    assert finished["exit"] == 3
    from xaytune.runtimes.local.runtime import finished_status

    assert finished_status(finished).state == "failed"
    assert types[0] == "WorkerReady" and types[-1] == "IncidentObserved"


def test_a_cancellation_alone_still_cancels_the_group(root: Path) -> None:
    def script(paths: WorkloadPaths, ranks: list[Any], write: Any) -> Any:
        def run() -> None:
            for rank in ranks:
                rank.spawning.touch()
                write(rank.started, {"pid": 1, "host": "h"})
            write(paths.cancel, {"requested_at": "now"})
            for rank in ranks:
                write(
                    rank.finished,
                    {"exit_code": -15, "signal": 15, "cancelled": True, "stopped_by": "cancel"},
                )

        return run

    finished, _ = _drive_scripted(root, script)
    assert finished["cancelled"] is True and finished["exit"] == 0


def test_a_group_that_failed_before_every_rank_started_was_never_ready(root: Path) -> None:
    """Rank 0 started, rank 1 never did, and Ray Train failed the group: no WorkerReady."""

    def script(paths: WorkloadPaths, ranks: list[Any], write: Any) -> Any:
        def run() -> None:
            ranks[0].spawning.touch()
            write(ranks[0].started, {"pid": 1, "host": "h"})
            write(paths.observations, {})  # rank 0 had begun to report
            write(
                ranks[0].finished,
                {"exit_code": -15, "signal": 15, "cancelled": False, "stopped_by": "group"},
            )
            raise RuntimeError("a worker actor was lost before rank 1 started")

        return run

    finished, types = _drive_scripted(root, script)
    assert (finished["exit_code"], finished.get("group_failed")) == (1, True)
    assert "WorkerReady" not in types
    assert types == ["IncidentObserved"]
    assert not WorkloadPaths(root / "op_scripted").started.exists()
