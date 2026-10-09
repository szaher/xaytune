"""RayJobsRuntime: Ray as one more RuntimeBackend, adopted after any restart, never doubled.

Every scenario runs real supervisors and workers. ``ProcessJobs`` stands in
for Ray's job manager in the ordinary matrix; the ``ray`` marked tests at the
end repeat the essential ones against a real local Ray head.
"""

from __future__ import annotations

import ast
import asyncio
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.test_ray.ray_support import ProcessJobs, RayHead
from xaytune.core.capabilities import PLUGIN_API_VERSIONS, PluginDescriptor
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.errors import IdempotencyConflictError
from xaytune.core.execution import (
    CommandEntrypoint,
    CompilerIdentity,
    ResolvedExecutionPlan,
    ResourceRequirements,
    TrainingExecutionSpec,
)
from xaytune.core.ids import OperationId
from xaytune.core.refs import RuntimeRef
from xaytune.ray import (
    RayJobsBackend,
    RayJobsConfig,
    RayJobsRuntime,
    RayUnavailableError,
    ray_jobs_runtime,
)
from xaytune.ray.runtime.jobs import ACCEPTED, REJECTED
from xaytune.runtimes import RuntimeBackend, StreamCursor, UnsupportedPlanError
from xaytune.runtimes.local import LocalRuntime
from xaytune.runtimes.local.paths import WorkloadPaths, read_json

_DESCRIPTOR = PluginDescriptor(
    api_version=PLUGIN_API_VERSIONS[0],
    name="fake-compiler",
    plugin_version="0.1.0",
    provider="tests",
    xaytune_version="0.6.0",
)
_TERMINAL = frozenset({"succeeded", "failed", "cancelled", "unknown"})
_SETTLE_SECONDS = 60.0
_PRELUDE = (
    "import sys, time\n"
    "from xaytune.core import telemetry as t\n"
    "from xaytune.runtimes.worker import ObservationWriter\n"
    "writer = ObservationWriter.from_environment()\n"
)


def _plan(
    body: str = "pass",
    *,
    runtime: str = "ray-jobs",
    target: RuntimeOperationTarget | None = None,
    **spec: Any,
) -> ResolvedExecutionPlan:
    return ResolvedExecutionPlan(
        spec=TrainingExecutionSpec(
            compiler=CompilerIdentity(name="fake", version="0.1.0", descriptor=_DESCRIPTOR),
            candidate_fingerprint="sha256:" + "0" * 64,
            entrypoint=CommandEntrypoint(argv=(sys.executable, "-c", _PRELUDE + body)),
            **spec,
        ),
        runtime=runtime,
        target=target or RuntimeOperationTarget(kind="training-attempt", id="ra_ray"),
    )


_TRAINS = (
    "writer.write(t.TrainingStartedPayload())\n"
    "for step in range(3):\n"
    "    writer.write(t.TrainingMetricObserved(optimizer_step=step + 1, loss=1.0 / (step + 1)))\n"
    "writer.write(t.TrainingCompletedPayload())\n"
)
_FAILS = "writer.write(t.TrainingStartedPayload())\nsys.exit(3)\n"
_SLEEPS = "writer.write(t.TrainingStartedPayload())\ntime.sleep(60)\n"


def _config(root: Path, address: str = "http://ray.test:8265") -> RayJobsConfig:
    return RayJobsConfig(address=address, runtime_env={}, shared_state_root=str(root))


@pytest.fixture
def jobs() -> Iterator[ProcessJobs]:
    fake = ProcessJobs()
    yield fake
    fake.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "ray-storage"


def _runtime(root: Path, jobs: Any) -> RayJobsRuntime:
    return RayJobsRuntime(_config(root), submission=jobs)


async def _settle(runtime: RayJobsRuntime, ref: RuntimeRef) -> Any:
    deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
    status = await runtime.get_status(ref)
    while status.state not in _TERMINAL and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
        status = await runtime.get_status(ref)
    return status


async def _await(predicate: Any, what: str) -> None:
    deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"never: {what}")


def _types(events: list[Any]) -> list[str]:
    return [event.payload.data.type for event in events]


# ---- the contract ------------------------------------------------------------------------


def test_the_ray_runtime_is_a_runtime_backend(root: Path, jobs: ProcessJobs) -> None:
    runtime = _runtime(root, jobs)
    assert isinstance(runtime, RuntimeBackend)
    assert runtime.descriptor.name == "ray-jobs"
    capabilities = runtime.capabilities()
    assert capabilities.resilience.supports_event_replay is True
    assert capabilities.resilience.reports_completed_operations is True
    assert capabilities.distributed.max_workers == 1


def test_a_successful_workload_succeeds_with_the_workers_telemetry(
    root: Path, jobs: ProcessJobs
) -> None:
    runtime = _runtime(root, jobs)

    async def scenario() -> None:
        ref = await runtime.submit_or_get(OperationId.generate(), _plan(_TRAINS))
        assert ref.backend == "ray-jobs"
        status = await _settle(runtime, ref)
        assert status.state == "succeeded" and status.exit_code == 0
        events = [event async for event in runtime.watch(ref)]
        assert _types(events) == [
            "WorkerReady",
            "TrainingStarted",
            "TrainingMetricObserved",
            "TrainingMetricObserved",
            "TrainingMetricObserved",
            "TrainingCompleted",
        ]
        assert [event.sequence for event in events] == list(range(6))
        assert {event.target.id for event in events} == {"ra_ray"}

    asyncio.run(scenario())


def test_a_failed_worker_fails_with_its_exit_code(root: Path, jobs: ProcessJobs) -> None:
    runtime = _runtime(root, jobs)

    async def scenario() -> None:
        ref = await runtime.submit_or_get(OperationId.generate(), _plan(_FAILS))
        status = await _settle(runtime, ref)
        assert (status.state, status.exit_code) == ("failed", 3)
        events = [event async for event in runtime.watch(ref)]
        assert _types(events)[-1] == "IncidentObserved"

    asyncio.run(scenario())


def test_an_evaluation_plan_runs_too(root: Path, jobs: ProcessJobs) -> None:
    """The same runtime runs an evaluation attempt: it never asks which kind of work it is."""
    from tests.test_runtimes.test_evaluation_transport import _evaluation_spec

    body = (
        "from xaytune.core.telemetry import EvaluationStartedPayload\n"
        "writer.write(EvaluationStartedPayload())\n"
    )
    spec = _evaluation_spec(
        entrypoint=CommandEntrypoint(argv=(sys.executable, "-c", _PRELUDE + body))
    )
    plan = ResolvedExecutionPlan(
        spec=spec,
        runtime="ray-jobs",
        target=RuntimeOperationTarget(kind="evaluation-attempt", id="ea_ray"),
    )
    runtime = _runtime(root, jobs)

    async def scenario() -> None:
        ref = await runtime.submit_or_get(OperationId.generate(), plan)
        assert (await _settle(runtime, ref)).state == "succeeded"
        events = [event async for event in runtime.watch(ref)]
        assert [e.payload.workload for e in events] == ["evaluation"] * len(events)
        assert "EvaluationStarted" in _types(events)

    asyncio.run(scenario())


# ---- one operation, one Ray job ------------------------------------------------------------


def test_the_same_operation_submitted_twice_is_one_job(root: Path, jobs: ProcessJobs) -> None:
    runtime = _runtime(root, jobs)
    operation = OperationId.generate()

    async def scenario() -> None:
        first = await runtime.submit_or_get(operation, _plan(_TRAINS))
        second = await runtime.submit_or_get(operation, _plan(_TRAINS))
        assert first == second
        assert jobs.submissions == 1 and len(jobs.jobs) == 1
        await _settle(runtime, first)

    asyncio.run(scenario())


def test_another_request_under_the_same_operation_is_a_conflict(
    root: Path, jobs: ProcessJobs
) -> None:
    runtime = _runtime(root, jobs)
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _plan(_TRAINS))
        with pytest.raises(IdempotencyConflictError):
            await runtime.submit_or_get(operation, _plan(_FAILS))
        assert len(jobs.jobs) == 1
        await _settle(runtime, ref)

    asyncio.run(scenario())


def test_a_recreated_runtime_looks_up_and_adopts_the_job(root: Path, jobs: ProcessJobs) -> None:
    """The controller died after Ray accepted the job, before it recorded the reference."""
    operation = OperationId.generate()

    async def scenario() -> None:
        first = _runtime(root, jobs)
        await first.submit_or_get(operation, _plan(_SLEEPS))
        # ... the controller dies here: the RuntimeRef is never recorded.

        restarted = _runtime(root, jobs)
        outcome = await restarted.lookup_operation(operation)
        assert outcome is not None and outcome.disposition == "accepted"
        assert outcome.runtime_ref == RuntimeRef(backend="ray-jobs", external_id=str(operation))
        adopted = await restarted.submit_or_get(operation, _plan(_SLEEPS))
        assert adopted == outcome.runtime_ref
        assert jobs.submissions == 1, "restarting never submits a second job"
        await restarted.cancel(adopted, OperationId.generate())
        assert (await _settle(restarted, adopted)).state == "cancelled"

    asyncio.run(scenario())


def test_a_crash_before_ray_was_asked_leaves_nothing_to_adopt(
    root: Path, jobs: ProcessJobs
) -> None:
    """The plan was written; Ray never heard. ``None`` -- and re-submitting starts it once."""
    operation = OperationId.generate()
    plan = _plan(_TRAINS)
    original = jobs.submit

    def never_arrives(*args: Any, **kwargs: Any) -> None:
        raise RayUnavailableError("cannot reach Ray at http://fake: ConnectionError")

    jobs.submit = never_arrives  # type: ignore[method-assign]

    async def scenario() -> None:
        runtime = _runtime(root, jobs)
        with pytest.raises(RayUnavailableError):
            await runtime.submit_or_get(operation, plan)
        assert read_json(WorkloadPaths(root / str(operation)).plan) is not None
        jobs.submit = original  # type: ignore[method-assign]
        restarted = _runtime(root, jobs)
        assert await restarted.lookup_operation(operation) is None
        ref = await restarted.submit_or_get(operation, plan)
        assert (await _settle(restarted, ref)).state == "succeeded"
        assert len(jobs.jobs) == 1

    asyncio.run(scenario())


def test_a_lost_answer_from_ray_is_resolved_by_its_job_store(root: Path, jobs: ProcessJobs) -> None:
    """Ray accepted the job, and the answer was lost: the retry finds it, never a second."""
    operation = OperationId.generate()
    runtime = _runtime(root, jobs)
    original = jobs.submit

    def accepted_then_lost(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        raise RayUnavailableError("the response never arrived")

    jobs.submit = accepted_then_lost  # type: ignore[method-assign]

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _plan(_TRAINS))
        assert ref.external_id == str(operation)
        assert len(jobs.jobs) == 1
        await _settle(runtime, ref)

    asyncio.run(scenario())


# ---- telemetry across restarts -------------------------------------------------------------


def test_watching_resumes_after_the_cursor_without_duplicates(
    root: Path, jobs: ProcessJobs
) -> None:
    runtime = _runtime(root, jobs)

    async def scenario() -> None:
        ref = await runtime.submit_or_get(OperationId.generate(), _plan(_TRAINS))
        await _settle(runtime, ref)
        everything = [event async for event in runtime.watch(ref)]
        cursor = StreamCursor(generation=0, sequence=2)
        restarted = _runtime(root, jobs)
        rest = [event async for event in restarted.watch(ref, cursor)]
        assert [e.sequence for e in rest] == [3, 4, 5]
        assert rest == everything[3:]

    asyncio.run(scenario())


def test_a_restarted_runtime_follows_a_running_workload_to_its_end(
    root: Path, jobs: ProcessJobs
) -> None:
    body = (
        "writer.write(t.TrainingStartedPayload())\n"
        "import pathlib, os\n"
        "gate = pathlib.Path(os.environ['XAYTUNE_WORKER_CONFIG_PATH']).parent / 'go'\n"
        "while not gate.exists():\n"
        "    time.sleep(0.02)\n"
        "writer.write(t.TrainingCompletedPayload())\n"
    )
    operation = OperationId.generate()

    async def scenario() -> None:
        first = _runtime(root, jobs)
        ref = await first.submit_or_get(operation, _plan(body))
        paths = WorkloadPaths(root / str(operation))
        await _await(
            lambda: paths.events.exists() and paths.events.read_text().count("\n") >= 2,
            "two events",
        )
        seen = []
        async for event in first.watch(ref):
            seen.append(event)
            if len(seen) == 2:
                break
        # The controller restarts holding the cursor it durably recorded.
        restarted = _runtime(root, jobs)
        (paths.directory / "go").write_text("")
        rest = [e async for e in restarted.watch(ref, StreamCursor(sequence=seen[-1].sequence))]
        assert _types(seen + rest) == ["WorkerReady", "TrainingStarted", "TrainingCompleted"]
        assert (await restarted.get_status(ref)).state == "succeeded"

    asyncio.run(scenario())


# ---- cancellation ---------------------------------------------------------------------------


def test_cancelling_a_running_workload_is_graceful_and_idempotent(
    root: Path, jobs: ProcessJobs
) -> None:
    runtime = _runtime(root, jobs)
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _plan(_SLEEPS))
        paths = WorkloadPaths(root / str(operation))
        await _await(lambda: read_json(paths.started) is not None, "the worker started")
        cancel = OperationId.generate()
        await runtime.cancel(ref, cancel)
        await runtime.cancel(ref, cancel)
        status = await _settle(runtime, ref)
        assert status.state == "cancelled"
        await runtime.cancel(ref, cancel)  # after the end: still nothing new
        assert (await runtime.get_status(ref)).state == "cancelled"
        assert read_json(paths.finished)["cancelled"] is True  # type: ignore[index]

    asyncio.run(scenario())


def test_a_job_ray_cannot_schedule_stays_pending_until_ray_ends_it_then_is_cancelled(
    root: Path,
) -> None:
    """Ray ignores a stop for a job it cannot schedule; its start timeout ends it later."""
    held = ProcessJobs(hold=True)
    held.stop_ignores_pending = True
    runtime = _runtime(root, held)
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _plan(_SLEEPS))
        await runtime.cancel(ref, OperationId.generate())
        waiting = await runtime.get_status(ref)
        assert waiting.state == "pending" and "cancellation requested" in (waiting.detail or "")
        held.time_out(str(operation))
        ended = await runtime.get_status(ref)
        assert ended.state == "cancelled" and "before it started" in (ended.detail or "")

    try:
        asyncio.run(scenario())
    finally:
        held.close()


def test_a_job_that_failed_to_start_without_a_cancel_is_failed(root: Path) -> None:
    held = ProcessJobs(hold=True)
    runtime = _runtime(root, held)
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _plan(_SLEEPS))
        held.time_out(str(operation))
        assert (await runtime.get_status(ref)).state == "failed"

    try:
        asyncio.run(scenario())
    finally:
        held.close()


def test_cancelling_a_job_ray_has_not_started_stops_it_there(root: Path) -> None:
    held = ProcessJobs(hold=True)
    runtime = _runtime(root, held)
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _plan(_SLEEPS))
        assert (await runtime.get_status(ref)).state == "pending"
        await runtime.cancel(ref, OperationId.generate())
        assert (await runtime.get_status(ref)).state == "cancelled"
        outcome = await runtime.lookup_operation(operation)
        assert outcome is not None and outcome.disposition == "completed"

    try:
        asyncio.run(scenario())
    finally:
        held.close()


# ---- failing closed -------------------------------------------------------------------------


def test_an_unreachable_cluster_is_unknown_and_never_never_received(
    root: Path, jobs: ProcessJobs
) -> None:
    runtime = _runtime(root, jobs)
    operation = OperationId.generate()

    async def scenario() -> None:
        held = ProcessJobs(hold=True)
        try:
            pending = _runtime(root, held)
            ref = await pending.submit_or_get(operation, _plan(_SLEEPS))
            held.unreachable = True
            status = await pending.get_status(ref)
            assert status.state == "unknown" and "cannot reach Ray" in (status.detail or "")
            with pytest.raises(RayUnavailableError):
                await pending.lookup_operation(operation)
        finally:
            held.close()
        assert runtime is not None

    asyncio.run(scenario())


def test_a_job_ray_forgot_is_unknown_and_never_never_received(
    root: Path, jobs: ProcessJobs
) -> None:
    """Ray's 404 for a job it accepted is not "never received": nothing is resubmitted."""
    held = ProcessJobs(hold=True)
    runtime = _runtime(root, held)
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _plan(_SLEEPS))
        assert read_json(root / str(operation) / ACCEPTED) is not None
        held.forget(str(operation))
        status = await runtime.get_status(ref)
        assert status.state == "unknown" and "no record" in (status.detail or "")
        outcome = await _runtime(root, held).lookup_operation(operation)
        assert outcome is not None and outcome.disposition == "accepted"
        assert outcome.runtime_ref == ref and outcome.status is not None
        assert outcome.status.state == "unknown" and ACCEPTED in (outcome.status.detail or "")
        assert await _runtime(root, held).submit_or_get(operation, _plan(_SLEEPS)) == ref
        assert held.submissions == 1

    try:
        asyncio.run(scenario())
    finally:
        held.close()


def test_a_finished_job_ray_forgot_is_completed_and_never_rerun(
    root: Path, jobs: ProcessJobs
) -> None:
    operation = OperationId.generate()

    async def scenario() -> None:
        runtime = _runtime(root, jobs)
        ref = await runtime.submit_or_get(operation, _plan(_TRAINS))
        assert (await _settle(runtime, ref)).state == "succeeded"
        jobs.forget(str(operation))
        restarted = _runtime(root, jobs)
        outcome = await restarted.lookup_operation(operation)
        assert outcome is not None and outcome.disposition == "completed"
        assert outcome.status is not None and outcome.status.state == "succeeded"
        assert await restarted.submit_or_get(operation, _plan(_TRAINS)) == ref
        assert jobs.submissions == 1

    asyncio.run(scenario())


def test_a_job_whose_supervisor_ran_is_not_new_even_without_an_acceptance_record(
    root: Path, jobs: ProcessJobs
) -> None:
    """Belt and braces: the supervisor's own claim is evidence Ray had the job."""
    operation = OperationId.generate()

    async def scenario() -> None:
        runtime = _runtime(root, jobs)
        ref = await runtime.submit_or_get(operation, _plan(_SLEEPS))
        paths = WorkloadPaths(root / str(operation))
        await _await(lambda: read_json(paths.started) is not None, "the worker started")
        (root / str(operation) / ACCEPTED).unlink()
        jobs.forget(str(operation))
        outcome = await runtime.lookup_operation(operation)
        assert outcome is not None and outcome.disposition == "accepted"
        assert outcome.runtime_ref == ref

    asyncio.run(scenario())


def test_the_same_operation_in_another_runtime_env_is_a_conflict(
    root: Path, jobs: ProcessJobs
) -> None:
    """The runtime_env is part of what was submitted, so part of the request."""
    operation = OperationId.generate()

    def runtime(environment: dict[str, Any]) -> RayJobsRuntime:
        return RayJobsRuntime(
            RayJobsConfig(
                address="http://ray.test:8265",
                runtime_env=environment,
                shared_state_root=str(root),
            ),
            submission=jobs,
        )

    async def scenario() -> None:
        first = runtime({"pip": ["xaytune==1"]})
        ref = await first.submit_or_get(operation, _plan(_SLEEPS))
        assert (
            await runtime({"pip": ["xaytune==1"]}).submit_or_get(operation, _plan(_SLEEPS)) == ref
        )
        with pytest.raises(IdempotencyConflictError) as conflict:
            await runtime({"pip": ["xaytune==2"]}).submit_or_get(operation, _plan(_SLEEPS))
        assert conflict.value.differing == ("runtime_env",)
        # ... and still after Ray forgot the job: the acceptance record says so.
        jobs.forget(str(operation))
        with pytest.raises(IdempotencyConflictError):
            await runtime({}).submit_or_get(operation, _plan(_SLEEPS))
        assert jobs.submissions == 1

    asyncio.run(scenario())


def test_a_cancellation_id_is_bound_to_one_workload(root: Path, jobs: ProcessJobs) -> None:
    runtime = _runtime(root, jobs)

    async def scenario() -> None:
        first_op, second_op = OperationId.generate(), OperationId.generate()
        first = await runtime.submit_or_get(first_op, _plan(_SLEEPS))
        second = await runtime.submit_or_get(second_op, _plan(_SLEEPS))
        cancel = OperationId.generate()
        await runtime.cancel(first, cancel)
        # The same id for the same workload is a retry, and reasserts the effect.
        WorkloadPaths(root / str(first_op)).cancel.unlink()
        await _runtime(root, jobs).cancel(first, cancel)
        assert read_json(WorkloadPaths(root / str(first_op)).cancel) is not None
        # The same id for another workload is not.
        with pytest.raises(IdempotencyConflictError) as conflict:
            await _runtime(root, jobs).cancel(second, cancel)
        assert conflict.value.differing == ("request_digest",)
        assert read_json(WorkloadPaths(root / str(second_op)).cancel) is None
        # Nor is an id that already names a submission, or the reverse.
        with pytest.raises(IdempotencyConflictError) as conflict:
            await runtime.cancel(second, first_op)
        assert conflict.value.differing == ("operation_type",)
        with pytest.raises(IdempotencyConflictError) as conflict:
            await runtime.submit_or_get(cancel, _plan(_SLEEPS))
        assert conflict.value.differing == ("operation_type",)

    asyncio.run(scenario())


@pytest.mark.parametrize("variable", ["RAY_API_SERVER_ADDRESS", "RAY_ADDRESS"])
def test_the_configured_address_is_authoritative(
    variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ray's SDK prefers these variables to its argument; another address is refused."""
    monkeypatch.delenv("RAY_API_SERVER_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setenv(variable, "http://user:secret@elsewhere:8265")
    backend = RayJobsBackend("http://ray.test:8265")
    for ask in (
        lambda: backend.info("op_x"),
        lambda: backend.stop("op_x"),
        lambda: backend.submit("op_x", "true", metadata={}, resources={}, runtime_env={}),
    ):
        with pytest.raises(RayUnavailableError, match=variable) as refused:
            ask()
        assert "secret" not in str(refused.value) and "elsewhere" not in str(refused.value)
    assert backend._client is None  # refused before a client existed


def test_a_reference_from_another_backend_is_not_answered(root: Path, jobs: ProcessJobs) -> None:
    runtime = _runtime(root, jobs)
    with pytest.raises(KeyError):
        asyncio.run(runtime.get_status(RuntimeRef(backend="local", external_id="op_x")))


@pytest.mark.parametrize(
    ("plan", "reason"),
    [
        (_plan(runtime="local"), "resolved for 'local'"),
        (_plan(resources=ResourceRequirements(workers=2)), "runs one worker"),
        (_plan(resources=ResourceRequirements(gpu_type="a100")), "gpu_type"),
        (_plan(resources=ResourceRequirements(max_runtime_seconds=60)), "max_runtime_seconds"),
        (_plan(environment={"WORLD_SIZE": "8"}), "WORLD_SIZE"),
    ],
    ids=["other-runtime", "two-workers", "gpu-type", "deadline", "topology"],
)
def test_a_plan_this_runtime_cannot_honour_is_refused_and_recorded(
    root: Path, jobs: ProcessJobs, plan: ResolvedExecutionPlan, reason: str
) -> None:
    runtime = _runtime(root, jobs)
    operation = OperationId.generate()

    async def scenario() -> None:
        with pytest.raises(UnsupportedPlanError, match=reason):
            await runtime.submit_or_get(operation, plan)
        outcome = await _runtime(root, jobs).lookup_operation(operation)
        assert outcome is not None and outcome.disposition == "rejected" and outcome.may_reissue
        assert jobs.submissions == 0
        assert (root / str(operation) / REJECTED).exists()

    asyncio.run(scenario())


def test_the_plans_resources_become_ray_scheduling_options_and_nothing_else(
    root: Path,
) -> None:
    held = ProcessJobs(hold=True)
    runtime = _runtime(root, held)
    operation = OperationId.generate()
    plan = _plan(resources=ResourceRequirements(cpus=2.0, gpus=1, memory_bytes=2**30))
    try:
        asyncio.run(runtime.submit_or_get(operation, plan))
        job = held.jobs[str(operation)]
        assert job.resources == {
            "entrypoint_num_cpus": 2.0,
            "entrypoint_num_gpus": 1,
            "entrypoint_memory": 2**30,
        }
        assert job.metadata["xaytune.request_digest"] == plan.request_digest("submit")
        assert job.entrypoint.split()[1:3] == ["-m", "xaytune.ray.runtime.supervisor"]
    finally:
        held.close()


# ---- configuration and boundaries ----------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        {"runtime_env": {}, "shared_state_root": "/srv/x"},
        {"address": "ray://head:10001", "runtime_env": {}, "shared_state_root": "/srv/x"},
        {"address": "http://h:8265", "runtime_env": {}, "shared_state_root": "relative"},
        {"address": "http://h:8265", "shared_state_root": "/srv/x"},
        {"address": "http://h:8265", "runtime_env": {}},
        {"address": "http://h:8265", "runtime_env": {}, "shared_state_root": "/x", "token": "s"},
        # No interpreter path: the job's environment provides ``python``.
        {"address": "http://h:8265", "runtime_env": {}, "shared_state_root": "/x", "python": "/p"},
    ],
    ids=[
        "no-address",
        "client-address",
        "relative-root",
        "no-runtime-env",
        "no-state-root",
        "unknown-key",
        "python-path",
    ],
)
def test_configuration_is_explicit_and_closed(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ray_jobs_runtime(config)


def test_the_factory_builds_the_runtime_a_host_registers(tmp_path: Path) -> None:
    runtime = ray_jobs_runtime(
        {"address": "http://127.0.0.1:1", "runtime_env": {}, "shared_state_root": str(tmp_path)}
    )
    assert isinstance(runtime, RayJobsRuntime) and isinstance(runtime, RuntimeBackend)


def test_the_configured_runtime_env_goes_to_every_job_and_no_path_is_assumed(
    tmp_path: Path,
) -> None:
    """The code travels in Ray's runtime_env; the entrypoint names no controller-side path."""
    held = ProcessJobs(hold=True)
    environment = {"working_dir": "s3://bucket/code.zip", "env_vars": {"MODE": "x"}}
    runtime = RayJobsRuntime(
        RayJobsConfig(
            address="http://ray.test:8265",
            runtime_env=environment,
            shared_state_root=str(tmp_path),
        ),
        submission=held,
    )
    operation = OperationId.generate()
    try:
        asyncio.run(runtime.submit_or_get(operation, _plan()))
        job = held.jobs[str(operation)]
        assert job.runtime_env == {"working_dir": "s3://bucket/code.zip", "env_vars": {"MODE": "x"}}
        assert job.entrypoint.split()[0] == "python"
    finally:
        held.close()


def test_the_ray_runtime_never_names_a_candidate() -> None:
    """Mechanical translation only: no candidate, compiler, planner or controller in reach."""
    forbidden_modules = (
        "xaytune.core.domain.candidate",
        "xaytune.compilation",
        "xaytune.planning",
        "xaytune.experiment",
        "xaytune.storage",
        "xaytune.policy",
        "xaytune.decision",
        "xaytune.resilience",
    )
    forbidden_names = {"CandidateSpec", "TrainingSpec", "AdapterSpec", "TrainingKind"}
    package = Path(__file__).resolve().parents[2] / "xaytune" / "ray"
    offences = []
    for source in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                forbidden_modules
            ):
                offences.append(f"{source.name} imports {node.module}")
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                offences.append(f"{source.name} refers to {node.id}")
    assert offences == []


def test_ray_is_imported_by_the_ray_runtime_alone() -> None:
    """Nothing above the runtime boundary -- nor xaytune.ray itself -- loads Ray on import."""
    probe = (
        "import sys\n"
        "import xaytune.core, xaytune.planning, xaytune.agent, xaytune.storage\n"
        "import xaytune.experiment, xaytune.daemon, xaytune.runtimes.local\n"
        "import xaytune.ray, xaytune.runtimes\n"
        "print(sorted(m for m in sys.modules if m == 'ray' or m.startswith('ray.')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
    root = Path(__file__).resolve().parents[2] / "xaytune"
    importers = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if any(
            (
                isinstance(node, ast.Import)
                and any(a.name.split(".")[0] == "ray" for a in node.names)
            )
            or (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "ray")
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
    )
    # The Jobs client, the Ray Train driver Ray runs on the cluster, and Ray Tune's
    # searcher (PR-034) -- each lazily.
    assert importers == ["ray/runtime/train_driver.py", "ray/search.py", "ray/submission/jobs.py"]


def test_the_local_runtime_still_refuses_ray_plans(tmp_path: Path) -> None:
    runtime = LocalRuntime(tmp_path / "local")
    try:
        with pytest.raises(UnsupportedPlanError, match="resolved for 'ray-jobs'"):
            asyncio.run(runtime.submit_or_get(OperationId.generate(), _plan()))
    finally:
        runtime.close()


# ---- a real Ray head ------------------------------------------------------------------------


def _real(root: Path, head: RayHead) -> RayJobsRuntime:
    return RayJobsRuntime(_config(root, head.address))


@pytest.mark.ray
def test_on_ray_a_workload_runs_once_is_adopted_and_reports_its_telemetry(
    root: Path,
    ray_head: RayHead,
) -> None:
    operation = OperationId.generate()

    async def scenario() -> None:
        first = _real(root, ray_head)
        ref = await first.submit_or_get(operation, _plan(_TRAINS))
        restarted = _real(root, ray_head)
        outcome = await restarted.lookup_operation(operation)
        assert outcome is not None and outcome.runtime_ref == ref
        assert await restarted.submit_or_get(operation, _plan(_TRAINS)) == ref
        status = await _settle(restarted, ref)
        assert status.state == "succeeded", status
        events = [event async for event in restarted.watch(ref)]
        assert _types(events)[-1] == "TrainingCompleted"
        assert len([e for e in events if e.payload.data.type == "WorkerReady"]) == 1
        assert await restarted.lookup_operation(OperationId.generate()) is None

    asyncio.run(scenario())


@pytest.mark.ray
def test_on_ray_a_failed_worker_fails_and_a_running_one_cancels(
    root: Path,
    ray_head: RayHead,
) -> None:
    runtime = _real(root, ray_head)

    async def scenario() -> None:
        failed = await runtime.submit_or_get(OperationId.generate(), _plan(_FAILS))
        sleeping_op = OperationId.generate()
        sleeping = await runtime.submit_or_get(sleeping_op, _plan(_SLEEPS))
        assert (
            (await _settle(runtime, failed)).state,
            (await runtime.get_status(failed)).exit_code,
        ) == (
            "failed",
            3,
        )
        paths = WorkloadPaths(root / str(sleeping_op))
        await _await(lambda: read_json(paths.started) is not None, "the worker started")
        await runtime.cancel(sleeping, OperationId.generate())
        await runtime.cancel(sleeping, OperationId.generate())
        assert (await _settle(runtime, sleeping)).state == "cancelled"

    asyncio.run(scenario())


@pytest.mark.ray
def test_on_ray_another_address_in_the_environment_is_refused_before_any_job(
    root: Path,
    ray_head: RayHead,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAY_ADDRESS naming another cluster would redirect Ray's SDK; nothing is submitted."""
    monkeypatch.setenv("RAY_ADDRESS", "http://127.0.0.1:1")
    runtime = _real(root, ray_head)
    operation = OperationId.generate()
    with pytest.raises(RayUnavailableError, match="RAY_ADDRESS"):
        asyncio.run(runtime.submit_or_get(operation, _plan(_TRAINS)))
    monkeypatch.setenv("RAY_ADDRESS", ray_head.address + "/")  # the same cluster is fine
    assert asyncio.run(_real(root, ray_head).lookup_operation(operation)) is None


@pytest.mark.ray
def test_on_ray_a_job_waiting_for_resources_is_pending_and_cancels_there(
    root: Path,
    ray_head: RayHead,
) -> None:
    """The head has no GPU, so a job asking for one waits in Ray until it is stopped."""
    runtime = _real(root, ray_head)

    async def scenario() -> None:
        ref = await runtime.submit_or_get(
            OperationId.generate(), _plan(_SLEEPS, resources=ResourceRequirements(gpus=1))
        )
        assert (await runtime.get_status(ref)).state == "pending"
        await runtime.cancel(ref, OperationId.generate())
        status = await runtime.get_status(ref)
        assert status.state in ("pending", "cancelled"), status
        # Ray decides when it ends; this head ends it at its start timeout.
        assert (await _settle(runtime, ref)).state == "cancelled"

    asyncio.run(scenario())


def test_the_ray_tests_are_required_where_ray_is() -> None:
    """The ``ray`` CI job sets XAYTUNE_REQUIRE_RAY: there, Ray must import."""
    if os.environ.get("XAYTUNE_REQUIRE_RAY"):
        import ray  # noqa: F401
