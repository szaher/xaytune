"""LocalRuntime: the first backend that actually runs something.

These are not unit tests around a mock. Every one of them spawns real
processes and reads real files, because the properties being checked --
an operation that survives a restart, an exit code nobody was waiting for,
a cancellation that reaches a process group -- are exactly the ones a mock
would grant for free.
"""

from __future__ import annotations

import ast
import asyncio
import os
import signal
import sys
from pathlib import Path

import pytest

from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.errors import IdempotencyConflictError
from xaytune.core.execution import (
    CommandEntrypoint,
    CompilerIdentity,
    ContainerSpec,
    PythonModuleEntrypoint,
    ResolvedExecutionPlan,
    ResourceRequirements,
    SecretRef,
    TrainingExecutionSpec,
)
from xaytune.core.ids import OperationId
from xaytune.runtimes import RuntimeBackend, RuntimeStatus
from xaytune.runtimes.local import BACKEND, LocalRuntime, UnsupportedPlanError
from xaytune.runtimes.local.launcher import run as run_launcher
from xaytune.runtimes.local.paths import WorkloadPaths

_TERMINAL = frozenset({"succeeded", "failed", "cancelled", "unknown"})
_SETTLE_SECONDS = 20.0


def _plan(
    *args: str,
    target: RuntimeOperationTarget | None = None,
    **spec_kwargs: object,
) -> ResolvedExecutionPlan:
    """A plan that runs *args* as a command."""
    spec = TrainingExecutionSpec(
        compiler=CompilerIdentity(name="fake", version="0.1.0"),
        candidate_fingerprint="sha256:" + "0" * 64,
        entrypoint=CommandEntrypoint(argv=args),
        **spec_kwargs,  # type: ignore[arg-type]
    )
    return ResolvedExecutionPlan(
        spec=spec,
        runtime="local",
        target=target or RuntimeOperationTarget(kind="training-attempt", id="ra_1"),
    )


def _python(source: str, *arguments: str) -> tuple[str, ...]:
    return (sys.executable, "-c", source, *arguments)


async def _settle(runtime: LocalRuntime, ref: object) -> RuntimeStatus:
    """Wait for a workload to reach a state it will not leave."""
    deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
    status = await runtime.get_status(ref)  # type: ignore[arg-type]
    while status.state not in _TERMINAL and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.02)
        status = await runtime.get_status(ref)  # type: ignore[arg-type]
    return status


async def _await_state(runtime: LocalRuntime, ref: object, state: str) -> None:
    deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
    while asyncio.get_running_loop().time() < deadline:
        if (await runtime.get_status(ref)).state == state:  # type: ignore[arg-type]
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"workload never reached {state!r}")


@pytest.fixture
def runtime(tmp_path: Path) -> LocalRuntime:
    backend = LocalRuntime(tmp_path / "runtime")
    yield backend
    backend.close()


# ---- the boundary holds --------------------------------------------------


def test_the_local_runtime_is_a_runtime_backend(runtime: LocalRuntime) -> None:
    assert isinstance(runtime, RuntimeBackend)


def test_the_local_runtime_never_names_a_candidate() -> None:
    """The execute half must not be able to interpret scientific intent.

    Checked against the source rather than against ``sys.modules``: importing
    anything from ``xaytune.core`` loads the whole package, candidate included,
    so a module-loading check would pass or fail for reasons that have nothing
    to do with this package. What matters is whether the runtime can *name*
    those types, and that is a property of what it imports and refers to.
    """
    forbidden_modules = {"xaytune.core.domain.candidate", "xaytune.compilation"}
    forbidden_names = {
        "AlgorithmSpec",
        "CandidateSpec",
        "EnvironmentSpec",
        "OptimizationSpec",
        "RewardSpec",
        "TrainingSpec",
        "TrainingKind",
    }

    package = Path(__file__).resolve().parents[2] / "xaytune" / "runtimes" / "local"
    offences: list[str] = []

    for source_file in sorted(package.glob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
                offences.append(f"{source_file.name} imports {node.module}")
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                offences.append(f"{source_file.name} refers to {node.id}")

    assert offences == []


def test_the_local_runtime_does_not_import_the_control_plane() -> None:
    """A backend is driven by the control plane, not the other way round.

    Run in a subprocess because this process has already imported half the
    repository. A runtime that pulled in ``xaytune.storage`` would invert the
    dependency and make every worker process pay to import a write surface it
    is not allowed to use.
    """
    result = __import__("subprocess").run(
        [
            sys.executable,
            "-c",
            "import sys, xaytune.runtimes.local\n"
            "print([m for m in sys.modules if m.startswith('xaytune.storage')])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"


def test_capabilities_declare_what_recovery_may_assume(runtime: LocalRuntime) -> None:
    """Both honest positives here, and both are backed by files."""
    resilience = runtime.capabilities().resilience
    assert resilience is not None
    assert resilience.supports_event_replay is True
    assert resilience.reports_completed_operations is True


def test_this_backend_runs_one_worker(runtime: LocalRuntime) -> None:
    """No torchrun. Choosing a world size means reading the training config."""
    distributed = runtime.capabilities().distributed
    assert distributed is not None
    assert distributed.max_workers == 1


# ---- operation identity --------------------------------------------------


def test_resubmitting_returns_the_same_workload_and_starts_one_process(
    runtime: LocalRuntime, tmp_path: Path
) -> None:
    """Get-or-create, proved by counting what the workload actually did."""
    marker = tmp_path / "ran.txt"
    plan = _plan(*_python("import sys; open(sys.argv[1], 'a').write('x')", str(marker)))
    operation_id = OperationId.generate()

    async def scenario() -> tuple[object, object]:
        first = await runtime.submit_or_get(operation_id, plan)
        second = await runtime.submit_or_get(operation_id, plan)
        await _settle(runtime, first)
        return first, second

    first, second = asyncio.run(scenario())

    assert first == second
    assert marker.read_text() == "x", "the worker ran exactly once"


def test_reusing_an_operation_id_for_another_request_is_refused(
    runtime: LocalRuntime,
) -> None:
    """Guessing which one the caller meant would start a second workload."""
    operation_id = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation_id, _plan(*_python("pass")))
        await _settle(runtime, ref)
        with pytest.raises(IdempotencyConflictError, match="request_digest"):
            await runtime.submit_or_get(operation_id, _plan(*_python("print(1)")))

    asyncio.run(scenario())


def test_an_operation_that_never_arrived_is_not_a_rejection(
    runtime: LocalRuntime,
) -> None:
    """None and 'rejected' are different facts; only one is a record."""
    assert asyncio.run(runtime.lookup_operation(OperationId.generate())) is None


# ---- what the runtime refuses --------------------------------------------


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("secrets", {"secrets": (SecretRef(name="HF_TOKEN", source="env"),)}),
        ("container", {"container": ContainerSpec(image="trainer:1")}),
        ("workers", {"resources": ResourceRequirements(workers=4)}),
    ],
)
def test_a_plan_it_cannot_honour_is_refused_and_recorded(
    runtime: LocalRuntime, label: str, kwargs: dict[str, object]
) -> None:
    """Refusing beats ignoring, and the refusal has to be durable.

    A rejection the runtime forgot would read as "never received" after a
    restart, and never-received is the one answer that makes re-issuing the
    identical request look safe.
    """
    operation_id = OperationId.generate()

    async def scenario() -> object:
        with pytest.raises(UnsupportedPlanError):
            await runtime.submit_or_get(operation_id, _plan(*_python("pass"), **kwargs))
        return await runtime.lookup_operation(operation_id)

    outcome = asyncio.run(scenario())

    assert outcome is not None
    assert outcome.disposition == "rejected"
    assert outcome.runtime_ref is None, "nothing was started, so there is nothing to name"
    assert outcome.may_reissue is True


# ---- process outcomes ----------------------------------------------------


def test_a_successful_worker_completes_its_operation(runtime: LocalRuntime) -> None:
    operation_id = OperationId.generate()

    async def scenario() -> tuple[RuntimeStatus, object]:
        ref = await runtime.submit_or_get(operation_id, _plan(*_python("pass")))
        status = await _settle(runtime, ref)
        return status, await runtime.lookup_operation(operation_id)

    status, outcome = asyncio.run(scenario())

    assert status.state == "succeeded"
    assert status.exit_code == 0
    assert outcome is not None
    assert outcome.disposition == "completed"
    assert outcome.may_reissue is False, "a finished workload is adopted, not repeated"


def test_a_failing_worker_reports_its_exit_code(runtime: LocalRuntime) -> None:
    async def scenario() -> RuntimeStatus:
        ref = await runtime.submit_or_get(
            OperationId.generate(), _plan(*_python("import sys; sys.exit(3)"))
        )
        return await _settle(runtime, ref)

    status = asyncio.run(scenario())

    assert status.state == "failed"
    assert status.exit_code == 3


def test_a_worker_that_cannot_be_started_fails_rather_than_vanishes(
    runtime: LocalRuntime,
) -> None:
    """An exec that never happened still has to produce an answer."""

    async def scenario() -> RuntimeStatus:
        ref = await runtime.submit_or_get(
            OperationId.generate(), _plan("/nonexistent/xaytune-no-such-binary")
        )
        return await _settle(runtime, ref)

    status = asyncio.run(scenario())

    assert status.state == "failed"
    assert "could not be started" in (status.detail or "")


# ---- cancellation --------------------------------------------------------


def test_cancelling_stops_the_workload(runtime: LocalRuntime) -> None:
    async def scenario() -> RuntimeStatus:
        ref = await runtime.submit_or_get(
            OperationId.generate(), _plan(*_python("import time; time.sleep(120)"))
        )
        await _await_state(runtime, ref, "running")
        await runtime.cancel(ref, OperationId.generate())
        return await _settle(runtime, ref)

    status = asyncio.run(scenario())

    assert status.state == "cancelled"


def test_retrying_one_cancellation_does_not_signal_twice(
    runtime: LocalRuntime, tmp_path: Path
) -> None:
    """The second call must be a no-op, and the worker is what proves it.

    A worker that counts the signals it receives is the only witness that
    cannot be satisfied by bookkeeping: the registry could look idempotent
    while two signals still reached the process group.
    """
    counter = tmp_path / "signals.txt"
    source = (
        "import signal, sys, time\n"
        "path = sys.argv[1]\n"
        "def handler(signum, frame):\n"
        "    open(path, 'a').write('term\\n')\n"
        "signal.signal(signal.SIGTERM, handler)\n"
        "time.sleep(5)\n"
    )
    cancellation = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(
            OperationId.generate(), _plan(*_python(source, str(counter)))
        )
        await _await_state(runtime, ref, "running")
        await asyncio.sleep(0.5)  # let the handler be installed
        await runtime.cancel(ref, cancellation)
        await runtime.cancel(ref, cancellation)
        await runtime.cancel(ref, cancellation)
        await _settle(runtime, ref)

    asyncio.run(scenario())

    received = counter.read_text().count("term") if counter.exists() else 0
    assert received == 1, f"one cancellation, {received} signals"


def test_a_workload_that_finished_first_is_not_recorded_as_cancelled(
    runtime: LocalRuntime,
) -> None:
    """The cancel arrived too late, which means the work is done (ADR-013 §5).

    Recording it as cancelled would discard artifacts that exist.
    """

    async def scenario() -> RuntimeStatus:
        ref = await runtime.submit_or_get(OperationId.generate(), _plan(*_python("pass")))
        await _settle(runtime, ref)
        await runtime.cancel(ref, OperationId.generate())
        return await runtime.get_status(ref)

    status = asyncio.run(scenario())

    assert status.state == "succeeded"


# ---- surviving a restart -------------------------------------------------


def test_a_recreated_runtime_finds_a_finished_workload(tmp_path: Path) -> None:
    """The property an in-memory registry cannot have.

    The second runtime never spawned this worker and never waited for it, so
    its exit code exists only because the launcher wrote it down.
    """
    root = tmp_path / "runtime"
    operation_id = OperationId.generate()

    async def submit() -> None:
        first = LocalRuntime(root)
        try:
            ref = await first.submit_or_get(
                operation_id, _plan(*_python("import sys; sys.exit(7)"))
            )
            await _settle(first, ref)
        finally:
            first.close()

    asyncio.run(submit())

    second = LocalRuntime(root)
    try:
        outcome = asyncio.run(second.lookup_operation(operation_id))
    finally:
        second.close()

    assert outcome is not None
    assert outcome.disposition == "completed"
    assert outcome.status is not None
    assert outcome.status.exit_code == 7


def test_a_recreated_runtime_adopts_a_running_workload(tmp_path: Path) -> None:
    """Still running is not the same answer as finished, and neither is lost."""
    root = tmp_path / "runtime"
    operation_id = OperationId.generate()

    async def submit() -> None:
        first = LocalRuntime(root)
        try:
            ref = await first.submit_or_get(
                operation_id, _plan(*_python("import time; time.sleep(120)"))
            )
            await _await_state(first, ref, "running")
        finally:
            first.close()

    asyncio.run(submit())

    second = LocalRuntime(root)
    try:
        outcome = asyncio.run(second.lookup_operation(operation_id))
        assert outcome is not None
        assert outcome.disposition == "accepted"
        assert outcome.is_running is True
        assert outcome.may_reissue is False
        asyncio.run(second.cancel(outcome.runtime_ref, OperationId.generate()))
        asyncio.run(_settle(second, outcome.runtime_ref))
    finally:
        second.close()


def test_a_recreated_runtime_resubmitting_starts_nothing_new(
    tmp_path: Path,
) -> None:
    """Idempotency has to survive the process that promised it."""
    root = tmp_path / "runtime"
    marker = tmp_path / "ran.txt"
    plan = _plan(*_python("import sys; open(sys.argv[1], 'a').write('x')", str(marker)))
    operation_id = OperationId.generate()

    async def submit(backend: LocalRuntime) -> object:
        ref = await backend.submit_or_get(operation_id, plan)
        await _settle(backend, ref)
        return ref

    first = LocalRuntime(root)
    try:
        original = asyncio.run(submit(first))
    finally:
        first.close()

    second = LocalRuntime(root)
    try:
        again = asyncio.run(submit(second))
    finally:
        second.close()

    assert original == again
    assert marker.read_text() == "x"


def test_a_workload_whose_launcher_died_is_unknown_not_failed(
    runtime: LocalRuntime,
) -> None:
    """ "I cannot tell" is an answer, and it is not "it failed".

    The controller's response to an unobserved ending is to reconcile. Reading
    it as a failure would retire a workload that may still have produced
    artifacts, and reading it as success would be worse.
    """

    async def scenario() -> RuntimeStatus:
        ref = await runtime.submit_or_get(
            OperationId.generate(), _plan(*_python("import time; time.sleep(120)"))
        )
        await _await_state(runtime, ref, "running")

        workload = runtime._registry.workload(ref.external_id)
        assert workload is not None and workload.launcher_pid is not None
        os.killpg(os.getpgid(workload.launcher_pid), signal.SIGKILL)

        return await _settle(runtime, ref)

    status = asyncio.run(scenario())

    assert status.state == "unknown"
    assert "never started" in (status.detail or "")


# ---- telemetry -----------------------------------------------------------


def test_telemetry_is_ordered_and_names_the_plans_target(runtime: LocalRuntime) -> None:
    target = RuntimeOperationTarget(kind="training-attempt", id="ra_77")

    async def scenario() -> list[object]:
        ref = await runtime.submit_or_get(
            OperationId.generate(), _plan(*_python("import sys; sys.exit(2)"), target=target)
        )
        await _settle(runtime, ref)
        return [event async for event in runtime.watch(ref)]

    events = asyncio.run(scenario())

    assert [event.payload.type for event in events] == ["WorkerReady", "IncidentObserved"]
    assert [event.sequence for event in events] == [0, 1]
    assert all(event.target == target for event in events)


def test_an_evaluation_target_gets_evaluation_telemetry(runtime: LocalRuntime) -> None:
    """The family follows the target, and the envelope will not let it not."""
    target = RuntimeOperationTarget(kind="evaluation-attempt", id="ea_1")

    async def scenario() -> list[object]:
        ref = await runtime.submit_or_get(
            OperationId.generate(), _plan(*_python("pass"), target=target)
        )
        await _settle(runtime, ref)
        return [event async for event in runtime.watch(ref)]

    events = asyncio.run(scenario())

    assert events
    assert all(event.payload.workload == "evaluation" for event in events)


def test_watching_from_a_cursor_replays_only_what_follows_it(
    runtime: LocalRuntime,
) -> None:
    """A reconnecting controller receives what it missed, not the whole stream."""
    from xaytune.runtimes import StreamCursor

    async def scenario() -> tuple[list[object], list[object]]:
        ref = await runtime.submit_or_get(
            OperationId.generate(), _plan(*_python("import sys; sys.exit(1)"))
        )
        await _settle(runtime, ref)
        everything = [event async for event in runtime.watch(ref)]
        after = [
            event async for event in runtime.watch(ref, StreamCursor(generation=0, sequence=0))
        ]
        return everything, after

    everything, after = asyncio.run(scenario())

    assert len(everything) == 2
    assert [event.sequence for event in after] == [1], "strictly after the recorded sequence"


def test_a_relaunch_starts_a_new_generation(runtime: LocalRuntime) -> None:
    """Sequences restart at zero, so the stream must not (ADR-014 §1a).

    Without a new generation a controller holding a cursor would read the
    relaunched worker's events as duplicates of ones it had already recorded.
    """
    from xaytune.runtimes import StreamCursor

    async def scenario() -> tuple[list[object], list[object]]:
        ref = await runtime.submit_or_get(OperationId.generate(), _plan(*_python("pass")))
        await _settle(runtime, ref)

        workload = runtime._registry.workload(ref.external_id)
        assert workload is not None
        run_launcher(workload.directory)

        everything = [event async for event in runtime.watch(ref)]
        after = [
            event async for event in runtime.watch(ref, StreamCursor(generation=0, sequence=99))
        ]
        return everything, after

    everything, after = asyncio.run(scenario())

    assert sorted({event.stream_generation for event in everything}) == [0, 1]
    assert {event.stream_generation for event in after} == {1}


# ---- logs ----------------------------------------------------------------


def test_logs_stream_both_worker_streams(runtime: LocalRuntime) -> None:
    source = "import sys; print('out'); print('err', file=sys.stderr)"

    async def scenario() -> list[object]:
        ref = await runtime.submit_or_get(OperationId.generate(), _plan(*_python(source)))
        await _settle(runtime, ref)
        return [line async for line in runtime.get_logs(ref)]

    logs = asyncio.run(scenario())

    assert {(entry.stream, entry.line) for entry in logs} == {
        ("stdout", "out"),
        ("stderr", "err"),
    }


def test_logs_are_not_registry_state(runtime: LocalRuntime) -> None:
    """Logs are for humans. A controller rate-limited by its own log ingestion
    would be slower than the workloads it supervises."""

    async def scenario() -> Path:
        ref = await runtime.submit_or_get(OperationId.generate(), _plan(*_python("print('hello')")))
        await _settle(runtime, ref)
        workload = runtime._registry.workload(ref.external_id)
        assert workload is not None
        return WorkloadPaths(workload.directory).stdout

    stdout_path = asyncio.run(scenario())
    database = (stdout_path.parents[2] / "registry.db").read_bytes()

    assert stdout_path.read_text().strip() == "hello"
    assert b"hello" not in database


# ---- the plan crosses the boundary intact --------------------------------


def test_a_module_entrypoint_runs_and_receives_its_arguments(
    runtime: LocalRuntime, tmp_path: Path
) -> None:
    """The other entrypoint kind, exercised for real rather than asserted about."""
    marker = tmp_path / "module-ran.txt"
    module_dir = tmp_path / "pkg"
    module_dir.mkdir()
    (module_dir / "worker.py").write_text(
        "import sys\n\n\ndef main(path):\n    open(path, 'w').write('ran')\n    return 0\n",
        encoding="utf-8",
    )

    spec = TrainingExecutionSpec(
        compiler=CompilerIdentity(name="fake", version="0.1.0"),
        candidate_fingerprint="sha256:" + "0" * 64,
        entrypoint=PythonModuleEntrypoint(module="worker", function="main"),
        arguments=(str(marker),),
        environment={"PYTHONPATH": str(module_dir)},
    )
    plan = ResolvedExecutionPlan(
        spec=spec,
        runtime="local",
        target=RuntimeOperationTarget(kind="training-attempt", id="ra_9"),
    )

    async def scenario() -> RuntimeStatus:
        ref = await runtime.submit_or_get(OperationId.generate(), plan)
        return await _settle(runtime, ref)

    status = asyncio.run(scenario())

    assert status.state == "succeeded", status.detail
    assert marker.read_text() == "ran"


def test_the_workers_environment_carries_the_plans_target(
    runtime: LocalRuntime, tmp_path: Path
) -> None:
    """So a worker can name itself in the telemetry it emits later."""
    marker = tmp_path / "env.txt"
    source = "import os, sys; open(sys.argv[1], 'w').write(os.environ['XAYTUNE_TARGET_ID'])"

    async def scenario() -> None:
        ref = await runtime.submit_or_get(
            OperationId.generate(),
            _plan(
                *_python(source, str(marker)),
                target=RuntimeOperationTarget(kind="training-attempt", id="ra_env"),
            ),
        )
        await _settle(runtime, ref)

    asyncio.run(scenario())

    assert marker.read_text() == "ra_env"


def test_a_reference_from_another_backend_is_refused(runtime: LocalRuntime) -> None:
    """This runtime answers for the workloads it issued, and no others."""
    from xaytune.core.refs import RuntimeRef

    with pytest.raises(KeyError, match="not a local workload"):
        asyncio.run(runtime.get_status(RuntimeRef(backend="ray", external_id="job-1")))

    with pytest.raises(KeyError, match="unknown local workload"):
        asyncio.run(runtime.get_status(RuntimeRef(backend=BACKEND, external_id="op_missing")))
