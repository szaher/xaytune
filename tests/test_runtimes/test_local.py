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
import subprocess
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
from xaytune.core.refs import RuntimeRef
from xaytune.core.sqlite import write_transaction
from xaytune.runtimes import OperationOutcome, RuntimeBackend, RuntimeStatus
from xaytune.runtimes.local import BACKEND, LocalRuntime, UnsupportedPlanError
from xaytune.runtimes.local.launcher import run as run_launcher
from xaytune.runtimes.local.paths import WorkloadPaths, read_json, write_atomic
from xaytune.runtimes.local.registry import LocalWorkloadRecord

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


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _release_ownership(runtime: LocalRuntime, external_id: str) -> None:
    """Clear the launcher claim, as a deliberate replacement would."""
    with write_transaction(runtime._registry._connection) as connection:
        connection.execute(
            "UPDATE workloads SET launcher_pid = NULL, spawned_pid = NULL WHERE external_id = ?",
            (external_id,),
        )


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


@pytest.mark.parametrize(
    ("label", "update"),
    [
        ("another runtime", {"runtime": "ray"}),
        ("an unimplemented launcher", {"runtime_options": {"launcher": "torchrun"}}),
        ("an unknown option", {"runtime_options": {"gpu_affinity": "0,1"}}),
    ],
)
def test_a_plan_asking_for_something_else_is_refused(
    runtime: LocalRuntime, label: str, update: dict[str, object]
) -> None:
    """Silently doing something different is the failure mode being closed.

    A plan resolved for another runtime carries decisions made against that
    runtime's capabilities, and an unknown runtime option is a caller asking
    for behaviour. Running a plain subprocess for either would be the
    "honour some of the request and ignore the rest" failure this backend
    refuses everywhere else.
    """
    operation_id = OperationId.generate()
    plan = _plan(*_python("pass")).model_copy(update=update)

    async def scenario() -> object:
        with pytest.raises(UnsupportedPlanError):
            await runtime.submit_or_get(operation_id, plan)
        return await runtime.lookup_operation(operation_id)

    outcome = asyncio.run(scenario())

    assert outcome is not None
    assert outcome.disposition == "rejected"
    assert outcome.may_reissue is True


def test_the_runtime_option_it_does_implement_is_accepted(
    runtime: LocalRuntime, tmp_path: Path
) -> None:
    """The closed set is a boundary, not a blanket refusal."""
    workdir = tmp_path / "elsewhere"
    workdir.mkdir()
    source = "import os, sys; open(sys.argv[1], 'w').write(os.getcwd())"
    marker = tmp_path / "cwd.txt"

    async def scenario() -> RuntimeStatus:
        ref = await runtime.submit_or_get(
            OperationId.generate(),
            _plan(*_python(source, str(marker))).model_copy(
                update={"runtime_options": {"working_directory": str(workdir)}}
            ),
        )
        return await _settle(runtime, ref)

    status = asyncio.run(scenario())

    assert status.state == "succeeded", status.detail
    assert Path(marker.read_text()).resolve() == workdir.resolve()


def test_one_idempotency_conflict_for_one_condition() -> None:
    """Both layers raise the same type, so a caller catches one thing.

    The control plane's journal and this runtime's registry both implement
    get-or-create, and a controller should not have to learn which layer
    refused it.
    """
    from xaytune.core.errors import IdempotencyConflictError as CoreError
    from xaytune.storage.journal import IdempotencyConflictError as StoragePath

    assert CoreError is StoragePath


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
    while two signals still reached the worker's process group.
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


def _force_finished(runtime: LocalRuntime, external_id: str, **record: object) -> None:
    """Stage a terminal record, so classification can be tested without a race."""
    workload = runtime._registry.workload(external_id)
    assert workload is not None
    write_atomic(WorkloadPaths(workload.directory).finished, dict(record))


def _mark_cancel_requested(runtime: LocalRuntime, external_id: str) -> None:
    with write_transaction(runtime._registry._connection) as connection:
        connection.execute(
            "UPDATE workloads SET cancel_requested_at = 'requested' WHERE external_id = ?",
            (external_id,),
        )


def test_a_cancellation_that_never_arrived_does_not_rewrite_the_outcome(
    runtime: LocalRuntime,
) -> None:
    """Observed ending beats pending intent (ADR-013 §5).

    The worker was already failing when the cancellation was requested and
    exited 3 before the launcher could deliver anything. Classifying from the
    request rather than the delivery would file a genuine failure as a
    cancellation, and an operator looking for why training stopped would find
    a decision they made instead of the fault that caused it.
    """

    async def scenario() -> RuntimeStatus:
        ref = await runtime.submit_or_get(OperationId.generate(), _plan(*_python("pass")))
        await _settle(runtime, ref)
        _force_finished(runtime, ref.external_id, exit_code=3, cancelled=False)
        _mark_cancel_requested(runtime, ref.external_id)
        return await runtime.get_status(ref)

    status = asyncio.run(scenario())

    assert status.state == "failed"
    assert status.exit_code == 3


def test_a_worker_that_shut_down_gracefully_is_still_cancelled(
    runtime: LocalRuntime,
) -> None:
    """Exit zero after a delivered SIGTERM is a cancellation, not a success.

    A worker that handles the signal and shuts down tidily did not finish its
    work. Reading the exit code alone would put a partial run into the record
    as a complete one, and whatever it had produced would be treated as the
    result of the whole thing.
    """

    async def scenario() -> RuntimeStatus:
        ref = await runtime.submit_or_get(OperationId.generate(), _plan(*_python("pass")))
        await _settle(runtime, ref)
        _force_finished(runtime, ref.external_id, exit_code=0, cancelled=True)
        return await runtime.get_status(ref)

    assert asyncio.run(scenario()).state == "cancelled"


def test_cancelling_reaches_the_workers_own_children(runtime: LocalRuntime, tmp_path: Path) -> None:
    """Cancellation must leave nothing executing.

    An ordinary training worker starts dataloader workers and helpers.
    Signalling only the process the launcher can see would leave those running
    while the workload reported itself cancelled -- a status that would be
    false at the moment it was written.
    """
    pidfile = tmp_path / "child.pid"
    source = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "open(sys.argv[1], 'w').write(str(child.pid))\n"
        "time.sleep(120)\n"
    )

    async def scenario() -> tuple[RuntimeStatus, int]:
        ref = await runtime.submit_or_get(
            OperationId.generate(), _plan(*_python(source, str(pidfile)))
        )
        deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
        while not pidfile.exists() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.02)
        spawned = int(pidfile.read_text())
        assert _process_exists(spawned), "the worker's child should be running"

        await runtime.cancel(ref, OperationId.generate())
        status = await _settle(runtime, ref)
        await asyncio.sleep(0.5)
        return status, spawned

    status, spawned = asyncio.run(scenario())

    try:
        assert status.state == "cancelled"
        assert not _process_exists(spawned), "the worker's child outlived the cancellation"
    finally:
        if _process_exists(spawned):
            os.kill(spawned, signal.SIGKILL)


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


def test_a_worker_outliving_its_launcher_is_still_running(
    runtime: LocalRuntime,
) -> None:
    """The supervisor is not the workload (ADR-014 §1a).

    Killing the launcher ends the telemetry stream and reparents the worker;
    it does not stop the work. Reading the supervisor's absence as the
    workload's end would retire a training job that is still running, and its
    artifacts with it.
    """

    async def scenario() -> tuple[RuntimeStatus, bool]:
        ref = await runtime.submit_or_get(
            OperationId.generate(), _plan(*_python("import time; time.sleep(120)"))
        )
        await _await_state(runtime, ref, "running")

        workload = runtime._registry.workload(ref.external_id)
        assert workload is not None and workload.launcher_pid is not None
        worker_pid = (read_json(WorkloadPaths(workload.directory).started) or {})["pid"]

        os.kill(workload.launcher_pid, signal.SIGKILL)
        await asyncio.sleep(0.3)

        status = await runtime.get_status(ref)
        os.kill(worker_pid, signal.SIGKILL)
        return status, _process_exists(workload.launcher_pid)

    status, launcher_alive = asyncio.run(scenario())

    assert launcher_alive is False, "the supervisor really is gone"
    assert status.state == "running"
    assert "supervisor" in (status.detail or "")


def test_watching_ends_when_the_supervisor_is_gone(runtime: LocalRuntime) -> None:
    """Otherwise the controller waits forever for a writer that no longer exists.

    ADR-014 treats a lost supervisor as a gap to reconcile, which the
    controller can only do once ``watch()`` hands control back.
    """

    async def scenario() -> int:
        ref = await runtime.submit_or_get(
            OperationId.generate(), _plan(*_python("import time; time.sleep(120)"))
        )
        await _await_state(runtime, ref, "running")

        workload = runtime._registry.workload(ref.external_id)
        assert workload is not None and workload.launcher_pid is not None
        worker_pid = (read_json(WorkloadPaths(workload.directory).started) or {})["pid"]
        os.kill(workload.launcher_pid, signal.SIGKILL)
        await asyncio.sleep(0.3)

        async def drain() -> list[object]:
            return [event async for event in runtime.watch(ref)]

        try:
            return len(await asyncio.wait_for(drain(), timeout=10))
        finally:
            os.kill(worker_pid, signal.SIGKILL)

    assert asyncio.run(scenario()) >= 1


def test_a_worker_awaiting_its_epitaph_is_not_unknown(
    runtime: LocalRuntime, tmp_path: Path
) -> None:
    """ "No answer yet" is not "no answer possible".

    Between the worker exiting and the launcher recording why, the worker pid
    is already dead. Calling that ``unknown`` would send a controller to
    reconcile a workload that is moments from reporting a clean exit -- and
    since the supervisor is still alive, an answer is coming.

    Built rather than raced: the window is milliseconds wide, and a test that
    tried to land inside it would pass by luck.
    """
    supervisor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()

    async def scenario() -> RuntimeStatus:
        ref = await runtime.submit_or_get(OperationId.generate(), _plan(*_python("pass")))
        await _settle(runtime, ref)

        workload = runtime._registry.workload(ref.external_id)
        assert workload is not None
        paths = WorkloadPaths(workload.directory)
        paths.finished.unlink()
        write_atomic(paths.started, {"pid": dead.pid, "launcher_pid": supervisor.pid})
        with write_transaction(runtime._registry._connection) as connection:
            connection.execute(
                "UPDATE workloads SET launcher_pid = ? WHERE external_id = ?",
                (supervisor.pid, ref.external_id),
            )
        return await runtime.get_status(ref)

    try:
        status = asyncio.run(scenario())
    finally:
        supervisor.kill()
        supervisor.wait()

    assert status.state == "running"
    assert "being recorded" in (status.detail or "")


def test_a_workload_with_no_observed_ending_is_unknown(runtime: LocalRuntime) -> None:
    """ "I cannot tell" is an answer, and it is not "it failed".

    Reading it as a failure would retire a workload that may have produced
    artifacts; reading it as success would be worse.
    """

    async def scenario() -> RuntimeStatus:
        ref = await runtime.submit_or_get(
            OperationId.generate(), _plan(*_python("import time; time.sleep(120)"))
        )
        await _await_state(runtime, ref, "running")

        workload = runtime._registry.workload(ref.external_id)
        assert workload is not None and workload.launcher_pid is not None
        worker_pid = (read_json(WorkloadPaths(workload.directory).started) or {})["pid"]

        # Both, and separately: the worker leads its own process group now, so
        # killing the launcher's no longer takes it down -- which is the point
        # of the split, and means an unobserved ending has to be staged rather
        # than assumed to follow from one signal.
        os.killpg(worker_pid, signal.SIGKILL)
        os.kill(workload.launcher_pid, signal.SIGKILL)

        return await _settle(runtime, ref)

    status = asyncio.run(scenario())

    assert status.state == "unknown"
    assert status.exit_code is None


def test_an_unknown_workload_is_not_reported_as_running() -> None:
    """``is_running`` must not manufacture a certainty the runtime refused.

    A backend that accepted an operation and then lost track of it reports
    ``accepted`` with a status of ``unknown``. Collapsing that to ``True``
    contradicts the whole reason the ``unknown`` state exists.
    """
    lost = OperationOutcome(
        operation_id=OperationId.generate(),
        disposition="accepted",
        runtime_ref=RuntimeRef(backend=BACKEND, external_id="op_x"),
        status=RuntimeStatus(state="unknown"),
    )
    running = lost.model_copy(update={"status": RuntimeStatus(state="running")})
    done = lost.model_copy(update={"status": RuntimeStatus(state="succeeded")})

    assert lost.is_running is None
    assert running.is_running is True
    assert done.is_running is False


# ---- the crash windows ---------------------------------------------------


def test_two_launchers_racing_produce_exactly_one_worker(
    runtime: LocalRuntime, tmp_path: Path
) -> None:
    """The window between ``Popen`` returning and anything recording it.

    A controller that dies in there restarts, sees no owner, and starts a
    second launcher -- so this is the state that window leaves behind, and both
    launchers are started at once to make them race for real. Ownership is
    claimed by the launcher, so exactly one can proceed.
    """
    marker = tmp_path / "ran.txt"
    plan = _plan(*_python("import sys; open(sys.argv[1], 'a').write('x')", str(marker)))
    operation_id = OperationId.generate()
    directory = runtime._workloads / str(operation_id)

    runtime._registry.claim_submission(
        operation_id=operation_id,
        request_digest=plan.request_digest("submit"),
        external_id=str(operation_id),
        target_kind=plan.target.kind,
        target_id=plan.target.id,
        directory=directory,
    )
    directory.mkdir(parents=True, exist_ok=True)
    WorkloadPaths(directory).plan.write_text(plan.model_dump_json(), encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "xaytune.runtimes.local.launcher",
        str(directory),
        str(runtime._root / "registry.db"),
        str(operation_id),
    ]
    racers = [subprocess.Popen(command) for _ in range(2)]
    for racer in racers:
        racer.wait(timeout=_SETTLE_SECONDS)

    assert marker.read_text() == "x", "two launchers, one worker"


def test_a_cancellation_interrupted_before_delivery_is_not_lost(
    runtime: LocalRuntime,
) -> None:
    """Durable intent that never became an effect must stay retryable.

    A crash between recording the claim and the request reaching the workload
    used to be permanent: the retry found the claim already recorded and
    declined to act. That is the same "intent without effect" ambiguity one
    layer down that ADR-013 exists to remove.
    """
    cancellation = OperationId.generate()

    async def scenario() -> RuntimeStatus:
        ref = await runtime.submit_or_get(
            OperationId.generate(), _plan(*_python("import time; time.sleep(120)"))
        )
        await _await_state(runtime, ref, "running")

        # The claim commits, and then the controller dies before the request
        # reaches the workload.
        runtime._registry.claim_cancellation(
            operation_id=cancellation,
            request_digest=f"cancel:{ref.external_id}",
            external_id=ref.external_id,
        )
        assert not WorkloadPaths(
            (runtime._registry.workload(ref.external_id) or _missing()).directory
        ).cancel.exists()

        await runtime.cancel(ref, cancellation)
        return await _settle(runtime, ref)

    assert asyncio.run(scenario()).state == "cancelled"


def _missing() -> LocalWorkloadRecord:
    raise AssertionError("workload disappeared")


def test_cancelling_never_signals_a_remembered_pid(runtime: LocalRuntime, tmp_path: Path) -> None:
    """A pid from before a restart is a number, not a process.

    The operating system may have given it to something unrelated, so the
    runtime writes a request the launcher acts on rather than signalling
    anything itself. Proved with a bystander: its pid is planted in the
    registry, and it must survive the cancellation.
    """
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    async def scenario() -> None:
        ref = await runtime.submit_or_get(OperationId.generate(), _plan(*_python("pass")))
        await _settle(runtime, ref)
        with write_transaction(runtime._registry._connection) as connection:
            connection.execute(
                "UPDATE workloads SET launcher_pid = ?, spawned_pid = ? WHERE external_id = ?",
                (bystander.pid, bystander.pid, ref.external_id),
            )
        await runtime.cancel(ref, OperationId.generate())

    try:
        asyncio.run(scenario())
        assert bystander.poll() is None, "the cancellation reached an unrelated process"
    finally:
        bystander.kill()
        bystander.wait()


def test_a_cancellation_arriving_before_the_worker_is_honoured(
    runtime: LocalRuntime, tmp_path: Path
) -> None:
    """A request made during startup must not be outrun by the spawn.

    The launcher checks before it spawns, so a workload cancelled in that
    window never starts rather than starting and having to be stopped.
    """
    marker = tmp_path / "ran.txt"
    plan = _plan(*_python("import sys; open(sys.argv[1], 'a').write('x')", str(marker)))
    operation_id = OperationId.generate()
    directory = runtime._workloads / str(operation_id)

    runtime._registry.claim_submission(
        operation_id=operation_id,
        request_digest=plan.request_digest("submit"),
        external_id=str(operation_id),
        target_kind=plan.target.kind,
        target_id=plan.target.id,
        directory=directory,
    )
    paths = WorkloadPaths(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths.plan.write_text(plan.model_dump_json(), encoding="utf-8")

    async def scenario() -> RuntimeStatus:
        ref = RuntimeRef(backend=BACKEND, external_id=str(operation_id))
        await runtime.cancel(ref, OperationId.generate())
        run_launcher(directory, runtime._root / "registry.db", str(operation_id))
        return await runtime.get_status(ref)

    status = asyncio.run(scenario())

    assert status.state == "cancelled"
    assert not marker.exists(), "the worker was never started"


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

        # A relaunch is only possible once ownership is released: the claim
        # that stops two launchers racing also stops one being replaced. When
        # PR-012a defines who may release it, this is the step it performs.
        _release_ownership(runtime, ref.external_id)
        run_launcher(workload.directory, runtime._root / "registry.db", ref.external_id)

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
