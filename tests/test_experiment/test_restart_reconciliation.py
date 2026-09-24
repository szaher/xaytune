"""A controller that dies does not orphan or duplicate its workload (PR-012a).

```text
submit ─► attempt RUNNING ─► controller killed (SIGKILL)
                                   │   workload keeps running
                                   ▼
new EmbeddedControllerHost ─► attach(experiment)
   ├── persisted RuntimeSpec, version checked      fail closed on mismatch
   ├── submit operation + RuntimeRef, or lookup_operation() if unconfirmed
   ├── adopt the existing workload                  never a second one
   ├── watch() from the durable cursor
   └── settle the same Attempt and Run
```

The invariant: **restart may rediscover or adopt an external effect; it never
blindly re-issues one whose outcome is uncertain.** An operation is issued
again only when the runtime says it never received it -- and then under the
same identity, so a second issue is the same request, not a new one.

The controller is killed for real, in a subprocess, with SIGKILL: no
``finally``, no clean shutdown, no chance to write anything on the way out.
"""

from __future__ import annotations

import asyncio
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.training_fixtures import sft_candidate, tiny_dataset, tiny_model
from xaytune.core.domain.objective import Objective, ObjectiveMetric
from xaytune.core.state.status import RunAttemptStatus, RunStatus

_REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _spec(tmp_path: Path):
    from xaytune.experiment import CompilerSpec, ExperimentSpec, RuntimeSpec

    return ExperimentSpec(
        name="tiny-sft",
        objective=Objective(primary=ObjectiveMetric(name="loss", direction="minimize")),
        candidate=sft_candidate(
            tiny_model(tmp_path / "model"),
            tiny_dataset(tmp_path / "data" / "train.jsonl", "text"),
            "text",
        ),
        seed=7,
        compiler=CompilerSpec(name="native"),
        runtime=RuntimeSpec(kind="local", config={"root": str(tmp_path / "runtime")}),
        artifact_root=str(tmp_path / "artifacts"),
    )


def _crash(tmp_path: Path, mode: str) -> str:
    """Submit in a controller process that is killed at *mode*; return the experiment id."""
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec(tmp_path).model_dump_json())
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.test_experiment.crashing_host",
            str(tmp_path / "state.db"),
            str(spec_path),
            mode,
        ],
        cwd=_REPOSITORY,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == -signal.SIGKILL, completed.stderr

    from xaytune.core.sqlite import connect

    connection = connect(tmp_path / "state.db")
    try:
        (row,) = connection.execute("SELECT id FROM experiments").fetchall()
        return str(row["id"])
    finally:
        connection.close()


def _workloads(tmp_path: Path) -> list[Path]:
    """Every workload the runtime ever started for this test."""
    root = tmp_path / "runtime" / "workloads"
    return sorted(root.iterdir()) if root.exists() else []


class _CountingSubmissions:
    """LocalRuntime, counting what the adopting host issues.

    The one-workload check alone cannot tell adoption from re-issue:
    LocalRuntime's own get-or-create would return the same workload for a
    second submit under the same operation id. The invariant is about the
    controller, so the controller's calls are what is counted.
    """

    issued: list[Any] = []

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    async def submit_or_get(self, operation_id: Any, plan: Any) -> Any:
        _CountingSubmissions.issued.append(operation_id)
        return await self._runtime.submit_or_get(operation_id, plan)


def _adopt(tmp_path: Path, experiment_id: str, **host_options: Any):
    from xaytune.experiment import EmbeddedControllerHost
    from xaytune.experiment.host import _local_runtime

    _CountingSubmissions.issued = []
    host_options.setdefault(
        "runtimes", {"local": lambda config: _CountingSubmissions(_local_runtime(config))}
    )

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db", **host_options)
        try:
            handle = await host.attach(experiment_id)
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            repo = host.repository
            (attempt,) = [
                a
                for node in repo.aggregates.nodes_for_experiment(experiment_id)
                for run in repo.aggregates.runs_for_node(str(node.id))
                for a in repo.aggregates.attempts_for_run(str(run.id))
            ]
            operations = repo.operations.for_target("training-attempt", str(attempt.id))
            return result, attempt, operations
        finally:
            await host.close()

    return asyncio.run(scenario())


def _settled_once(result: Any, attempt: Any) -> None:
    (node,) = result.nodes
    (run,) = node.runs
    assert run.status is RunStatus.SUCCEEDED
    assert attempt.status is RunAttemptStatus.SUCCEEDED
    assert attempt.attempt_number == 1, "the same attempt, not a new one"
    (artifact,) = run.artifacts
    assert artifact.producer_attempt_id == attempt.id


# ---- the acceptance case -----------------------------------------------------


def test_a_restarted_host_adopts_the_live_workload_instead_of_starting_another(
    tmp_path: Path,
) -> None:
    experiment_id = _crash(tmp_path, "running")
    assert len(_workloads(tmp_path)) == 1

    result, attempt, operations = _adopt(tmp_path, experiment_id)

    _settled_once(result, attempt)
    assert len(_workloads(tmp_path)) == 1, "adopted, not duplicated"
    assert _CountingSubmissions.issued == [], "a confirmed workload is adopted, never re-issued"
    (submit,) = operations
    assert submit.state == "confirmed"


def test_adoption_resumes_from_the_durable_cursor(tmp_path: Path) -> None:
    """Not from the start of the stream: what was recorded is not re-read.

    The dead controller had applied at least ``TrainingStarted`` -- that is
    what made the attempt RUNNING -- so the new one asks for what came after.
    """
    from xaytune.experiment.host import _local_runtime

    asked: list[Any] = []

    class Recording:
        def __init__(self, runtime: Any) -> None:
            self._runtime = runtime

        def __getattr__(self, name: str) -> Any:
            return getattr(self._runtime, name)

        def watch(self, reference: Any, cursor: Any = None) -> Any:
            asked.append(cursor)
            return self._runtime.watch(reference, cursor)

    experiment_id = _crash(tmp_path, "running")
    result, attempt, _ = _adopt(
        tmp_path, experiment_id, runtimes={"local": lambda c: Recording(_local_runtime(c))}
    )

    _settled_once(result, attempt)
    (cursor,) = asked
    assert cursor is not None and cursor.sequence >= 0, "resumed, not replayed from zero"


def test_restarting_twice_still_leaves_one_workload(tmp_path: Path) -> None:
    """A host that adopted and then went away is just another restart."""
    from xaytune.experiment import EmbeddedControllerHost

    experiment_id = _crash(tmp_path, "running")

    async def adopt_and_leave():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        await host.attach(experiment_id)
        await host.close()

    asyncio.run(adopt_and_leave())
    result, attempt, _ = _adopt(tmp_path, experiment_id)

    _settled_once(result, attempt)
    assert len(_workloads(tmp_path)) == 1


# ---- an unconfirmed submission ------------------------------------------------


def test_a_lost_submit_response_is_found_by_lookup_not_reissued(tmp_path: Path) -> None:
    """The runtime started it; the controller died before recording that."""
    experiment_id = _crash(tmp_path, "lost-response")
    assert len(_workloads(tmp_path)) == 1

    result, attempt, operations = _adopt(tmp_path, experiment_id)

    _settled_once(result, attempt)
    assert len(_workloads(tmp_path)) == 1
    assert _CountingSubmissions.issued == [], "found by lookup, not re-issued"
    (submit,) = operations
    assert submit.state == "confirmed" and submit.runtime_ref is not None


def test_a_submission_the_runtime_never_received_is_issued_once_under_its_own_identity(
    tmp_path: Path,
) -> None:
    """The only case in which reconciliation may issue: the runtime says it never arrived.

    Issued under the recorded operation id and against the recorded digest, so
    it is the same request the dead controller meant to make -- a lookup racing
    a late arrival would find one workload, not two.
    """
    experiment_id = _crash(tmp_path, "never-sent")
    assert _workloads(tmp_path) == [], "nothing was started before the crash"

    result, attempt, operations = _adopt(tmp_path, experiment_id)

    _settled_once(result, attempt)
    (submit,) = operations
    assert _CountingSubmissions.issued == [submit.id], "issued exactly once, under its own id"
    (workload,) = _workloads(tmp_path)
    assert workload.name == str(submit.id), "issued under the recorded operation id"


# ---- fail closed ---------------------------------------------------------------


class _RenumberedRuntime:
    """LocalRuntime, claiming a version other than the one recorded."""

    constructed = 0

    def __init__(self, runtime: Any) -> None:
        _RenumberedRuntime.constructed += 1
        self._runtime = runtime
        self.descriptor = runtime.descriptor.model_copy(update={"plugin_version": "9.9.9"})

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)


def _renumbered_runtimes() -> dict[str, Any]:
    from xaytune.experiment.host import _local_runtime

    _RenumberedRuntime.constructed = 0
    return {"local": lambda config: _RenumberedRuntime(_local_runtime(config))}


@pytest.mark.parametrize(
    ("runtimes", "error"),
    [
        (lambda: {}, "UnknownImplementationError"),
        (_renumbered_runtimes, "ImplementationMismatchError"),
    ],
    ids=["runtime-unavailable", "runtime-renumbered"],
)
def test_a_live_workload_without_its_recorded_runtime_fails_closed(
    tmp_path: Path, runtimes: Any, error: str
) -> None:
    """The record says which implementation ran it; a different one does not adopt it."""
    import xaytune.experiment as experiment_module
    from xaytune.experiment import EmbeddedControllerHost

    experiment_id = _crash(tmp_path, "running")

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db", runtimes=runtimes())
        try:
            with pytest.raises(getattr(experiment_module, error)):
                await host.attach(experiment_id)
        finally:
            await host.close()

    asyncio.run(scenario())
    assert len(_workloads(tmp_path)) == 1


def test_a_runtime_that_cannot_report_completed_operations_escalates(tmp_path: Path) -> None:
    """ADR-013 AC-5: "not found" from such a runtime might mean "finished and forgotten".

    Re-issuing would re-run it, so reconciliation stops and says so. The
    operation stays INTENDED: it is still unresolved, and nothing claims
    otherwise.
    """
    from xaytune.experiment import EmbeddedControllerHost, ReconciliationEscalatedError
    from xaytune.experiment.host import _local_runtime

    class Forgetful:
        def __init__(self, runtime: Any) -> None:
            self._runtime = runtime

        def __getattr__(self, name: str) -> Any:
            return getattr(self._runtime, name)

        def capabilities(self) -> Any:
            document = self._runtime.capabilities()
            resilience = document.resilience.model_copy(
                update={"reports_completed_operations": False}
            )
            return document.model_copy(update={"resilience": resilience})

        async def lookup_operation(self, _operation_id: Any) -> None:
            return None

    experiment_id = _crash(tmp_path, "never-sent")

    async def scenario():
        host = EmbeddedControllerHost(
            tmp_path / "state.db", runtimes={"local": lambda c: Forgetful(_local_runtime(c))}
        )
        try:
            handle = await host.attach(experiment_id)
            with pytest.raises(ReconciliationEscalatedError, match="completed operations"):
                await handle.wait()
            return host.repository.operations.unresolved()
        finally:
            await host.close()

    (unresolved,) = asyncio.run(scenario())
    assert unresolved.state == "intended"
    assert _workloads(tmp_path) == [], "escalated, not re-issued"


# ---- a dead stream over a live workload (ADR-014 §1a) ----------------------------


def test_a_dead_supervisor_degrades_telemetry_without_minting_an_attempt(
    tmp_path: Path,
) -> None:
    """The launcher dies mid-training; the worker it supervised runs on.

    ```text
    stream ends, workload live   ->  telemetry_generation 0 -> 1, TelemetryDegraded
    worker exits, nothing saw it ->  outcome unknown: escalate, never guess
    ```

    Still one attempt: the workload is the same one, and a new attempt would
    claim an execution that never happened. A second host attaching later
    reaches the same escalation from the record, and adds nothing to it.
    """
    import os

    from xaytune.experiment import EmbeddedControllerHost, ReconciliationEscalatedError

    async def first():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.submit(_spec(tmp_path))
            repo = host.repository
            deadline = asyncio.get_running_loop().time() + 60
            while True:
                (run,) = host._result(handle.experiment_id).nodes[0].runs
                if run.attempt_status is RunAttemptStatus.RUNNING:
                    break
                assert asyncio.get_running_loop().time() < deadline, "never started"
                await asyncio.sleep(0.02)
            (attempt,) = _attempts(repo, str(handle.experiment_id))
            (submit,) = repo.operations.for_target("training-attempt", str(attempt.id))
            (runtime,) = host._runtimes.values()
            workload = runtime._registry.workload(submit.runtime_ref.external_id)
            os.kill(workload.launcher_pid, signal.SIGKILL)

            with pytest.raises(ReconciliationEscalatedError, match="no recorded outcome"):
                await asyncio.wait_for(handle.wait(), timeout=120)
            return str(handle.experiment_id), attempt.id
        finally:
            await host.close()

    async def second(experiment_id):
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.attach(experiment_id)
            with pytest.raises(ReconciliationEscalatedError):
                await asyncio.wait_for(handle.wait(), timeout=60)
            repo = host.repository
            attempts = _attempts(repo, experiment_id)
            position = repo.aggregates.telemetry_position(str(attempts[0].id))
            degraded = [
                e
                for e in repo.events.events_for_experiment(experiment_id)
                if e.event_type == "TelemetryDegraded"
            ]
            return attempts, position, degraded
        finally:
            await host.close()

    experiment_id, attempt_id = asyncio.run(first())
    attempts, position, degraded = asyncio.run(second(experiment_id))

    (attempt,) = attempts
    assert attempt.id == attempt_id, "no new attempt was minted"
    assert not attempt.is_terminal, "an unobserved ending is not written down as an outcome"
    assert position == (1, -1), "the stream moved to the next generation"
    (event,) = degraded
    assert (event.payload["from_generation"], event.payload["to_generation"]) == (0, 1)
    assert len(_workloads(tmp_path)) == 1


def _attempts(repo: Any, experiment_id: str) -> list:
    return [
        a
        for node in repo.aggregates.nodes_for_experiment(experiment_id)
        for run in repo.aggregates.runs_for_node(str(node.id))
        for a in repo.aggregates.attempts_for_run(str(run.id))
    ]


# ---- cancellation survives the controller -----------------------------------------


def test_a_cancellation_recorded_before_the_crash_is_carried_out_after_it(
    tmp_path: Path,
) -> None:
    """ADR-013 AC-6: the intent is in the record, so a restart knows about it.

    The dead controller recorded the cancellation and its intended effect,
    and died before issuing it. The next host issues it, observes the workload
    stop, and only then calls the experiment CANCELLED.
    """
    from xaytune.core.state.status import ExperimentStatus

    experiment_id = _crash(tmp_path, "cancel-intended")

    result, attempt, operations = _adopt(tmp_path, experiment_id)

    assert result.status is ExperimentStatus.CANCELLED
    assert attempt.status is RunAttemptStatus.CANCELLED
    (cancel,) = [op for op in operations if op.type == "cancel"]
    assert cancel.state == "confirmed"
    assert len(_workloads(tmp_path)) == 1
    assert _CountingSubmissions.issued == [], "cancelling adopts; it never re-submits"


# ---- the compiler is a dependency of re-issue only -----------------------------------


class _RenumberedCompiler:
    """The native compiler, claiming a version other than the one recorded."""

    def __init__(self) -> None:
        from xaytune.compilation.native import NativeCompiler

        self._compiler = NativeCompiler()
        self.descriptor = self._compiler.descriptor.model_copy(update={"plugin_version": "9.9.9"})

    def __getattr__(self, name: str) -> Any:
        return getattr(self._compiler, name)


_NO_COMPILERS: dict[str, Any] = {}
_RENUMBERED = {"native": _RenumberedCompiler}


@pytest.mark.parametrize(
    "compilers", [_NO_COMPILERS, _RENUMBERED], ids=["compiler-unavailable", "compiler-renumbered"]
)
def test_a_running_workload_is_adopted_without_its_compiler(
    tmp_path: Path, compilers: dict
) -> None:
    """Rediscovering an effect that exists needs the runtime, not the compiler.

    The unavailable case is the strong one: with no compiler registered at
    all, adoption can only succeed if nothing on its path resolves one.
    """
    experiment_id = _crash(tmp_path, "running")

    result, attempt, _ = _adopt(tmp_path, experiment_id, compilers=compilers)

    _settled_once(result, attempt)
    assert _CountingSubmissions.issued == []
    assert len(_workloads(tmp_path)) == 1


@pytest.mark.parametrize(
    "compilers", [_NO_COMPILERS, _RENUMBERED], ids=["compiler-unavailable", "compiler-renumbered"]
)
def test_a_workload_found_by_lookup_is_adopted_without_its_compiler(
    tmp_path: Path, compilers: dict
) -> None:
    """Not "confirmed operations skip the compiler": finding an existing effect does.

    The submission was never confirmed, so this goes through lookup -- and the
    lookup finding the workload is what makes the compiler irrelevant.
    """
    experiment_id = _crash(tmp_path, "lost-response")

    result, attempt, operations = _adopt(tmp_path, experiment_id, compilers=compilers)

    _settled_once(result, attempt)
    (submit,) = operations
    assert submit.state == "confirmed"
    assert _CountingSubmissions.issued == []


@pytest.mark.parametrize(
    ("compilers", "error"),
    [(_NO_COMPILERS, "UnknownImplementationError"), (_RENUMBERED, "ImplementationMismatchError")],
    ids=["compiler-unavailable", "compiler-renumbered"],
)
def test_re_issuing_without_the_original_compiler_fails_closed(
    tmp_path: Path, compilers: dict, error: str
) -> None:
    """Rebuilding the request is where the compiler matters, so that is where it is required.

    Nothing was started, so the submission would have to be issued -- which
    means rebuilding it, which needs the compiler that built the original. An
    unavailable one or a different version refuses, and nothing is submitted.
    """
    import xaytune.experiment as experiment_module
    from xaytune.experiment import EmbeddedControllerHost
    from xaytune.experiment.host import _local_runtime

    experiment_id = _crash(tmp_path, "never-sent")
    _CountingSubmissions.issued = []

    async def scenario():
        host = EmbeddedControllerHost(
            tmp_path / "state.db",
            compilers=compilers,
            runtimes={"local": lambda c: _CountingSubmissions(_local_runtime(c))},
        )
        try:
            with pytest.raises(getattr(experiment_module, error)):
                await host.attach(experiment_id)
            return host.repository.operations.unresolved()
        finally:
            await host.close()

    (unresolved,) = asyncio.run(scenario())
    assert unresolved.state == "intended", "left exactly as recorded"
    assert _CountingSubmissions.issued == []
    assert _workloads(tmp_path) == []


# ---- the runtime is a dependency of touching an effect, not of the record ----------


@pytest.mark.parametrize(
    "runtimes",
    [lambda: {}, _renumbered_runtimes],
    ids=["runtime-unavailable", "runtime-renumbered"],
)
def test_a_settled_experiment_attaches_without_its_runtime(tmp_path: Path, runtimes: Any) -> None:
    """A handle asks the durable record, and a settled record needs no runtime to read.

    Training finished under one host; a later host that has no local runtime,
    or a different version of it, attaches, reads the status, the result and
    the history, and waits -- and never resolves a runtime to do it.
    """
    from xaytune.core.state.status import ExperimentStatus
    from xaytune.experiment import EmbeddedControllerHost

    async def train() -> str:
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.submit(_spec(tmp_path))
            await asyncio.wait_for(handle.wait(), timeout=180)
            return str(handle.experiment_id)
        finally:
            await host.close()

    experiment_id = asyncio.run(train())
    registry = runtimes()
    _RenumberedRuntime.constructed = 0

    async def reattach():
        host = EmbeddedControllerHost(tmp_path / "state.db", runtimes=registry)
        try:
            handle = await host.attach(experiment_id)
            status = await handle.status()
            result = await asyncio.wait_for(handle.wait(), timeout=10)
            first = await asyncio.wait_for(anext(aiter(handle.events())), timeout=10)
            return status, result, first
        finally:
            await host.close()

    status, result, first = asyncio.run(reattach())

    assert status is ExperimentStatus.ACTIVE, "next_stage decides what follows, not this host"
    assert result.quiescent
    (node,) = result.nodes
    (run,) = node.runs
    assert run.status is RunStatus.SUCCEEDED
    assert first.sequence == 1, "the history is readable from its start"
    assert _RenumberedRuntime.constructed == 0, "no runtime was resolved"
    assert len(_workloads(tmp_path)) == 1


def test_a_recorded_refusal_is_settled_without_the_runtime(tmp_path: Path) -> None:
    """The controller died after recording the submission failed, before settling.

    The record already says what happened; settling it is bookkeeping, and
    needs no runtime -- this host has none.
    """
    from xaytune.core.refs import Actor
    from xaytune.experiment import EmbeddedControllerHost

    experiment_id = _crash(tmp_path, "never-sent")

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db", runtimes={})
        try:
            # What the dead controller recorded before dying, had the runtime
            # refused the request.
            (submission,) = host.repository.operations.unresolved()
            host.repository.fail_operation(
                submission.id,
                expected_revision=submission.revision,
                actor=Actor(type="system", id="test"),
            )
            handle = await host.attach(experiment_id)
            result = await asyncio.wait_for(handle.wait(), timeout=10)
            return result, _attempts(host.repository, experiment_id)
        finally:
            await host.close()

    result, (attempt,) = asyncio.run(scenario())
    (node,) = result.nodes
    (run,) = node.runs
    assert attempt.status is RunAttemptStatus.FAILED
    assert run.status is RunStatus.FAILED
    assert _workloads(tmp_path) == []
