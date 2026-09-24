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


def _adopt(tmp_path: Path, experiment_id: str, **host_options: Any):
    from xaytune.experiment import EmbeddedControllerHost

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
    (workload,) = _workloads(tmp_path)
    assert workload.name == str(submit.id), "issued under the recorded operation id"


# ---- fail closed ---------------------------------------------------------------


def test_a_runtime_version_the_host_cannot_provide_fails_closed(tmp_path: Path) -> None:
    """The record says which implementation ran it; a different one does not adopt it."""
    from xaytune.experiment import EmbeddedControllerHost, ImplementationMismatchError
    from xaytune.experiment.host import _local_runtime

    class Newer:
        def __init__(self, runtime: Any) -> None:
            self._runtime = runtime
            self.descriptor = runtime.descriptor.model_copy(update={"plugin_version": "9.9.9"})

        def __getattr__(self, name: str) -> Any:
            return getattr(self._runtime, name)

    experiment_id = _crash(tmp_path, "running")

    async def scenario():
        host = EmbeddedControllerHost(
            tmp_path / "state.db", runtimes={"local": lambda c: Newer(_local_runtime(c))}
        )
        try:
            with pytest.raises(ImplementationMismatchError, match="9.9.9"):
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
            runtime = document.runtime.model_copy(update={"reports_completed_operations": False})
            return document.model_copy(update={"runtime": runtime})

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
