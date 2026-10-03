"""The daemon as a real process: singleton, signals, death, restart (PR-027 exit criterion).

```text
start daemon on state.db           it takes <state.db>.lock; a second is refused
a client commits a submission      and is SIGKILLed; the daemon drives it anyway
SIGTERM                            observers stop, the workload is not cancelled,
                                   the lock is released
restart                            no sweep: a COMPLETED handoff is not adopted
attach request                     the existing reconciliation adopts it, once
SIGKILL at each crash point        the unfinished request is resumed, never
                                   duplicated
```

Workloads are :mod:`.file_runtime` files, so they outlive every daemon here.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.test_daemon.file_runtime import calls, finish, workloads
from tests.test_daemon.test_daemon_host import _file_spec
from xaytune.core.state.status import RunAttemptStatus, RunStatus
from xaytune.daemon import (
    ControllerRequestState,
    DaemonAlreadyRunningError,
    DaemonClient,
    StateDatabaseLock,
    lock_path,
)

_REPOSITORY = Path(__file__).resolve().parents[2]
_CONFIG = "tests.test_daemon.daemon_config:create_config"
_TIMEOUT = 60


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


class _Daemons:
    """Starts daemon processes and makes sure none outlives the test."""

    def __init__(self, tmp_path: Path) -> None:
        self.state = tmp_path / "state.db"
        self.processes: list[subprocess.Popen[str]] = []

    def start(
        self, *, fault: str | None = None, config: str = _CONFIG, ready: bool = True
    ) -> subprocess.Popen[str]:
        env = dict(os.environ)
        env.pop("XAYTUNE_TEST_FAULT", None)
        if fault is not None:
            env["XAYTUNE_TEST_FAULT"] = fault
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "xaytune.daemon",
                "--state",
                str(self.state),
                "--config",
                config,
                "--poll-interval",
                "0.05",
            ],
            cwd=_REPOSITORY,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.processes.append(process)
        if ready:
            _until(
                lambda: (
                    (StateDatabaseLock.holder(lock_path(self.state)) or {}).get("pid")
                    == process.pid
                    or process.poll() is not None
                )
            )
            assert process.poll() is None, process.communicate()[1]
        return process

    def stop(self, process: subprocess.Popen[str], signum: int = signal.SIGTERM) -> str:
        process.send_signal(signum)
        _, stderr = process.communicate(timeout=_TIMEOUT)
        return stderr

    def close(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=_TIMEOUT)


@pytest.fixture
def daemons(tmp_path: Path) -> Iterator[_Daemons]:
    started = _Daemons(tmp_path)
    try:
        yield started
    finally:
        started.close()


def _until(predicate: Callable[[], Any], timeout: float = _TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.05)


def _handoff(client: DaemonClient, request_id: Any) -> Any:
    _until(
        lambda: (
            client.request(request_id).state
            in (ControllerRequestState.COMPLETED, ControllerRequestState.FAILED)
        )
    )
    return client.request(request_id)


def _count(client: DaemonClient, table: str) -> int:
    return int(client._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _lock_is_free(state: Path) -> bool:
    lock = StateDatabaseLock(state)
    try:
        lock.acquire({"pid": os.getpid()})
    except DaemonAlreadyRunningError:
        return False
    lock.release()
    return True


# ---- singleton ----------------------------------------------------------------------


def test_one_daemon_per_database_and_the_lock_goes_with_the_process(
    tmp_path: Path, daemons: _Daemons
) -> None:
    first = daemons.start()
    assert not _lock_is_free(daemons.state)

    alias = tmp_path / "alias" / ".." / "state.db"
    (tmp_path / "alias").mkdir()
    second = subprocess.run(
        [sys.executable, "-m", "xaytune.daemon", "--state", str(alias), "--config", _CONFIG],
        cwd=_REPOSITORY,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=False,
    )
    assert second.returncode == 3, second.stderr
    assert "another daemon holds" in second.stderr and str(first.pid) in second.stderr

    stderr = daemons.stop(first)
    assert first.returncode == 0, stderr
    assert _lock_is_free(daemons.state), "released by a controlled shutdown"

    killed = daemons.start()
    killed.kill()
    killed.communicate(timeout=_TIMEOUT)
    assert _lock_is_free(daemons.state), "released by the kernel when the process died"
    restarted = daemons.start()
    assert restarted.poll() is None


def test_a_configuration_that_is_not_one_is_refused(daemons: _Daemons) -> None:
    process = daemons.start(config="tests.test_daemon.daemon_config:not_a_config", ready=False)
    _, stderr = process.communicate(timeout=_TIMEOUT)
    assert process.returncode == 2
    assert "not a DaemonConfig" in stderr


# ---- the exit criterion -------------------------------------------------------------


def test_a_dead_client_hands_off_and_shutdown_leaves_the_workload_running(
    tmp_path: Path, daemons: _Daemons
) -> None:
    root = tmp_path / "runtime"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_file_spec(tmp_path).model_dump_json())
    daemon = daemons.start()

    client_process = subprocess.run(
        [sys.executable, "-m", "tests.test_daemon.client_process", str(daemons.state), spec_path],
        cwd=_REPOSITORY,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=False,
    )
    assert client_process.returncode == -signal.SIGKILL, client_process.stderr
    request_id = client_process.stdout.strip()

    with DaemonClient(daemons.state) as client:
        done = _handoff(client, request_id)
        assert done.state is ControllerRequestState.COMPLETED
        experiment = client.aggregates.load_experiment(str(done.experiment_id))
        assert experiment.controller_host.kind == "local_daemon"
        (operation_id,) = workloads(root)
        (node,) = client.aggregates.nodes_for_experiment(str(experiment.id))
        (run,) = client.aggregates.runs_for_node(str(node.id))

        stderr = daemons.stop(daemon)
        assert daemon.returncode == 0, stderr
        assert workloads(root)[operation_id] == {
            **workloads(root)[operation_id],
            "state": "running",
            "cancelled": False,
        }, "the runtime owns the workload; shutdown does not cancel it"
        assert calls(root, "cancel") == []
        (attempt,) = client.aggregates.attempts_for_run(str(run.id))
        assert not attempt.is_terminal, "nothing synthetic recorded at shutdown"
        assert _lock_is_free(daemons.state)

        # Restart: no sweep. The workload ends while nobody watches it.
        finish(root, operation_id)
        watched = len(calls(root, "watch"))
        daemons.start()
        time.sleep(1.0)
        assert len(calls(root, "watch")) == watched
        assert client.aggregates.load_run(str(run.id)).status is RunStatus.ACTIVE

        # An explicit attach adopts it through reconciliation, and only once.
        attach = client.attach(experiment.id)
        assert _handoff(client, attach.id).state is ControllerRequestState.COMPLETED
        _until(lambda: client.aggregates.load_run(str(run.id)).status is RunStatus.SUCCEEDED)
        (attempt,) = client.aggregates.attempts_for_run(str(run.id))
        assert attempt.status is RunAttemptStatus.SUCCEEDED
        assert len(calls(root, "submit")) == 1
        assert len(workloads(root)) == 1
        assert _count(client, "runtime_operations") == 1


# ---- crash points -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("fault", "submissions"),
    [
        # Dies before the runtime gets it: never received, so the next daemon
        # issues it -- under the recorded operation id.
        ("kill-before-submit", 2),
        # Dies after the runtime started it, before recording that: the next
        # daemon finds it by lookup and adopts it.
        ("kill-after-submit", 1),
    ],
)
def test_a_daemon_killed_after_admission_leaves_one_intent_the_next_resumes(
    tmp_path: Path, daemons: _Daemons, fault: str, submissions: int
) -> None:
    root = tmp_path / "runtime"
    with DaemonClient(daemons.state) as client:
        request = client.submit(_file_spec(tmp_path))
        crashing = daemons.start(fault=fault, ready=False)
        crashing.communicate(timeout=_TIMEOUT)
        assert crashing.returncode == -signal.SIGKILL

        assert client.request(request.id).state is ControllerRequestState.ACCEPTED
        (operation,) = client._connection.execute(
            "SELECT id, state FROM runtime_operations"
        ).fetchall()
        assert operation["state"] == "intended"
        assert _lock_is_free(daemons.state)

        daemons.start()
        assert _handoff(client, request.id).state is ControllerRequestState.COMPLETED
        assert list(workloads(root)) == [operation["id"]], "one workload"
        submitted = calls(root, "submit")
        assert len(submitted) == submissions
        assert {entry["subject"] for entry in submitted} == {operation["id"]}
        for table in ("experiments", "experiment_nodes", "runs", "run_attempts"):
            assert _count(client, table) == 1, table
        (row,) = client._connection.execute("SELECT state FROM runtime_operations").fetchall()
        assert row["state"] == "confirmed"


def test_a_daemon_killed_before_admission_leaves_the_request_to_process_once(
    tmp_path: Path, daemons: _Daemons
) -> None:
    root = tmp_path / "runtime"
    with DaemonClient(daemons.state) as client:
        request = client.submit(_file_spec(tmp_path))
        crashing = daemons.start(fault="kill-before-admission", ready=False)
        crashing.communicate(timeout=_TIMEOUT)
        assert crashing.returncode == -signal.SIGKILL
        assert client.request(request.id).state is ControllerRequestState.PENDING
        assert _count(client, "experiments") == 0

        daemons.start()
        assert _handoff(client, request.id).state is ControllerRequestState.COMPLETED
        time.sleep(0.5)
        assert _count(client, "experiments") == 1
        assert _count(client, "runtime_operations") == 1
        assert len(workloads(root)) == 1 and len(calls(root, "submit")) == 1
        assert json.loads(lock_path(daemons.state).read_text())["state_db"] == str(
            daemons.state.resolve()
        )
