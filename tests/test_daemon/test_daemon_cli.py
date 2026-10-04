"""The CLI against a real daemon process: PR-029's exit criterion.

```text
start daemon
xaytune submit experiment.json      the experiment id; the client exits;
                                    the experiment continues
xaytune watch <id>                  killed part-way: nothing lost; again, it
                                    follows the record until the daemon rests
a proposal awaiting approval        xaytune actions shows it pending;
                                    xaytune approve resolves it
xaytune cancel <id>                 a request: the daemon, not the CLI, calls
                                    the runtime
a client killed waiting, no daemon  the request is still carried out when one
                                    starts; --request-id retries it, once
```
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.test_daemon.file_runtime import calls, finish, workloads
from tests.test_daemon.test_daemon_process import _REPOSITORY, _TIMEOUT, _Daemons, _until
from tests.test_daemon.test_daemon_server import _file_spec
from xaytune.core.domain.action import ActionStatus, ActionTarget
from xaytune.core.domain.actions import RejectCandidate
from xaytune.core.refs import Actor
from xaytune.daemon import ControllerRequestState, LocalDaemonControllerHost


@pytest.fixture
def daemons(tmp_path: Path) -> Iterator[_Daemons]:
    started = _Daemons(tmp_path)
    try:
        yield started
    finally:
        started.close()


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("XAYTUNE_TEST_POLICY", "review")


def _command(state: Path, *args: str) -> list[str]:
    return [sys.executable, "-m", "xaytune", *args, "--state", str(state)]


def _xaytune(state: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _command(state, *args),
        cwd=_REPOSITORY,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT * 2,
    )


def _ok(state: Path, *args: str) -> str:
    done = _xaytune(state, *args)
    assert done.returncode == 0, (args, done.stdout, done.stderr)
    return done.stdout


def test_submit_exit_reconnect_approve_and_cancel_from_the_command_line(
    tmp_path: Path, daemons: _Daemons
) -> None:
    state = daemons.state
    runtime = tmp_path / "runtime"
    spec_path = tmp_path / "experiment.json"
    spec_path.write_text(_file_spec(tmp_path).model_dump_json())
    daemon = daemons.start()

    # Submitted; the client exits; the experiment carries on in the daemon.
    experiment_id = _ok(state, "submit", str(spec_path)).strip()
    (operation_id,) = workloads(runtime)
    assert workloads(runtime)[operation_id]["state"] == "running"
    assert "status      active" in _ok(state, "status", experiment_id)

    # A watcher killed part-way changes nothing.
    watcher = subprocess.Popen(
        _command(state, "watch", experiment_id),
        cwd=_REPOSITORY,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _until(lambda: watcher.poll() is None and len(calls(runtime, "watch")) >= 1)
    watcher.send_signal(signal.SIGKILL)
    watcher.communicate(timeout=_TIMEOUT)

    finish(runtime, operation_id)
    watched = _ok(state, "watch", experiment_id, "--timeout", str(_TIMEOUT))
    assert "ExperimentCreated" in watched and "quiescent   true" in watched
    result = json.loads(_ok(state, "results", experiment_id))
    assert result["nodes"][0]["runs"][0]["status"] == "succeeded"

    # A proposal the daemon's policy sends to a human, approved from the CLI.
    node_id = result["nodes"][0]["node_id"]

    async def propose() -> str:
        async with LocalDaemonControllerHost(state, handoff_timeout=_TIMEOUT) as host:
            governed = await host.handle(experiment_id).propose(
                RejectCandidate(target=ActionTarget(kind="node", id=node_id)),
                reason="off target",
                proposed_by=Actor(type="llm_agent", id="planner"),
            )
            assert governed.action.status is ActionStatus.APPROVAL_PENDING
            return str(governed.action.id)

    action_id = asyncio.run(propose())
    listed = _ok(state, "actions", experiment_id)
    assert action_id in listed and "approval_pending" in listed
    approved = _ok(state, "approve", action_id, "--reason", "agreed", "--approver", "ana")
    assert approved.split() == [action_id, "approved"]
    refused = _xaytune(state, "reject", action_id, "--reason", "no", "--approver", "bo")
    assert refused.returncode == 1 and "ApprovalConflictError" in refused.stderr

    # Cancellation of a second experiment: a request the daemon carries out.
    second = _ok(state, "submit", str(spec_path)).strip()
    assert second != experiment_id
    _ok(state, "cancel", second, "--reason", "enough")
    (cancelled,) = calls(runtime, "cancel")
    assert cancelled["pid"] == daemon.pid, "the daemon called the runtime, not the CLI"
    assert "status      cancelled" in _ok(state, "watch", second, "--timeout", str(_TIMEOUT))


def test_a_client_killed_before_any_daemon_ran_still_hands_its_request_off(
    tmp_path: Path, daemons: _Daemons
) -> None:
    state = daemons.state
    runtime = tmp_path / "runtime"
    spec_path = tmp_path / "experiment.json"
    spec_path.write_text(_file_spec(tmp_path).model_dump_json())

    # No daemon: the submission is recorded, the client killed while it waits.
    client = subprocess.Popen(
        _command(state, "submit", str(spec_path)),
        cwd=_REPOSITORY,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ),
    )
    assert client.stderr is not None
    # Read up to the warning, past anything else on stderr -- a plugin that
    # failed to load, say: the request id is printed before the request is sent.
    lines: list[str] = []
    while not any("no daemon holds a live lease" in line for line in lines):
        line = client.stderr.readline()
        assert line, f"stderr ended: {lines}"
        lines.append(line)
    (request_id,) = [line.split()[1] for line in lines if line.startswith("request ")]
    client.send_signal(signal.SIGKILL)
    client.communicate(timeout=_TIMEOUT)

    daemons.start()
    # Retried by its id: the same request, so the same experiment, admitted once.
    experiment_id = _ok(state, "submit", str(spec_path), "--request-id", request_id).strip()
    again = _ok(state, "submit", str(spec_path), "--request-id", request_id).strip()
    assert again == experiment_id
    assert len(workloads(runtime)) == 1
    assert len(calls(runtime, "submit")) == 1

    async def requests() -> list[tuple[str, ControllerRequestState]]:
        async with LocalDaemonControllerHost(state) as host:
            return [(r.kind, r.state) for r in host.client.requests.for_experiment(experiment_id)]

    assert asyncio.run(requests()) == [("submit", ControllerRequestState.COMPLETED)]
