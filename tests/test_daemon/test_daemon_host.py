"""LocalDaemonControllerHost in one process: the mailbox, admission and provenance (PR-027).

The cross-process properties -- a second daemon refused, SIGTERM, SIGKILL, a
client that dies -- are in :mod:`.test_daemon_process`. These drive
:meth:`LocalDaemonControllerHost.serve` as a task.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation_fixtures import EVALUATORS
from tests.test_daemon.file_runtime import FileRuntime, calls, finish, workloads
from tests.test_experiment.adaptive_fixtures import AdaptiveRuntime, LoRACompiler, adaptive_spec
from tests.test_experiment.test_host_behaviour import _spec
from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
from xaytune.compilation.native import NativeCompiler
from xaytune.core.domain.policy import PolicyVerdict
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.core.ids import ExperimentId
from xaytune.core.immutable import FrozenDict
from xaytune.core.state.status import ExperimentStatus, RunAttemptStatus, RunStatus
from xaytune.daemon import (
    ControllerRequest,
    ControllerRequestState,
    DaemonClient,
    DaemonConfig,
    LocalDaemonControllerHost,
)
from xaytune.decision import AdaptiveThresholdDecisionEngine, ThresholdDecisionEngine
from xaytune.experiment import CompilerSpec, RuntimeSpec
from xaytune.planning import PLANNERS
from xaytune.policy import DenyAllPolicy, RulePolicyEngine
from xaytune.storage import IdempotencyConflictError

_TIMEOUT = 30


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _file_spec(tmp_path: Path, **overrides: Any) -> Any:
    return _spec(
        tmp_path,
        runtime=RuntimeSpec(kind="file", config={"root": str(tmp_path / "runtime")}),
        **overrides,
    )


def _config(*, fault: str | None = None, **overrides: Any) -> DaemonConfig:
    fields: dict[str, Any] = {
        "compilers": {"native": NativeCompiler},
        "runtimes": {"file": lambda config: FileRuntime(config["root"], fault=fault)},
        "evaluators": {},
        "planners": PLANNERS,
        "decision_engine": ThresholdDecisionEngine(),
        "policy": DenyAllPolicy(),
        "checkpoint_manager": None,
        "recovery_request_for_incident": None,
    }
    fields.update(overrides)
    return DaemonConfig(**fields)


@contextlib.asynccontextmanager
async def _serving(daemon: LocalDaemonControllerHost) -> AsyncIterator[LocalDaemonControllerHost]:
    stop = asyncio.Event()
    task = asyncio.create_task(daemon.serve(stop))
    try:
        for _ in range(_TIMEOUT * 20):
            if daemon._controller is not None or task.done():
                break
            await asyncio.sleep(0.05)
        if task.done():
            task.result()
        yield daemon
    finally:
        stop.set()
        await asyncio.wait_for(task, _TIMEOUT)


async def _until(predicate: Callable[[], bool]) -> None:
    for _ in range(_TIMEOUT * 20):
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("timed out")


def _state(client: DaemonClient, request: ControllerRequest) -> ControllerRequestState:
    return client.request(request.id).state


def _counts(client: DaemonClient) -> dict[str, int]:
    connection = client._connection
    return {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "experiments",
            "experiment_nodes",
            "runs",
            "run_attempts",
            "runtime_operations",
            "budget_ledger",
            "controller_requests",
        )
    }


def _daemon(tmp_path: Path, **config: Any) -> LocalDaemonControllerHost:
    return LocalDaemonControllerHost(tmp_path / "state.db", _config(**config), poll_interval=0.05)


# ---- submission ---------------------------------------------------------------------


def test_a_submitted_request_is_admitted_issued_and_recorded_as_the_daemons(
    tmp_path: Path,
) -> None:
    client = DaemonClient(tmp_path / "state.db")

    async def scenario() -> None:
        request = client.submit(_file_spec(tmp_path))
        assert _state(client, request) is ControllerRequestState.PENDING
        daemon = _daemon(tmp_path)
        async with _serving(daemon):
            done = await client.wait_for_handoff(request.id, timeout=_TIMEOUT)
            assert done.state is ControllerRequestState.COMPLETED
            experiment = client.aggregates.load_experiment(str(request.experiment_id))
            assert experiment.controller_host.kind == "local_daemon"
            assert experiment.controller_host.id == daemon.instance_id
            assert experiment.status is ExperimentStatus.ACTIVE
            (operation_id,) = workloads(tmp_path / "runtime")
            finish(tmp_path / "runtime", operation_id)
            (node,) = client.aggregates.nodes_for_experiment(str(experiment.id))
            (run,) = client.aggregates.runs_for_node(str(node.id))
            await _until(
                lambda: client.aggregates.load_run(str(run.id)).status is RunStatus.SUCCEEDED
            )

    try:
        asyncio.run(scenario())
    finally:
        client.close()


def test_replaying_a_request_admits_and_issues_nothing_twice(tmp_path: Path) -> None:
    client = DaemonClient(tmp_path / "state.db")

    async def scenario() -> None:
        request = client.submit(_file_spec(tmp_path))
        assert client.send(request) == request, "the same request, recorded once"
        async with _serving(_daemon(tmp_path)):
            await client.wait_for_handoff(request.id, timeout=_TIMEOUT)
        after_first = _counts(client)

        replayed = client.send(request)
        assert replayed.state is ControllerRequestState.COMPLETED
        daemon = _daemon(tmp_path)
        async with _serving(daemon):
            await daemon.process_requests()
            await daemon.process_requests()

        assert _counts(client) == after_first
        assert after_first["experiments"] == after_first["runtime_operations"] == 1
        assert after_first["controller_requests"] == 1
        assert len(workloads(tmp_path / "runtime")) == 1
        assert len(calls(tmp_path / "runtime", "submit")) == 1

    try:
        asyncio.run(scenario())
    finally:
        client.close()


def test_the_same_id_for_another_request_is_refused(tmp_path: Path) -> None:
    with DaemonClient(tmp_path / "state.db") as client:
        request = client.submit(_file_spec(tmp_path))
        other = ControllerRequest.submit(
            FrozenDict(_file_spec(tmp_path, seed=8).model_dump(mode="json")),
            experiment_id=request.experiment_id,
            request_id=request.id,
        )
        with pytest.raises(IdempotencyConflictError):
            client.send(other)


@pytest.mark.parametrize("problem", ["unknown-compiler", "unknown-runtime", "invalid-spec"])
def test_a_request_that_cannot_run_fails_with_nothing_admitted(
    tmp_path: Path, problem: str
) -> None:
    client = DaemonClient(tmp_path / "state.db")

    async def scenario() -> None:
        if problem == "unknown-compiler":
            request = client.submit(_file_spec(tmp_path, compiler=CompilerSpec(name="nope")))
        elif problem == "unknown-runtime":
            request = client.submit(_spec(tmp_path))  # kind "local": not configured
        else:
            request = client.send(ControllerRequest.submit(FrozenDict({"name": "half a spec"})))
        async with _serving(_daemon(tmp_path)):
            done = await client.wait_for_handoff(request.id, timeout=_TIMEOUT)
        assert done.state is ControllerRequestState.FAILED
        assert done.error is not None
        expected = {
            "unknown-compiler": "UnknownImplementationError",
            "unknown-runtime": "UnknownImplementationError",
            "invalid-spec": "ValidationError",
        }[problem]
        assert done.error["type"] == expected
        counts = _counts(client)
        assert counts["experiments"] == counts["runtime_operations"] == 0
        assert workloads(tmp_path / "runtime") == {}

    try:
        asyncio.run(scenario())
    finally:
        client.close()


# ---- a stop between admission and confirmation -------------------------------------


def test_a_daemon_stopped_mid_submission_leaves_one_intent_the_next_resumes(
    tmp_path: Path,
) -> None:
    """Stopped while the runtime is being asked: ACCEPTED, one INTENDED submit, no workload.

    The next daemon resumes the unfinished request through reconciliation:
    the runtime never received it, so it is issued under the recorded
    operation id -- one operation, one workload -- and the handoff completes.
    """
    client = DaemonClient(tmp_path / "state.db")

    async def scenario() -> None:
        request = client.submit(_file_spec(tmp_path))
        async with _serving(_daemon(tmp_path, fault="hang-before-submit")):
            await _until(lambda: _state(client, request) is ControllerRequestState.ACCEPTED)
            await _until(lambda: bool(calls(tmp_path / "runtime", "submit")))
        assert _state(client, request) is ControllerRequestState.ACCEPTED
        (operation,) = client._connection.execute(
            "SELECT id, state FROM runtime_operations"
        ).fetchall()
        assert operation["state"] == "intended"
        assert workloads(tmp_path / "runtime") == {}
        admitted = _counts(client)

        async with _serving(_daemon(tmp_path)):
            done = await client.wait_for_handoff(request.id, timeout=_TIMEOUT)
        assert done.state is ControllerRequestState.COMPLETED
        assert list(workloads(tmp_path / "runtime")) == [operation["id"]]
        assert {entry["subject"] for entry in calls(tmp_path / "runtime", "submit")} == {
            operation["id"]
        }
        resumed = _counts(client)
        assert {k: v for k, v in resumed.items() if k != "budget_ledger"} == {
            k: v for k, v in admitted.items() if k != "budget_ledger"
        }, "no second experiment, node, run, attempt or operation"
        (row,) = client._connection.execute("SELECT state FROM runtime_operations").fetchall()
        assert row["state"] == "confirmed"

    try:
        asyncio.run(scenario())
    finally:
        client.close()


# ---- attach -----------------------------------------------------------------------


def test_attaching_an_unknown_experiment_fails(tmp_path: Path) -> None:
    client = DaemonClient(tmp_path / "state.db")

    async def scenario() -> None:
        request = client.attach(ExperimentId.generate())
        async with _serving(_daemon(tmp_path)):
            done = await client.wait_for_handoff(request.id, timeout=_TIMEOUT)
        assert done.state is ControllerRequestState.FAILED
        assert done.error is not None and done.error["type"] == "AggregateNotFoundError"

    try:
        asyncio.run(scenario())
    finally:
        client.close()


def test_a_restarted_daemon_adopts_only_what_it_is_asked_to(tmp_path: Path) -> None:
    """COMPLETED handoffs are not swept on restart; an attach request adopts one."""
    client = DaemonClient(tmp_path / "state.db")
    root = tmp_path / "runtime"

    async def scenario() -> None:
        request = client.submit(_file_spec(tmp_path))
        async with _serving(_daemon(tmp_path)):
            await client.wait_for_handoff(request.id, timeout=_TIMEOUT)
        (operation_id,) = workloads(root)
        assert workloads(root)[operation_id]["cancelled"] is False
        finish(root, operation_id)
        (node,) = client.aggregates.nodes_for_experiment(str(request.experiment_id))
        (run,) = client.aggregates.runs_for_node(str(node.id))
        watched = len(calls(root, "watch"))

        daemon = _daemon(tmp_path)
        async with _serving(daemon):
            await daemon.process_requests()
            await asyncio.sleep(0.3)
            assert len(calls(root, "watch")) == watched, "nothing adopted it"
            assert client.aggregates.load_run(str(run.id)).status is RunStatus.ACTIVE

            attach = client.attach(request.experiment_id)
            done = await client.wait_for_handoff(attach.id, timeout=_TIMEOUT)
            assert done.state is ControllerRequestState.COMPLETED
            await _until(
                lambda: client.aggregates.load_run(str(run.id)).status is RunStatus.SUCCEEDED
            )
        (attempt,) = client.aggregates.attempts_for_run(str(run.id))
        assert attempt.status is RunAttemptStatus.SUCCEEDED
        assert len(calls(root, "submit")) == 1
        assert calls(root, "cancel") == []

    try:
        asyncio.run(scenario())
    finally:
        client.close()


# ---- the whole loop, under the daemon ---------------------------------------------


def test_the_daemon_drives_the_adaptive_loop_to_its_end(tmp_path: Path) -> None:
    """Spec 18 through the mailbox: submitted, then nobody else calls anything."""
    manager = CheckpointManager(SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles"))
    runtime = AdaptiveRuntime(manager, tmp_path)
    config = DaemonConfig(
        compilers={"native": lambda: LoRACompiler(64)},
        runtimes={"local": lambda config: runtime},
        evaluators=EVALUATORS,
        planners=PLANNERS,
        decision_engine=AdaptiveThresholdDecisionEngine(),
        policy=RulePolicyEngine(default=PolicyVerdict.ALLOW),
        checkpoint_manager=manager,
        recovery_request_for_incident=lambda incident: RecoveryRequest(
            restore_context=runtime.restore_context
        ),
    )
    client = DaemonClient(tmp_path / "state.db")

    async def scenario() -> None:
        request = client.submit(adaptive_spec(tmp_path))
        daemon = LocalDaemonControllerHost(tmp_path / "state.db", config, poll_interval=0.05)
        async with _serving(daemon):
            await client.wait_for_handoff(request.id, timeout=_TIMEOUT)
            await _until(
                lambda: client.aggregates.load_experiment(str(request.experiment_id)).is_terminal
            )
        experiment = client.aggregates.load_experiment(str(request.experiment_id))
        assert experiment.status is ExperimentStatus.SUCCEEDED
        assert experiment.controller_host.kind == "local_daemon"
        nodes = sorted(
            client.aggregates.nodes_for_experiment(str(experiment.id)), key=lambda n: n.created_at
        )
        assert len(nodes) == 2 and experiment.best_node_id == nodes[1].id

    try:
        asyncio.run(scenario())
    finally:
        client.close()
