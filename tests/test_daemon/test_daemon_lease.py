"""The daemon's lease and its startup sweep, in one process (PR-028).

```text
embedded host        reads; every write refused while the daemon's lease lives,
                     whether it opened before the daemon or after
DaemonClient         still commits requests: the mailbox is not controller state
lease lost           renewal refused, an observer's or a request's write fenced:
                     the daemon stops, raising LeaseLostError, and writes nothing
startup sweep        every nonterminal experiment a daemon admitted or adopted
                     is attached; an embedded one nobody attached is left alone;
                     sweeping twice repeats nothing
budget               the ledger repaired, and nothing else: a used-up quota
                     stops an effect when one is about to be created, never
                     because the controller restarted
```

The cross-process exit criterion -- a daemon SIGKILLed, its successor waiting
out the lease -- is in :mod:`.test_daemon_process`.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation_fixtures import EVALUATORS
from tests.test_daemon.file_runtime import FileRuntime, calls, finish, workloads
from tests.test_daemon.test_daemon_server import (
    _TIMEOUT,
    _config,
    _counts,
    _daemon,
    _file_spec,
    _serving,
    _until,
)
from tests.test_experiment.adaptive_fixtures import AdaptiveRuntime, LoRACompiler, adaptive_spec
from tests.test_storage.conftest import make_attempt, make_experiment, make_node, make_run
from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
from xaytune.compilation.native import NativeCompiler
from xaytune.core.clock import utc_now
from xaytune.core.domain.budget import BudgetDimension
from xaytune.core.domain.decision import DecisionOutcome
from xaytune.core.domain.experiment import Experiment
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.domain.policy import PolicyVerdict
from xaytune.core.ids import ExperimentId
from xaytune.core.refs import Actor, ControllerHostRef
from xaytune.core.state.status import (
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)
from xaytune.daemon import (
    ControllerRequestState,
    DaemonClient,
    DaemonConfig,
    DaemonConfigurationError,
    LocalDaemonControllerServer,
)
from xaytune.decision import AdaptiveThresholdDecisionEngine
from xaytune.experiment import EmbeddedControllerHost
from xaytune.planning import PLANNERS
from xaytune.policy import RulePolicyEngine
from xaytune.storage import (
    ControllerLeaseHeldError,
    ControllerLeaseStore,
    ControlPlaneRepository,
    LeaseLostError,
    write_transaction,
)

_ACTOR = Actor(type="system", id="test")
_OLD_DAEMON = ControllerHostRef(kind="local_daemon", id="daemon-gone")


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _embedded(tmp_path: Path, **options: Any) -> EmbeddedControllerHost:
    return EmbeddedControllerHost(
        tmp_path / "state.db",
        compilers={"native": NativeCompiler},
        runtimes={"file": lambda config: FileRuntime(config["root"])},
        evaluators={},
        **options,
    )


def _steal(client: DaemonClient) -> None:
    """Take the lease as another controller: as if this daemon's had expired."""
    later = utc_now() + timedelta(hours=1)
    thief = ControllerLeaseStore(client._connection, clock=lambda: later)
    assert thief.acquire("daemon-thief", timedelta(hours=2)) is not None


def _owner(client: DaemonClient) -> str:
    lease = ControllerLeaseStore(client._connection).current()
    assert lease is not None
    return lease.controller_id


async def _lost(task: asyncio.Task[None]) -> LeaseLostError:
    with pytest.raises(LeaseLostError) as lost:
        await asyncio.wait_for(task, _TIMEOUT)
    return lost.value


async def _started(daemon: LocalDaemonControllerServer) -> asyncio.Task[None]:
    task = asyncio.create_task(daemon.serve(asyncio.Event()))
    await _until(lambda: daemon._controller is not None or task.done())
    return task


# ---- the embedded host -----------------------------------------------------------------


def test_an_embedded_host_reads_a_daemons_database_but_cannot_control_it(
    tmp_path: Path,
) -> None:
    client = DaemonClient(tmp_path / "state.db")

    async def scenario() -> None:
        async with _serving(_daemon(tmp_path)) as daemon:
            # The mailbox is not controller state: a client writes under the lease.
            request = client.submit(_file_spec(tmp_path))
            done = await client.wait_for_handoff(request.id, timeout=_TIMEOUT)
            assert done.state is ControllerRequestState.COMPLETED
            assert _owner(client) == daemon.instance_id

            embedded = _embedded(tmp_path)
            try:
                before = _counts(client)
                with pytest.raises(ControllerLeaseHeldError, match=daemon.instance_id):
                    await embedded.submit(_file_spec(tmp_path))
                assert _counts(client) == before, "nothing recorded"
                experiment = embedded.repository.aggregates.load_experiment(
                    str(request.experiment_id)
                )
                assert experiment.controller_host == daemon.reference, "reads remain"
            finally:
                await embedded.close()

    try:
        asyncio.run(scenario())
    finally:
        client.close()


def test_an_embedded_host_opened_first_loses_its_write_authority_to_the_daemon(
    tmp_path: Path,
) -> None:
    client = DaemonClient(tmp_path / "state.db")
    root = tmp_path / "runtime"

    async def scenario() -> None:
        embedded = _embedded(tmp_path)
        try:
            handle = await embedded.submit(_file_spec(tmp_path))
            (operation_id,) = workloads(root)
            (node,) = client.aggregates.nodes_for_experiment(str(handle.experiment_id))
            (run,) = client.aggregates.runs_for_node(str(node.id))

            async with _serving(_daemon(tmp_path)):
                # Its workload ends while the embedded host still watches it:
                # the settlement it would write is refused.
                finish(root, operation_id)
                await _until(
                    lambda: (
                        embedded._controllers[str(handle.experiment_id)] == {}
                        or all(
                            task.done()
                            for task in embedded._controllers[str(handle.experiment_id)].values()
                        )
                    )
                )
                (attempt,) = client.aggregates.attempts_for_run(str(run.id))
                assert not attempt.is_terminal, "the embedded observer wrote nothing"
                assert client.aggregates.load_run(str(run.id)).status is RunStatus.ACTIVE
                with pytest.raises(ControllerLeaseHeldError):
                    await embedded.submit(_file_spec(tmp_path))
                # Not the daemon's either: an embedded experiment nobody attached.
                assert client.aggregates.load_run(str(run.id)).status is RunStatus.ACTIVE
        finally:
            await embedded.close()

    try:
        asyncio.run(scenario())
    finally:
        client.close()


# ---- losing the lease ------------------------------------------------------------------


def test_a_daemon_that_cannot_renew_its_lease_stops_without_writing(tmp_path: Path) -> None:
    client = DaemonClient(tmp_path / "state.db")
    root = tmp_path / "runtime"

    async def scenario() -> None:
        daemon = LocalDaemonControllerServer(
            tmp_path / "state.db", _config(lease_ttl_seconds=0.6), poll_interval=0.05
        )
        task = await _started(daemon)
        request = client.submit(_file_spec(tmp_path))
        await client.wait_for_handoff(request.id, timeout=_TIMEOUT)
        (operation_id,) = workloads(root)

        _steal(client)
        before = _counts(client)
        lost = await _lost(task)

        assert "daemon-thief" in str(lost)
        assert _counts(client) == before, "nothing written after the loss"
        assert _owner(client) == "daemon-thief", "stopping does not touch another's lease"
        assert workloads(root)[operation_id]["cancelled"] is False
        assert calls(root, "cancel") == []
        (node,) = client.aggregates.nodes_for_experiment(str(request.experiment_id))
        (run,) = client.aggregates.runs_for_node(str(node.id))
        (attempt,) = client.aggregates.attempts_for_run(str(run.id))
        assert not attempt.is_terminal, "no synthetic outcome"
        assert daemon._lock.held is False

    try:
        asyncio.run(scenario())
    finally:
        client.close()


def test_an_observer_that_finds_the_lease_gone_stops_the_daemon(tmp_path: Path) -> None:
    client = DaemonClient(tmp_path / "state.db")
    root = tmp_path / "runtime"

    async def scenario() -> None:
        daemon = _daemon(tmp_path)  # a 30 s lease: the heartbeat will not notice first
        task = await _started(daemon)
        request = client.submit(_file_spec(tmp_path))
        await client.wait_for_handoff(request.id, timeout=_TIMEOUT)
        (operation_id,) = workloads(root)
        (node,) = client.aggregates.nodes_for_experiment(str(request.experiment_id))
        (run,) = client.aggregates.runs_for_node(str(node.id))

        _steal(client)
        finish(root, operation_id)
        await _lost(task)

        (attempt,) = client.aggregates.attempts_for_run(str(run.id))
        assert attempt.status is not RunAttemptStatus.SUCCEEDED, "its settlement was fenced"
        assert client.aggregates.load_run(str(run.id)).status is RunStatus.ACTIVE
        assert _owner(client) == "daemon-thief"

    try:
        asyncio.run(scenario())
    finally:
        client.close()


def test_a_request_processed_after_the_lease_is_gone_stops_the_daemon(tmp_path: Path) -> None:
    client = DaemonClient(tmp_path / "state.db")

    async def scenario() -> None:
        daemon = _daemon(tmp_path)
        task = await _started(daemon)
        _steal(client)
        before = _counts(client)
        request = client.submit(_file_spec(tmp_path))
        await _lost(task)

        assert client.request(request.id).state is ControllerRequestState.PENDING
        after = _counts(client)
        assert after["experiments"] == before["experiments"] == 0
        assert after["controller_requests"] == before["controller_requests"] + 1

    try:
        asyncio.run(scenario())
    finally:
        client.close()


# ---- the startup sweep -----------------------------------------------------------------


def _admitted_never_issued(tmp_path: Path) -> ExperimentId:
    """An experiment a daemon admitted and died before issuing: an INTENDED submit."""

    async def admit() -> ExperimentId:
        host = _embedded(tmp_path, controller_host=_OLD_DAEMON)

        async def died_here(*args: Any, **kwargs: Any) -> None:
            return None

        host._issue = died_here  # type: ignore[method-assign]
        try:
            handle = await host.submit(_file_spec(tmp_path))
            return handle.experiment_id
        finally:
            await host.close()

    return asyncio.run(admit())


def test_the_sweep_reconciles_an_intended_submission_once_however_often_it_runs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    experiment_id = _admitted_never_issued(tmp_path)
    client = DaemonClient(tmp_path / "state.db")

    async def scenario() -> None:
        (node,) = client.aggregates.nodes_for_experiment(str(experiment_id))
        (run,) = client.aggregates.runs_for_node(str(node.id))
        (attempt,) = client.aggregates.attempts_for_run(str(run.id))
        (submission,) = client._repository.operations.for_target(
            "training-attempt", str(attempt.id)
        )
        assert submission.state == "intended"
        assert workloads(root) == {}

        async with _serving(_daemon(tmp_path)) as daemon:
            await _until(lambda: len(workloads(root)) == 1)
            assert list(workloads(root)) == [str(submission.id)], "under its recorded id"
            settled = _counts(client)
            assert await daemon.sweep() == (experiment_id,)
            assert _counts(client) == settled
        async with _serving(_daemon(tmp_path)):
            # A sweep interrupted by a restart: the next one repeats nothing.
            await _until(lambda: len(calls(root, "watch")) == 2)
            assert _counts(client) == settled
            finish(root, str(submission.id))
            await _until(
                lambda: client.aggregates.load_run(str(run.id)).status is RunStatus.SUCCEEDED
            )

        assert len(calls(root, "submit")) == 1
        assert [a.id for a in client.aggregates.attempts_for_run(str(run.id))] == [attempt.id]
        (confirmed,) = client._repository.operations.for_target("training-attempt", str(attempt.id))
        assert confirmed.id == submission.id and confirmed.state == "confirmed"
        assert client.aggregates.load_experiment(str(experiment_id)).controller_host == (
            _OLD_DAEMON
        ), "provenance is never rewritten by a takeover"

    try:
        asyncio.run(scenario())
    finally:
        client.close()


def test_an_embedded_experiment_is_swept_only_once_a_daemon_has_adopted_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    client = DaemonClient(tmp_path / "state.db")

    async def scenario() -> None:
        embedded = _embedded(tmp_path)
        try:
            handle = await embedded.submit(_file_spec(tmp_path))
        finally:
            await embedded.close()
        experiment_id = handle.experiment_id
        (operation_id,) = workloads(root)
        (node,) = client.aggregates.nodes_for_experiment(str(experiment_id))
        (run,) = client.aggregates.runs_for_node(str(node.id))

        watched = len(calls(root, "watch"))

        async with _serving(_daemon(tmp_path)) as daemon:
            assert await daemon.sweep() == ()
            assert len(calls(root, "watch")) == watched, "the daemon left it to its host"
            attach = client.attach(experiment_id)
            done = await client.wait_for_handoff(attach.id, timeout=_TIMEOUT)
            assert done.state is ControllerRequestState.COMPLETED

        async with _serving(_daemon(tmp_path)) as daemon:
            # The COMPLETED attach is durable adoption: swept without being asked.
            await _until(lambda: len(calls(root, "watch")) == watched + 2)
            assert await daemon.sweep() == (experiment_id,)
            finish(root, operation_id)
            await _until(
                lambda: client.aggregates.load_run(str(run.id)).status is RunStatus.SUCCEEDED
            )
        assert len(calls(root, "submit")) == 1

    try:
        asyncio.run(scenario())
    finally:
        client.close()


# ---- budget after downtime -------------------------------------------------------------


def _failed_around_the_ledger(tmp_path: Path, status: ExperimentStatus) -> ExperimentId:
    """A daemon's experiment with one failure the ledger never heard of, at *status*.

    An ordinary failed run, with no recovery to drive: the experiment rests,
    quiescent, in ``failure-handling``. ``max_failures=1``: once the failure
    is settled the quota is used up.
    """
    connection = DaemonClient(tmp_path / "state.db")._connection
    repository = ControlPlaneRepository(connection)
    experiment = Experiment.model_validate(
        {
            **make_experiment().model_dump(mode="python"),
            "budget": BudgetSpec(max_runs=3, max_failures=1),
            "controller_host": _OLD_DAEMON,
        }
    )
    repository.create_experiment(experiment, actor=_ACTOR)
    repository.transition_experiment(
        experiment.id, expected_revision=0, new_status=ExperimentStatus.ACTIVE, actor=_ACTOR
    )
    if status is ExperimentStatus.PAUSED:
        repository.transition_experiment(
            experiment.id, expected_revision=1, new_status=status, actor=_ACTOR
        )
    node = repository.create_node(make_node(experiment), actor=_ACTOR)
    for step in (
        ExperimentNodeStatus.PLANNED,
        ExperimentNodeStatus.READY,
        ExperimentNodeStatus.ACTIVE,
    ):
        node = repository.transition_node(
            node.id, expected_revision=node.revision, new_status=step, actor=_ACTOR
        )
    run = make_run(node)
    attempt = make_attempt(run)
    with write_transaction(connection):
        repository.aggregates._insert_run(run)
        repository.aggregates._insert_attempt(attempt)
        ended = run.with_status(RunStatus.ACTIVE).with_status(RunStatus.FAILED)
        repository.aggregates._update_run(
            type(ended).model_validate({**ended.model_dump(mode="python"), "revision": 1})
        )
        failed = attempt
        for step_status in (
            RunAttemptStatus.QUEUED,
            RunAttemptStatus.STARTING,
            RunAttemptStatus.RUNNING,
            RunAttemptStatus.FAILED,
        ):
            failed = failed.with_status(step_status)
        repository.aggregates._update_attempt(
            type(failed).model_validate({**failed.model_dump(mode="python"), "revision": 1})
        )
    assert repository.budget.entries(str(experiment.id)) == ()
    connection.close()
    return experiment.id


def _failures(client: DaemonClient, experiment_id: ExperimentId) -> Decimal:
    status = client._repository.budget_status(experiment_id)
    assert status is not None
    failures = status.of(BudgetDimension.FAILURES)
    assert failures is not None
    return failures.consumed


def _resting(client: DaemonClient, experiment_id: ExperimentId) -> None:
    """Still ACTIVE, and nothing ever said its budget was exhausted."""
    assert client.aggregates.load_experiment(str(experiment_id)).status is ExperimentStatus.ACTIVE
    events = client._repository.events.events_for_experiment(str(experiment_id))
    assert "BudgetExhausted" not in [event.event_type for event in events]


def test_a_restart_settles_the_ledger_but_does_not_end_a_resting_experiment(
    tmp_path: Path,
) -> None:
    """Failures used up, nothing running, nothing about to run: a restart changes no outcome.

    The live controller leaves this experiment ACTIVE in failure-handling;
    exhaustion is decided where an effect would be created, and none is.
    """
    experiment_id = _failed_around_the_ledger(tmp_path, ExperimentStatus.ACTIVE)
    client = DaemonClient(tmp_path / "state.db")

    async def scenario() -> None:
        async with _serving(_daemon(tmp_path)) as daemon:
            assert await daemon.sweep() == (experiment_id,)
        assert _failures(client, experiment_id) == 1, "the missing settlement was repaired"
        _resting(client, experiment_id)
        status = client._repository.budget_status(experiment_id)
        assert status is not None and [d.dimension for d in status.exhausted] == [
            BudgetDimension.FAILURES
        ]
        # Only what the ledger measures: no time, GPU, token or cost entry.
        assert set(BudgetDimension) == {
            BudgetDimension.RUNS,
            BudgetDimension.PARALLEL_RUNS,
            BudgetDimension.FAILURES,
        }

    try:
        asyncio.run(scenario())
    finally:
        client.close()


def _adaptive_world(tmp_path: Path) -> tuple[Any, DaemonConfig, dict[str, Any]]:
    """The scripted adaptive runtime, as a daemon configuration and as host options."""
    manager = CheckpointManager(SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles"))
    runtime = AdaptiveRuntime(manager, tmp_path, oom_rank=None)
    options: dict[str, Any] = {
        "compilers": {"native": lambda: LoRACompiler(64)},
        "runtimes": {"local": lambda config: runtime},
        "evaluators": EVALUATORS,
        "planners": PLANNERS,
        "decision_engine": AdaptiveThresholdDecisionEngine(),
        "policy": RulePolicyEngine(default=PolicyVerdict.ALLOW),
        "checkpoint_manager": manager,
        "recovery_request_for_incident": None,
    }
    return runtime, DaemonConfig(**options), options


async def _rest(tmp_path: Path, options: dict[str, Any], spec: Any, **stopped: Any) -> Any:
    """Run *spec* under a host that admitted it as a daemon, until it rests; return its id."""
    host = EmbeddedControllerHost(tmp_path / "state.db", controller_host=_OLD_DAEMON, **options)
    try:
        for name, replacement in stopped.items():
            setattr(host, name, replacement)
        handle = await host.submit(spec)
        await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)
        return handle.experiment_id
    finally:
        await host.close()


async def _held(*args: Any, **kwargs: Any) -> None:
    return None


def test_a_trained_node_whose_run_used_the_last_slot_is_still_evaluated_after_a_restart(
    tmp_path: Path,
) -> None:
    """max_runs=1, training succeeded, evaluation never began: evaluation needs no run."""
    runtime, config, options = _adaptive_world(tmp_path)
    spec = adaptive_spec(tmp_path, budget=BudgetSpec(max_runs=1))
    client = DaemonClient(tmp_path / "state.db")

    async def scenario() -> None:
        experiment_id = await _rest(tmp_path, options, spec, _continue_to_evaluation=_held)
        status = client._repository.budget_status(experiment_id)
        assert status is not None and status.of(BudgetDimension.RUNS).remaining == 0  # type: ignore[union-attr]
        assert runtime.evaluation_plans == []

        daemon = LocalDaemonControllerServer(tmp_path / "state.db", config, poll_interval=0.05)
        async with _serving(daemon):
            await _until(lambda: client.aggregates.load_experiment(str(experiment_id)).is_terminal)
        assert len(runtime.evaluation_plans) == 1, "evaluated after the restart"
        (node,) = client.aggregates.nodes_for_experiment(str(experiment_id))
        (decision,) = client.aggregates.decisions_for_node(str(node.id))
        assert decision.outcome is DecisionOutcome.BRANCH
        # Only now, planning another candidate, does the used-up run quota
        # stop anything -- at the effect, as without a restart.
        experiment = client.aggregates.load_experiment(str(experiment_id))
        assert experiment.status is ExperimentStatus.BUDGET_EXHAUSTED

    try:
        asyncio.run(scenario())
    finally:
        client.close()


def test_a_planning_experiment_with_no_planner_is_not_exhausted_by_a_restart(
    tmp_path: Path,
) -> None:
    """Runs used up, resting at planning with nobody to plan: nothing is about to run."""
    runtime, config, options = _adaptive_world(tmp_path)
    spec = adaptive_spec(tmp_path, budget=BudgetSpec(max_runs=1)).model_copy(
        update={"planner": None}
    )
    client = DaemonClient(tmp_path / "state.db")

    async def scenario() -> None:
        experiment_id = await _rest(tmp_path, options, spec)
        _resting(client, experiment_id)
        status = client._repository.budget_status(experiment_id)
        assert status is not None and status.exhausted, "the run quota is used up"

        daemon = LocalDaemonControllerServer(tmp_path / "state.db", config, poll_interval=0.05)
        async with _serving(daemon):
            assert await daemon.sweep() == (experiment_id,)
        _resting(client, experiment_id)
        assert len(runtime.training_plans) == 1

    try:
        asyncio.run(scenario())
    finally:
        client.close()


def test_a_used_up_quota_found_at_startup_leaves_a_paused_experiment_paused(
    tmp_path: Path,
) -> None:
    experiment_id = _failed_around_the_ledger(tmp_path, ExperimentStatus.PAUSED)
    client = DaemonClient(tmp_path / "state.db")

    async def scenario() -> None:
        async with _serving(_daemon(tmp_path)) as daemon:
            assert await daemon.sweep() == (experiment_id,)
        assert _failures(client, experiment_id) == 1, "settled all the same"
        experiment = client.aggregates.load_experiment(str(experiment_id))
        assert experiment.status is ExperimentStatus.PAUSED

    try:
        asyncio.run(scenario())
    finally:
        client.close()


# ---- the one timing setting ------------------------------------------------------------


@pytest.mark.parametrize("ttl", [0, -1.0, float("inf"), float("nan"), "30", True])
def test_a_lease_ttl_must_be_a_finite_positive_number(ttl: Any) -> None:
    with pytest.raises(DaemonConfigurationError, match="lease_ttl_seconds"):
        _config(lease_ttl_seconds=ttl)


def test_the_lease_is_renewed_every_third_of_its_ttl(tmp_path: Path) -> None:
    assert _config().lease_ttl_seconds == 30
    daemon = LocalDaemonControllerServer(tmp_path / "state.db", _config(lease_ttl_seconds=0.9))
    assert daemon._renew_every == pytest.approx(0.3)
