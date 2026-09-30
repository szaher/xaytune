"""The controller preserves incident evidence and settles unconfigured OOM recovery."""

from __future__ import annotations

import asyncio

import pytest

from tests.test_resilience.test_incidents import envelope
from tests.test_storage.conftest import make_attempt, make_experiment, make_node, make_run
from xaytune.core.domain.incident import IncidentCategory
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.refs import Actor, RuntimeRef
from xaytune.core.state.status import RunAttemptStatus, RunStatus
from xaytune.core.telemetry import IncidentObservedPayload
from xaytune.experiment import EmbeddedControllerHost
from xaytune.runtimes import RuntimeEventEnvelope, RuntimeStatus
from xaytune.storage import write_transaction
from xaytune.storage.control_plane import ProvenanceError


def _seed(host):
    experiment = make_experiment()
    node = make_node(experiment)
    run = make_run(node)
    attempt = make_attempt(run)
    with write_transaction(host.repository._connection):
        store = host.repository.aggregates
        store._insert_experiment(experiment)
        store._insert_node(node)
        store._insert_run(run)
        store._insert_attempt(attempt)
    run = host.repository.transition_run(
        run.id,
        expected_revision=run.revision,
        new_status=RunStatus.ACTIVE,
        actor=Actor(type="system", id="test"),
    )
    target = RuntimeOperationTarget(kind="training-attempt", id=str(attempt.id))
    return experiment, run, attempt, host.repository.incident_context(target)


class _ReplayRuntime:
    def __init__(self, event):
        self.event = event

    async def watch(self, reference, cursor):
        # Deliberately redeliver even after the controller's durable cursor.
        yield RuntimeEventEnvelope.model_validate_json(self.event.model_dump_json())

    async def get_status(self, reference):
        return RuntimeStatus(state="failed", exit_code=1)


def test_a_controller_restart_replays_oom_incident_and_fails_unconfigured_run(tmp_path) -> None:
    async def scenario():
        first = EmbeddedControllerHost(tmp_path / "state.db")
        experiment, run, attempt, owner = _seed(first)
        event = envelope(owner)
        first._record_incident(owner.target, event)
        (original,) = first.repository.incidents.for_attempt(owner.target)
        await first.close()
        second = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            await second._observe(
                experiment.id,
                attempt.id,
                run.id,
                _ReplayRuntime(event),
                RuntimeRef(backend="local", external_id="failed-workload"),
            )
            assert second.repository.incidents.for_attempt(owner.target) == (original,)
            assert (
                second.repository.aggregates.load_attempt(str(attempt.id)).status
                is RunAttemptStatus.FAILED
            )
            assert second.repository.aggregates.load_run(str(run.id)).status is RunStatus.FAILED
            connection = second.repository._connection
            assert connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[0] == 1
            assert connection.execute("SELECT COUNT(*) FROM experiment_nodes").fetchone()[0] == 1
            assert connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 0
            assert connection.execute("SELECT COUNT(*) FROM runtime_operations").fetchone()[0] == 0
            events = second.repository.events.events_for_aggregate(str(attempt.id))
            assert sum(e.event_type == "IncidentRecorded" for e in events) == 1
        finally:
            await second.close()

    asyncio.run(scenario())


def test_cuda_oom_without_recovery_consumer_settles_the_logical_run(tmp_path) -> None:
    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            experiment, run, attempt, owner = _seed(host)
            event = envelope(owner, IncidentObservedPayload(reason="cuda-oom"))
            await host._observe(
                experiment.id,
                attempt.id,
                run.id,
                _ReplayRuntime(event),
                RuntimeRef(backend="local", external_id="oom-workload"),
            )
            assert (
                host.repository.aggregates.load_attempt(str(attempt.id)).status
                is RunAttemptStatus.FAILED
            )
            assert host.repository.aggregates.load_run(str(run.id)).status is RunStatus.FAILED
            assert len(host.repository.incidents.for_attempt(owner.target)) == 1
            assert host.repository.aggregates.attempts_for_run(str(run.id)) == (
                host.repository.aggregates.load_attempt(str(attempt.id)),
            )
        finally:
            await host.close()

    asyncio.run(scenario())


def test_a_controller_cannot_take_incident_evidence_from_another_attempt(tmp_path) -> None:
    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            experiment, run, attempt, owner = _seed(host)
            wrong = RuntimeOperationTarget(kind="training-attempt", id="another-attempt")
            event = envelope(owner).model_copy(update={"target": wrong})
            with pytest.raises(ProvenanceError, match="another attempt"):
                host._record_incident(owner.target, event)
            assert host.repository.incidents.for_attempt(owner.target) == ()
        finally:
            await host.close()

    asyncio.run(scenario())


def test_the_local_runtime_reports_process_failure_as_an_incident(tmp_path, monkeypatch) -> None:
    from tests.test_experiment.test_host_behaviour import _spec

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.submit(_spec(tmp_path, dataset=tmp_path / "missing.jsonl"))
            result = await asyncio.wait_for(handle.wait(), timeout=120)
            incidents = host.repository.incidents.for_experiment(str(handle.experiment_id))
            assert any(i.category is IncidentCategory.PROCESS_FAILURE for i in incidents)
            assert any(i.category is IncidentCategory.UNKNOWN for i in incidents)
            assert all(i.context.experiment_id == handle.experiment_id for i in incidents)
            assert result.next_stage == "failure-handling"
            assert await handle.actions() == ()
            assert (
                host.repository._connection.execute("SELECT COUNT(*) FROM run_attempts").fetchone()[
                    0
                ]
                == 1
            )
        finally:
            await host.close()

    asyncio.run(scenario())
