"""Training telemetry reaches the metric-point store through the controller (Track B, 020).

The host writes points from the same telemetry loop that records incidents,
so delivery is at-least-once: a restarted controller replays events past its
cursor, and every point must still be recorded exactly once.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tests.test_experiment.test_incidents import _seed
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.refs import RuntimeRef
from xaytune.core.telemetry import IncidentObservedPayload, TrainingMetricObserved
from xaytune.experiment import EmbeddedControllerHost
from xaytune.runtimes import RuntimeEventEnvelope, RuntimeStatus, TrainingEventPayload
from xaytune.storage.control_plane import ProvenanceError


def _event(target, sequence: int, data) -> RuntimeEventEnvelope:
    return RuntimeEventEnvelope(
        protocol_version="xaytune.telemetry/v1alpha3",
        event_id=f"event-{sequence}",
        target=target,
        sequence=sequence,
        payload=TrainingEventPayload(data=data),
    )


class _Redelivering:
    """Delivers every event it holds, whatever the cursor says (at-least-once)."""

    def __init__(self, events) -> None:
        self.events = events

    async def watch(self, reference, cursor):
        for event in self.events:
            yield RuntimeEventEnvelope.model_validate_json(event.model_dump_json())

    async def get_status(self, reference):
        return RuntimeStatus(state="failed", exit_code=1)


def test_a_restarted_controller_replays_telemetry_without_duplicating_points(tmp_path) -> None:
    async def scenario():
        first = EmbeddedControllerHost(tmp_path / "state.db")
        experiment, run, attempt, owner = _seed(first)
        target = owner.target
        events = [
            _event(target, 0, TrainingMetricObserved(optimizer_step=1, loss=2.0)),
            _event(target, 1, TrainingMetricObserved(optimizer_step=2, loss=1.5)),
            _event(target, 2, IncidentObservedPayload(reason="process-failure")),
        ]
        # The first controller records the first two events' points, then
        # dies before anything moves its cursor.
        for event in events[:2]:
            first._record_metric_points(attempt.id, target, event)
        await first.close()

        second = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            await second._observe(
                experiment.id,
                attempt.id,
                run.id,
                _Redelivering(events),
                RuntimeRef(backend="local", external_id="replayed-workload"),
            )
            await second._observe(
                experiment.id,
                attempt.id,
                run.id,
                _Redelivering(events),
                RuntimeRef(backend="local", external_id="replayed-workload"),
            )
            store = second.repository.metric_points
            points = store.points(str(attempt.id), "training.loss", unit=None)
            assert [(p.optimizer_step, p.value, p.event_id) for p in points] == [
                (1, 2.0, "event-0"),
                (2, 1.5, "event-1"),
            ]
            assert len(second.repository.incidents.for_attempt(target)) == 1
        finally:
            await second.close()

    asyncio.run(scenario())


def test_a_controller_cannot_take_metric_evidence_from_another_attempt(tmp_path) -> None:
    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            _, _, attempt, owner = _seed(host)
            other = RuntimeOperationTarget(kind="training-attempt", id="another-attempt")
            event = _event(other, 0, TrainingMetricObserved(optimizer_step=1, loss=2.0))
            with pytest.raises(ProvenanceError, match="another attempt"):
                host._record_metric_points(attempt.id, owner.target, event)
            assert host.repository.metric_points.series(str(attempt.id)) == ()
        finally:
            await host.close()

    asyncio.run(scenario())


# ---- a real training run ----------------------------------------------------------------------


@pytest.fixture
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _reported_losses(runtime_root: Path) -> list[tuple[int, float]]:
    """What the worker wrote to its telemetry file: the ground truth, read directly."""
    reported = []
    for path in sorted(runtime_root.rglob("events.jsonl")):
        for line in path.read_text().splitlines():
            data = json.loads(line)["payload"]["data"]
            if data["type"] == "TrainingMetricObserved" and data.get("loss") is not None:
                reported.append((data["optimizer_step"], data["loss"]))
    return reported


@pytest.mark.usefixtures("_offline")
def test_a_real_training_runs_loss_history_is_recorded_once_across_a_controller_restart(
    tmp_path: Path,
) -> None:
    """A native worker trains under the local runtime; one host submits and
    closes mid-run, the next adopts the workload and replays its telemetry."""
    from tests.test_experiment.test_host_behaviour import _spec

    async def orphan():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        handle = await host.submit(_spec(tmp_path))
        await asyncio.sleep(0.5)
        await host.close()
        return handle.experiment_id

    async def adopt(experiment_id):
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.attach(experiment_id)
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            ((attempt_id, _),) = [
                (attempt.id, attempt)
                for node in host.repository.aggregates.nodes_for_experiment(str(experiment_id))
                for run in host.repository.aggregates.runs_for_node(str(node.id))
                for attempt in host.repository.aggregates.attempts_for_run(str(run.id))
            ]
            points = host.repository.metric_points.points(
                str(attempt_id), "training.loss", unit=None
            )
            return result, points
        finally:
            await host.close()

    experiment_id = asyncio.run(orphan())
    result, points = asyncio.run(adopt(experiment_id))

    assert result.quiescent
    reported = _reported_losses(tmp_path / "runtime")
    assert reported, "the native worker reported no loss"
    assert [(p.optimizer_step, p.value) for p in points] == reported
    assert len({p.optimizer_step for p in points}) == len(points)
