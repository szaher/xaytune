"""``GET /api/v1/attempts/{id}/metrics``: recorded series, read-only, over real telemetry.

The database is a real native training run (``tests.studio_record
--training``): 24 optimizer steps, telemetry carried through a controller
restart into the metric-point store. No point is written by hand.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.studio_record import record_training
from tests.test_api.conftest import Recorded, client
from tests.test_experiment.test_incidents import _seed
from tests.test_experiment.test_metric_ingestion import _event, _Redelivering
from xaytune.api.v1.mapping import downsample
from xaytune.storage.metrics import StoredMetricPoint

STEPS = 24


@pytest.fixture(scope="module")
def training(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return record_training(tmp_path_factory.mktemp("studio-metrics"), steps=STEPS)


def _attempt(state: Path) -> str:
    api = client(state)
    (experiment,) = api.get("/api/v1/experiments").json()["items"]
    graph = api.get(f"/api/v1/experiments/{experiment['id']}/graph").json()
    (node,) = graph["nodes"]
    detail = api.get(f"/api/v1/nodes/{node['id']}").json()
    (run,) = detail["runs"]
    (attempt,) = run["attempts"]
    return attempt["id"]


def test_a_training_runs_series_are_served_with_steps_and_units(training: Path) -> None:
    attempt = _attempt(training)
    body = client(training).get(f"/api/v1/attempts/{attempt}/metrics").json()
    assert body["schema"] == "xaytune.studio-api/v1"
    assert body["attempt_id"] == attempt
    names = [series["name"] for series in body["series"]]
    assert names == sorted(names)
    assert {"training.loss", "training.learning_rate"} <= set(names)
    (loss,) = [series for series in body["series"] if series["name"] == "training.loss"]
    assert (loss["count"], loss["downsampled"], loss["unit"]) == (STEPS, False, None)
    assert [point["optimizer_step"] for point in loss["points"]] == list(range(1, STEPS + 1))
    assert all(isinstance(point["value"], float) for point in loss["points"])
    assert len({point["event_id"] for point in loss["points"]}) == STEPS


def test_a_name_filter_keeps_only_those_series(training: Path) -> None:
    attempt = _attempt(training)
    body = (
        client(training)
        .get(f"/api/v1/attempts/{attempt}/metrics", params={"name": "training.loss"})
        .json()
    )
    assert [series["name"] for series in body["series"]] == ["training.loss"]
    nothing = (
        client(training)
        .get(f"/api/v1/attempts/{attempt}/metrics", params={"name": "training.nothing"})
        .json()
    )
    assert nothing["series"] == []


def test_a_downsampled_series_keeps_its_extremes_and_says_so(training: Path) -> None:
    attempt = _attempt(training)
    full = client(training).get(
        f"/api/v1/attempts/{attempt}/metrics", params={"name": "training.loss"}
    )
    small = client(training).get(
        f"/api/v1/attempts/{attempt}/metrics", params={"name": "training.loss", "max_points": 6}
    )
    (everything,) = full.json()["series"]
    (kept,) = small.json()["series"]
    assert (kept["count"], kept["downsampled"]) == (STEPS, True)
    assert len(kept["points"]) <= 6
    values = [point["value"] for point in everything["points"]]
    kept_values = [point["value"] for point in kept["points"]]
    assert min(values) in kept_values and max(values) in kept_values
    assert kept["points"][0] == everything["points"][0]
    assert kept["points"][-1] == everything["points"][-1]


def test_the_same_series_after_a_studio_restart(training: Path) -> None:
    attempt = _attempt(training)
    first = client(training).get(f"/api/v1/attempts/{attempt}/metrics").json()
    second = client(training).get(f"/api/v1/attempts/{attempt}/metrics").json()
    assert first == second


def test_an_attempt_without_telemetry_metrics_has_no_series(recorded: Recorded) -> None:
    api = client(recorded.state)
    (experiment,) = api.get("/api/v1/experiments").json()["items"]
    graph = api.get(f"/api/v1/experiments/{experiment['id']}/graph").json()
    for node in graph["nodes"]:
        for run in api.get(f"/api/v1/nodes/{node['id']}").json()["runs"]:
            for attempt in run["attempts"]:
                body = api.get(f"/api/v1/attempts/{attempt['id']}/metrics").json()
                assert body["series"] == []


def test_one_custom_name_in_two_units_is_served_as_two_series(tmp_path: Path) -> None:
    """A worker reports ``duration`` in ms, then in s, then in ms: through the
    controller into the store, and out of the API as two series that never
    share a value."""
    from xaytune.core.refs import RuntimeRef
    from xaytune.core.telemetry import MetricObservedPayload
    from xaytune.experiment import EmbeddedControllerHost

    state = tmp_path / "state.db"
    reported = [(1000.0, "ms", 1), (1.0, "s", 2), (1500.0, "ms", 3)]

    async def scenario() -> str:
        host = EmbeddedControllerHost(state)
        try:
            experiment, run, attempt, owner = _seed(host)
            events = [
                _event(
                    owner.target,
                    index,
                    MetricObservedPayload(
                        name="duration", value=value, unit=unit, optimizer_step=step
                    ),
                )
                for index, (value, unit, step) in enumerate(reported)
            ]
            await host._observe(
                experiment.id,
                attempt.id,
                run.id,
                _Redelivering(events),
                RuntimeRef(backend="local", external_id="two-units"),
            )
            return str(attempt.id)
        finally:
            await host.close()

    attempt = asyncio.run(scenario())
    body = client(state).get(f"/api/v1/attempts/{attempt}/metrics").json()
    served = [
        (series["name"], series["unit"], series["count"], [p["value"] for p in series["points"]])
        for series in body["series"]
    ]
    assert served == [
        ("custom.duration", "ms", 2, [1000.0, 1500.0]),
        ("custom.duration", "s", 1, [1.0]),
    ]


def test_an_unknown_attempt_is_not_found_without_echoing_the_id(training: Path) -> None:
    answer = client(training).get("/api/v1/attempts/att_secret-looking-id/metrics")
    assert answer.status_code == 404
    assert answer.headers["content-type"].startswith("application/problem+json")
    assert "secret-looking-id" not in answer.text


@pytest.mark.parametrize("max_points", [0, 3, 5001])
def test_a_max_points_outside_the_bounds_is_refused(training: Path, max_points: int) -> None:
    attempt = _attempt(training)
    answer = client(training).get(
        f"/api/v1/attempts/{attempt}/metrics", params={"max_points": max_points}
    )
    assert answer.status_code == 422
    assert str(max_points) not in answer.json()["detail"]


def _series(values: list[float]) -> list[StoredMetricPoint]:
    return [
        StoredMetricPoint(
            stream_generation=0,
            sequence=index,
            optimizer_step=index + 1,
            value=value,
            worker=None,
            rank=None,
            event_id=f"e{index}",
            emitted_at=None,
        )
        for index, value in enumerate(values)
    ]


@pytest.mark.parametrize("limit", [4, 5, 10, 50])
def test_downsampling_is_bounded_ordered_deterministic_and_keeps_spikes(limit: int) -> None:
    values = [1.0] * 200
    values[37], values[150] = 50.0, -50.0
    points = _series(values)
    kept = downsample(points, limit)
    assert len(kept) <= limit
    assert [p.sequence for p in kept] == sorted(p.sequence for p in kept)
    assert kept[0] == points[0] and kept[-1] == points[-1]
    assert {50.0, -50.0} <= {p.value for p in kept}
    assert downsample(points, limit) == kept


def test_a_short_series_is_not_downsampled() -> None:
    points = _series([3.0, 2.0, 1.0])
    assert downsample(points, 4) == points
