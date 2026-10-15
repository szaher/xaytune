"""What a telemetry observation gives the metric-point store (Track B, migration 020)."""

from __future__ import annotations

import pytest

from xaytune.core.metric_points import MAX_NAME_LENGTH, metric_points, metric_unit
from xaytune.core.telemetry import (
    CheckpointStartedPayload,
    DataMetricObserved,
    DistributedMetricObserved,
    HeartbeatPayload,
    IncidentObservedPayload,
    MetricObservedPayload,
    ResourceMetricObserved,
    TrainingMetricObserved,
)


def _named(observation) -> dict[str, tuple[float, str | None, int | None]]:
    return {p.name: (p.value, p.unit, p.optimizer_step) for p in metric_points(observation)}


def test_a_training_metric_gives_one_point_per_reported_field() -> None:
    observation = TrainingMetricObserved(
        optimizer_step=3, loss=1.25, learning_rate=2e-4, tokens_per_second=512.0
    )
    assert _named(observation) == {
        "training.loss": (1.25, None, 3),
        "training.learning_rate": (2e-4, None, 3),
        "training.tokens_per_second": (512.0, "tokens/s", 3),
    }
    assert {p.observation for p in metric_points(observation)} == {"TrainingMetricObserved"}


def test_an_absent_field_is_no_point_and_an_absent_step_is_never_zero() -> None:
    assert metric_points(TrainingMetricObserved(optimizer_step=1)) == ()
    (point,) = metric_points(TrainingMetricObserved(loss=0.0))
    assert point.value == 0.0  # a recorded zero is a point
    assert point.optimizer_step is None


def test_where_a_measurement_was_taken_is_not_a_measurement() -> None:
    observation = ResourceMetricObserved(
        worker="w-1", rank=0, gpu_index=1, gpu_utilization=0.5, host_memory_used_bytes=1024
    )
    assert _named(observation) == {
        "resource.gpu_utilization@gpu1": (0.5, "fraction", None),
        "resource.host_memory_used_bytes": (1024.0, "bytes", None),
    }
    assert {(p.worker, p.rank) for p in metric_points(observation)} == {("w-1", 0)}
    distributed = DistributedMetricObserved(rank=1, world_size=2, straggler_ratio=1.5)
    assert _named(distributed) == {"distributed.straggler_ratio": (1.5, "ratio", None)}


def test_a_custom_metric_cannot_pass_for_a_trainers_own() -> None:
    custom = MetricObservedPayload(name="loss", value=9.0, unit="nats", optimizer_step=2)
    assert _named(custom) == {"custom.loss": (9.0, "nats", 2)}
    too_long = MetricObservedPayload(name="x" * MAX_NAME_LENGTH, value=1.0)
    assert metric_points(too_long) == ()


@pytest.mark.parametrize(
    "observation",
    [
        HeartbeatPayload(expected_interval_seconds=5.0),
        CheckpointStartedPayload(optimizer_step=1),
        IncidentObservedPayload(reason="cuda-oom"),
    ],
)
def test_observations_that_measure_nothing_give_no_points(observation) -> None:
    assert metric_points(observation) == ()


def test_units_follow_the_field_and_a_loss_has_none() -> None:
    assert metric_unit("training.loss") is None
    assert metric_unit("training.step_duration_seconds") == "s"
    assert metric_unit("data.padding_fraction") == "fraction"
    assert metric_unit("resource.gpu_power_watts@gpu0") == "W"
    assert _named(DataMetricObserved(batch_tokens=64))["data.batch_tokens"][1] == "tokens"
