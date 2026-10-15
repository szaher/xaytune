"""Metric points: the scalars one telemetry event reported, named and ready to store.

Track B's metric-point store (ADR-023 §S12) keeps every scalar a training
attempt's telemetry reported, so loss and throughput survive the runtime's
event files. This module only *reads* an observation; it decides nothing and
touches no storage. Writing happens in the control-plane repository, inside
the telemetry ingestion path.

Names are ``<family>.<field>``, so a custom metric can never be mistaken for
a trainer's own: ``training.loss`` and ``custom.loss`` are different series.
A per-device field carries its device in the name (``…@gpu0``). The step is
the observation's ``optimizer_step`` when it has one, and nothing otherwise:
an absent step is never 0.
"""

from __future__ import annotations

from typing import Any, Final

from xaytune.core.immutable import FrozenDomainModel
from xaytune.core.telemetry import (
    AlignmentMetricObserved,
    DataMetricObserved,
    DistributedMetricObserved,
    MetricObservedPayload,
    ResourceMetricObserved,
    TrainingMetricObserved,
)

__all__ = ["MAX_NAME_LENGTH", "MetricPoint", "metric_points", "metric_unit"]

MAX_NAME_LENGTH: Final = 200
"""The longest series name stored. A longer custom name is not a series."""

_FAMILIES: Final[dict[type[Any], str]] = {
    TrainingMetricObserved: "training",
    ResourceMetricObserved: "resource",
    DataMetricObserved: "data",
    DistributedMetricObserved: "distributed",
    AlignmentMetricObserved: "alignment",
}

_NOT_MEASUREMENTS: Final = frozenset(
    {"type", "metadata", "optimizer_step", "worker", "rank", "world_size", "gpu_index"}
)
"""Fields that say *where* or *when* a measurement was taken, not what it measured."""

_GPU_FIELDS: Final = frozenset(
    {
        "gpu_utilization",
        "gpu_memory_used_bytes",
        "gpu_memory_reserved_bytes",
        "gpu_power_watts",
        "gpu_temperature_celsius",
    }
)

_UNITS: Final[dict[str, str]] = {
    "training.samples_per_second": "samples/s",
    "training.tokens_per_second": "tokens/s",
    "training.optimizer_steps_per_second": "steps/s",
    "training.examples_seen": "examples",
    "training.tokens_seen": "tokens",
    "data.examples_consumed": "examples",
    "data.tokens_consumed": "tokens",
    "data.batch_tokens": "tokens",
    "alignment.rollout_count": "rollouts",
    "alignment.rollout_success_rate": "fraction",
    "alignment.rollouts_per_second": "rollouts/s",
    "alignment.generation_tokens_per_second": "tokens/s",
}


class MetricPoint(FrozenDomainModel):
    """One scalar from one telemetry event. Its position and provenance come with it."""

    name: str
    value: float
    unit: str | None
    optimizer_step: int | None
    worker: str | None
    rank: int | None
    observation: str
    """The observation type that reported it, e.g. ``TrainingMetricObserved``."""


def metric_unit(name: str) -> str | None:
    """The unit of a built-in series, by its name; ``None`` when it has none (a loss)."""
    if name in _UNITS:
        return _UNITS[name]
    field = name.split(".", 1)[-1].split("@", 1)[0]
    for suffix, unit in (
        ("_seconds", "s"),
        ("_bytes", "bytes"),
        ("_utilization", "fraction"),
        ("_fraction", "fraction"),
        ("_watts", "W"),
        ("_celsius", "celsius"),
        ("_ratio", "ratio"),
    ):
        if field.endswith(suffix):
            return unit
    return None


def metric_points(observation: Any) -> tuple[MetricPoint, ...]:
    """Every scalar *observation* reported, in field order. Empty for anything else.

    A custom ``MetricObserved`` gives one point named ``custom.<name>`` with
    its own unit; one whose name would exceed :data:`MAX_NAME_LENGTH` gives
    none. Booleans are never measurements.
    """
    if isinstance(observation, MetricObservedPayload):
        name = f"custom.{observation.name}"
        if len(name) > MAX_NAME_LENGTH:
            return ()
        return (
            MetricPoint(
                name=name,
                value=float(observation.value),
                unit=observation.unit,
                optimizer_step=observation.optimizer_step,
                worker=None,
                rank=None,
                observation=observation.type,
            ),
        )
    family = _FAMILIES.get(type(observation))
    if family is None:
        return ()
    step = getattr(observation, "optimizer_step", None)
    worker = getattr(observation, "worker", None)
    rank = getattr(observation, "rank", None)
    gpu = getattr(observation, "gpu_index", None)
    points: list[MetricPoint] = []
    for field in type(observation).model_fields:
        if field in _NOT_MEASUREMENTS:
            continue
        value = getattr(observation, field)
        if value is None or isinstance(value, bool) or not isinstance(value, int | float):
            continue
        name = f"{family}.{field}"
        if field in _GPU_FIELDS and gpu is not None:
            name = f"{name}@gpu{gpu}"
        points.append(
            MetricPoint(
                name=name,
                value=float(value),
                unit=metric_unit(name),
                optimizer_step=step,
                worker=worker,
                rank=rank,
                observation=observation.type,
            )
        )
    return tuple(points)
