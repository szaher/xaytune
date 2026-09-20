"""What an experiment is optimizing, and what it is allowed to spend."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field

from xaytune.core.immutable import FrozenDomainModel

__all__ = [
    "BudgetSpec",
    "MetricConstraint",
    "Objective",
    "ObjectiveMetric",
]

ConstraintOperator = Literal["<", "<=", ">", ">=", "==", "!="]


class _Frozen(FrozenDomainModel):
    """Immutable, with a validating ``model_copy``."""


class ObjectiveMetric(_Frozen):
    """The metric an experiment optimizes, and which way is better."""

    name: str
    direction: Literal["maximize", "minimize"]


class MetricConstraint(_Frozen):
    """A bound a candidate must satisfy to be acceptable.

    Example:
        ``MetricConstraint(name="latency_ms", operator="<=", value=120)``
    """

    name: str
    operator: ConstraintOperator
    value: float


class Objective(_Frozen):
    """The optimization target plus any constraints on acceptable candidates."""

    primary: ObjectiveMetric
    target: float | None = None
    constraints: tuple[MetricConstraint, ...] = Field(default_factory=tuple)


class BudgetSpec(_Frozen):
    """Declared limits for an experiment.

    Every field is optional. On-prem environments commonly bound GPU-hours with
    no currency cost attached.
    """

    max_runs: int | None = None
    max_parallel_runs: int | None = None
    max_gpu_hours: float | None = None
    max_wall_time_seconds: int | None = None
    max_tokens: int | None = None
    max_cost: Decimal | None = None
    max_failures: int | None = None
