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

    Every field is optional, and every limit is finite and non-negative. A
    zero limit is valid and admits nothing. ``max_parallel_runs`` is a
    capacity, not a quota: at least one run must be able to run.

    What each limit means, and which are enforced, is
    :mod:`xaytune.core.domain.budget`. On-prem environments commonly bound
    GPU-hours with no currency cost attached.
    """

    max_runs: int | None = Field(default=None, ge=0)
    max_parallel_runs: int | None = Field(default=None, ge=1)
    max_gpu_hours: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_wall_time_seconds: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=0)
    max_cost: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_failures: int | None = Field(default=None, ge=0)
