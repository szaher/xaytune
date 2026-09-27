"""The budget ledger's vocabulary: what is limited, what each entry means, what is left.

A budget is not a counter (spec 09 §10). Each consequence of a run or an
attempt is an **entry** in an append-only journal, and every balance is
derived from the entries, never stored:

```text
reserve   set aside before an effect         a claim, not yet spent
commit    the effect took place              provenance; subtracts nothing again
consume   what was actually spent            the quota's usage
release   a reservation no longer needed     returns what was set aside

outstanding = max(reserved - consumed - released, 0)
remaining   = limit - consumed - outstanding
```

**Two kinds of limit.**

- A **quota** is spent: once it is used up, the next effect is not taken.
  ``runs`` is reserved before a run is recorded; ``failures`` is consumed as
  attempts fail. Consuming past a limit is an **overrun**; reaching it
  exactly is only exhaustion.
- A **capacity** is held, never spent: ``parallel_runs`` is a semaphore on
  live training attempts. When it is full, the next attempt waits for a slot;
  it never exhausts the experiment.

**What is not enforced** is refused rather than pretended:
``max_wall_time_seconds``, ``max_gpu_hours``, ``max_tokens`` and ``max_cost``
have no authoritative measure yet -- attempt timestamps are stamped by the
controller when it observes a change, so a restart can shrink a workload's
duration to seconds -- so a budget that sets one is refused at submission.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import Field

from xaytune.core.clock import utc_now
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.immutable import FrozenDomainModel

__all__ = [
    "BudgetDimension",
    "BudgetExhaustedError",
    "BudgetLedgerEntry",
    "BudgetStatus",
    "BudgetSubjectKind",
    "CapacityUnavailableError",
    "DimensionStatus",
    "LedgerEntryKind",
    "UnsupportedBudgetError",
    "budget_refusals",
    "budget_status",
    "limits",
]


class BudgetDimension(str, Enum):
    """What a limit bounds. Each is one ``BudgetSpec`` field the ledger enforces."""

    RUNS = "runs"
    PARALLEL_RUNS = "parallel_runs"
    FAILURES = "failures"

    @property
    def is_capacity(self) -> bool:
        """Held and given back, never spent: a full capacity means wait, not exhaustion."""
        return self is BudgetDimension.PARALLEL_RUNS


class LedgerEntryKind(str, Enum):
    RESERVE = "reserve"
    COMMIT = "commit"
    CONSUME = "consume"
    RELEASE = "release"


class BudgetSubjectKind(str, Enum):
    """What an entry is about: a training run, or an attempt of either kind."""

    RUN = "run"
    TRAINING_ATTEMPT = "training-attempt"
    EVALUATION_ATTEMPT = "evaluation-attempt"


_LIMIT_FIELDS: dict[BudgetDimension, str] = {
    BudgetDimension.RUNS: "max_runs",
    BudgetDimension.PARALLEL_RUNS: "max_parallel_runs",
    BudgetDimension.FAILURES: "max_failures",
}

_UNMEASURED: dict[str, str] = {
    "max_wall_time_seconds": (
        "workload duration is not yet reported authoritatively across runtime and "
        "controller restarts: an attempt's timestamps are when the controller "
        "observed it, not when it ran"
    ),
    "max_gpu_hours": (
        "GPU usage is not measured: GPUs requested times wall time is not what a "
        "workload used, and a worker may choose its device itself"
    ),
    "max_tokens": "tokens are not metered across trainers and evaluators",
    "max_cost": "no runtime reports a price",
}


class UnsupportedBudgetError(ValueError):
    """A budget sets a limit the ledger cannot enforce. Carries every reason."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("the budget cannot be enforced as declared: " + "; ".join(reasons))


class BudgetExhaustedError(RuntimeError):
    """A quota is used up, so the effect that needed it was not taken."""

    def __init__(self, experiment_id: str, reasons: tuple[str, ...]) -> None:
        self.experiment_id = experiment_id
        self.reasons = reasons
        super().__init__(f"experiment {experiment_id}'s budget is exhausted: " + "; ".join(reasons))


class CapacityUnavailableError(RuntimeError):
    """Every slot of a capacity is held. Nothing was written; wait for one to be released."""

    def __init__(self, experiment_id: str, dimension: BudgetDimension, limit: Decimal) -> None:
        self.experiment_id = experiment_id
        self.dimension = dimension
        self.limit = limit
        super().__init__(
            f"experiment {experiment_id} has all {limit} {dimension.value} slots in use"
        )


def budget_refusals(spec: BudgetSpec) -> tuple[str, ...]:
    """Every limit *spec* sets that the ledger cannot enforce, with why."""
    return tuple(
        f"{field} is set; {reason}"
        for field, reason in _UNMEASURED.items()
        if getattr(spec, field) is not None
    )


def limits(spec: BudgetSpec | None) -> dict[BudgetDimension, Decimal]:
    """The enforced limits *spec* sets, as decimals. Unset dimensions are unlimited."""
    if spec is None:
        return {}
    found: dict[BudgetDimension, Decimal] = {}
    for dimension, field in _LIMIT_FIELDS.items():
        value = getattr(spec, field)
        if value is not None:
            found[dimension] = Decimal(value)
    return found


class BudgetLedgerEntry(FrozenDomainModel):
    """One consequence, for one subject, on one dimension. Never changed once written.

    ``(subject_kind, subject_id, dimension, kind)`` identifies it: settling
    the same thing twice writes nothing the second time.
    """

    id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    dimension: BudgetDimension
    kind: LedgerEntryKind
    amount: Decimal = Field(gt=0, allow_inf_nan=False)
    subject_kind: BudgetSubjectKind
    subject_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)


class DimensionStatus(FrozenDomainModel):
    """One dimension's balance, derived from the ledger. ``remaining`` may be negative."""

    dimension: BudgetDimension
    kind: Literal["quota", "capacity"]
    limit: Decimal
    reserved: Decimal
    committed: Decimal
    consumed: Decimal
    released: Decimal
    outstanding: Decimal
    remaining: Decimal

    @property
    def exhausted(self) -> bool:
        """A quota with nothing left. A capacity is never exhausted, only full."""
        return self.kind == "quota" and self.remaining <= 0

    @property
    def overrun(self) -> bool:
        """More was consumed than the limit allows. Reaching it exactly is not an overrun."""
        return self.consumed > self.limit


class BudgetStatus(FrozenDomainModel):
    """Every enforced dimension of an experiment's budget, as the ledger stands."""

    dimensions: tuple[DimensionStatus, ...] = ()

    def of(self, dimension: BudgetDimension) -> DimensionStatus | None:
        return next((d for d in self.dimensions if d.dimension is dimension), None)

    @property
    def exhausted(self) -> tuple[DimensionStatus, ...]:
        return tuple(d for d in self.dimensions if d.exhausted)


def budget_status(spec: BudgetSpec | None, entries: Iterable[BudgetLedgerEntry]) -> BudgetStatus:
    """The balance of each limited dimension, from the ledger's entries alone."""
    bounded = limits(spec)
    totals: dict[tuple[BudgetDimension, LedgerEntryKind], Decimal] = {}
    for entry in entries:
        key = (entry.dimension, entry.kind)
        totals[key] = totals.get(key, Decimal(0)) + entry.amount
    dimensions = []
    for dimension, limit in bounded.items():

        def total(kind: LedgerEntryKind, dimension: BudgetDimension = dimension) -> Decimal:
            return totals.get((dimension, kind), Decimal(0))

        reserved = total(LedgerEntryKind.RESERVE)
        consumed = total(LedgerEntryKind.CONSUME)
        released = total(LedgerEntryKind.RELEASE)
        outstanding = max(reserved - consumed - released, Decimal(0))
        dimensions.append(
            DimensionStatus(
                dimension=dimension,
                kind="capacity" if dimension.is_capacity else "quota",
                limit=limit,
                reserved=reserved,
                committed=total(LedgerEntryKind.COMMIT),
                consumed=consumed,
                released=released,
                outstanding=outstanding,
                remaining=limit - consumed - outstanding,
            )
        )
    return BudgetStatus(dimensions=tuple(dimensions))
