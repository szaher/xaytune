"""The budget's arithmetic: derived from entries, never double-counted (PR-016).

```text
outstanding = max(reserved - consumed - released, 0)
remaining   = limit - consumed - outstanding
```
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from xaytune.core.domain.budget import (
    BudgetDimension,
    BudgetLedgerEntry,
    BudgetSubjectKind,
    LedgerEntryKind,
    budget_refusals,
    budget_status,
    limits,
)
from xaytune.core.domain.objective import BudgetSpec

RUNS = BudgetDimension.RUNS


def _entry(kind: LedgerEntryKind, amount: Any = 1, **fields: Any) -> BudgetLedgerEntry:
    values: dict[str, Any] = {
        "id": f"ledger_{kind.value}_{fields.get('subject_id', 'run_1')}",
        "experiment_id": "exp_1",
        "dimension": RUNS,
        "kind": kind,
        "amount": Decimal(amount),
        "subject_kind": BudgetSubjectKind.RUN,
        "subject_id": "run_1",
    }
    values.update(fields)
    return BudgetLedgerEntry(**values)


def _runs(spec: BudgetSpec, *entries: BudgetLedgerEntry) -> Any:
    status = budget_status(spec, entries).of(RUNS)
    assert status is not None
    return status


# ---- the limits themselves -------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"max_runs": -1},
        {"max_failures": -1},
        {"max_wall_time_seconds": -1},
        {"max_gpu_hours": float("inf")},
        {"max_gpu_hours": float("nan")},
        {"max_cost": Decimal("NaN")},
        {"max_cost": Decimal("-1")},
        {"max_parallel_runs": 0},
    ],
    ids=["runs", "failures", "wall", "gpu-inf", "gpu-nan", "cost-nan", "cost-neg", "parallel-0"],
)
def test_a_limit_is_finite_and_non_negative_and_capacity_is_at_least_one(
    fields: dict,
) -> None:
    with pytest.raises(ValidationError):
        BudgetSpec(**fields)


def test_only_runs_failures_and_parallel_runs_are_enforced() -> None:
    every = BudgetSpec(
        max_runs=1,
        max_parallel_runs=1,
        max_failures=1,
        max_wall_time_seconds=1,
        max_gpu_hours=1.0,
        max_tokens=1,
        max_cost=Decimal(1),
    )
    assert set(limits(every)) == {RUNS, BudgetDimension.PARALLEL_RUNS, BudgetDimension.FAILURES}
    assert {d.value for d in BudgetDimension} == {"runs", "parallel_runs", "failures"}


def test_zero_is_a_limit_not_the_absence_of_one() -> None:
    assert limits(BudgetSpec(max_runs=0)) == {RUNS: Decimal(0)}
    assert limits(BudgetSpec()) == {}
    assert limits(None) == {}


def test_what_nothing_measures_is_refused_with_why() -> None:
    reasons = budget_refusals(
        BudgetSpec(max_wall_time_seconds=60, max_gpu_hours=1.0, max_tokens=10, max_cost=Decimal(1))
    )
    assert [reason.split(" ")[0] for reason in reasons] == [
        "max_wall_time_seconds",
        "max_gpu_hours",
        "max_tokens",
        "max_cost",
    ]
    assert "not when it ran" in reasons[0], "controller-observed timestamps are not duration"
    assert "GPUs requested times wall time is not what a workload used" in reasons[1]
    assert budget_refusals(BudgetSpec(max_runs=1, max_failures=1, max_parallel_runs=1)) == ()


# ---- derived, not counted ----------------------------------------------------------------


def test_a_reservation_holds_what_it_will_spend() -> None:
    status = _runs(BudgetSpec(max_runs=3), _entry(LedgerEntryKind.RESERVE))
    assert (status.reserved, status.outstanding, status.consumed, status.remaining) == (1, 1, 0, 2)


def test_a_commit_subtracts_nothing_again() -> None:
    status = _runs(
        BudgetSpec(max_runs=3),
        _entry(LedgerEntryKind.RESERVE),
        _entry(LedgerEntryKind.COMMIT),
    )
    assert (status.committed, status.outstanding, status.remaining) == (1, 1, 2)


def test_reserved_then_consumed_is_counted_once() -> None:
    status = _runs(
        BudgetSpec(max_runs=3),
        _entry(LedgerEntryKind.RESERVE),
        _entry(LedgerEntryKind.COMMIT),
        _entry(LedgerEntryKind.CONSUME),
    )
    assert (status.consumed, status.outstanding, status.remaining) == (1, 0, 2)


def test_a_released_reservation_is_returned() -> None:
    status = _runs(
        BudgetSpec(max_runs=3),
        _entry(LedgerEntryKind.RESERVE),
        _entry(LedgerEntryKind.RELEASE),
    )
    assert (status.consumed, status.outstanding, status.remaining) == (0, 0, 3)


def test_reaching_a_limit_exhausts_it_and_only_passing_it_is_an_overrun() -> None:
    reached = _runs(BudgetSpec(max_runs=1), _entry(LedgerEntryKind.CONSUME))
    assert reached.exhausted and not reached.overrun

    passed = _runs(
        BudgetSpec(max_runs=1),
        _entry(LedgerEntryKind.CONSUME),
        _entry(LedgerEntryKind.CONSUME, subject_id="run_2"),
    )
    assert passed.overrun and passed.remaining == -1


def test_a_capacity_is_full_but_never_exhausted() -> None:
    slot = _entry(
        LedgerEntryKind.RESERVE,
        dimension=BudgetDimension.PARALLEL_RUNS,
        subject_kind=BudgetSubjectKind.TRAINING_ATTEMPT,
        subject_id="attempt_1",
    )
    status = budget_status(BudgetSpec(max_parallel_runs=1), [slot]).of(
        BudgetDimension.PARALLEL_RUNS
    )
    assert status is not None
    assert (status.kind, status.remaining, status.exhausted) == ("capacity", 0, False)


def test_only_limited_dimensions_are_reported() -> None:
    status = budget_status(BudgetSpec(max_failures=2), [_entry(LedgerEntryKind.RESERVE)])
    assert [d.dimension for d in status.dimensions] == [BudgetDimension.FAILURES]


def test_nothing_is_recorded_for_nothing() -> None:
    with pytest.raises(ValidationError):
        _entry(LedgerEntryKind.CONSUME, amount=0)
