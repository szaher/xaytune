"""An experiment's budget, through the public API (PR-016).

```text
submit      a limit nothing measures (GPU-hours, tokens, cost)   refused, nothing recorded
            no run left for the first run                         recorded, BUDGET_EXHAUSTED
            wall time, which nothing reports authoritatively yet  refused, nothing recorded
train       runs reserved → committed → consumed; slot held and released
evaluate    a used-up quota refuses the evaluation                BUDGET_EXHAUSTED, node trained
restart     every entry was written with its change               nothing to settle, nothing twice
```

With the scripted evaluator; what each entry is, and the arithmetic, are
``tests/test_storage/test_budget_ledger.py`` and ``tests/test_core/test_budget.py``.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation_fixtures import EVALUATORS, evaluation
from tests.test_experiment.test_evaluation_lifecycle import _drive
from tests.test_experiment.test_evaluation_restart import (
    _adopt,
    _decided_once,
    _evaluated,
    _evaluator_workloads,
)
from tests.test_experiment.test_restart_reconciliation import _crash, _spec
from xaytune.core.domain.budget import (
    BudgetDimension,
    BudgetSubjectKind,
    LedgerEntryKind,
    UnsupportedBudgetError,
)
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus, RunStatus
from xaytune.experiment import EmbeddedControllerHost
from xaytune.experiment.host import _ACTOR
from xaytune.storage import write_transaction

RUNS = BudgetDimension.RUNS
PARALLEL = BudgetDimension.PARALLEL_RUNS
FAILURES = BudgetDimension.FAILURES


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _budgeted(tmp_path: Path, **limits: Any) -> Any:
    return _spec(tmp_path).model_copy(
        update={"evaluation": evaluation(value=0.8), "budget": BudgetSpec(**limits)}
    )


def _dimension(result: Any, dimension: BudgetDimension) -> Any:
    assert result.budget is not None
    found = result.budget.of(dimension)
    assert found is not None
    return found


@pytest.mark.parametrize(
    "limit",
    [{"max_wall_time_seconds": 3600}, {"max_gpu_hours": 2.0}],
    ids=["wall-time", "gpu-hours"],
)
def test_a_limit_nothing_measures_is_refused_before_anything_is_recorded(
    tmp_path: Path, limit: dict
) -> None:
    (field,) = limit

    async def scenario() -> tuple:
        host = EmbeddedControllerHost(tmp_path / "state.db", evaluators=EVALUATORS)
        try:
            with pytest.raises(UnsupportedBudgetError, match=field) as refused:
                await host.submit(_budgeted(tmp_path, max_runs=1, **limit))
            return refused.value, host._connection.execute("SELECT id FROM experiments").fetchall()
        finally:
            await host.close()

    refused, experiments = asyncio.run(scenario())
    assert len(refused.reasons) == 1
    assert experiments == []


def test_no_run_left_for_the_first_run_exhausts_the_experiment_before_any_effect(
    tmp_path: Path,
) -> None:
    result, experiment, events = _drive(tmp_path, _budgeted(tmp_path, max_runs=0))

    assert experiment.status is ExperimentStatus.BUDGET_EXHAUSTED
    assert result.next_stage is None
    (node,) = result.nodes
    assert node.runs == (), "nothing was started, or even recorded as a run"
    runs = _dimension(result, RUNS)
    assert (runs.limit, runs.consumed, runs.remaining) == (0, 0, 0)
    (exhausted,) = [e for e in events if e.event_type == "BudgetExhausted"]
    assert "no run is left" in exhausted.payload["reasons"][0]


def test_a_budgeted_experiment_trains_evaluates_and_accounts_for_both(tmp_path: Path) -> None:
    """Reaching max_runs does not stop the evaluation: it is not a run."""
    spec = _budgeted(tmp_path, max_runs=1, max_parallel_runs=1, max_failures=1)
    result, experiment, events = _drive(tmp_path, spec)

    assert experiment.status is ExperimentStatus.ACTIVE
    assert result.next_stage == "decision", "trained, evaluated, and waiting to be decided"
    runs = _dimension(result, RUNS)
    assert (runs.reserved, runs.committed, runs.consumed, runs.remaining) == (1, 1, 1, 0)
    slots = _dimension(result, PARALLEL)
    assert (slots.reserved, slots.released, slots.outstanding) == (1, 1, 0)
    assert _dimension(result, FAILURES).consumed == 0
    assert result.budget is not None and len(result.budget.dimensions) == 3
    assert not any(e.event_type in ("BudgetExhausted", "BudgetOverrun") for e in events)


class _FailuresRunOut(EmbeddedControllerHost):
    """An earlier attempt failed before training succeeded, using the last failure allowed."""

    async def _continue_to_evaluation(self, experiment_id: Any, node_id: Any) -> None:
        with write_transaction(self.repository._connection):
            self.repository._ledger(
                str(experiment_id),
                FAILURES,
                LedgerEntryKind.CONSUME,
                Decimal(1),
                BudgetSubjectKind.TRAINING_ATTEMPT,
                "an-earlier-failed-attempt",
                _ACTOR,
                (),
            )
        await super()._continue_to_evaluation(experiment_id, node_id)


def test_a_used_up_quota_refuses_the_evaluation_and_ends_the_experiment(tmp_path: Path) -> None:
    result, experiment, events = _drive(
        tmp_path, _budgeted(tmp_path, max_failures=1), host_class=_FailuresRunOut
    )

    assert experiment.status is ExperimentStatus.BUDGET_EXHAUSTED
    assert result.quiescent and result.next_stage is None
    (node,) = result.nodes
    assert node.status is ExperimentNodeStatus.ACTIVE, "trained; nothing evaluated it"
    assert node.evaluations == ()
    assert [run.status for run in node.runs] == [RunStatus.SUCCEEDED]
    assert _evaluator_workloads(tmp_path) == []
    (exhausted,) = [e for e in events if e.event_type == "BudgetExhausted"]
    assert "failures" in exhausted.payload["reasons"][0]
    failures = _dimension(result, FAILURES)
    assert failures.exhausted and not failures.overrun, "reaching the limit is enough"
    assert not any(e.event_type == "BudgetOverrun" for e in events)


def test_a_restart_finds_every_cost_already_written_and_writes_none_twice(
    tmp_path: Path,
) -> None:
    spec = _evaluated(tmp_path, hold=True).model_copy(
        update={"budget": BudgetSpec(max_runs=1, max_parallel_runs=1, max_failures=3)}
    )
    experiment_id = _crash(tmp_path, "evaluating", spec)

    result, run, attempts, _ = _adopt(tmp_path, experiment_id)

    _decided_once(result, run, attempts)
    from xaytune.core.sqlite import connect
    from xaytune.storage import ControlPlaneRepository

    connection = connect(tmp_path / "state.db")
    try:
        repo = ControlPlaneRepository(connection)
        entries = repo.budget.entries(experiment_id)
        keys = Counter((e.subject_kind, e.subject_id, e.dimension, e.kind) for e in entries)
        assert max(keys.values()) == 1, "no entry was written twice"
        assert {e.subject_kind for e in entries} == {
            BudgetSubjectKind.RUN,
            BudgetSubjectKind.TRAINING_ATTEMPT,
        }, "the run and its slot; the evaluation spent nothing it did not fail"
        assert repo.settle_budget(experiment_id, actor=_ACTOR) == 0
    finally:
        connection.close()
    runs = _dimension(result, RUNS)
    assert (runs.consumed, runs.outstanding) == (1, 0)
    assert _dimension(result, PARALLEL).outstanding == 0
