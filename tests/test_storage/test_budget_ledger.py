"""The budget ledger, written with the changes that cause it (PR-016).

```text
run recorded                     runs          reserve 1
training attempt recorded        parallel_runs reserve 1   (full → CapacityUnavailableError)
training submission confirmed    both          commit     (subtracts nothing again)
attempt ends                     failures      consume 1 if FAILED (not PREEMPTED, CANCELLED)
training attempt ends            parallel_runs release
run ends                         runs          consume if committed, else release
```

Each entry commits in the same transaction as the change; a used-up quota
refuses the next run, attempt or evaluation cycle before anything is written.
"""

from __future__ import annotations

import sqlite3
from decimal import Decimal
from typing import Any

import pytest

from tests.test_storage.test_evaluation_lifecycle import _attempt as _evaluation_attempt
from tests.test_storage.test_evaluation_lifecycle import _begin
from tests.test_storage.test_evaluation_lifecycle import _run as _evaluation_run
from tests.test_storage.test_evaluation_lifecycle import _running as _evaluation_running
from xaytune.core import Actor, ExperimentStatus, RunAttemptStatus, RuntimeRef
from xaytune.core.domain.budget import (
    BudgetDimension,
    BudgetExhaustedError,
    BudgetLedgerEntry,
    BudgetSubjectKind,
    CapacityUnavailableError,
    LedgerEntryKind,
)
from xaytune.core.domain.experiment import Experiment
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.state.status import (
    EvaluationAttemptStatus,
    ExperimentNodeStatus,
    RunStatus,
)
from xaytune.storage import ControlPlaneRepository, write_transaction
from xaytune.storage.budget import LedgerConflictError

from .conftest import make_attempt, make_experiment, make_node, make_run

ACTOR = Actor(type="system", id="controller")
RUNS = BudgetDimension.RUNS
PARALLEL = BudgetDimension.PARALLEL_RUNS
FAILURES = BudgetDimension.FAILURES


@pytest.fixture
def repo(connection: sqlite3.Connection) -> ControlPlaneRepository:
    return ControlPlaneRepository(connection)


def _node(repo: ControlPlaneRepository, **limits: Any) -> Any:
    """An ACTIVE experiment limited by *limits*, and one ACTIVE node in it."""
    plain = make_experiment()
    experiment = Experiment.model_validate(
        {**plain.model_dump(mode="python"), "budget": BudgetSpec(**limits)}
    )
    repo.create_experiment(experiment, actor=ACTOR)
    repo.transition_experiment(
        experiment.id, expected_revision=0, new_status=ExperimentStatus.ACTIVE, actor=ACTOR
    )
    node = repo.create_node(make_node(experiment), actor=ACTOR)
    for status in (
        ExperimentNodeStatus.PLANNED,
        ExperimentNodeStatus.READY,
        ExperimentNodeStatus.ACTIVE,
    ):
        node = repo.transition_node(
            node.id, expected_revision=node.revision, new_status=status, actor=ACTOR
        )
    return node


def _submitted(repo: ControlPlaneRepository, run: Any, number: int = 1) -> Any:
    """A training attempt of *run*, recorded with its intent and confirmed by the runtime."""
    attempt, operation = repo.create_attempt_with_submit_intent(
        make_attempt(run, number), request_digest=f"sha256:{run.id}-{number}", actor=ACTOR
    )
    repo.confirm_operation(
        operation.id,
        expected_revision=operation.revision,
        actor=ACTOR,
        runtime_ref=RuntimeRef(backend="local", external_id=f"pid-{attempt.id}"),
    )
    return attempt


def _end(repo: ControlPlaneRepository, attempt: Any, status: RunAttemptStatus) -> Any:
    """Run *attempt* through RUNNING, then to *status*."""
    for step in (RunAttemptStatus.QUEUED, RunAttemptStatus.STARTING, RunAttemptStatus.RUNNING):
        attempt = repo.transition_attempt(
            attempt.id, expected_revision=attempt.revision, new_status=step, actor=ACTOR
        )
    return repo.transition_attempt(
        attempt.id, expected_revision=attempt.revision, new_status=status, actor=ACTOR
    )


def _finish(repo: ControlPlaneRepository, run: Any, status: RunStatus) -> Any:
    run = repo.aggregates.load_run(str(run.id))
    if run.status is RunStatus.CREATED and status is not RunStatus.CANCELLED:
        run = repo.transition_run(
            run.id, expected_revision=run.revision, new_status=RunStatus.ACTIVE, actor=ACTOR
        )
    return repo.transition_run(
        run.id, expected_revision=run.revision, new_status=status, actor=ACTOR
    )


def _status(repo: ControlPlaneRepository, node: Any, dimension: BudgetDimension) -> Any:
    status = repo.budget_status(node.experiment_id)
    assert status is not None
    found = status.of(dimension)
    assert found is not None
    return found


def _entries(repo: ControlPlaneRepository, node: Any) -> list[tuple[str, str, str]]:
    return [
        (e.dimension.value, e.kind.value, e.subject_kind.value)
        for e in repo.budget.entries(str(node.experiment_id))
    ]


# ---- nothing limited, nothing written ---------------------------------------------------


def test_an_experiment_without_a_budget_has_no_ledger(repo: ControlPlaneRepository) -> None:
    node = _node(repo)
    run = repo.create_run(make_run(node), actor=ACTOR)
    _end(repo, _submitted(repo, run), RunAttemptStatus.FAILED)

    assert repo.budget_status(node.experiment_id) is None
    assert _entries(repo, node) == []


# ---- runs: a hard quota, per Run ---------------------------------------------------------


def test_a_run_is_reserved_committed_and_consumed_once(repo: ControlPlaneRepository) -> None:
    node = _node(repo, max_runs=2)
    run = repo.create_run(make_run(node), actor=ACTOR)
    reserved = _status(repo, node, RUNS)
    assert (reserved.reserved, reserved.outstanding, reserved.remaining) == (1, 1, 1)

    attempt = _submitted(repo, run)
    committed = _status(repo, node, RUNS)
    assert (committed.committed, committed.remaining) == (1, 1), "a commit subtracts nothing"

    _end(repo, attempt, RunAttemptStatus.SUCCEEDED)
    _finish(repo, run, RunStatus.SUCCEEDED)
    spent = _status(repo, node, RUNS)
    assert (spent.consumed, spent.outstanding, spent.remaining) == (1, 0, 1)


def test_a_retry_under_the_same_run_spends_no_second_run(repo: ControlPlaneRepository) -> None:
    node = _node(repo, max_runs=1)
    run = repo.create_run(make_run(node), actor=ACTOR)
    _end(repo, _submitted(repo, run, 1), RunAttemptStatus.PREEMPTED)
    _end(repo, _submitted(repo, run, 2), RunAttemptStatus.SUCCEEDED)
    _finish(repo, run, RunStatus.SUCCEEDED)

    assert _status(repo, node, RUNS).consumed == 1


def test_a_run_that_never_reached_the_runtime_is_released_not_spent(
    repo: ControlPlaneRepository,
) -> None:
    node = _node(repo, max_runs=1)
    run = repo.create_run(make_run(node), actor=ACTOR)
    _finish(repo, run, RunStatus.CANCELLED)

    status = _status(repo, node, RUNS)
    assert (status.consumed, status.released, status.remaining) == (0, 1, 1)


def test_a_used_up_run_quota_refuses_the_next_run_before_writing_it(
    repo: ControlPlaneRepository,
) -> None:
    node = _node(repo, max_runs=1)
    repo.create_run(make_run(node), actor=ACTOR)
    before = _entries(repo, node)

    second = make_run(node)
    with pytest.raises(BudgetExhaustedError, match="no run is left"):
        repo.create_run(second, actor=ACTOR)

    assert repo.aggregates.get_run(str(second.id)) is None
    assert _entries(repo, node) == before


# ---- parallel runs: a capacity, never spent ------------------------------------------------


def test_a_full_capacity_refuses_an_attempt_until_a_slot_is_released(
    repo: ControlPlaneRepository,
) -> None:
    node = _node(repo, max_parallel_runs=1)
    first = repo.create_run(make_run(node), actor=ACTOR)
    second = repo.create_run(make_run(node), actor=ACTOR)
    live = _submitted(repo, first)

    waiting = make_attempt(second)
    with pytest.raises(CapacityUnavailableError):
        repo.create_attempt_with_submit_intent(waiting, request_digest="sha256:w", actor=ACTOR)
    assert repo.aggregates.get_attempt(str(waiting.id)) is None, "nothing written while it waits"

    _end(repo, live, RunAttemptStatus.SUCCEEDED)
    repo.create_attempt_with_submit_intent(waiting, request_digest="sha256:w", actor=ACTOR)

    slots = _status(repo, node, PARALLEL)
    assert (slots.consumed, slots.outstanding, slots.remaining) == (0, 1, 0)
    assert not slots.exhausted, "full is not exhausted"


def test_retrying_a_recorded_attempt_takes_no_new_slot(repo: ControlPlaneRepository) -> None:
    node = _node(repo, max_parallel_runs=1)
    run = repo.create_run(make_run(node), actor=ACTOR)
    attempt = make_attempt(run)
    _, operation = repo.create_attempt_with_submit_intent(
        attempt, request_digest="sha256:a", actor=ACTOR
    )

    again, same = repo.create_attempt_with_submit_intent(
        attempt, request_digest="sha256:a", actor=ACTOR, operation_id=operation.id
    )
    assert (again.id, same.id) == (attempt.id, operation.id)
    assert _status(repo, node, PARALLEL).reserved == 1


# ---- failures: counted as attempts end ---------------------------------------------------


@pytest.mark.parametrize(
    ("status", "failures"),
    [
        (RunAttemptStatus.FAILED, 1),
        (RunAttemptStatus.PREEMPTED, 0),
        (RunAttemptStatus.CANCELLED, 0),
        (RunAttemptStatus.SUCCEEDED, 0),
    ],
    ids=["failed", "preempted", "cancelled", "succeeded"],
)
def test_only_a_failure_is_a_failure(
    repo: ControlPlaneRepository, status: RunAttemptStatus, failures: int
) -> None:
    node = _node(repo, max_failures=5)
    run = repo.create_run(make_run(node), actor=ACTOR)
    _end(repo, _submitted(repo, run), status)

    assert _status(repo, node, FAILURES).consumed == failures


def test_an_evaluation_failure_counts(repo: ControlPlaneRepository) -> None:
    node = _node(repo, max_failures=5)
    evaluation = _evaluation_run(node)
    _begin(repo, node, evaluation)
    attempt, _ = _evaluation_attempt(repo, evaluation)
    attempt = _evaluation_running(repo, attempt)
    repo.transition_evaluation_attempt(
        attempt.id,
        expected_revision=attempt.revision,
        new_status=EvaluationAttemptStatus.FAILED,
        actor=ACTOR,
    )

    assert _status(repo, node, FAILURES).consumed == 1


def test_used_up_failures_refuse_the_next_run_and_the_next_evaluation(
    repo: ControlPlaneRepository,
) -> None:
    node = _node(repo, max_failures=1)
    run = repo.create_run(make_run(node), actor=ACTOR)
    _end(repo, _submitted(repo, run), RunAttemptStatus.FAILED)

    with pytest.raises(BudgetExhaustedError, match="failures: 1 consumed of 1"):
        repo.create_run(make_run(node), actor=ACTOR)
    with pytest.raises(BudgetExhaustedError, match="failures"):
        _begin(repo, node, _evaluation_run(node))
    assert repo.aggregates.load_node(str(node.id)).status is ExperimentNodeStatus.ACTIVE


# ---- overrun, and exhaustion once nothing runs ----------------------------------------


def test_passing_a_limit_is_an_overrun_recorded_once_and_nothing_is_stopped(
    repo: ControlPlaneRepository,
) -> None:
    node = _node(repo, max_failures=1)
    runs = [repo.create_run(make_run(node), actor=ACTOR) for _ in range(3)]
    attempts = [_submitted(repo, run) for run in runs]  # all authorized before any failed

    _end(repo, attempts[0], RunAttemptStatus.FAILED)  # reaches the limit
    _end(repo, attempts[1], RunAttemptStatus.FAILED)  # passes it
    _end(repo, attempts[2], RunAttemptStatus.FAILED)  # still past it

    kinds = [e.event_type for e in repo.events.events_for_experiment(str(node.experiment_id))]
    assert kinds.count("BudgetOverrun") == 1
    status = _status(repo, node, FAILURES)
    assert (status.consumed, status.overrun) == (3, True)


def test_the_experiment_is_exhausted_only_once_nothing_is_running(
    repo: ControlPlaneRepository,
) -> None:
    node = _node(repo, max_runs=1)
    run = repo.create_run(make_run(node), actor=ACTOR)
    attempt = _submitted(repo, run)

    assert not repo.exhaust_budget(node.experiment_id, reasons=("runs",), actor=ACTOR)
    experiment = repo.aggregates.load_experiment(str(node.experiment_id))
    assert experiment.status is ExperimentStatus.ACTIVE, "a worker is still alive"

    _end(repo, attempt, RunAttemptStatus.SUCCEEDED)
    assert repo.exhaust_budget(node.experiment_id, reasons=("runs",), actor=ACTOR)
    experiment = repo.aggregates.load_experiment(str(node.experiment_id))
    assert experiment.status is ExperimentStatus.BUDGET_EXHAUSTED
    events = repo.events.events_for_experiment(str(node.experiment_id))
    (exhausted,) = [e for e in events if e.event_type == "BudgetExhausted"]
    assert tuple(exhausted.payload["reasons"]) == ("runs",)


# ---- one commit, append-only, idempotent -------------------------------------------------


def test_an_attempt_ends_with_its_costs_or_not_at_all(
    repo: ControlPlaneRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = _node(repo, max_failures=2, max_parallel_runs=1)
    run = repo.create_run(make_run(node), actor=ACTOR)
    attempt = _submitted(repo, run)
    for step in (RunAttemptStatus.QUEUED, RunAttemptStatus.STARTING, RunAttemptStatus.RUNNING):
        attempt = repo.transition_attempt(
            attempt.id, expected_revision=attempt.revision, new_status=step, actor=ACTOR
        )
    before = _entries(repo, node)

    real = repo.budget._append

    def fail_on_release(entry: BudgetLedgerEntry) -> BudgetLedgerEntry:
        if entry.kind is LedgerEntryKind.RELEASE:
            raise RuntimeError("the disk filled up")
        return real(entry)

    monkeypatch.setattr(repo.budget, "_append", fail_on_release)
    with pytest.raises(RuntimeError, match="disk"):
        repo.transition_attempt(
            attempt.id,
            expected_revision=attempt.revision,
            new_status=RunAttemptStatus.FAILED,
            actor=ACTOR,
        )

    assert repo.aggregates.load_attempt(str(attempt.id)).status is RunAttemptStatus.RUNNING
    assert _entries(repo, node) == before, "no failure without the slot's release"


def test_settling_again_writes_nothing(repo: ControlPlaneRepository) -> None:
    node = _node(repo, max_runs=2, max_failures=2)
    run = repo.create_run(make_run(node), actor=ACTOR)
    _end(repo, _submitted(repo, run), RunAttemptStatus.FAILED)
    _finish(repo, run, RunStatus.FAILED)
    written = _entries(repo, node)

    assert repo.settle_budget(node.experiment_id, actor=ACTOR) == 0
    assert _entries(repo, node) == written


def test_a_record_the_ledger_missed_is_settled_by_the_safety_net(
    repo: ControlPlaneRepository, connection: sqlite3.Connection
) -> None:
    """A run and attempt written around the repository -- as before the ledger existed."""
    node = _node(repo, max_failures=2)
    run = make_run(node)
    attempt = make_attempt(run)
    with write_transaction(connection):
        repo.aggregates._insert_run(run)
        repo.aggregates._insert_attempt(attempt)
        failed = attempt
        for step in (
            RunAttemptStatus.QUEUED,
            RunAttemptStatus.STARTING,
            RunAttemptStatus.RUNNING,
            RunAttemptStatus.FAILED,
        ):
            failed = failed.with_status(step)
        repo.aggregates._update_attempt(
            type(failed).model_validate({**failed.model_dump(mode="python"), "revision": 1})
        )
    assert _entries(repo, node) == []

    assert repo.settle_budget(node.experiment_id, actor=ACTOR) == 1
    assert _status(repo, node, FAILURES).consumed == 1
    assert repo.settle_budget(node.experiment_id, actor=ACTOR) == 0


def test_the_database_keeps_the_ledger_append_only(
    repo: ControlPlaneRepository, connection: sqlite3.Connection
) -> None:
    node = _node(repo, max_runs=1)
    repo.create_run(make_run(node), actor=ACTOR)

    for statement in ("UPDATE budget_ledger SET amount = '2'", "DELETE FROM budget_ledger"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with write_transaction(connection):
                connection.execute(statement)
    for dimension, amount in (("runs", "0"), ("wall_time_seconds", "1")):
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            with write_transaction(connection):
                connection.execute(
                    "INSERT INTO budget_ledger VALUES "
                    "('x', ?, ?, 'consume', ?, 'run', 'run_x', '2026-01-01')",
                    (str(node.experiment_id), dimension, amount),
                )


def test_the_same_entry_with_another_amount_is_a_conflict(repo: ControlPlaneRepository) -> None:
    node = _node(repo, max_runs=5)
    run = repo.create_run(make_run(node), actor=ACTOR)
    (reserved,) = repo.budget.entries(str(node.experiment_id))

    with pytest.raises(LedgerConflictError, match="would rewrite what happened"):
        with write_transaction(repo._connection):
            repo.budget._append(
                reserved.model_copy(update={"id": "ledger_other", "amount": Decimal(2)})
            )
    assert reserved.subject_kind is BudgetSubjectKind.RUN and reserved.subject_id == str(run.id)
