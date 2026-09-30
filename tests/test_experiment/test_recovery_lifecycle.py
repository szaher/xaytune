"""An infrastructure failure may leave its logical Run open for recovery."""

from __future__ import annotations

import asyncio

from xaytune.core.domain.budget import BudgetDimension
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.domain.run import RunAttempt
from xaytune.core.ids import RunAttemptId
from xaytune.core.refs import Actor
from xaytune.core.state.status import RunAttemptStatus, RunStatus
from xaytune.experiment import EmbeddedControllerHost

from .test_host_behaviour import _spec

ACTOR = Actor(type="system", id="recovery-lifecycle-test")


def test_failed_attempt_can_leave_run_active_and_release_capacity(tmp_path):
    state = tmp_path / "state.db"

    async def first():
        host = EmbeddedControllerHost(state)
        try:
            spec = _spec(
                tmp_path,
                budget=BudgetSpec(max_runs=1, max_parallel_runs=1, max_failures=2),
            )
            experiment = host._record_experiment(
                spec, host._compiler(spec.compiler.name), host._runtime(spec.runtime), None
            )
            node = host._record_node(experiment, spec)
            run = host._record_run(node, spec.seed)
            attempt = RunAttempt(id=RunAttemptId.generate(), run_id=run.id, attempt_number=1)
            host.repository.create_attempt_with_submit_intent(
                attempt, request_digest="first-attempt", actor=ACTOR
            )
            host._settle(attempt.id, run.id, RunAttemptStatus.FAILED, None)
            current = host.repository.aggregates.load_run(str(run.id))
            failed = host.repository.aggregates.load_attempt(str(attempt.id))
            budget = host.repository.budget_status(experiment.id)
            assert budget is not None
            assert current.status is RunStatus.ACTIVE
            assert current.revision == run.revision
            assert failed.status is RunAttemptStatus.FAILED
            assert budget.of(BudgetDimension.RUNS).reserved == 1
            assert budget.of(BudgetDimension.FAILURES).consumed == 1
            assert budget.of(BudgetDimension.PARALLEL_RUNS).outstanding == 0
            return experiment.id, run.id, attempt.id
        finally:
            await host.close()

    experiment_id, run_id, first_attempt_id = asyncio.run(first())

    async def restart():
        host = EmbeddedControllerHost(state)
        try:
            run = host.repository.aggregates.load_run(str(run_id))
            assert run.status is RunStatus.ACTIVE
            assert (
                host.repository.aggregates.load_attempt(str(first_attempt_id)).status
                is RunAttemptStatus.FAILED
            )
            successor = RunAttempt(id=RunAttemptId.generate(), run_id=run.id, attempt_number=2)
            host.repository.create_attempt_with_submit_intent(
                successor, request_digest="second-attempt", actor=ACTOR
            )
            assert host.repository.aggregates.load_run(str(run_id)).status is RunStatus.ACTIVE
            budget = host.repository.budget_status(experiment_id)
            assert budget is not None
            assert budget.of(BudgetDimension.RUNS).reserved == 1
            assert budget.of(BudgetDimension.PARALLEL_RUNS).outstanding == 1
            host._settle(successor.id, run.id, RunAttemptStatus.SUCCEEDED, RunStatus.SUCCEEDED)
            assert host.repository.aggregates.load_run(str(run_id)).status is RunStatus.SUCCEEDED
        finally:
            await host.close()

    asyncio.run(restart())


def test_run_fails_only_when_pending_recovery_is_definitively_refused(tmp_path):
    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            spec = _spec(tmp_path)
            experiment = host._record_experiment(
                spec, host._compiler(spec.compiler.name), host._runtime(spec.runtime), None
            )
            run = host._record_run(host._record_node(experiment, spec), spec.seed)
            attempt = RunAttempt(id=RunAttemptId.generate(), run_id=run.id, attempt_number=1)
            host.repository.create_attempt_with_submit_intent(
                attempt, request_digest="failed-attempt", actor=ACTOR
            )
            host._settle(attempt.id, run.id, RunAttemptStatus.FAILED, None)
            assert host.repository.aggregates.load_run(str(run.id)).status is RunStatus.ACTIVE
            host._settle(attempt.id, run.id, RunAttemptStatus.FAILED, RunStatus.FAILED)
            assert host.repository.aggregates.load_run(str(run.id)).status is RunStatus.FAILED
        finally:
            await host.close()

    asyncio.run(scenario())
