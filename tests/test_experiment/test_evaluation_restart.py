"""A controller killed mid-evaluation neither orphans nor duplicates it (PR-013).

```text
submit ─► training SUCCEEDED ─► evaluation attempt RUNNING ─► SIGKILL
                                        │   the evaluator keeps running
                                        ▼
new EmbeddedControllerHost ─► attach(experiment)
   ├── submit operation: CONFIRMED → adopt       INTENDED/SENT → lookup
   ├── watch() from the evaluation attempt's durable cursor
   ├── EvaluationCompleted(metrics) + runtime "succeeded"
   └── the same EvaluationAttempt and EvaluationRun settle; node → DECIDING
```

The same reconciliation as training, not a second one: these tests are the
training restart tests' evaluation counterparts, and they count
``submit_or_get`` calls for the same reason -- LocalRuntime's get-or-create
would hide a wrongful re-issue behind "still one workload".
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation_fixtures import EVALUATORS, RenumberedEvaluator, evaluation
from tests.test_experiment.test_restart_reconciliation import (
    _CountingSubmissions,
    _crash,
    _spec,
)
from xaytune.core.state.status import (
    EvaluationAttemptStatus,
    EvaluationRunStatus,
    ExperimentNodeStatus,
    ExperimentStatus,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _evaluated(tmp_path: Path, *, hold: bool) -> Any:
    config: dict[str, object] = {"value": 0.8}
    if hold:
        config["hold"] = str(tmp_path / "release")
    return _spec(tmp_path).model_copy(update={"evaluation": evaluation(**config)})


def _release(tmp_path: Path) -> None:
    (tmp_path / "release").touch()


def _evaluator_workloads(tmp_path: Path) -> list[str]:
    """Every evaluator workload the runtime ever started for this test."""
    registry = tmp_path / "runtime" / "registry.db"
    if not registry.exists():
        return []
    connection = sqlite3.connect(registry)
    try:
        rows = connection.execute(
            "SELECT external_id FROM workloads WHERE target_kind = 'evaluation-attempt'"
        ).fetchall()
    finally:
        connection.close()
    return [row[0] for row in rows]


def _counting_runtimes() -> dict[str, Any]:
    from xaytune.experiment.host import _local_runtime

    _CountingSubmissions.issued = []
    return {"local": lambda config: _CountingSubmissions(_local_runtime(config))}


def _evaluation_record(repo: Any, experiment_id: str) -> tuple[Any, tuple, tuple]:
    (node,) = repo.aggregates.nodes_for_experiment(experiment_id)
    (run,) = repo.aggregates.evaluation_runs_for_node(str(node.id))
    attempts = repo.aggregates.evaluation_attempts_for_run(str(run.id))
    operations = tuple(
        operation
        for attempt in attempts
        for operation in repo.operations.for_target("evaluation-attempt", str(attempt.id))
    )
    return run, attempts, operations


def _adopt(tmp_path: Path, experiment_id: str, *, release: bool = True, **host_options: Any):
    """Attach a new host, let the evaluation finish, and return what it recorded."""
    from xaytune.experiment import EmbeddedControllerHost

    host_options.setdefault("runtimes", _counting_runtimes())
    host_options.setdefault("evaluators", EVALUATORS)

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db", **host_options)
        try:
            handle = await host.attach(experiment_id)
            if release:
                _release(tmp_path)
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            return (result, *_evaluation_record(host.repository, experiment_id))
        finally:
            await host.close()

    return asyncio.run(scenario())


def _decided_once(result: Any, run: Any, attempts: tuple) -> None:
    (node,) = result.nodes
    assert node.status is ExperimentNodeStatus.DECIDING
    assert run.status is EvaluationRunStatus.SUCCEEDED
    (attempt,) = attempts
    assert attempt.status is EvaluationAttemptStatus.SUCCEEDED
    assert attempt.attempt_number == 1, "the same attempt, not a new one"
    (evaluated,) = node.evaluations
    assert evaluated.result is not None
    assert evaluated.result.evaluation_run_id == run.id


# ---- the acceptance case -------------------------------------------------------------


def test_a_restarted_host_adopts_the_live_evaluation(tmp_path: Path) -> None:
    experiment_id = _crash(tmp_path, "evaluating", _evaluated(tmp_path, hold=True))
    assert len(_evaluator_workloads(tmp_path)) == 1, "the evaluator outlived its controller"

    result, run, attempts, operations = _adopt(tmp_path, experiment_id)

    _decided_once(result, run, attempts)
    assert len(_evaluator_workloads(tmp_path)) == 1, "adopted, not duplicated"
    assert _CountingSubmissions.issued == [], "a confirmed evaluation is adopted, never re-issued"
    (submit,) = operations
    assert submit.state == "confirmed"


def test_adoption_resumes_the_evaluation_from_its_durable_cursor(tmp_path: Path) -> None:
    """The dead controller recorded STARTING and RUNNING; the next one asks for what followed."""
    from xaytune.experiment.host import _local_runtime

    asked: list[Any] = []

    class Recording:
        def __init__(self, runtime: Any) -> None:
            self._runtime = runtime

        def __getattr__(self, name: str) -> Any:
            return getattr(self._runtime, name)

        def watch(self, reference: Any, cursor: Any = None) -> Any:
            asked.append(cursor)
            return self._runtime.watch(reference, cursor)

    experiment_id = _crash(tmp_path, "evaluating", _evaluated(tmp_path, hold=True))
    result, run, attempts, _ = _adopt(
        tmp_path, experiment_id, runtimes={"local": lambda c: Recording(_local_runtime(c))}
    )

    _decided_once(result, run, attempts)
    (cursor,) = asked
    assert cursor is not None and cursor.sequence >= 0, "resumed, not replayed from the start"


def test_restarting_twice_still_leaves_one_evaluator(tmp_path: Path) -> None:
    from xaytune.experiment import EmbeddedControllerHost

    experiment_id = _crash(tmp_path, "evaluating", _evaluated(tmp_path, hold=True))

    async def attach_and_leave() -> None:
        host = EmbeddedControllerHost(
            tmp_path / "state.db", runtimes=_counting_runtimes(), evaluators=EVALUATORS
        )
        try:
            await host.attach(experiment_id)
        finally:
            await host.close()

    asyncio.run(attach_and_leave())
    assert _CountingSubmissions.issued == []

    result, run, attempts, _ = _adopt(tmp_path, experiment_id)

    _decided_once(result, run, attempts)
    assert len(_evaluator_workloads(tmp_path)) == 1
    assert _CountingSubmissions.issued == []


def test_a_lost_evaluation_response_is_found_by_lookup_not_reissued(tmp_path: Path) -> None:
    experiment_id = _crash(tmp_path, "eval-lost-response", _evaluated(tmp_path, hold=False))

    result, run, attempts, operations = _adopt(tmp_path, experiment_id, release=False)

    _decided_once(result, run, attempts)
    (submit,) = operations
    assert submit.state == "confirmed"
    assert _CountingSubmissions.issued == []
    assert len(_evaluator_workloads(tmp_path)) == 1


def test_an_evaluation_the_runtime_never_received_is_issued_once_under_its_own_identity(
    tmp_path: Path,
) -> None:
    experiment_id = _crash(tmp_path, "eval-never-sent", _evaluated(tmp_path, hold=False))
    assert _evaluator_workloads(tmp_path) == []

    result, run, attempts, operations = _adopt(tmp_path, experiment_id, release=False)

    _decided_once(result, run, attempts)
    (submit,) = operations
    assert _CountingSubmissions.issued == [submit.id], "exactly once, under the recorded id"
    assert len(_evaluator_workloads(tmp_path)) == 1


def test_an_uncertain_evaluation_absence_escalates(tmp_path: Path) -> None:
    """ADR-013 AC-5 for evaluation: "not found" may mean "finished and forgotten"."""
    from xaytune.experiment import EmbeddedControllerHost, ReconciliationEscalatedError
    from xaytune.experiment.host import _local_runtime

    class Forgetful(_CountingSubmissions):
        def capabilities(self) -> Any:
            document = self._runtime.capabilities()
            resilience = document.resilience.model_copy(
                update={"reports_completed_operations": False}
            )
            return document.model_copy(update={"resilience": resilience})

        async def lookup_operation(self, _operation_id: Any) -> None:
            return None

    experiment_id = _crash(tmp_path, "eval-never-sent", _evaluated(tmp_path, hold=False))
    _CountingSubmissions.issued = []

    async def scenario():
        host = EmbeddedControllerHost(
            tmp_path / "state.db",
            runtimes={"local": lambda c: Forgetful(_local_runtime(c))},
            evaluators=EVALUATORS,
        )
        try:
            handle = await host.attach(experiment_id)
            with pytest.raises(ReconciliationEscalatedError, match="completed operations"):
                await handle.wait()
            return host.repository.operations.unresolved()
        finally:
            await host.close()

    (unresolved,) = asyncio.run(scenario())
    assert unresolved.target.kind == "evaluation-attempt"
    assert unresolved.state == "intended", "left exactly as recorded"
    assert _CountingSubmissions.issued == []
    assert _evaluator_workloads(tmp_path) == []


# ---- a completion survives a lost stream and a dead controller ------------------------


def test_a_completion_survives_a_lost_stream_and_a_dead_controller(tmp_path: Path) -> None:
    """ADR-014 §1a meets ADR-015: the result reported before the stream died is kept.

    The completion arrives; the stream then ends while the evaluator still
    runs, so the attempt's telemetry moves to a new generation; the
    controller dies before the evaluator exits. The next host must record
    the same result -- not call the evaluation failed for lacking a
    completion that is only in a generation it no longer reads.
    """
    spec = _spec(tmp_path).model_copy(
        update={
            "evaluation": evaluation(value=0.8, hold_after_completion=str(tmp_path / "release"))
        }
    )
    experiment_id = _crash(tmp_path, "eval-stream-lost", spec)

    result, run, attempts, operations = _adopt(tmp_path, experiment_id)

    _decided_once(result, run, attempts)
    (evaluated,) = result.nodes[0].evaluations
    assert [metric.value for metric in evaluated.result.metrics] == [0.8]
    assert _CountingSubmissions.issued == [], "recovered, not re-run"
    assert len(_evaluator_workloads(tmp_path)) == 1
    (submit,) = operations
    assert submit.state == "confirmed"


def test_a_preempted_attempt_found_after_a_crash_fails_its_run(tmp_path: Path) -> None:
    """A run left ACTIVE over a PREEMPTED attempt has nothing executing it: settle it."""
    from xaytune.core.refs import Actor
    from xaytune.storage import connect
    from xaytune.storage.control_plane import ControlPlaneRepository

    experiment_id = _crash(tmp_path, "evaluating", _evaluated(tmp_path, hold=True))
    repo = ControlPlaneRepository(connect(tmp_path / "state.db"))
    _, (attempt,), _ = _evaluation_record(repo, experiment_id)
    repo.transition_evaluation_attempt(
        attempt.id,
        expected_revision=attempt.revision,
        new_status=EvaluationAttemptStatus.PREEMPTED,
        actor=Actor(type="system", id="test"),
    )

    try:
        result, run, attempts, _ = _adopt(tmp_path, experiment_id, release=False)
    finally:
        _release(tmp_path)

    assert run.status is EvaluationRunStatus.FAILED
    (preempted,) = attempts
    assert preempted.status is EvaluationAttemptStatus.PREEMPTED, "never recovered"
    (node,) = result.nodes
    assert node.status is ExperimentNodeStatus.EVALUATING
    assert result.next_stage == "failure-handling"
    assert _CountingSubmissions.issued == []


# ---- cancellation survives the controller ---------------------------------------------


def test_a_cancellation_recorded_mid_evaluation_is_carried_out_after_the_crash(
    tmp_path: Path,
) -> None:
    """ADR-015 AC-7: intent in the record, the effect issued by the next host."""
    experiment_id = _crash(tmp_path, "eval-cancel-intended", _evaluated(tmp_path, hold=True))

    result, run, attempts, operations = _adopt(tmp_path, experiment_id, release=False)

    assert result.status is ExperimentStatus.CANCELLED
    assert run.status is EvaluationRunStatus.CANCELLED
    (attempt,) = attempts
    assert attempt.status is EvaluationAttemptStatus.CANCELLED
    (cancel,) = [op for op in operations if op.type == "cancel"]
    assert cancel.state == "confirmed"
    assert _CountingSubmissions.issued == [], "cancelling adopts; it never re-submits"
    assert len(_evaluator_workloads(tmp_path)) == 1


# ---- the evaluator is a dependency of re-issue only ------------------------------------


@pytest.mark.parametrize(
    "evaluators",
    [{}, {"scripted": RenumberedEvaluator}],
    ids=["evaluator-unavailable", "evaluator-renumbered"],
)
def test_a_live_evaluation_is_adopted_without_its_evaluator(
    tmp_path: Path, evaluators: dict
) -> None:
    """Rediscovering an evaluation that exists needs the runtime, not the evaluator."""
    experiment_id = _crash(tmp_path, "evaluating", _evaluated(tmp_path, hold=True))

    result, run, attempts, _ = _adopt(tmp_path, experiment_id, evaluators=evaluators)

    _decided_once(result, run, attempts)
    assert _CountingSubmissions.issued == []


@pytest.mark.parametrize(
    ("evaluators", "error"),
    [
        ({}, "UnknownImplementationError"),
        ({"scripted": RenumberedEvaluator}, "ImplementationMismatchError"),
    ],
    ids=["evaluator-unavailable", "evaluator-renumbered"],
)
def test_re_issuing_without_the_original_evaluator_fails_closed(
    tmp_path: Path, evaluators: dict, error: str
) -> None:
    """Rebuilding the request is where the evaluator matters, so that is where it is required."""
    import xaytune.experiment as experiment_module
    from xaytune.experiment import EmbeddedControllerHost

    experiment_id = _crash(tmp_path, "eval-never-sent", _evaluated(tmp_path, hold=False))

    async def scenario():
        host = EmbeddedControllerHost(
            tmp_path / "state.db", runtimes=_counting_runtimes(), evaluators=evaluators
        )
        try:
            with pytest.raises(getattr(experiment_module, error)):
                await host.attach(experiment_id)
            return host.repository.operations.unresolved()
        finally:
            await host.close()

    (unresolved,) = asyncio.run(scenario())
    assert unresolved.state == "intended"
    assert _CountingSubmissions.issued == []
    assert _evaluator_workloads(tmp_path) == []
