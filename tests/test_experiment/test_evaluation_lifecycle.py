"""A trained model is evaluated, durably, through the public API (PR-013).

```text
host.submit(ExperimentSpec(..., evaluation=EvaluationSpec(...)))
   training run SUCCEEDED
      ↓
   node ACTIVE → EVALUATING (cycle 1), EvaluationRun on the trained model
      ↓
   EvaluationAttempt + INTENDED → submit_or_get → scripted worker (real process)
      ↓
   EvaluationCompleted(metrics) + runtime "succeeded"
      ↓
   result + attempt + run SUCCEEDED, one commit
      ↓
   node EVALUATING → DECIDING;  wait() → next_stage="decision"
```

Crashing the controller mid-evaluation is in ``test_evaluation_restart.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation_fixtures import EVALUATORS, evaluation
from tests.test_experiment.test_restart_reconciliation import _spec
from xaytune.core.domain.evaluation import EvaluatorDeterminism, EvaluatorSpec
from xaytune.core.state.status import (
    EvaluationAttemptStatus,
    EvaluationRunStatus,
    ExperimentNodeStatus,
    ExperimentStatus,
    RunStatus,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _evaluated(tmp_path: Path, **config: object) -> Any:
    return _spec(tmp_path).model_copy(update={"evaluation": evaluation(**config)})


def _drive(tmp_path: Path, spec: Any, **host_options: Any) -> tuple[Any, Any, list]:
    from xaytune.experiment import EmbeddedControllerHost

    host_options.setdefault("evaluators", EVALUATORS)

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db", **host_options)
        try:
            handle = await host.submit(spec)
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            events = host.repository.events.events_for_experiment(str(handle.experiment_id))
            return (
                result,
                host.repository.aggregates.load_experiment(str(handle.experiment_id)),
                events,
            )
        finally:
            await host.close()

    return asyncio.run(scenario())


def _stalls(events: list) -> list:
    return [event for event in events if event.event_type == "EvaluationStalled"]


# ---- the lifecycle ------------------------------------------------------------------


def test_a_trained_model_is_evaluated_and_the_node_moves_to_deciding(tmp_path: Path) -> None:
    result, experiment, events = _drive(tmp_path, _evaluated(tmp_path, value=0.8))

    assert result.quiescent
    assert result.next_stage == "decision"
    assert result.status is ExperimentStatus.ACTIVE, "deciding is PR-015's; nothing decides yet"
    (node,) = result.nodes
    assert node.status is ExperimentNodeStatus.DECIDING
    (run,) = node.runs
    assert run.status is RunStatus.SUCCEEDED
    (model,) = [artifact for artifact in run.artifacts if artifact.kind == "model"]

    (evaluated,) = node.evaluations
    assert evaluated.evaluation_cycle == 1
    assert evaluated.status is EvaluationRunStatus.SUCCEEDED
    assert evaluated.attempt_status is EvaluationAttemptStatus.SUCCEEDED
    measured = evaluated.result
    assert measured is not None
    assert measured.evaluation_run_id == evaluated.evaluation_run_id
    assert measured.subject == model, "the trained model is what was measured"
    (accuracy,) = measured.metrics
    assert (accuracy.name, accuracy.value, accuracy.seed) == ("accuracy", 0.8, 7)
    (report,) = measured.artifacts
    assert report.kind == "evaluation_report"
    assert report.producer_evaluation_id == measured.id

    assert "EvaluationResultRecorded" in [event.event_type for event in events]
    assert _stalls(events) == []


def test_the_result_is_recorded_with_the_cursor_at_its_completion(tmp_path: Path) -> None:
    """The completion's position commits with the result, so a crash before it replays it.

    WorkerReady, EvaluationStarted, MetricObserved, EvaluationCompleted: the
    completion is sequence 3, and nothing after it has an effect to record.
    """
    from xaytune.experiment import EmbeddedControllerHost

    result, _, _ = _drive(tmp_path, _evaluated(tmp_path))
    (evaluated,) = result.nodes[0].evaluations

    async def cursor() -> tuple[int, int]:
        host = EmbeddedControllerHost(tmp_path / "state.db", evaluators=EVALUATORS)
        try:
            (attempt,) = host.repository.aggregates.evaluation_attempts_for_run(
                str(evaluated.evaluation_run_id)
            )
            return host.repository.aggregates.telemetry_position(
                str(attempt.id), kind="evaluation-attempt"
            )
        finally:
            await host.close()

    assert asyncio.run(cursor()) == (0, 3)


def test_without_an_evaluation_the_trained_node_waits_for_one(tmp_path: Path) -> None:
    result, experiment, _ = _drive(tmp_path, _spec(tmp_path))

    assert experiment.evaluation is None
    assert result.next_stage == "evaluation"
    (node,) = result.nodes
    assert node.status is ExperimentNodeStatus.ACTIVE
    assert node.evaluations == ()


def test_the_host_records_the_evaluator_it_resolved(tmp_path: Path) -> None:
    """ADR-016: the record says which implementation measured, not only its name."""
    _, experiment, _ = _drive(tmp_path, _evaluated(tmp_path))

    (bound,) = experiment.evaluation.evaluators
    assert bound.version == "1.0.0"
    assert bound.determinism is EvaluatorDeterminism.SEEDED


def test_the_evaluation_is_not_part_of_the_candidate(tmp_path: Path) -> None:
    """Changing how a model is evaluated never means retraining it."""
    with_evaluation, _, _ = _drive(tmp_path / "a", _evaluated(tmp_path / "a"))
    without, _, _ = _drive(tmp_path / "b", _spec(tmp_path / "a"))

    from xaytune.experiment import EmbeddedControllerHost

    async def fingerprints() -> list[str]:
        found = []
        for root, result in ((tmp_path / "a", with_evaluation), (tmp_path / "b", without)):
            host = EmbeddedControllerHost(root / "state.db", evaluators=EVALUATORS)
            try:
                (node,) = host.repository.aggregates.nodes_for_experiment(str(result.experiment_id))
                found.append(node.candidate_fingerprint)
            finally:
                await host.close()
        return found

    first, second = asyncio.run(fingerprints())
    assert first == second


# ---- an evaluation that did not produce a result -------------------------------------


def test_exit_zero_without_a_result_is_not_a_successful_evaluation(tmp_path: Path) -> None:
    result, _, events = _drive(tmp_path, _evaluated(tmp_path, mode="no-result"))

    (node,) = result.nodes
    (evaluated,) = node.evaluations
    assert evaluated.status is EvaluationRunStatus.FAILED
    assert evaluated.result is None
    assert node.status is ExperimentNodeStatus.EVALUATING, "stalled, visibly, not advanced"
    assert len(_stalls(events)) == 1
    assert result.next_stage == "failure-handling"
    assert result.quiescent


def test_a_completion_is_not_success_until_the_workload_succeeds(tmp_path: Path) -> None:
    """A result reported by a workload that then failed is not a result."""
    result, _, events = _drive(tmp_path, _evaluated(tmp_path, mode="complete-then-fail"))

    (node,) = result.nodes
    (evaluated,) = node.evaluations
    assert evaluated.status is EvaluationRunStatus.FAILED
    assert evaluated.result is None
    assert "EvaluationResultRecorded" not in [event.event_type for event in events]


def test_a_failed_evaluation_fails_the_evaluation_not_the_training(tmp_path: Path) -> None:
    result, _, events = _drive(tmp_path, _evaluated(tmp_path, mode="fail"))

    (node,) = result.nodes
    (run,) = node.runs
    assert run.status is RunStatus.SUCCEEDED
    (evaluated,) = node.evaluations
    assert evaluated.status is EvaluationRunStatus.FAILED
    assert len(_stalls(events)) == 1


# ---- refused at submission ---------------------------------------------------------


def test_an_unknown_evaluator_is_refused_before_anything_is_recorded(tmp_path: Path) -> None:
    from xaytune.experiment import EmbeddedControllerHost, UnknownImplementationError

    async def scenario() -> tuple:
        host = EmbeddedControllerHost(tmp_path / "state.db", evaluators={})
        try:
            with pytest.raises(UnknownImplementationError, match="scripted"):
                await host.submit(_evaluated(tmp_path))
            return tuple(host._connection.execute("SELECT id FROM experiments").fetchall())
        finally:
            await host.close()

    assert asyncio.run(scenario()) == ()


@pytest.mark.parametrize(
    "evaluators",
    [
        (EvaluatorSpec(name="scripted", version="1.0.0"),),
        (EvaluatorSpec(name="scripted", determinism=EvaluatorDeterminism.DETERMINISTIC),),
        (EvaluatorSpec(name="scripted"), EvaluatorSpec(name="other")),
    ],
    ids=["version-supplied", "determinism-supplied", "two-evaluators"],
)
def test_the_caller_names_one_evaluator_and_the_host_binds_it(
    tmp_path: Path, evaluators: tuple
) -> None:
    from pydantic import ValidationError

    from xaytune.core.domain.evaluation import EvaluationSpec

    with pytest.raises(ValidationError):
        _spec(tmp_path).model_validate(
            {
                **_spec(tmp_path).model_dump(),
                "evaluation": EvaluationSpec(evaluators=evaluators).model_dump(),
            }
        )


# ---- cancelling while it evaluates -------------------------------------------------


def test_cancelling_during_evaluation_leaves_no_evaluator_running(tmp_path: Path) -> None:
    """ADR-015 AC-7, through the handle: intent, effect, and nothing left executing."""
    from xaytune.experiment import EmbeddedControllerHost

    spec = _evaluated(tmp_path, hold=str(tmp_path / "never-released"))

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db", evaluators=EVALUATORS)
        try:
            handle = await host.submit(spec)
            repo = host.repository
            attempt = await _evaluation_running(repo, str(handle.experiment_id))
            await handle.cancel("changed our mind")
            result = await asyncio.wait_for(handle.wait(), timeout=60)
            (operation,) = [
                op
                for op in repo.operations.for_target("evaluation-attempt", str(attempt.id))
                if op.type == "submit"
            ]
            runtime = host._recorded_runtime(
                repo.aggregates.load_experiment(str(handle.experiment_id))
            )
            status = await runtime.get_status(operation.runtime_ref)
            events = repo.events.events_for_experiment(str(handle.experiment_id))
            return result, repo.aggregates.load_evaluation_attempt(str(attempt.id)), status, events
        finally:
            await host.close()

    result, attempt, status, events = asyncio.run(scenario())

    assert result.status is ExperimentStatus.CANCELLED
    assert attempt.status is EvaluationAttemptStatus.CANCELLED
    (node,) = result.nodes
    assert node.status is ExperimentNodeStatus.CANCELLED
    (evaluated,) = node.evaluations
    assert evaluated.status is EvaluationRunStatus.CANCELLED
    assert status.state == "cancelled", "the evaluator process is stopped, not abandoned"
    assert _stalls(events) == [], "a cancellation is not an evaluation stall"


async def _evaluation_running(repo: Any, experiment_id: str) -> Any:
    """Wait until the experiment's evaluation attempt reports RUNNING."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 120
    while loop.time() < deadline:
        for node in repo.aggregates.nodes_for_experiment(experiment_id):
            for run in repo.aggregates.evaluation_runs_for_node(str(node.id)):
                for attempt in repo.aggregates.evaluation_attempts_for_run(str(run.id)):
                    if attempt.status is EvaluationAttemptStatus.RUNNING:
                        return attempt
        await asyncio.sleep(0.05)
    raise AssertionError("the evaluation never started running")
