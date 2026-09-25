"""Train, then evaluate with the built-in evaluator: the path from PR-013 to a decision.

```text
ExperimentSpec(evaluation=native)
      ↓ real training, real model artifact
NativeEvaluator.prepare()  →  EvaluationExecutionSpec
      ↓ LocalRuntime
xaytune.workers.eval_native  →  EvaluationStarted … EvaluationCompleted(metrics)
      ↓
durable EvaluationResult   →   node DECIDING
```

No test fixture stands in for the evaluator here: the host resolves
``native`` from its defaults, as a user's would. The restart cases are the
PR-013 crash points, with the real evaluator's worker in place of the
scripted one.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation_fixtures import native_evaluation
from tests.test_experiment.test_evaluation_restart import _evaluator_workloads
from tests.test_experiment.test_restart_reconciliation import (
    _CountingSubmissions,
    _crash,
    _spec,
)
from xaytune.core.domain.evaluation import EvaluatorDeterminism
from xaytune.core.refs import Actor
from xaytune.core.state.status import (
    EvaluationAttemptStatus,
    EvaluationRunStatus,
    ExperimentNodeStatus,
    ExperimentStatus,
)
from xaytune.evaluation import UnsupportedEvaluationError
from xaytune.evaluation.native import NativeEvaluator


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _evaluated(tmp_path: Path, **config: Any):
    return _spec(tmp_path).model_copy(
        update={"evaluation": native_evaluation(tmp_path / "eval", **config)}
    )


def _run(tmp_path: Path, spec: Any, *, after_submit: Any = None, **host_options: Any):
    """Submit *spec* on a default host, wait, and return the result and the repository's view."""
    from xaytune.experiment import EmbeddedControllerHost

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db", **host_options)
        try:
            handle = await host.submit(spec)
            if after_submit is not None:
                after_submit()
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            return result, _record(host.repository, str(handle.experiment_id))
        finally:
            await host.close()

    return asyncio.run(scenario())


def _record(repo: Any, experiment_id: str) -> dict[str, Any]:
    (node,) = repo.aggregates.nodes_for_experiment(experiment_id)
    runs = repo.aggregates.evaluation_runs_for_node(str(node.id))
    return {
        "experiment": repo.aggregates.load_experiment(experiment_id),
        "node": node,
        "runs": runs,
        "attempts": [
            attempt
            for run in runs
            for attempt in repo.aggregates.evaluation_attempts_for_run(str(run.id))
        ],
        "results": [repo.aggregates.evaluation_result_for_run(str(run.id)) for run in runs],
        "events": repo.events.events_for_experiment(experiment_id),
    }


# ---- the acceptance case -------------------------------------------------------------


def test_a_trained_model_is_evaluated_by_the_built_in_evaluator_and_decided(
    tmp_path: Path,
) -> None:
    result, record = _run(tmp_path, _evaluated(tmp_path))

    assert result.status is ExperimentStatus.ACTIVE
    assert result.next_stage == "decision"
    (node,) = result.nodes
    assert node.status is ExperimentNodeStatus.DECIDING
    (training,) = node.runs
    (model,) = training.artifacts
    (evaluated,) = node.evaluations
    assert evaluated.status is EvaluationRunStatus.SUCCEEDED
    assert evaluated.attempt_status is EvaluationAttemptStatus.SUCCEEDED

    stored = evaluated.result
    assert stored is not None
    assert stored.subject == model, "it measured the model training produced"
    metrics = {metric.name: metric for metric in stored.metrics}
    assert set(metrics) == {"loss", "perplexity", "token_accuracy"}
    loss = metrics["loss"].value
    assert loss > 0 and metrics["perplexity"].value == pytest.approx(2.718281828459045**loss)
    assert 0.0 <= metrics["token_accuracy"].value <= 1.0

    (run,) = record["runs"]
    for metric in stored.metrics:
        assert (metric.evaluator_name, metric.evaluator_version) == ("native", "0.1.0")
        assert metric.seed == run.seed == 7, "the training run's seed, recorded"
        assert metric.sample_count == 3
        assert metric.dataset_ref == run.spec.dataset

    (report,) = stored.artifacts
    assert report.kind == "evaluation_report"
    assert report.producer_evaluation_id == stored.id
    written = json.loads(Path(report.uri).read_text(encoding="utf-8"))
    assert written["metrics"]["loss"] == pytest.approx(loss)
    assert written["data"]["content_digest"] == run.spec.dataset.content_digest
    assert written["tokens_scored"] == metrics["loss"].metadata["tokens_scored"] > 0


def test_the_record_names_the_evaluator_that_measured(tmp_path: Path) -> None:
    """Bound at submission: version and determinism are the evaluator's own declarations."""
    _, record = _run(tmp_path, _evaluated(tmp_path))
    bound = record["experiment"].evaluation.evaluator
    assert (bound.name, bound.version) == ("native", "0.1.0")
    assert bound.determinism is EvaluatorDeterminism.SEEDED


# ---- refusals -------------------------------------------------------------------------


def test_an_evaluation_it_cannot_run_is_refused_at_submission_before_anything_trains(
    tmp_path: Path,
) -> None:
    """Refused before hours of training, not after them -- and nothing is recorded."""
    from xaytune.experiment import EmbeddedControllerHost

    spec = _evaluated(tmp_path)
    unpinned = spec.evaluation.model_copy(
        update={"dataset": spec.evaluation.dataset.model_copy(update={"content_digest": None})}
    )

    async def scenario() -> None:
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            with pytest.raises(UnsupportedEvaluationError, match="content_digest"):
                await host.submit(spec.model_copy(update={"evaluation": unpinned}))
            (count,) = host._connection.execute("SELECT count(*) FROM experiments").fetchone()
            assert count == 0, "nothing was recorded"
        finally:
            await host.close()

    asyncio.run(scenario())
    workloads = tmp_path / "runtime" / "workloads"
    assert not workloads.exists() or not any(workloads.iterdir()), "no workload was started"


def test_data_changed_after_submission_fails_the_evaluation_instead_of_measuring_it(
    tmp_path: Path,
) -> None:
    spec = _evaluated(tmp_path)
    held_out = Path(spec.evaluation.dataset.uri)

    result, record = _run(
        tmp_path,
        spec,
        after_submit=lambda: held_out.write_text('{"text": "hello ."}\n', encoding="utf-8"),
    )

    (run,) = record["runs"]
    assert run.status is EvaluationRunStatus.FAILED
    assert record["results"] == [None]
    assert result.nodes[0].status is ExperimentNodeStatus.EVALUATING
    assert result.next_stage == "failure-handling"
    assert any(event.event_type == "EvaluationStalled" for event in record["events"])
    failed = [
        event
        for event in record["events"]
        if event.aggregate_type == "EvaluationAttempt" and event.payload.get("status") == "failed"
    ]
    assert failed, "the attempt's failure is recorded"


class _RefusesTheSubject(NativeEvaluator):
    """Accepts the spec, refuses every subject: a refusal only prepare() can make."""

    def prepare(self, subject: Any, spec: Any, context: Any) -> Any:
        raise UnsupportedEvaluationError(self.descriptor.name, ("this subject cannot be read",))


def test_a_subject_the_evaluator_refuses_fails_its_run_before_any_effect(tmp_path: Path) -> None:
    result, record = _run(tmp_path, _evaluated(tmp_path), evaluators={"native": _RefusesTheSubject})

    (run,) = record["runs"]
    assert run.status is EvaluationRunStatus.FAILED
    assert record["attempts"] == [], "nothing was issued, so there is no attempt"
    assert _evaluator_workloads(tmp_path) == []
    (failure,) = [
        event
        for event in record["events"]
        if event.aggregate_type == "EvaluationRun" and event.payload.get("status") == "failed"
    ]
    assert "this subject cannot be read" in failure.payload["reason"]
    assert result.nodes[0].status is ExperimentNodeStatus.EVALUATING
    assert result.next_stage == "failure-handling"
    assert any(event.event_type == "EvaluationStalled" for event in record["events"])


# ---- no reuse, yet ------------------------------------------------------------------


def test_a_second_evaluation_of_the_same_model_really_runs(tmp_path: Path) -> None:
    """Same artifact, same fingerprint, same seed: still a second workload.

    Reuse is ADR-015 AC-4, deliberately not implemented: a result that
    suppressed a requested evaluation would be a scientific-correctness bug,
    so until that policy exists no existing result may stand in for a run.
    The second cycle is started the way the host starts one -- the node back
    to ACTIVE, then ``_continue_to_evaluation`` -- because nothing public
    re-evaluates a node yet.
    """
    from xaytune.experiment import EmbeddedControllerHost

    async def scenario() -> dict[str, Any]:
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            handle = await host.submit(_evaluated(tmp_path))
            await asyncio.wait_for(handle.wait(), timeout=180)
            repo = host.repository
            (node,) = repo.aggregates.nodes_for_experiment(str(handle.experiment_id))
            repo.transition_node(
                node.id,
                expected_revision=node.revision,
                new_status=ExperimentNodeStatus.ACTIVE,
                actor=Actor(type="system", id="test"),
            )
            await host._continue_to_evaluation(handle.experiment_id, node.id)
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            assert result.nodes[0].status is ExperimentNodeStatus.DECIDING
            return _record(repo, str(handle.experiment_id))
        finally:
            await host.close()

    record = asyncio.run(scenario())

    first, second = record["runs"]
    assert (first.evaluation_cycle, second.evaluation_cycle) == (1, 2)
    assert first.subject == second.subject
    assert first.evaluation_fingerprint == second.evaluation_fingerprint
    assert (first.seed, first.replicate) == (second.seed, second.replicate)
    assert len(_evaluator_workloads(tmp_path)) == 2, "the second evaluation executed"
    assert len(record["attempts"]) == 2
    first_result, second_result = record["results"]
    assert first_result is not None and second_result is not None
    assert first_result.id != second_result.id
    assert first_result.artifacts[0].uri != second_result.artifacts[0].uri


# ---- restart safety, with the real evaluator ---------------------------------------


@pytest.mark.parametrize(
    "mode",
    ["evaluating", "eval-lost-response", "eval-never-sent", "eval-stream-lost"],
)
def test_a_controller_killed_mid_evaluation_is_succeeded_by_one_that_finishes_it_once(
    tmp_path: Path, mode: str
) -> None:
    from xaytune.experiment import EmbeddedControllerHost
    from xaytune.experiment.host import _local_runtime

    experiment_id = _crash(tmp_path, mode, _evaluated(tmp_path))
    _CountingSubmissions.issued = []

    async def adopt() -> tuple[Any, dict[str, Any]]:
        host = EmbeddedControllerHost(
            tmp_path / "state.db",
            runtimes={"local": lambda config: _CountingSubmissions(_local_runtime(config))},
        )
        try:
            handle = await host.attach(experiment_id)
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            return result, _record(host.repository, experiment_id)
        finally:
            await host.close()

    result, record = asyncio.run(adopt())

    assert result.nodes[0].status is ExperimentNodeStatus.DECIDING
    (run,) = record["runs"]
    assert run.status is EvaluationRunStatus.SUCCEEDED
    (attempt,) = record["attempts"]
    assert attempt.attempt_number == 1, "the same attempt, finished; not a new one"
    (stored,) = record["results"]
    assert stored is not None and {m.name for m in stored.metrics} == {
        "loss",
        "perplexity",
        "token_accuracy",
    }
    assert len(_evaluator_workloads(tmp_path)) == 1, "one evaluation workload, ever"
    reissued = len(_CountingSubmissions.issued)
    assert reissued == (1 if mode == "eval-never-sent" else 0)
