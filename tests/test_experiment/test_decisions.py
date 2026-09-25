"""The loop closes: train, evaluate, decide -- and the decision is durable (PR-015).

```text
ExperimentSpec(objective, evaluation)
   ↓ training → evaluation → node DECIDING
DecisionContext from the record → DecisionEngine.decide()
   ↓ one commit
Decision + node + experiment:
   STOP_SUCCEEDED  node COMPLETED  experiment SUCCEEDED, best_node_id = node
   STOP_FAILED     node REJECTED   experiment FAILED
   REJECT          node REJECTED   experiment ACTIVE, next_stage "planning"
```

A cycle the engine cannot decide stays ``DECIDING``, with a
``DecisionDeferred`` event saying why. A controller that dies before
deciding is succeeded by one that decides once; one that decided is
succeeded by one that does not decide again.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation_fixtures import EVALUATORS, evaluation, native_evaluation
from tests.test_experiment.test_restart_reconciliation import _crash, _spec
from xaytune.core.domain.decision import DecisionContext, DecisionOutcome
from xaytune.core.domain.objective import MetricConstraint, Objective, ObjectiveMetric
from xaytune.core.refs import Actor
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus
from xaytune.decision import ThresholdDecisionEngine, UndecidableError


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def _objective(
    name: str = "accuracy",
    direction: str = "maximize",
    target: float | None = 0.5,
    *constraints: MetricConstraint,
) -> Objective:
    return Objective(
        primary=ObjectiveMetric(name=name, direction=direction),  # type: ignore[arg-type]
        target=target,
        constraints=constraints,
    )


def _scripted(tmp_path: Path, objective: Objective, value: float = 0.8) -> Any:
    """Tiny training, then the scripted evaluator reporting accuracy=*value*."""
    return _spec(tmp_path).model_copy(
        update={"objective": objective, "evaluation": evaluation(value=value)}
    )


def _run(tmp_path: Path, spec: Any, **host_options: Any) -> tuple[Any, dict[str, Any]]:
    from xaytune.experiment import EmbeddedControllerHost

    host_options.setdefault("evaluators", EVALUATORS)

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db", **host_options)
        try:
            handle = await host.submit(spec)
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            return result, _record(host.repository, str(handle.experiment_id))
        finally:
            await host.close()

    return asyncio.run(scenario())


def _record(repo: Any, experiment_id: str) -> dict[str, Any]:
    (node,) = repo.aggregates.nodes_for_experiment(experiment_id)
    events = repo.events.events_for_experiment(experiment_id)
    return {
        "experiment": repo.aggregates.load_experiment(experiment_id),
        "node": node,
        "decisions": repo.aggregates.decisions_for_node(str(node.id)),
        "results": repo.aggregates.evaluation_results_for_node(str(node.id)),
        "events": [e.event_type for e in events],
        "deferred": [e for e in events if e.event_type == "DecisionDeferred"],
    }


# ---- the milestone ------------------------------------------------------------------


def test_train_evaluate_and_decide_end_to_end(tmp_path: Path) -> None:
    """Tiny training, the native evaluator, a target met: the experiment succeeds."""
    spec = _spec(tmp_path).model_copy(
        update={
            "objective": _objective("loss", "minimize", target=100.0),
            "evaluation": native_evaluation(tmp_path / "eval"),
        }
    )
    result, record = _run(tmp_path, spec, evaluators=None)

    assert result.status is ExperimentStatus.SUCCEEDED
    assert result.next_stage is None
    assert result.quiescent
    (node,) = result.nodes
    assert node.status is ExperimentNodeStatus.COMPLETED
    assert record["experiment"].best_node_id == node.node_id

    (decision,) = record["decisions"]
    assert decision.outcome is DecisionOutcome.STOP_SUCCEEDED
    assert record["node"].decision_ids == (decision.id,)
    (stored,) = record["results"]
    assert decision.evaluation_result_ids == (stored.id,)
    (evidence,) = decision.evidence
    loss = next(m for m in stored.metrics if m.name == "loss")
    assert (evidence.metric, evidence.value, evidence.operator) == ("loss", loss.value, "<=")
    assert evidence.evaluator_name == "native"
    assert (decision.engine_name, decision.engine_version) == ("threshold", "1.0.0")
    assert record["events"].count("DecisionRecorded") == 1


# ---- outcomes ---------------------------------------------------------------------


def test_a_missed_target_fails_the_experiment(tmp_path: Path) -> None:
    result, record = _run(tmp_path, _scripted(tmp_path, _objective(target=0.9), value=0.8))

    assert result.status is ExperimentStatus.FAILED
    assert result.next_stage is None
    assert result.nodes[0].status is ExperimentNodeStatus.REJECTED
    assert record["experiment"].best_node_id is None
    (decision,) = record["decisions"]
    assert decision.outcome is DecisionOutcome.STOP_FAILED
    assert "target not met" in decision.reason


def test_a_violated_constraint_rejects_the_candidate_and_leaves_the_experiment_open(
    tmp_path: Path,
) -> None:
    """REJECT judges the candidate; proposing another is a planner's, so planning is next."""
    objective = _objective(
        "accuracy", "maximize", 0.5, MetricConstraint(name="accuracy", operator=">=", value=0.95)
    )
    result, record = _run(tmp_path, _scripted(tmp_path, objective, value=0.8))

    assert result.status is ExperimentStatus.ACTIVE
    assert result.next_stage == "planning"
    assert result.quiescent
    assert result.nodes[0].status is ExperimentNodeStatus.REJECTED
    assert record["experiment"].best_node_id is None
    (decision,) = record["decisions"]
    assert decision.outcome is DecisionOutcome.REJECT
    assert "ExperimentStatusChanged" not in record["events"][-2:]


# ---- nothing is guessed -------------------------------------------------------------


@pytest.mark.parametrize(
    ("objective", "reason"),
    [
        (_objective("bleu"), "objective metric 'bleu'"),
        (
            _objective(
                "accuracy",
                "maximize",
                0.5,
                MetricConstraint(name="latency", operator="<", value=100),
            ),
            "constraint metric 'latency'",
        ),
        (_objective(target=None), "no target"),
    ],
    ids=["missing-objective-metric", "missing-constraint-metric", "no-target"],
)
def test_an_undecidable_cycle_waits_with_its_reasons(
    tmp_path: Path, objective: Objective, reason: str
) -> None:
    result, record = _run(tmp_path, _scripted(tmp_path, objective))

    assert result.status is ExperimentStatus.ACTIVE
    assert result.next_stage == "decision"
    assert result.nodes[0].status is ExperimentNodeStatus.DECIDING
    assert record["decisions"] == ()
    (deferred,) = record["deferred"]
    assert any(reason in r for r in deferred.payload["reasons"])
    assert deferred.payload["evaluation_cycle"] == 1


def test_only_the_current_cycles_results_reach_the_engine(tmp_path: Path) -> None:
    """ADR-015 §5: a second round is decided on the second round's results alone."""
    from xaytune.experiment import EmbeddedControllerHost

    seen: list[DecisionContext] = []

    class Recording:
        name, version = "recording", "1"

        def decide(self, context: DecisionContext) -> Any:
            seen.append(context)
            raise UndecidableError(self.name, ("recording only",))

    async def scenario() -> Any:
        host = EmbeddedControllerHost(
            tmp_path / "state.db", evaluators=EVALUATORS, decision_engine=Recording()
        )
        try:
            handle = await host.submit(_scripted(tmp_path, _objective()))
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
            await asyncio.wait_for(handle.wait(), timeout=180)
            return repo.aggregates.evaluation_results_for_node(str(node.id))
        finally:
            await host.close()

    first_result, second_result = asyncio.run(scenario())
    first, second = seen
    assert (first.evaluation_cycle, second.evaluation_cycle) == (1, 2)
    assert [r.id for r in first.results] == [first_result.id]
    assert [r.id for r in second.results] == [second_result.id], "cycle 1's result excluded"


# ---- restart ----------------------------------------------------------------------


def _attach(tmp_path: Path, experiment_id: str) -> tuple[Any, dict[str, Any]]:
    from xaytune.experiment import EmbeddedControllerHost

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db", evaluators=EVALUATORS)
        try:
            handle = await host.attach(experiment_id)
            result = await asyncio.wait_for(handle.wait(), timeout=180)
            return result, _record(host.repository, experiment_id)
        finally:
            await host.close()

    return asyncio.run(scenario())


def test_a_controller_that_died_deciding_is_succeeded_by_one_that_decides_once(
    tmp_path: Path,
) -> None:
    experiment_id = _crash(tmp_path, "deciding", _scripted(tmp_path, _objective(target=0.5)))

    result, record = _attach(tmp_path, experiment_id)

    assert result.status is ExperimentStatus.SUCCEEDED
    assert result.nodes[0].status is ExperimentNodeStatus.COMPLETED
    assert record["experiment"].best_node_id == result.nodes[0].node_id
    (decision,) = record["decisions"]
    assert decision.outcome is DecisionOutcome.STOP_SUCCEEDED
    assert record["events"].count("DecisionRecorded") == 1


def test_a_decided_experiment_is_not_decided_again_after_a_restart(tmp_path: Path) -> None:
    before, record = _run(tmp_path, _scripted(tmp_path, _objective(target=0.5)))
    (decision,) = record["decisions"]

    after, again = _attach(tmp_path, str(before.experiment_id))

    assert after.status is ExperimentStatus.SUCCEEDED
    assert again["decisions"] == (decision,)
    assert again["events"].count("DecisionRecorded") == 1
    assert again["events"] == record["events"], "attaching recorded nothing new"


def test_the_host_decides_with_the_engine_it_is_given(tmp_path: Path) -> None:
    """The default is the threshold engine; any DecisionEngine can stand in."""

    class Strict(ThresholdDecisionEngine):
        name = "strict"

    _, record = _run(
        tmp_path, _scripted(tmp_path, _objective(target=0.5)), decision_engine=Strict()
    )
    (decision,) = record["decisions"]
    assert decision.engine_name == "strict"


# ---- what next_stage says, pinned ----------------------------------------------------

_PATHS: dict[ExperimentNodeStatus, tuple[ExperimentNodeStatus, ...]] = {
    ExperimentNodeStatus.DECIDING: (
        ExperimentNodeStatus.PLANNED,
        ExperimentNodeStatus.READY,
        ExperimentNodeStatus.ACTIVE,
        ExperimentNodeStatus.EVALUATING,
        ExperimentNodeStatus.DECIDING,
    ),
    ExperimentNodeStatus.REJECTED: (
        ExperimentNodeStatus.PLANNED,
        ExperimentNodeStatus.READY,
        ExperimentNodeStatus.ACTIVE,
        ExperimentNodeStatus.EVALUATING,
        ExperimentNodeStatus.DECIDING,
        ExperimentNodeStatus.REJECTED,
    ),
    ExperimentNodeStatus.FAILED: (
        ExperimentNodeStatus.PLANNED,
        ExperimentNodeStatus.READY,
        ExperimentNodeStatus.ACTIVE,
        ExperimentNodeStatus.FAILED,
    ),
    ExperimentNodeStatus.CANCELLED: (
        ExperimentNodeStatus.PLANNED,
        ExperimentNodeStatus.READY,
        ExperimentNodeStatus.ACTIVE,
        ExperimentNodeStatus.CANCELLED,
    ),
}


def _next_stage(
    tmp_path: Path, nodes: tuple[ExperimentNodeStatus, ...], experiment: ExperimentStatus
) -> Any:
    """next_stage for an experiment whose record holds candidates in *nodes*."""
    from tests.test_storage.conftest import make_experiment, make_node
    from xaytune.experiment import EmbeddedControllerHost

    actor = Actor(type="system", id="test")

    async def scenario() -> Any:
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            repo = host.repository
            recorded = repo.create_experiment(make_experiment(), actor=actor)
            recorded = repo.transition_experiment(
                recorded.id,
                expected_revision=recorded.revision,
                new_status=ExperimentStatus.ACTIVE,
                actor=actor,
            )
            for index, status in enumerate(nodes):
                node = repo.create_node(make_node(recorded, fingerprint=str(index)), actor=actor)
                for step in _PATHS[status]:
                    node = repo.transition_node(
                        node.id, expected_revision=node.revision, new_status=step, actor=actor
                    )
            if experiment is not ExperimentStatus.ACTIVE:
                repo.transition_experiment(
                    recorded.id,
                    expected_revision=recorded.revision,
                    new_status=experiment,
                    actor=actor,
                )
            return host._result(recorded.id).next_stage
        finally:
            await host.close()

    return asyncio.run(scenario())


REJECTED, FAILED, DECIDING, CANCELLED = (
    ExperimentNodeStatus.REJECTED,
    ExperimentNodeStatus.FAILED,
    ExperimentNodeStatus.DECIDING,
    ExperimentNodeStatus.CANCELLED,
)


@pytest.mark.parametrize(
    ("nodes", "experiment", "stage"),
    [
        ((REJECTED,), ExperimentStatus.ACTIVE, "planning"),
        ((REJECTED, REJECTED), ExperimentStatus.ACTIVE, "planning"),
        ((FAILED,), ExperimentStatus.ACTIVE, "failure-handling"),
        ((CANCELLED,), ExperimentStatus.ACTIVE, "failure-handling"),
        ((REJECTED, FAILED), ExperimentStatus.ACTIVE, "failure-handling"),
        ((DECIDING,), ExperimentStatus.ACTIVE, "decision"),
        ((REJECTED, DECIDING), ExperimentStatus.ACTIVE, "decision"),
        ((REJECTED,), ExperimentStatus.FAILED, None),
        ((DECIDING,), ExperimentStatus.CANCELLED, None),
    ],
    ids=[
        "rejected-is-planning",
        "all-rejected-is-planning",
        "failed-is-failure-handling",
        "cancelled-is-failure-handling",
        "a-failure-beside-a-rejection-is-failure-handling",
        "deferred-is-decision",
        "a-deferral-beside-a-rejection-is-decision",
        "terminal-is-none",
        "terminal-is-none-even-while-deciding",
    ],
)
def test_next_stage_follows_what_the_record_says(
    tmp_path: Path,
    nodes: tuple[ExperimentNodeStatus, ...],
    experiment: ExperimentStatus,
    stage: str | None,
) -> None:
    """Planning only from a scientific outcome; a failure is never a planner request."""
    assert _next_stage(tmp_path, nodes, experiment) == stage
