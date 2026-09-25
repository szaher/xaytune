"""ThresholdDecisionEngine: thresholds on recorded values, and nothing invented.

Pure tests: a context in, a decision (or a refusal) out. What the decision
does to the record is ``tests/test_storage/test_decisions.py``; the loop end
to end is ``tests/test_experiment/test_decisions.py``.
"""

from __future__ import annotations

import pytest

from xaytune.core.domain.decision import DecisionContext, DecisionOutcome
from xaytune.core.domain.evaluation import EvaluationResult, MetricResult
from xaytune.core.domain.objective import MetricConstraint, Objective, ObjectiveMetric
from xaytune.core.ids import (
    ArtifactId,
    EvaluationId,
    EvaluationRunId,
    ExperimentId,
    ExperimentNodeId,
)
from xaytune.core.refs import ArtifactRef
from xaytune.decision import DecisionEngine, ThresholdDecisionEngine, UndecidableError

EXPERIMENT = ExperimentId.generate()
NODE = ExperimentNodeId.generate()
SUBJECT = ArtifactRef(id=ArtifactId.generate(), kind="model", uri="/m")


def _result(*metrics: tuple[str, float], **metric_fields: object) -> EvaluationResult:
    return EvaluationResult(
        id=EvaluationId.generate(),
        evaluation_run_id=EvaluationRunId.generate(),
        node_id=NODE,
        subject=SUBJECT,
        evaluation_fingerprint="sha256:e",
        metrics=tuple(
            MetricResult(
                name=name,
                value=value,
                evaluator_name="native",
                evaluator_version="0.1.0",
                seed=7,
                **metric_fields,  # type: ignore[arg-type]
            )
            for name, value in metrics
        ),
    )


def _objective(
    name: str = "accuracy",
    direction: str = "maximize",
    target: float | None = 0.8,
    *constraints: MetricConstraint,
) -> Objective:
    return Objective(
        primary=ObjectiveMetric(name=name, direction=direction),  # type: ignore[arg-type]
        target=target,
        constraints=constraints,
    )


def _context(objective: Objective, *results: EvaluationResult, cycle: int = 1) -> DecisionContext:
    return DecisionContext(
        experiment_id=EXPERIMENT,
        node_id=NODE,
        evaluation_cycle=cycle,
        objective=objective,
        results=results,
    )


def _decide(objective: Objective, *results: EvaluationResult):
    return ThresholdDecisionEngine().decide(_context(objective, *results))


def test_it_is_a_decision_engine() -> None:
    assert isinstance(ThresholdDecisionEngine(), DecisionEngine)


# ---- the target -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "target", "value", "outcome"),
    [
        ("maximize", 0.80, 0.85, DecisionOutcome.STOP_SUCCEEDED),
        ("maximize", 0.80, 0.80, DecisionOutcome.STOP_SUCCEEDED),
        ("maximize", 0.80, 0.79, DecisionOutcome.STOP_FAILED),
        ("minimize", 2.0, 1.5, DecisionOutcome.STOP_SUCCEEDED),
        ("minimize", 2.0, 2.0, DecisionOutcome.STOP_SUCCEEDED),
        ("minimize", 2.0, 2.1, DecisionOutcome.STOP_FAILED),
    ],
    ids=[
        "maximize-reached",
        "maximize-exactly",
        "maximize-missed",
        "minimize-reached",
        "minimize-exactly",
        "minimize-missed",
    ],
)
def test_the_target_decides_by_direction(
    direction: str, target: float, value: float, outcome: DecisionOutcome
) -> None:
    decision = _decide(_objective("m", direction, target), _result(("m", value)))
    assert decision.outcome is outcome
    (evidence,) = decision.evidence
    assert evidence.operator == (">=" if direction == "maximize" else "<=")
    assert (evidence.value, evidence.threshold) == (value, target)
    assert evidence.satisfied is (outcome is DecisionOutcome.STOP_SUCCEEDED)


def test_outcomes_map_to_where_the_candidate_ends_up() -> None:
    from xaytune.core.state.status import ExperimentNodeStatus

    assert DecisionOutcome.STOP_SUCCEEDED.node_status is ExperimentNodeStatus.COMPLETED
    assert DecisionOutcome.STOP_FAILED.node_status is ExperimentNodeStatus.REJECTED
    assert DecisionOutcome.REJECT.node_status is ExperimentNodeStatus.REJECTED


# ---- constraints ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operator", "value", "holds"),
    [
        ("<", 1.0, True),
        ("<", 2.0, False),
        ("<=", 2.0, True),
        ("<=", 2.1, False),
        (">", 3.0, True),
        (">", 2.0, False),
        (">=", 2.0, True),
        (">=", 1.9, False),
        ("==", 2.0, True),
        ("==", 2.0000001, False),
        ("!=", 2.5, True),
        ("!=", 2.0, False),
    ],
)
def test_every_operator_is_applied_exactly(operator: str, value: float, holds: bool) -> None:
    objective = _objective(
        "accuracy",
        "maximize",
        0.5,
        MetricConstraint(name="latency", operator=operator, value=2.0),  # type: ignore[arg-type]
    )
    decision = _decide(objective, _result(("accuracy", 0.9), ("latency", value)))
    expected = DecisionOutcome.STOP_SUCCEEDED if holds else DecisionOutcome.REJECT
    assert decision.outcome is expected
    constraint = next(e for e in decision.evidence if e.role == "constraint")
    assert (constraint.operator, constraint.satisfied) == (operator, holds)


def test_a_violated_constraint_rejects_even_when_the_target_is_met() -> None:
    objective = _objective(
        "accuracy", "maximize", 0.5, MetricConstraint(name="latency", operator="<=", value=100)
    )
    decision = _decide(objective, _result(("accuracy", 0.99), ("latency", 250.0)))
    assert decision.outcome is DecisionOutcome.REJECT
    assert "latency 250.0 is not <= 100" in decision.reason


def test_a_violated_constraint_rejects_without_a_target() -> None:
    """Unacceptable is decidable even when "good enough" is not."""
    objective = _objective(
        "loss", "minimize", None, MetricConstraint(name="latency", operator="<=", value=100)
    )
    decision = _decide(objective, _result(("loss", 1.0), ("latency", 250.0)))
    assert decision.outcome is DecisionOutcome.REJECT
    assert [e.role for e in decision.evidence] == ["constraint"]


def test_constraints_that_hold_are_part_of_the_evidence() -> None:
    objective = _objective(
        "accuracy", "maximize", 0.5, MetricConstraint(name="latency", operator="<", value=100)
    )
    decision = _decide(objective, _result(("accuracy", 0.9), ("latency", 20.0)))
    assert [e.role for e in decision.evidence] == ["objective", "constraint"]
    assert "constraints held" in decision.reason


# ---- nothing is invented ----------------------------------------------------------


def test_a_missing_objective_metric_is_not_read_as_passing() -> None:
    with pytest.raises(UndecidableError, match="objective metric 'accuracy'"):
        _decide(_objective(), _result(("loss", 1.0)))


def test_a_missing_constraint_metric_is_not_read_as_passing() -> None:
    objective = _objective(
        "accuracy", "maximize", 0.5, MetricConstraint(name="latency", operator="<", value=100)
    )
    with pytest.raises(UndecidableError, match="constraint metric 'latency'"):
        _decide(objective, _result(("accuracy", 0.9)))


def test_a_missing_metric_is_undecidable_even_if_a_constraint_is_violated() -> None:
    """The rules are ordered: absent evidence first, then judgement on what is present."""
    objective = _objective(
        "accuracy", "maximize", 0.5, MetricConstraint(name="latency", operator="<", value=100)
    )
    with pytest.raises(UndecidableError):
        _decide(objective, _result(("latency", 500.0)))


def test_an_objective_without_a_target_is_not_decided() -> None:
    """ "Optimize this" is not "this is good enough"."""
    with pytest.raises(UndecidableError, match="no target"):
        _decide(_objective("loss", "minimize", None), _result(("loss", 0.01)))


def test_a_metric_reported_twice_is_ambiguous() -> None:
    with pytest.raises(UndecidableError, match="reported 2 times"):
        _decide(_objective(), _result(("accuracy", 0.9)), _result(("accuracy", 0.1)))


def test_a_sliced_metric_does_not_stand_in_for_the_metric() -> None:
    with pytest.raises(UndecidableError, match="objective metric 'accuracy'"):
        _decide(_objective(), _result(("accuracy", 0.9), slice="hard"))


def test_every_reason_is_given_at_once() -> None:
    objective = _objective(
        "accuracy", "maximize", 0.5, MetricConstraint(name="latency", operator="<", value=100)
    )
    with pytest.raises(UndecidableError) as refused:
        _decide(objective, _result(("loss", 1.0)))
    assert len(refused.value.reasons) == 2


def test_no_results_at_all_is_undecidable() -> None:
    with pytest.raises(UndecidableError):
        _decide(_objective())


# ---- determinism and provenance ---------------------------------------------------


def test_the_same_context_gives_an_identical_proposal() -> None:
    """Pure: nothing minted -- no id, no time, no actor -- so byte-equivalent output."""
    context = _context(_objective(), _result(("accuracy", 0.9)))
    first, second = (ThresholdDecisionEngine().decide(context) for _ in range(2))
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert not {"id", "created_at", "actor"} & set(type(first).model_fields)


def test_the_input_fingerprint_is_the_objective_and_the_exact_results() -> None:
    result = _result(("accuracy", 0.9))
    other = _result(("accuracy", 0.9))
    base = _context(_objective(), result).input_fingerprint()

    assert _context(_objective(), result).input_fingerprint() == base
    assert _context(_objective(target=0.7), result).input_fingerprint() != base
    assert _context(_objective(), other).input_fingerprint() != base, "another result"
    assert _context(_objective(), result, cycle=2).input_fingerprint() != base


def test_result_order_does_not_change_the_input() -> None:
    a, b = _result(("accuracy", 0.9)), _result(("loss", 1.0))
    assert (
        _context(_objective(), a, b).input_fingerprint()
        == _context(_objective(), b, a).input_fingerprint()
    )


def test_a_decision_names_the_engine_the_results_and_the_evidence() -> None:
    result = _result(("accuracy", 0.9))
    context = _context(_objective(), result)
    decision = ThresholdDecisionEngine().decide(context)

    assert (decision.engine_name, decision.engine_version) == ("threshold", "1.0.0")
    assert decision.evaluation_result_ids == (result.id,)
    assert decision.input_fingerprint == context.input_fingerprint()
    assert (decision.experiment_id, decision.node_id, decision.evaluation_cycle) == (
        EXPERIMENT,
        NODE,
        1,
    )
    (evidence,) = decision.evidence
    assert evidence.evaluation_result_id == result.id
    assert (evidence.evaluator_name, evidence.evaluator_version) == ("native", "0.1.0")


def test_sample_count_does_not_enter_the_decision() -> None:
    """Point estimates only: a count means different things for different evaluators."""
    few = _decide(_objective(), _result(("accuracy", 0.9), sample_count=3))
    many = _decide(_objective(), _result(("accuracy", 0.9), sample_count=30_000))
    assert few.outcome is many.outcome is DecisionOutcome.STOP_SUCCEEDED


# ---- the input identity is an explicit projection ----------------------------------


def _changed(result: EvaluationResult, **metric_update: object) -> EvaluationResult:
    (metric,) = result.metrics
    return result.model_copy(update={"metrics": (metric.model_copy(update=metric_update),)})


@pytest.mark.parametrize(
    "update",
    [
        {"value": 0.91},
        {"seed": 8},
        {"sample_count": 4},
        {"slice": "hard"},
        {"evaluator_version": "0.2.0"},
        {"standard_error": 0.01},
        {"confidence_interval": (0.8, 0.95)},
    ],
    ids=lambda u: next(iter(u)),
)
def test_what_is_evidence_changes_the_input_identity(update: dict) -> None:
    result = _result(("accuracy", 0.9))
    base = _context(_objective(), result).input_fingerprint()
    assert _context(_objective(), _changed(result, **update)).input_fingerprint() != base


def test_what_merely_describes_a_result_does_not() -> None:
    """A timestamp, report artifacts or free-form metadata: not evidence, not identity."""
    from datetime import datetime, timezone

    result = _result(("accuracy", 0.9))
    base = _context(_objective(), result).input_fingerprint()
    report = ArtifactRef(id=ArtifactId.generate(), kind="evaluation_report", uri="/r.json")
    described = _changed(result, metadata={"tokens_scored": 42}).model_copy(
        update={
            "created_at": datetime(2020, 1, 1, tzinfo=timezone.utc),
            "artifacts": (report,),
        }
    )
    assert _context(_objective(), described).input_fingerprint() == base


def test_metric_and_constraint_order_is_not_identity() -> None:
    both = _result(("accuracy", 0.9), ("latency", 20.0))
    reordered = both.model_copy(update={"metrics": tuple(reversed(both.metrics))})
    a = MetricConstraint(name="latency", operator="<", value=100)
    b = MetricConstraint(name="accuracy", operator=">", value=0.1)
    assert (
        _context(_objective("accuracy", "maximize", 0.5, a, b), both).input_fingerprint()
        == _context(_objective("accuracy", "maximize", 0.5, b, a), reordered).input_fingerprint()
    )
