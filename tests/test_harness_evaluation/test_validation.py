"""ADR-022 acceptance criterion 12: the whole pipeline, on two harness configurations.

A small pinned static suite; two harnesses that differ in one behavior field
(``stopping.max_turns``); a seeded sample, so each seed draws its own tasks;
matched seeds and three execution replicates each; recorded trajectories;
assembled, verified evidence; the outcome evaluation; and a comparison report
that explains success, resource use and comparability.

**With the scripted model this demonstrates the pipeline, not real-world
quality**, and the report says so. A second report on a deliberately
mismatched pair is ``not-comparable``, with its reasons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.test_harness.fixtures import harness
from tests.test_harness_evaluation import fixtures as fx
from xaytune.core.domain.benchmark import SeededSample
from xaytune.core.harness_comparison import (
    CandidateRuns,
    ComparisonSpec,
    HarnessComparisonReport,
    ObjectiveMetric,
    ReplicateRuns,
    compare,
)
from xaytune.evaluation.harness import (
    HarnessOutcomeEvaluator,
    bind_evaluation,
    run_harness_evaluation,
)

SEEDS = (101, 102, 103)
SAMPLING = SeededSample(count=2)
SCRIPTED_CAVEAT = (
    "scripted model: this comparison demonstrates the evaluation pipeline, "
    "not real-world harness quality"
)
OBJECTIVES = ComparisonSpec(
    objectives=(
        ObjectiveMetric(metric="task_success", direction="maximize", min_effect=0.1),
        ObjectiveMetric(metric="total_tokens", direction="minimize", min_effect=1.0),
        ObjectiveMetric(metric="turns", direction="minimize", min_effect=0.5),
        ObjectiveMetric(metric="model_latency_ms", direction="minimize", min_effect=0.0),
    ),
    k=2.0,
)


def _harness(max_turns: int) -> Any:
    return harness(
        stopping={
            "max_turns": max_turns,
            "max_total_tokens": 100000,
            "max_tool_calls": 30,
            "terminate_on": ["final-answer"],
        }
    )


def _candidate(
    root: Path, harness_spec: Any, seeds: tuple[int, ...], normalization: str = "casefold-strip"
) -> CandidateRuns:
    config = fx.outcome_config(
        sampling=SAMPLING.model_dump(mode="json"),
        success={"kind": "exact-match", "normalization": normalization},
    )
    evaluator = HarnessOutcomeEvaluator(fx.source())
    spec = bind_evaluation(evaluator, fx.evaluation_spec(config))
    replicates = []
    for seed in seeds:
        replicate = fx.Replicate(
            root / f"seed-{seed}", seed=seed, harness_spec=harness_spec, sampling=SAMPLING
        )
        evaluation = run_harness_evaluation(
            evaluator,
            spec,
            replicate.subject,
            source=replicate.source,
            output_uri=str(root / f"eval-{seed}"),
            seed=seed,
            replicate=0,
        )
        assert evaluation.result is not None, evaluation.report.failure
        replicates.append(
            ReplicateRuns(
                evidence=replicate.evidence,
                results=(evaluation.result,),
                restricted=(evaluation.restricted,),
            )
        )
    return CandidateRuns(evaluations=(spec,), replicates=tuple(replicates))


@pytest.fixture(scope="module")
def candidates(tmp_path_factory: pytest.TempPathFactory) -> dict[str, CandidateRuns]:
    root = tmp_path_factory.mktemp("validation")
    return {
        "patient": _candidate(root / "patient", _harness(20), SEEDS),
        "hasty": _candidate(root / "hasty", _harness(1), SEEDS),
        "elsewhere": _candidate(root / "elsewhere", _harness(1), (201, 202, 203), "none"),
    }


@pytest.fixture(scope="module")
def report(candidates: dict[str, CandidateRuns]) -> HarnessComparisonReport:
    return compare(
        candidates["patient"], candidates["hasty"], OBJECTIVES, fx.suite(), notes=(SCRIPTED_CAVEAT,)
    )


def _metric(report: HarnessComparisonReport, name: str) -> Any:
    return next(m for m in report.metrics if m.metric == name)


def test_the_two_configurations_differ_in_one_behavior_field() -> None:
    patient, hasty = _harness(20), _harness(1)
    assert patient.harness_fingerprint() != hasty.harness_fingerprint()
    assert patient.model_copy(update={"stopping": hasty.stopping}) == hasty


def test_seeds_draw_different_tasks_and_candidates_are_matched_by_seed(
    candidates: dict[str, CandidateRuns],
) -> None:
    patient, hasty = candidates["patient"], candidates["hasty"]
    drawn = {r.evidence.assignment.task_ids for r in patient.replicates}
    assert len(drawn) > 1, "the seeded sample makes the seed matter"
    for a, b in zip(patient.replicates, hasty.replicates):
        assert a.evidence.seed == b.evidence.seed
        assert a.evidence.assignment == b.evidence.assignment
        assert len(a.evidence.entries) == 2


def test_every_run_was_recorded_and_verified(candidates: dict[str, CandidateRuns]) -> None:
    for candidate in candidates.values():
        for replicate in candidate.replicates:
            for entry in replicate.evidence.entries:
                assert entry.state == "attested"
                assert entry.outcome is not None and entry.attempts


def test_the_report_explains_comparability(report: HarnessComparisonReport) -> None:
    assert report.comparability.comparable, report.comparability.reasons
    assert report.comparability.comparison_key is not None
    assert (report.claim, report.rule, report.replicates, report.k) == (
        "exploratory",
        "paired-difference/v1",
        3,
        2.0,
    )
    assert report.seeds == SEEDS
    assert SCRIPTED_CAVEAT in report.notes


def test_the_report_explains_success(report: HarnessComparisonReport) -> None:
    success = _metric(report, "task_success")
    assert success.a is not None and success.b is not None
    assert success.b.estimate == 0.0, "one turn stops on the budget before any answer"
    assert success.a.estimate > 0.0
    assert success.verdict == "favors-a"
    counts = report.task_success
    assert counts is not None and counts.losses == 0 and counts.wins >= 1


def test_the_report_explains_resource_use(report: HarnessComparisonReport) -> None:
    tokens, turns = _metric(report, "total_tokens"), _metric(report, "turns")
    assert tokens.a.estimate > tokens.b.estimate, "stopping early spends fewer tokens"
    assert tokens.verdict == "favors-b" and turns.verdict == "favors-b"
    assert tokens.a.unit == "tokens(xaytune.whitespace-tokens/v1)"
    latency = _metric(report, "model_latency_ms")
    assert latency.verdict == "not-comparable", "no timing environment was declared"
    assert report.relation == "not-comparable", "a time metric is in the vector"


def test_without_time_in_the_vector_the_relation_is_a_trade_off(
    candidates: dict[str, CandidateRuns],
) -> None:
    spec = ComparisonSpec(objectives=OBJECTIVES.objectives[:3], k=2.0)
    relation = compare(candidates["patient"], candidates["hasty"], spec, fx.suite()).relation
    assert relation == "trade-off", "better success, more tokens: never collapsed into one score"


def test_a_deliberately_mismatched_pair_is_not_comparable(
    candidates: dict[str, CandidateRuns],
) -> None:
    mismatched = compare(
        candidates["patient"],
        candidates["elsewhere"],
        OBJECTIVES,
        fx.suite(),
        notes=(SCRIPTED_CAVEAT,),
    )
    assert not mismatched.comparability.comparable
    assert mismatched.comparability.comparison_key is None
    reasons = mismatched.comparability.reasons
    assert any(r.startswith("C2") and "seed" in r for r in reasons), reasons
    assert any(r.startswith("C3") for r in reasons), reasons
    assert mismatched.relation == "not-comparable"
    assert {m.verdict for m in mismatched.metrics} == {"not-comparable"}
