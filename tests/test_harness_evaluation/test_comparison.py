"""ADR-022 V8–V10: summaries, comparability and exploratory comparison, on constructed values.

The evidence is real and verified; the metric values attached to it are
constructed, so every verdict and relation can be reached exactly.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import pytest

from tests.test_harness_evaluation import fixtures as fx
from xaytune.core.domain.benchmark import FirstTasks
from xaytune.core.domain.evaluation import EvaluationResult, EvaluationSpec, MetricResult
from xaytune.core.harness_comparison import (
    CandidateRuns,
    ComparisonSpec,
    ObjectiveMetric,
    ReplicateRuns,
    SummaryRefusedError,
    comparability,
    compare,
    summarize,
)
from xaytune.core.harness_evaluation import (
    HarnessRunEvidence,
    MetricObservation,
    RestrictedEvaluationReport,
    TaskObservations,
)
from xaytune.core.ids import ArtifactId, EvaluationId, EvaluationRunId, ExperimentNodeId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import ArtifactRef
from xaytune.evaluation.harness import (
    HarnessJudgeEvaluator,
    HarnessOutcomeEvaluator,
    bind_evaluation,
)

SEEDS = (1, 2, 3, 4)
UNIT = "tokens(xaytune.whitespace-tokens/v1)"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[int, HarnessRunEvidence]:
    root = tmp_path_factory.mktemp("comparison")
    return {seed: fx.Replicate(root / f"seed-{seed}", seed=seed).evidence for seed in SEEDS}


@pytest.fixture(scope="module")
def two_task_evidence(tmp_path_factory: pytest.TempPathFactory) -> HarnessRunEvidence:
    root = tmp_path_factory.mktemp("two")
    return fx.Replicate(root, seed=1, sampling=FirstTasks(count=2)).evidence


@pytest.fixture(scope="module")
def suite() -> Any:
    return fx.suite()


@pytest.fixture(scope="module")
def outcome() -> EvaluationSpec:
    evaluator = HarnessOutcomeEvaluator(fx.source())
    return bind_evaluation(evaluator, fx.evaluation_spec(fx.outcome_config()))


@pytest.fixture(scope="module")
def judge() -> EvaluationSpec:
    evaluator = HarnessJudgeEvaluator(fx.source())
    return bind_evaluation(evaluator, fx.evaluation_spec(fx.judge_config()))


def _result(
    evidence: HarnessRunEvidence,
    spec: EvaluationSpec,
    values: dict[str, float],
    *,
    unit: str = UNIT,
) -> EvaluationResult:
    return EvaluationResult(
        id=EvaluationId.generate(),
        evaluation_run_id=EvaluationRunId.generate(),
        node_id=ExperimentNodeId.generate(),
        subject=ArtifactRef(
            id=ArtifactId.generate(),
            kind="agent_run_evidence",
            uri="/x/run-evidence.json",
            digest=evidence.evidence_digest,
        ),
        evaluation_fingerprint=spec.evaluation_fingerprint(),
        metrics=tuple(
            MetricResult(
                name=name,
                value=value,
                sample_count=len(evidence.entries),
                evaluator_name=spec.evaluator.name,
                evaluator_version=spec.evaluator.version,
                metadata=FrozenDict({"unit": unit}),
            )
            for name, value in values.items()
        ),
    )


def _candidate(
    evidence: dict[int, HarnessRunEvidence],
    spec: EvaluationSpec,
    per_seed: dict[int, dict[str, float]],
    *,
    restricted: dict[int, dict[str, float]] | None = None,
) -> CandidateRuns:
    replicates = []
    for seed, values in per_seed.items():
        reports = ()
        if restricted is not None:
            reports = (
                RestrictedEvaluationReport(
                    evaluation_fingerprint=spec.evaluation_fingerprint(),
                    evidence_digest=evidence[seed].evidence_digest,
                    tasks=tuple(
                        TaskObservations(
                            task_id=task_id,
                            position=i,
                            observations=(MetricObservation(metric="task_success", value=v),),
                        )
                        for i, (task_id, v) in enumerate(restricted[seed].items())
                    ),
                ),
            )
        replicates.append(
            ReplicateRuns(
                evidence=evidence[seed],
                results=(_result(evidence[seed], spec, values),),
                restricted=reports,
            )
        )
    return CandidateRuns(evaluations=(spec,), replicates=tuple(replicates))


def _series(name: str, values: list[float]) -> dict[int, dict[str, float]]:
    return {seed: {name: v} for seed, v in zip(SEEDS, values)}


def _spec(*objectives: tuple[str, str, float], k: float = 2.0) -> ComparisonSpec:
    return ComparisonSpec(
        objectives=tuple(
            ObjectiveMetric(metric=m, direction=d, min_effect=e)  # type: ignore[arg-type]
            for m, d, e in objectives
        ),
        k=k,
    )


# ---- V8: summary ------------------------------------------------------------------------------


class TestSummary:
    def test_known_values(self, evidence: Any, outcome: EvaluationSpec) -> None:
        candidate = _candidate(evidence, outcome, _series("total_tokens", [1, 2, 3, 6]))
        summary = summarize(candidate.replicates, outcome.evaluation_fingerprint())
        tokens = summary.get("total_tokens")
        assert tokens is not None
        assert tokens.estimate == 3.0
        assert tokens.execution_standard_error == pytest.approx(statistics.stdev([1, 2, 3, 6]) / 2)
        assert tokens.judge_standard_deviation is None
        assert (tokens.replicates, tokens.judging_runs, tokens.tasks) == (4, 1, 3)
        assert tokens.unit == UNIT
        assert summary.method == "replicate-mean/v1"

    def test_g13_one_replicate_has_unknown_uncertainty(
        self, evidence: Any, outcome: EvaluationSpec
    ) -> None:
        candidate = _candidate(evidence, outcome, {1: {"total_tokens": 5.0}})
        tokens = summarize(candidate.replicates, outcome.evaluation_fingerprint()).get(
            "total_tokens"
        )
        assert tokens is not None and tokens.estimate == 5.0
        assert tokens.execution_standard_error is None, "unknown, never 0"

    def test_judging_replicates_are_averaged_and_their_spread_reported_apart(
        self, evidence: Any, judge: EvaluationSpec
    ) -> None:
        replicates = (
            ReplicateRuns(
                evidence=evidence[1],
                results=(
                    _result(evidence[1], judge, {"quality": 4}),
                    _result(evidence[1], judge, {"quality": 2}),
                ),
            ),
            ReplicateRuns(
                evidence=evidence[2],
                results=(
                    _result(evidence[2], judge, {"quality": 3}),
                    _result(evidence[2], judge, {"quality": 3}),
                ),
            ),
        )
        quality = summarize(replicates, judge.evaluation_fingerprint()).get("quality")
        assert quality is not None
        assert quality.values == (3.0, 3.0)
        assert quality.estimate == 3.0
        assert quality.execution_standard_error == 0.0
        assert quality.judge_standard_deviation == pytest.approx(1.0)
        assert quality.judging_runs == 2

    def test_unbalanced_judging_is_refused(self, evidence: Any, judge: EvaluationSpec) -> None:
        replicates = (
            ReplicateRuns(
                evidence=evidence[1],
                results=(
                    _result(evidence[1], judge, {"quality": 4}),
                    _result(evidence[1], judge, {"quality": 2}),
                ),
            ),
            ReplicateRuns(
                evidence=evidence[2], results=(_result(evidence[2], judge, {"quality": 3}),)
            ),
        )
        with pytest.raises(SummaryRefusedError, match="unbalanced"):
            summarize(replicates, judge.evaluation_fingerprint())

    def test_a_metric_missing_from_one_replicate_is_absent(
        self, evidence: Any, outcome: EvaluationSpec
    ) -> None:
        per_seed = {1: {"turns": 2.0, "total_tokens": 9.0}, 2: {"turns": 2.0}}
        candidate = _candidate(evidence, outcome, per_seed)
        summary = summarize(candidate.replicates, outcome.evaluation_fingerprint())
        tokens = summary.get("total_tokens")
        assert tokens is not None and tokens.estimate is None
        assert "unbalanced" in (tokens.absent or "")
        assert summary.get("turns").estimate == 2.0

    def test_different_task_counts_are_absent(
        self, evidence: Any, two_task_evidence: HarnessRunEvidence, outcome: EvaluationSpec
    ) -> None:
        replicates = (
            ReplicateRuns(
                evidence=evidence[2], results=(_result(evidence[2], outcome, {"turns": 2}),)
            ),
            ReplicateRuns(
                evidence=two_task_evidence,
                results=(_result(two_task_evidence, outcome, {"turns": 2}),),
            ),
        )
        turns = summarize(replicates, outcome.evaluation_fingerprint()).get("turns")
        assert turns is not None and "task counts" in (turns.absent or "")

    def test_a_repeated_seed_is_refused(self, evidence: Any, outcome: EvaluationSpec) -> None:
        one = ReplicateRuns(
            evidence=evidence[1], results=(_result(evidence[1], outcome, {"turns": 2}),)
        )
        with pytest.raises(SummaryRefusedError, match="seed"):
            summarize((one, one), outcome.evaluation_fingerprint())


# ---- V9: comparability ----------------------------------------------------------------------


def _with_facets(evidence: HarnessRunEvidence, **changes: Any) -> HarnessRunEvidence:
    return evidence.model_copy(update={"facets": evidence.facets.model_copy(update=changes)})


class TestComparability:
    def test_g19_an_aa_comparison_is_comparable_with_a_key(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        a = _candidate(evidence, outcome, _series("turns", [2, 2, 2, 2]))
        verdict = comparability(a, a, suite)
        assert verdict.comparable and verdict.reasons == ()
        assert verdict.comparison_key is not None
        assert verdict.comparison_key == comparability(a, a, suite).comparison_key

    def test_g9_a_missing_seed(self, evidence: Any, outcome: EvaluationSpec, suite: Any) -> None:
        a = _candidate(evidence, outcome, _series("turns", [2, 2, 2, 2]))
        b = _candidate(evidence, outcome, {s: {"turns": 2.0} for s in SEEDS[:3]})
        verdict = comparability(a, b, suite)
        assert not verdict.comparable and verdict.comparison_key is None
        assert "C2: B has no replicate at seed 4" in verdict.reasons

    def test_g10_another_evaluator_version(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        bumped = outcome.model_copy(
            update={"evaluator": outcome.evaluator.model_copy(update={"version": "0.2.0"})}
        )
        a = _candidate(evidence, outcome, _series("turns", [2, 2, 2, 2]))
        b = _candidate(evidence, bumped, _series("turns", [2, 2, 2, 2]))
        verdict = comparability(a, b, suite)
        assert any(r.startswith("C3") and "fingerprints" in r for r in verdict.reasons)

    def test_g10_another_catalog_or_judge_pin(
        self, evidence: Any, judge: EvaluationSpec, suite: Any
    ) -> None:
        config = fx.judge_config()
        config["judge_model"]["revision"] = "2026-11-01"
        repinned = bind_evaluation(HarnessJudgeEvaluator(fx.source()), fx.evaluation_spec(config))
        a = _candidate(evidence, judge, _series("quality", [3, 3, 3, 3]))
        b = _candidate(evidence, repinned, _series("quality", [3, 3, 3, 3]))
        assert any("C3" in r for r in comparability(a, b, suite).reasons)

    def test_g11_a_facet_differs_between_or_within_candidates(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        other = {s: _with_facets(e, runner_version="9.9") for s, e in evidence.items()}
        a = _candidate(evidence, outcome, _series("turns", [2, 2, 2, 2]))
        b = _candidate(other, outcome, _series("turns", [2, 2, 2, 2]))
        assert (
            "C4: the facet runner_version differs across runs" in comparability(a, b, suite).reasons
        )
        mixed = dict(evidence)
        mixed[4] = _with_facets(evidence[4], token_counter="another-counter/v1")
        within = _candidate(mixed, outcome, _series("turns", [2, 2, 2, 2]))
        assert (
            "C4: the facet token_counter differs across runs"
            in comparability(within, within, suite).reasons
        )

    def test_every_failure_is_reported_at_once(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        bumped = outcome.model_copy(
            update={"evaluator": outcome.evaluator.model_copy(update={"version": "0.2.0"})}
        )
        other = {s: _with_facets(e, runner_version="9.9") for s, e in evidence.items()}
        a = _candidate(evidence, outcome, _series("turns", [2, 2, 2, 2]))
        b = _candidate(other, bumped, {s: {"turns": 2.0} for s in SEEDS[:3]})
        reasons = comparability(a, b, suite).reasons
        assert {r[:2] for r in reasons} >= {"C2", "C3", "C4"}

    def test_c1_another_cohort(
        self,
        evidence: Any,
        two_task_evidence: HarnessRunEvidence,
        outcome: EvaluationSpec,
        suite: Any,
    ) -> None:
        a = _candidate(evidence, outcome, {1: {"turns": 2.0}})
        b = _candidate({1: two_task_evidence}, outcome, {1: {"turns": 2.0}})
        assert any(r.startswith("C1") for r in comparability(a, b, suite).reasons)


# ---- V10: comparison --------------------------------------------------------------------------


def _verdict(report: Any, metric: str) -> str:
    return next(m.verdict for m in report.metrics if m.metric == metric)


class TestComparison:
    def test_favors_a_favors_b_and_inconclusive(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        a = _candidate(evidence, outcome, _series("task_success", [0.9, 0.8, 0.9, 0.85]))
        b = _candidate(evidence, outcome, _series("task_success", [0.5, 0.45, 0.5, 0.55]))
        spec = _spec(("task_success", "maximize", 0.05))
        assert _verdict(compare(a, b, spec, suite), "task_success") == "favors-a"
        assert _verdict(compare(b, a, spec, suite), "task_success") == "favors-b"
        noisy = _candidate(evidence, outcome, _series("task_success", [0.9, 0.4, 0.95, 0.3]))
        assert _verdict(compare(noisy, b, spec, suite), "task_success") == "inconclusive"

    def test_minimize_is_oriented_so_positive_favors_a(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        cheap = _candidate(evidence, outcome, _series("total_tokens", [10, 11, 10, 12]))
        costly = _candidate(evidence, outcome, _series("total_tokens", [40, 42, 41, 39]))
        report = compare(cheap, costly, _spec(("total_tokens", "minimize", 1)), suite)
        metric = report.metrics[0]
        assert metric.verdict == "favors-a"
        assert all(d > 0 for d in metric.differences)

    def test_the_minimum_effect_can_dominate(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        a = _candidate(evidence, outcome, _series("turns", [3.0, 3.0, 3.0, 3.0]))
        b = _candidate(evidence, outcome, _series("turns", [2.0, 2.0, 2.0, 2.0]))
        report = compare(a, b, _spec(("turns", "maximize", 5.0)), suite)
        assert report.metrics[0].verdict == "inconclusive"
        assert report.metrics[0].threshold == 5.0

    def test_g13_one_replicate(self, evidence: Any, outcome: EvaluationSpec, suite: Any) -> None:
        a = _candidate(evidence, outcome, {1: {"turns": 9.0}})
        b = _candidate(evidence, outcome, {1: {"turns": 1.0}})
        metric = compare(a, b, _spec(("turns", "maximize", 0)), suite).metrics[0]
        assert metric.verdict == "inconclusive" and metric.standard_error is None

    def test_g13a_two_replicates_and_a_huge_difference(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        a = _candidate(evidence, outcome, {1: {"turns": 100.0}, 2: {"turns": 101.0}})
        b = _candidate(evidence, outcome, {1: {"turns": 1.0}, 2: {"turns": 1.5}})
        metric = compare(a, b, _spec(("turns", "maximize", 0)), suite).metrics[0]
        assert metric.verdict == "inconclusive"

    def test_g13b_three_replicates_give_a_provisional_direction(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        a = _candidate(evidence, outcome, {s: {"turns": 10.0 + s} for s in SEEDS[:3]})
        b = _candidate(evidence, outcome, {s: {"turns": 1.0 + s / 10} for s in SEEDS[:3]})
        report = compare(a, b, _spec(("turns", "maximize", 0), k=2), suite)
        assert report.metrics[0].verdict == "favors-a"
        assert (report.claim, report.rule, report.replicates, report.k) == (
            "exploratory",
            "paired-difference/v1",
            3,
            2.0,
        )

    def test_g12_time_needs_a_shared_timing_environment(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        values = {s: {"model_latency_ms": 5.0 + s, "turns": 9.0} for s in SEEDS}
        slower = {s: {"model_latency_ms": 50.0 + s, "turns": 1.0} for s in SEEDS}
        a = _candidate(evidence, outcome, values)
        b = _candidate(evidence, outcome, slower)
        spec = _spec(("model_latency_ms", "minimize", 0), ("turns", "maximize", 0))
        report = compare(a, b, spec, suite)
        assert _verdict(report, "model_latency_ms") == "not-comparable"
        assert _verdict(report, "turns") == "favors-a"
        labelled = {s: _with_facets(e, timing_environment="ci-a") for s, e in evidence.items()}
        timed = compare(
            _candidate(labelled, outcome, values),
            _candidate(labelled, outcome, slower),
            spec,
            suite,
        )
        assert _verdict(timed, "model_latency_ms") == "favors-a"

    @pytest.mark.parametrize(
        ("a_values", "b_values", "relation"),
        [
            ((9, 9), (1, 1), "favors-a"),
            ((1, 1), (9, 9), "favors-b"),
            ((9, 1), (1, 9), "trade-off"),
            ((5, 5), (5, 5), "inconclusive"),
        ],
    )
    def test_vector_relations(
        self,
        evidence: Any,
        outcome: EvaluationSpec,
        suite: Any,
        a_values: tuple[int, int],
        b_values: tuple[int, int],
        relation: str,
    ) -> None:
        def series(x: int, y: int, jitter: float) -> dict[int, dict[str, float]]:
            return {s: {"turns": x + jitter * s, "tool_calls": y + jitter * s} for s in SEEDS}

        a = _candidate(evidence, outcome, series(*a_values, 0.01))
        b = _candidate(evidence, outcome, series(*b_values, 0.02))
        spec = _spec(("turns", "maximize", 0.5), ("tool_calls", "maximize", 0.5))
        assert compare(a, b, spec, suite).relation == relation

    def test_a_missing_vector_metric_makes_the_relation_not_comparable(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        a = _candidate(evidence, outcome, _series("turns", [9, 9, 9, 9]))
        b = _candidate(evidence, outcome, _series("turns", [1, 1, 1, 1]))
        report = compare(a, b, _spec(("turns", "maximize", 0), ("cost", "minimize", 0)), suite)
        assert _verdict(report, "cost") == "not-comparable"
        assert report.relation == "not-comparable"

    def test_an_incomparable_pair_is_not_comparable_everywhere(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        a = _candidate(evidence, outcome, _series("turns", [9, 9, 9, 9]))
        b = _candidate(evidence, outcome, {s: {"turns": 1.0} for s in SEEDS[:3]})
        report = compare(a, b, _spec(("turns", "maximize", 0)), suite)
        assert report.relation == "not-comparable"
        assert not report.comparability.comparable

    def test_task_success_counts_are_paired_by_seed_and_task(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        ids = fx.TASK_IDS
        restricted_a = {s: {ids[0]: 1.0, ids[1]: 1.0, ids[2]: 0.0} for s in SEEDS}
        restricted_b = {s: {ids[0]: 1.0, ids[1]: 0.0, ids[2]: 1.0} for s in SEEDS}
        a = _candidate(
            evidence, outcome, _series("task_success", [2 / 3] * 4), restricted=restricted_a
        )
        b = _candidate(
            evidence, outcome, _series("task_success", [2 / 3] * 4), restricted=restricted_b
        )
        counts = compare(a, b, _spec(("task_success", "maximize", 0)), suite).task_success
        assert counts is not None
        assert (counts.wins, counts.losses, counts.ties, counts.unscored) == (4, 4, 4, 0)

    def test_no_scalar_and_no_claim_of_significance(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        a = _candidate(evidence, outcome, _series("turns", [9, 8, 9, 9]))
        b = _candidate(evidence, outcome, _series("turns", [1, 2, 1, 1]))
        report = compare(a, b, _spec(("turns", "maximize", 0)), suite)
        text = report.model_dump_json().lower()
        for word in ("significan", "confiden", "proven", "p_value", "weighted", "scalar"):
            assert word not in text
        assert not any("score" in field for field in type(report).model_fields)

    def test_the_report_is_deterministic_and_fingerprinted(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        a = _candidate(evidence, outcome, _series("turns", [9, 8, 9, 9]))
        b = _candidate(evidence, outcome, _series("turns", [1, 2, 1, 1]))
        spec = _spec(("turns", "maximize", 0))
        first, second = compare(a, b, spec, suite), compare(a, b, spec, suite)
        assert first == second
        assert first.report_fingerprint.startswith("sha256:")

    def test_a_shrinking_difference_is_measured_exactly(
        self, evidence: Any, outcome: EvaluationSpec, suite: Any
    ) -> None:
        a = _candidate(evidence, outcome, _series("turns", [4, 5, 6, 7]))
        b = _candidate(evidence, outcome, _series("turns", [1, 1, 1, 1]))
        metric = compare(a, b, _spec(("turns", "maximize", 0)), suite).metrics[0]
        assert metric.differences == (3.0, 4.0, 5.0, 6.0)
        assert metric.mean_difference == 4.5
        assert metric.standard_error == pytest.approx(statistics.stdev([3, 4, 5, 6]) / math.sqrt(4))


def test_compare_refuses_unbalanced_judging(
    evidence: Any, judge: EvaluationSpec, suite: Any
) -> None:
    unbalanced = CandidateRuns(
        evaluations=(judge,),
        replicates=(
            ReplicateRuns(
                evidence=evidence[1],
                results=(
                    _result(evidence[1], judge, {"quality": 4}),
                    _result(evidence[1], judge, {"quality": 2}),
                ),
            ),
            ReplicateRuns(
                evidence=evidence[2], results=(_result(evidence[2], judge, {"quality": 3}),)
            ),
        ),
    )
    with pytest.raises(SummaryRefusedError, match="unbalanced"):
        compare(unbalanced, unbalanced, _spec(("quality", "maximize", 0)), suite)
