"""Summarizing replicates and comparing two harness candidates (H06, ADR-022 V8–V10).

```text
summarize(one candidate's results, one evaluation fingerprint)   replicate-mean/v1     (V8)
comparability(A, B)            C1–C4, every failure, and the comparison key            (V9)
compare(A, B, ComparisonSpec)  paired-difference/v1 per metric, the vector relation    (V10)
                               → HarnessComparisonReport, claim "exploratory"
```

**A verdict here is an exploratory decision rule, not a statistical claim.**
It says which way the evidence currently points, under a declared
threshold, for choosing what to try next. With few replicates the standard
error is itself a noisy estimate, and comparing many candidates multiplies
the chance that some difference crosses the threshold by luck; v1 controls
neither. ``k`` is a decision threshold, not a quantile of any distribution.

Everything here is pure and deterministic. It never scalarizes: a weighted
score belongs to a provider that declares weights (H07). ``Objective`` and
the threshold decision engines are untouched; a comparison report is
evidence, not a ``Decision``, and moves no node.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

from pydantic import ConfigDict, Field, model_validator

from xaytune.core.domain.benchmark import (
    BenchmarkSuiteRef,
    ResolvedBenchmarkSuite,
    UnverifiedAssignmentError,
    assignment_mismatches,
    benchmark_cohort,
    verify_assignment,
)
from xaytune.core.domain.evaluation import EvaluationResult, EvaluationSpec
from xaytune.core.fingerprint import fingerprint
from xaytune.core.harness_evaluation import (
    TIME_METRICS,
    Digest,
    HarnessRunEvidence,
    RestrictedEvaluationReport,
    parse_evaluation_config,
)
from xaytune.core.immutable import FrozenDomainModel, thaw

__all__ = [
    "COMPARISON_SCHEMA",
    "PAIRED_DIFFERENCE",
    "SUMMARY_METHOD",
    "CandidateRuns",
    "Comparability",
    "ComparisonSpec",
    "HarnessComparisonBody",
    "HarnessComparisonReport",
    "HarnessSummary",
    "MetricComparison",
    "MetricSummary",
    "ObjectiveMetric",
    "ReplicateRuns",
    "SummaryRefusedError",
    "TaskOutcomeCounts",
    "comparability",
    "compare",
    "summarize",
]

SUMMARY_METHOD: Final = "replicate-mean/v1"
PAIRED_DIFFERENCE: Final = "paired-difference/v1"
COMPARISON_SCHEMA: Final = "xaytune.harness-comparison/v1alpha1"
_MIN_REPLICATES: Final = 3
"""Fewer execution replicates give no direction at all, not even a provisional one."""


class _Closed(FrozenDomainModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class SummaryRefusedError(ValueError):
    """A summary was asked over results that are not one balanced design. Every reason."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("the replicates cannot be summarized: " + "; ".join(reasons))


# ---- inputs ---------------------------------------------------------------------------------


class ReplicateRuns(_Closed):
    """One execution replicate: its verified evidence and every evaluation of it.

    *results* holds one result per evaluation run: the outcome evaluation's,
    and each judging replicate's. *restricted* holds their restricted
    reports, read only to count paired task outcomes.
    """

    evidence: HarnessRunEvidence
    results: tuple[EvaluationResult, ...]
    restricted: tuple[RestrictedEvaluationReport, ...] = ()


class CandidateRuns(_Closed):
    """Every replicate of one candidate, and the bound evaluation specs that measured them."""

    evaluations: tuple[EvaluationSpec, ...] = Field(min_length=1)
    replicates: tuple[ReplicateRuns, ...] = Field(min_length=1)


# ---- V8: summary ----------------------------------------------------------------------------


class MetricSummary(_Closed):
    """One metric over one candidate's execution replicates (``replicate-mean/v1``).

    ``execution_standard_error`` is ``sd(v_r) / sqrt(R)``: unknown (``None``)
    with one replicate, never 0. ``judge_standard_deviation`` is the pooled
    within-replicate spread of judging runs, reported separately: the two
    sources of variation are never added into one number. No interval is
    reported; a standard error is descriptive.
    """

    metric: str
    slice: str | None = None
    unit: str | None = None
    estimate: float | None = None
    execution_standard_error: float | None = None
    judge_standard_deviation: float | None = None
    replicates: int = Field(strict=True, ge=0)
    judging_runs: int = Field(strict=True, ge=0)
    tasks: int | None = None
    values: tuple[float, ...] = ()
    """``v_r`` per execution replicate, in seed order."""
    absent: str | None = None

    @model_validator(mode="after")
    def _present_or_absent(self) -> MetricSummary:
        if (self.estimate is None) == (self.absent is None):
            raise ValueError("a summary has an estimate or an absence reason, never both")
        return self


class HarnessSummary(_Closed):
    """One candidate under one evaluation fingerprint: every metric, summarized."""

    method: Literal["replicate-mean/v1"] = "replicate-mean/v1"
    evaluation_fingerprint: Digest
    seeds: tuple[int, ...]
    metrics: tuple[MetricSummary, ...]

    def get(self, metric: str, slice: str | None = None) -> MetricSummary | None:
        return next((m for m in self.metrics if (m.metric, m.slice) == (metric, slice)), None)


def summarize(replicates: Sequence[ReplicateRuns], evaluation_fingerprint: str) -> HarnessSummary:
    """Summarize one candidate's results for *evaluation_fingerprint* (V8).

    For each metric (and slice): ``v_r`` is the run value, or the mean of the
    replicate's J judging runs; the estimate is the mean of ``v_r`` over the
    R replicates in seed order (``math.fsum``). A metric that some run lacks,
    or replicates with different task counts, give an absent summary with
    the reason: a summary over whichever replicates happened to finish
    would quietly change what is being estimated.

    Raises:
        SummaryRefusedError: If replicates repeat a seed, or the judging is
            unbalanced (replicates with different numbers of runs).
    """
    ordered = sorted(replicates, key=lambda r: (r.evidence.seed, r.evidence.replicate))
    seeds = tuple(r.evidence.seed for r in ordered)
    runs = [
        tuple(res for res in r.results if res.evaluation_fingerprint == evaluation_fingerprint)
        for r in ordered
    ]
    reasons: list[str] = []
    if len(set(seeds)) != len(seeds):
        reasons.append("a seed has more than one execution replicate")
    counts = {len(found) for found in runs}
    if 0 in counts:
        reasons.append("an execution replicate has no evaluation under this fingerprint")
    if len(counts) > 1:
        reasons.append(
            "judging is unbalanced: execution replicates have different numbers of judging runs"
        )
    if reasons:
        raise SummaryRefusedError(tuple(reasons))
    judging = counts.pop()
    task_counts = {len(r.evidence.entries) for r in ordered}
    task_count = next(iter(task_counts)) if len(task_counts) == 1 else None
    keys = sorted(
        {(m.name, m.slice) for found in runs for res in found for m in res.metrics},
        key=lambda key: (key[0], key[1] or ""),
    )
    summaries: list[MetricSummary] = []
    for name, slice_ in keys:
        base: dict[str, Any] = {
            "metric": name,
            "slice": slice_,
            "replicates": len(ordered),
            "judging_runs": judging,
        }
        if task_count is None:
            summaries.append(
                MetricSummary(**base, absent="unbalanced: replicates ran different task counts")
            )
            continue
        per_replicate: list[list[float]] = []
        units: set[str] = set()
        missing = False
        for found in runs:
            values = []
            for res in found:
                metric = next((m for m in res.metrics if (m.name, m.slice) == (name, slice_)), None)
                if metric is None:
                    missing = True
                    break
                values.append(metric.value)
                units.add(str(metric.metadata.get("unit")))
            per_replicate.append(values)
        if missing:
            summaries.append(
                MetricSummary(**base, absent="unbalanced: some run has no value for this metric")
            )
            continue
        if len(units) != 1:
            summaries.append(MetricSummary(**base, absent="the runs report different units"))
            continue
        v = [math.fsum(values) / len(values) for values in per_replicate]
        execution_se = statistics.stdev(v) / math.sqrt(len(v)) if len(v) >= 2 else None
        judge_sd = None
        if judging >= 2:
            judge_sd = math.sqrt(
                math.fsum(statistics.variance(values) for values in per_replicate)
                / len(per_replicate)
            )
        summaries.append(
            MetricSummary(
                **base,
                unit=units.pop(),
                estimate=math.fsum(v) / len(v),
                execution_standard_error=execution_se,
                judge_standard_deviation=judge_sd,
                tasks=task_count,
                values=tuple(v),
            )
        )
    return HarnessSummary(
        evaluation_fingerprint=evaluation_fingerprint, seeds=seeds, metrics=tuple(summaries)
    )


# ---- V9: comparability ------------------------------------------------------------------------


class Comparability(_Closed):
    """Whether A and B may be compared at all (C1–C4), every failed condition, and the key."""

    comparable: bool
    reasons: tuple[str, ...]
    comparison_key: Digest | None = None
    timing_environment: str | None = None
    """The shared declared timing environment, when every run of both has the same one."""


def _facets(evidence: HarnessRunEvidence) -> Mapping[str, object]:
    """C4's ``exact-facets/v1`` projection: every facet but the timing label (per metric)."""
    facets = evidence.facets.model_dump(mode="json")
    facets.pop("timing_environment")
    return facets


def _by_evaluator(candidate: CandidateRuns) -> dict[str, list[EvaluationSpec]]:
    found: dict[str, list[EvaluationSpec]] = {}
    for spec in candidate.evaluations:
        found.setdefault(spec.evaluator.name, []).append(spec)
    return found


def comparability(
    a: CandidateRuns,
    b: CandidateRuns,
    suite: ResolvedBenchmarkSuite,
    *,
    rule: str = PAIRED_DIFFERENCE,
) -> Comparability:
    """C1–C4 for A and B, every failure at once; the comparison key only when all hold (V9).

    Comparing a candidate with itself is allowed: an A/A comparison is how
    the noise floor is measured. *suite* is the verified suite both claim.
    """
    reasons: list[str] = []
    both = [*a.replicates, *b.replicates]

    # C1: the same benchmark.
    cohorts = {r.evidence.assignment.cohort for r in both}
    if len(cohorts) != 1:
        reasons.append("C1: the replicates ran different benchmark cohorts")
    for side, candidate in (("A", a), ("B", b)):
        for spec in candidate.evaluations:
            config = parse_evaluation_config(thaw(spec.evaluator.config))
            if not isinstance(config.suite, BenchmarkSuiteRef):
                reasons.append(f"C1: {side}'s {spec.evaluator.name} names an unpinned suite")
                continue
            if len(cohorts) == 1 and benchmark_cohort(config.suite, config.sampling) not in cohorts:
                reasons.append(
                    f"C1: {side}'s {spec.evaluator.name} measures another suite or sampling"
                )

    # C2: matched assignments, paired by seed.
    seeds_a = [r.evidence.seed for r in a.replicates]
    seeds_b = [r.evidence.seed for r in b.replicates]
    for side, seeds in (("A", seeds_a), ("B", seeds_b)):
        if len(set(seeds)) != len(seeds):
            reasons.append(f"C2: {side} has more than one execution replicate at a seed")
    for seed in sorted(set(seeds_a) - set(seeds_b)):
        reasons.append(f"C2: B has no replicate at seed {seed}")
    for seed in sorted(set(seeds_b) - set(seeds_a)):
        reasons.append(f"C2: A has no replicate at seed {seed}")
    for side, candidate in (("A", a), ("B", b)):
        for replicate in candidate.replicates:
            evidence = replicate.evidence
            try:
                verify_assignment(suite, evidence.sampling, evidence.assignment)
            except UnverifiedAssignmentError:
                reasons.append(f"C2: {side}'s assignment at seed {evidence.seed} does not verify")
    by_seed_b = {r.evidence.seed: r.evidence for r in b.replicates}
    for replicate in a.replicates:
        other = by_seed_b.get(replicate.evidence.seed)
        if other is not None:
            for mismatch in assignment_mismatches(replicate.evidence.assignment, other.assignment):
                reasons.append(f"C2: at seed {replicate.evidence.seed}: {mismatch}")

    # C3: the same scoring and units.
    specs_a, specs_b = _by_evaluator(a), _by_evaluator(b)
    if set(specs_a) != set(specs_b):
        reasons.append("C3: the candidates were measured by different evaluators")
    for name in sorted(set(specs_a) & set(specs_b)):
        prints_a = {s.evaluation_fingerprint() for s in specs_a[name]}
        prints_b = {s.evaluation_fingerprint() for s in specs_b[name]}
        if len(prints_a) != 1 or len(prints_b) != 1 or prints_a != prints_b:
            reasons.append(f"C3: {name} has different evaluation fingerprints")
        determinism = {s.evaluator.determinism for s in [*specs_a[name], *specs_b[name]]}
        if len(determinism) != 1:
            reasons.append(f"C3: {name} declares different determinism")
    declared = {s.evaluation_fingerprint() for s in [*a.evaluations, *b.evaluations]}
    for side, candidate in (("A", a), ("B", b)):
        for replicate in candidate.replicates:
            for result in replicate.results:
                if result.evaluation_fingerprint not in declared:
                    reasons.append(f"C3: a result of {side} has an undeclared evaluation")
                if result.subject.digest != replicate.evidence.evidence_digest:
                    reasons.append(f"C3: a result of {side} measured another subject")

    # C4: compatible execution conditions, declared and never inferred.
    facets = [_facets(r.evidence) for r in both]
    for name in sorted(facets[0]):
        if len({repr(f[name]) for f in facets}) > 1:
            reasons.append(f"C4: the facet {name} differs across runs")
    labels = {r.evidence.facets.timing_environment for r in both}
    timing = labels.pop() if len(labels) == 1 else None

    reasons = list(dict.fromkeys(reasons))
    if reasons:
        return Comparability(comparable=False, reasons=tuple(reasons), timing_environment=timing)
    key = fingerprint(
        {
            "type": "harness-comparison",
            "version": 1,
            "cohort": next(iter(cohorts)),
            "evaluation_fingerprints": sorted(declared),
            "facets": facets[0],
            "timing_environment": timing,
            "seeds": sorted(seeds_a),
            "summary_method": SUMMARY_METHOD,
            "rule": rule,
        }
    )
    return Comparability(comparable=True, reasons=(), comparison_key=key, timing_environment=timing)


# ---- V10: comparison ----------------------------------------------------------------------------

Verdict = Literal["favors-a", "favors-b", "inconclusive", "not-comparable"]
Relation = Literal["favors-a", "favors-b", "trade-off", "inconclusive", "not-comparable"]


class ObjectiveMetric(_Closed):
    """One metric of the objective vector: its direction is declared, never inferred."""

    metric: str
    slice: str | None = None
    direction: Literal["maximize", "minimize"]
    min_effect: float = Field(ge=0, allow_inf_nan=False)


class ComparisonSpec(_Closed):
    """What is compared and how: an ordered objective vector, the rule and its threshold ``k``."""

    objectives: tuple[ObjectiveMetric, ...] = Field(min_length=1)
    rule: Literal["paired-difference/v1"] = "paired-difference/v1"
    k: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    """A decision threshold on the standard error, not a quantile of any distribution."""

    @model_validator(mode="after")
    def _each_once(self) -> ComparisonSpec:
        keys = [(o.metric, o.slice) for o in self.objectives]
        if len(set(keys)) != len(keys):
            raise ValueError("each objective metric is listed once")
        return self


class TaskOutcomeCounts(_Closed):
    """``task_success`` paired by seed and task, as counts only (V5 governs which task)."""

    wins: int = Field(strict=True, ge=0)
    losses: int = Field(strict=True, ge=0)
    ties: int = Field(strict=True, ge=0)
    unscored: int = Field(strict=True, ge=0)


class MetricComparison(_Closed):
    """One objective metric: both summaries, the paired differences and the verdict."""

    metric: str
    slice: str | None = None
    direction: Literal["maximize", "minimize"]
    min_effect: float
    verdict: Verdict
    a: MetricSummary | None = None
    b: MetricSummary | None = None
    differences: tuple[float, ...] = ()
    """``d_r = v_A,r − v_B,r`` per seed, oriented so that positive favors A."""
    mean_difference: float | None = None
    standard_error: float | None = None
    threshold: float | None = None
    reasons: tuple[str, ...] = ()


class CandidateRecord(_Closed):
    evidence_digests: tuple[Digest, ...]
    result_ids: tuple[str, ...]


class HarnessComparisonBody(_Closed):
    """Everything a comparison report says except its fingerprint."""

    schema_version: Literal["xaytune.harness-comparison/v1alpha1"] = (
        "xaytune.harness-comparison/v1alpha1"
    )
    claim: Literal["exploratory"] = "exploratory"
    rule: Literal["paired-difference/v1"]
    summary_method: Literal["replicate-mean/v1"] = "replicate-mean/v1"
    k: float
    replicates: int
    seeds: tuple[int, ...]
    a: CandidateRecord
    b: CandidateRecord
    comparability: Comparability
    metrics: tuple[MetricComparison, ...]
    relation: Relation
    task_success: TaskOutcomeCounts | None = None
    notes: tuple[str, ...] = ()


class HarnessComparisonReport(HarnessComparisonBody):
    """A deterministic, fingerprinted comparison of two candidates (V10).

    ``claim`` is always ``"exploratory"``: verdicts are provisional directions
    under the named rule, R and ``k``, for choosing what to try next. They are
    not statistical claims about either harness.
    """

    report_fingerprint: Digest


def _summaries(candidate: CandidateRuns) -> list[HarnessSummary]:
    """Every evaluation's summary. An unbalanced design is refused, not skipped."""
    return [
        summarize(candidate.replicates, spec.evaluation_fingerprint())
        for spec in candidate.evaluations
    ]


def _find(
    summaries: Sequence[HarnessSummary], metric: str, slice_: str | None
) -> MetricSummary | None:
    for summary in summaries:
        found = summary.get(metric, slice_)
        if found is not None:
            return found
    return None


def _values_by_seed(candidate: CandidateRuns, summary: MetricSummary) -> dict[int, float]:
    seeds = sorted(r.evidence.seed for r in candidate.replicates)
    return dict(zip(seeds, summary.values))


def _verdict(
    objective: ObjectiveMetric,
    a: CandidateRuns,
    b: CandidateRuns,
    sa: MetricSummary | None,
    sb: MetricSummary | None,
    k: float,
    comparable: Comparability,
) -> MetricComparison:
    base: dict[str, Any] = {
        "metric": objective.metric,
        "slice": objective.slice,
        "direction": objective.direction,
        "min_effect": objective.min_effect,
        "a": sa,
        "b": sb,
    }
    if not comparable.comparable:
        return MetricComparison(**base, verdict="not-comparable", reasons=("the pair fails C1-C4",))
    if objective.metric in TIME_METRICS and comparable.timing_environment is None:
        return MetricComparison(
            **base,
            verdict="not-comparable",
            reasons=("time is compared only under one declared timing environment",),
        )
    if sa is None or sb is None or sa.estimate is None or sb.estimate is None:
        return MetricComparison(
            **base, verdict="not-comparable", reasons=("the metric is absent on a side",)
        )
    if sa.unit != sb.unit:
        return MetricComparison(
            **base, verdict="not-comparable", reasons=("the sides report different units",)
        )
    va, vb = _values_by_seed(a, sa), _values_by_seed(b, sb)
    sign = 1.0 if objective.direction == "maximize" else -1.0
    differences = tuple(sign * (va[seed] - vb[seed]) for seed in sorted(va))
    r = len(differences)
    mean = math.fsum(differences) / r
    se = statistics.stdev(differences) / math.sqrt(r) if r >= 2 else None
    if r < _MIN_REPLICATES or se is None:
        return MetricComparison(
            **base,
            verdict="inconclusive",
            differences=differences,
            mean_difference=mean,
            standard_error=se,
            reasons=(f"R = {r}: at least {_MIN_REPLICATES} replicates give a direction",),
        )
    threshold = max(objective.min_effect, k * se)
    verdict: Verdict = "inconclusive"
    if mean > threshold:
        verdict = "favors-a"
    elif -mean > threshold:
        verdict = "favors-b"
    return MetricComparison(
        **base,
        verdict=verdict,
        differences=differences,
        mean_difference=mean,
        standard_error=se,
        threshold=threshold,
    )


def _relation(verdicts: Sequence[Verdict], comparable: bool) -> Relation:
    if not comparable or "not-comparable" in verdicts:
        return "not-comparable"
    favors_a, favors_b = "favors-a" in verdicts, "favors-b" in verdicts
    if favors_a and favors_b:
        return "trade-off"
    if favors_a:
        return "favors-a"
    if favors_b:
        return "favors-b"
    return "inconclusive"


def _task_outcomes(a: CandidateRuns, b: CandidateRuns) -> TaskOutcomeCounts | None:
    def scores(candidate: CandidateRuns) -> dict[tuple[int, str], float | None]:
        found: dict[tuple[int, str], float | None] = {}
        for replicate in candidate.replicates:
            for report in replicate.restricted:
                for task in report.tasks:
                    observation = task.get("task_success")
                    if observation is not None:
                        found[(replicate.evidence.seed, task.task_id)] = observation.value
        return found

    left, right = scores(a), scores(b)
    if not left or not right:
        return None
    wins = losses = ties = unscored = 0
    for key in sorted(set(left) | set(right)):
        x, y = left.get(key), right.get(key)
        if x is None or y is None:
            unscored += 1
        elif x > y:
            wins += 1
        elif x < y:
            losses += 1
        else:
            ties += 1
    return TaskOutcomeCounts(wins=wins, losses=losses, ties=ties, unscored=unscored)


def compare(
    a: CandidateRuns,
    b: CandidateRuns,
    spec: ComparisonSpec,
    suite: ResolvedBenchmarkSuite,
    *,
    notes: Sequence[str] = (),
) -> HarnessComparisonReport:
    """Compare A with B under *spec*: one verdict per objective metric, and their relation.

    ``paired-difference/v1``: for each seed, ``d_r = v_A,r − v_B,r`` oriented
    so that positive favors A; with R ≥ 3, A is favored when the mean
    difference exceeds ``max(min_effect, k·se)``, B when its negation does,
    and otherwise the metric is inconclusive. Fewer than three replicates
    are always inconclusive.

    Raises:
        SummaryRefusedError: If either candidate's design is unbalanced
            (V8): it is refused, never compared on whatever finished.
    """
    comparable = comparability(a, b, suite, rule=spec.rule)
    summaries_a, summaries_b = _summaries(a), _summaries(b)
    metrics = tuple(
        _verdict(
            objective,
            a,
            b,
            _find(summaries_a, objective.metric, objective.slice),
            _find(summaries_b, objective.metric, objective.slice),
            spec.k,
            comparable,
        )
        for objective in spec.objectives
    )
    relation = _relation([m.verdict for m in metrics], comparable.comparable)
    body = HarnessComparisonBody(
        rule=spec.rule,
        k=spec.k,
        replicates=len(a.replicates),
        seeds=tuple(sorted(r.evidence.seed for r in a.replicates)),
        a=_record(a),
        b=_record(b),
        comparability=comparable,
        metrics=metrics,
        relation=relation,
        task_success=_task_outcomes(a, b),
        notes=tuple(notes),
    )
    report_fingerprint = fingerprint(
        {"type": "harness-comparison-report", "content": body.model_dump(mode="json")}
    )
    return HarnessComparisonReport(
        **{name: getattr(body, name) for name in HarnessComparisonBody.model_fields},
        report_fingerprint=report_fingerprint,
    )


def _record(candidate: CandidateRuns) -> CandidateRecord:
    ordered = sorted(candidate.replicates, key=lambda r: r.evidence.seed)
    return CandidateRecord(
        evidence_digests=tuple(r.evidence.evidence_digest for r in ordered),
        result_ids=tuple(str(res.id) for r in ordered for res in r.results),
    )
