"""Measuring a harness on a pinned benchmark: the pure contracts of H06 (ADR-022).

```text
HarnessTaskBinding        xaytune.harness-task-binding/v1alpha1     config["task"] of a bound spec
HarnessRunEvidence        xaytune.harness-run-evidence/v1alpha1     one execution replicate (V3)
HarnessEvaluationConfig   xaytune.harness-evaluation/v1alpha1       evaluator.config (V4)
metric catalog            xaytune.harness-metrics/v1                units and absences (V6)
HarnessEvaluationReport   xaytune.harness-evaluation-report/v1alpha1 per-task observations (V6)
```

**A score is a measurement of evidence under a declared rule. It is not a
fact about the harness.** Everything here is pure: records, their identities
and the arithmetic over them. Reading a trajectory store, and verifying
evidence against it, live in :mod:`xaytune.harness.evidence`; the evaluators
and the local evaluation driver in :mod:`xaytune.evaluation.harness`;
summaries and comparisons in :mod:`xaytune.core.harness_comparison`.

**Held-out references never appear here.** A binding names a task by its
fingerprint, which hashes the reference's digest; it never carries the
digest, a locator or the bytes (ADR-022 V5, ADR-020 B2).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Annotated, Any, Final, Literal

from pydantic import ConfigDict, Field, StrictInt, TypeAdapter, ValidationError, model_validator

from xaytune.core.agent_execution import (
    COMPLETIONS,
    HarnessRunResult,
    HarnessTaskInput,
    ResolvedContent,
    TerminationReason,
)
from xaytune.core.domain.benchmark import (
    BenchmarkSuiteRef,
    BenchmarkSuiteRequest,
    ResolvedBenchmarkSuite,
    TaskAssignment,
    TaskSampling,
    agent_visible,
)
from xaytune.core.domain.harness import HarnessModelBinding
from xaytune.core.fingerprint import canonical_encode, fingerprint
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw
from xaytune.core.refs import Actor
from xaytune.core.trajectory import Abandonment

__all__ = [
    "ABSENCE_REASONS",
    "CATALOG",
    "EVALUATION_CONFIG_SCHEMA",
    "EVALUATION_REPORT_SCHEMA",
    "METRICS",
    "REFERENCE_DERIVED",
    "RUN_EVIDENCE_SCHEMA",
    "SUCCESSFUL_TASKS",
    "TASK_BINDING_SCHEMA",
    "TIME_METRICS",
    "AbsenceCount",
    "AbsenceReason",
    "Disclosure",
    "HarnessEvaluationConfig",
    "HarnessEvaluationReport",
    "HarnessJudgeConfig",
    "HarnessOutcomeConfig",
    "JudgeExchange",
    "MetricObservation",
    "ReferenceEgressGrant",
    "RestrictedEvaluationReport",
    "RunValue",
    "ScoreScale",
    "SuccessRule",
    "TaskObservations",
    "aggregate_run",
    "canonical_config_refusals",
    "execution_observations",
    "metric_unit",
    "parse_evaluation_config",
    "success_observation",
    "EvidenceAttempt",
    "EvidenceEntry",
    "EvidenceOutcome",
    "ExecutionFacets",
    "HarnessRunEvidence",
    "HarnessRunEvidenceBody",
    "HarnessTaskBinding",
    "bind_task",
    "configuration_digest",
    "run_evidence_digest",
    "run_evidence_identity_v1",
    "task_binding_refusals",
]

TASK_BINDING_SCHEMA: Final = "xaytune.harness-task-binding/v1alpha1"
RUN_EVIDENCE_SCHEMA: Final = "xaytune.harness-run-evidence/v1alpha1"

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
Digest = Annotated[str, Field(pattern=_DIGEST.pattern)]


class _Closed(FrozenDomainModel):
    """Frozen, closed to unknown fields, and silent about refused input."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


# ---- V2: wiring a pinned task into execution -------------------------------------------------


class HarnessTaskBinding(_Closed):
    """Which pinned task a spec runs: carried in ``config["task"]`` (ADR-022 V2).

    Execution context, never harness identity. It enters the execution
    fingerprint and request digest because ``config`` does, and never
    ``HarnessFingerprint`` (ADR-020 B8). It names the task by fingerprint
    only: no reference digest, locator or bytes.
    """

    schema_version: Literal["xaytune.harness-task-binding/v1alpha1"] = (
        "xaytune.harness-task-binding/v1alpha1"
    )
    suite: BenchmarkSuiteRef
    task_id: str = Field(min_length=1, max_length=200)
    revision: str = Field(min_length=1, max_length=128)
    task_fingerprint: Digest
    position: int = Field(strict=True, ge=0)
    """The task's position in the suite manifest."""
    cohort: Digest
    """The assignment's benchmark cohort (ADR-020 B7)."""


def bind_task(
    suite: ResolvedBenchmarkSuite, assignment: TaskAssignment, task_id: str
) -> tuple[HarnessTaskInput, HarnessTaskBinding]:
    """The task input and binding for running *task_id* of *assignment* on *suite*.

    The input is exactly the task's agent-visible variables. The caller must
    have verified *assignment* (``verify_assignment``) first.

    Raises:
        KeyError: If *task_id* is not in the assignment or the suite.
    """
    if task_id not in assignment.task_ids:
        raise KeyError("the task is not in the assignment")
    task = suite.task(task_id)
    position = next(i for i, m in enumerate(suite.manifest.members) if m.task_id == task_id)
    task_input = HarnessTaskInput(task_id=task.id, variables=agent_visible(task).variables)
    binding = HarnessTaskBinding(
        suite=suite.ref,
        task_id=task.id,
        revision=task.revision,
        task_fingerprint=suite.manifest.members[position].task_fingerprint,
        position=position,
        cohort=assignment.cohort,
    )
    return task_input, binding


def task_binding_refusals(
    binding: HarnessTaskBinding, task: HarnessTaskInput, suite: ResolvedBenchmarkSuite
) -> tuple[str, ...]:
    """Every way *binding* and *task* disagree with the resolved *suite*. ``()`` if none.

    Refused, every reason listed (ADR-022 V2):

    - a binding for another suite;
    - a task fingerprint or position that disagrees with the suite manifest;
    - an id or variables that are not the task's agent-visible inputs;
    - a task with attachments: ``HarnessTaskInput`` has no place for them, and
      an input is never silently omitted.
    """
    reasons: list[str] = []
    if binding.suite != suite.ref:
        reasons.append("the binding names another suite than the resolved one")
    members = suite.manifest.members
    if binding.position >= len(members):
        reasons.append("the binding's position is beyond the suite manifest")
    else:
        member = members[binding.position]
        if (member.task_id, member.revision) != (binding.task_id, binding.revision):
            reasons.append("the binding's position is not that task's place in the suite")
        if member.task_fingerprint != binding.task_fingerprint:
            reasons.append("the binding's task fingerprint is not the suite's pin for that task")
    if task.task_id != binding.task_id:
        reasons.append("the task input's id is not the bound task's")
    try:
        resolved = suite.task(binding.task_id)
    except KeyError:
        reasons.append("the bound task is not a member of the suite")
    else:
        visible = agent_visible(resolved)
        if task.variables != visible.variables:
            reasons.append("the task input's variables are not the task's agent-visible variables")
        if visible.attachments:
            reasons.append(
                "the task has attachments, and a harness task input has no place for them: "
                "admitting them needs a versioned execution input (deferred), and an input "
                "is never silently omitted"
            )
    return tuple(reasons)


# ---- V3: the subject -- one execution replicate's evidence -----------------------------------


class ExecutionFacets(_Closed):
    """The declared conditions a replicate ran under: ADR-022 V9's C4 facets.

    Declared, never inferred from an execution fingerprint, which always
    differs between candidates. The runner, recording spec, token counter and
    compiler are rechecked against the trajectories and attested results by
    ``verify_run_evidence``; the rest are declared by assembly from the
    specs. ``timing_environment`` is a label the host declares, or ``None``:
    time metrics are compared only under the same declared label.
    """

    runner_name: str = Field(min_length=1)
    runner_version: str = Field(min_length=1)
    compiler_name: str = Field(min_length=1)
    compiler_version: str = Field(min_length=1)
    token_counter: str = Field(min_length=1)
    entrypoint: Digest
    container: Digest
    dependencies: Digest
    resources: Digest
    recording: Digest
    timing_environment: str | None = Field(default=None, min_length=1, max_length=200)


class EvidenceAttempt(_Closed):
    """One attempt instance of a task's request digest, as sealed."""

    trajectory_id: str = Field(min_length=1, max_length=200)
    trajectory_fingerprint: Digest
    completeness: Literal["complete", "partial"]
    abandonment: Abandonment | None = None


class EvidenceOutcome(_Closed):
    """The attempt whose result an ``attested`` entry is scored from."""

    trajectory_id: str = Field(min_length=1, max_length=200)
    trajectory_fingerprint: Digest
    result_digest: Digest
    termination: TerminationReason


class EvidenceEntry(_Closed):
    """One assigned task: every attempt instance of its request digest, and its state.

    ``attested``: one attempt is sealed ``complete`` and verifies; it is the
    outcome attempt, and the others are retries that count toward spend
    only. ``no-attested-result``: every attempt is sealed ``partial`` and
    verifies; the outcome is absent, the spend is kept (ADR-022 V3).

    Parsing checks shape only. Whether an entry is true -- its attempts, its
    state, its attribution -- is ``verify_run_evidence``'s to decide, with
    every reason, so nothing here refuses a record a verifier should report.
    """

    task_id: str = Field(min_length=1, max_length=200)
    revision: str = Field(min_length=1, max_length=128)
    task_fingerprint: Digest
    position: int = Field(strict=True, ge=0)
    request_digest: Digest
    attempts: tuple[EvidenceAttempt, ...]
    state: Literal["attested", "no-attested-result"]
    outcome: EvidenceOutcome | None = None


class HarnessRunEvidenceBody(_Closed):
    """Everything an evidence record says except its digest."""

    schema_version: Literal["xaytune.harness-run-evidence/v1alpha1"] = (
        "xaytune.harness-run-evidence/v1alpha1"
    )
    suite: BenchmarkSuiteRef
    sampling: TaskSampling
    assignment: TaskAssignment
    harness_fingerprint: Digest
    seed: int = Field(strict=True)
    replicate: int = Field(strict=True, ge=0)
    facets: ExecutionFacets
    configuration: FrozenDict
    """The harness's context, stopping and generation settings: what every
    attempt's ``run-started`` names by digest, so a verifier rechecks it."""
    entries: tuple[EvidenceEntry, ...]


class HarnessRunEvidence(HarnessRunEvidenceBody):
    """One execution replicate of one candidate: the subject an evaluation measures (V3).

    ``evidence_digest`` is :func:`run_evidence_digest` of the body: the
    subject's ``ArtifactRef.digest`` and the subject half of every reuse key.
    A digest proves self-consistency only; an edited record can be re-hashed.
    So no consumer trusts it: ``verify_run_evidence`` rebuilds every fact
    from the suite and the store.
    """

    evidence_digest: Digest

    @property
    def token_budget(self) -> int | None:
        """The harness's ``max_total_tokens``, from the verified configuration."""
        stopping = self.configuration.get("stopping")
        value = stopping.get("max_total_tokens") if isinstance(stopping, Mapping) else None
        return value if isinstance(value, int) and not isinstance(value, bool) else None


def configuration_digest(configuration: Mapping[str, Any]) -> str:
    """The digest a ``run-started`` event names the harness configuration by."""
    return fingerprint({"type": "harness-run-configuration", "configuration": configuration})


def run_evidence_identity_v1(evidence: HarnessRunEvidenceBody) -> Mapping[str, Any]:
    """The frozen v1 projection an evidence record is hashed by.

    Everything the record states: the suite, sampling and assignment; the
    harness, seed and replicate; the facets and configuration; and every
    entry with every attempt. Nothing is left out, because every field is
    something a measurement depends on.
    """
    return {
        "type": "harness-run-evidence",
        "version": 1,
        "suite": evidence.suite.model_dump(mode="json"),
        "sampling": evidence.sampling.model_dump(mode="json"),
        "assignment": evidence.assignment.model_dump(mode="json"),
        "harness_fingerprint": evidence.harness_fingerprint,
        "seed": evidence.seed,
        "replicate": evidence.replicate,
        "facets": evidence.facets.model_dump(mode="json"),
        "configuration": evidence.configuration,
        "entries": [entry.model_dump(mode="json") for entry in evidence.entries],
    }


def run_evidence_digest(evidence: HarnessRunEvidenceBody) -> str:
    """:func:`run_evidence_identity_v1`, hashed."""
    return fingerprint(run_evidence_identity_v1(evidence))


# ---- V4: the evaluators' typed configuration ------------------------------------------------

EVALUATION_CONFIG_SCHEMA: Final = "xaytune.harness-evaluation/v1alpha1"
CATALOG: Final = "xaytune.harness-metrics/v1"

Disclosure = Literal["aggregate", "per-task"]
"""Who sees reference-derived measurements (V5). Outside ``EvaluationFingerprint``."""


class SuccessRule(_Closed):
    """How an attested answer is judged against the held-out reference."""

    kind: Literal["exact-match"]
    normalization: Literal["none", "strip", "casefold-strip"]

    def matches(self, answer: str, reference: str) -> bool:
        if self.normalization == "strip":
            return answer.strip() == reference.strip()
        if self.normalization == "casefold-strip":
            return answer.strip().casefold() == reference.strip().casefold()
        return answer == reference


class ScoreScale(_Closed):
    """The judge's integer scale, both ends included."""

    minimum: StrictInt
    maximum: StrictInt

    @model_validator(mode="after")
    def _ordered(self) -> ScoreScale:
        if self.maximum <= self.minimum:
            raise ValueError("a score scale's maximum is above its minimum")
        return self


class _EvaluationConfig(_Closed):
    schema_version: Literal["xaytune.harness-evaluation/v1alpha1"]
    suite: BenchmarkSuiteRef | BenchmarkSuiteRequest
    """A request (an alias) until ``resolve()`` pins it to a ref."""
    sampling: TaskSampling
    catalog: Literal["xaytune.harness-metrics/v1"]
    success: SuccessRule
    partial_evidence: Literal["refuse"]
    timing_environment: str | None = Field(min_length=1, max_length=200)


class HarnessOutcomeConfig(_EvaluationConfig):
    """``evaluator.config`` of ``harness-outcome``. Every field explicit (canonical form)."""

    evaluator: Literal["harness-outcome"]


class HarnessJudgeConfig(_EvaluationConfig):
    """``evaluator.config`` of ``harness-judge``: the outcome config's pins, and the judge's.

    ``reference_access`` says what the measurement uses, and enters the
    fingerprint. It does not permit sending the reference anywhere: that
    needs a :class:`ReferenceEgressGrant` (V5).
    """

    evaluator: Literal["harness-judge"]
    judge_model: HarnessModelBinding
    prompt: ResolvedContent
    rubric: ResolvedContent
    scale: ScoreScale
    reference_access: Literal["none", "reference"]


HarnessEvaluationConfig = Annotated[
    HarnessOutcomeConfig | HarnessJudgeConfig, Field(discriminator="evaluator")
]


def parse_evaluation_config(
    config: Mapping[str, Any],
) -> HarnessOutcomeConfig | HarnessJudgeConfig:
    """*config* as its typed model. Raises ``pydantic.ValidationError`` if it is not one."""
    adapter: TypeAdapter[HarnessOutcomeConfig | HarnessJudgeConfig] = TypeAdapter(
        HarnessEvaluationConfig
    )
    return adapter.validate_python(dict(config))


def canonical_config_refusals(config: Mapping[str, Any], expected: str) -> tuple[str, ...]:
    """Every reason *config* is not exactly the canonical form of *expected*'s config (G17).

    Two configs that mean the same thing must not have two fingerprints, so a
    default left out, an extra key or any non-canonical spelling is refused:
    the config must equal its own typed model's dump, field for field.
    """
    raw = thaw(config)
    try:
        parsed = parse_evaluation_config(raw)
    except ValidationError as error:
        return tuple(
            f"evaluator.config.{'.'.join(str(part) for part in e['loc']) or 'config'}: {e['msg']}"
            for e in error.errors()
        )
    reasons: list[str] = []
    if parsed.evaluator != expected:
        reasons.append(f"evaluator.config is a {parsed.evaluator!r} config, not {expected!r}")
    if canonical_encode(parsed.model_dump(mode="json")) != canonical_encode(raw):
        # Typed, as the fingerprint is: ``0`` and ``0.0`` compare equal in Python
        # and encode differently, so they would be two fingerprints.
        reasons.append(
            "evaluator.config is not in canonical form: every field explicit, nothing extra, "
            "exactly as its typed model writes it"
        )
    return tuple(reasons)


class ReferenceEgressGrant(_Closed):
    """Permission to send a suite's held-out references to one pinned judge model (V5).

    Supplied by the host at evaluation time, recorded with the run's report
    and kept outside ``EvaluationFingerprint``: who may send data where is a
    different question from what is measured. Required for every judge with
    ``reference_access="reference"``, local or hosted.
    """

    suite: BenchmarkSuiteRef
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    granted_by: Actor
    granted_at: datetime

    def covers(self, config: HarnessJudgeConfig) -> bool:
        """Whether this grant names exactly *config*'s suite and pinned judge model."""
        binding = config.judge_model
        return (
            isinstance(config.suite, BenchmarkSuiteRef)
            and self.suite == config.suite
            and (self.provider, self.model, self.revision)
            == (binding.provider, binding.model, binding.revision)
        )


# ---- V6: the metric catalog ------------------------------------------------------------------

AbsenceReason = Literal[
    "no-attested-result",
    "unscored-cancelled",
    "judge-failed",
    "not-recorded",
    "not-applicable",
    "spend-unaccounted",
]
ABSENCE_REASONS: Final = frozenset(AbsenceReason.__args__)  # type: ignore[attr-defined]

EXECUTION_METRICS: Final = (
    "prompt_tokens",
    "output_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "cost",
    "model_latency_ms",
    "elapsed_ms",
    "turns",
    "tool_calls",
    "failed_tool_calls",
    "refused_tool_calls",
    "retries",
    "context_truncations",
    "peak_prompt_tokens",
    "token_budget_utilization",
    "spend_total_tokens",
)
METRICS: Final = ("task_success", "quality", *EXECUTION_METRICS)
"""Every metric of ``xaytune.harness-metrics/v1``, in catalog order."""
REFERENCE_DERIVED: Final = frozenset({"task_success", "quality"})
"""Computed against a held-out reference: their per-task values are restricted (V5)."""
TIME_METRICS: Final = frozenset({"model_latency_ms", "elapsed_ms"})
SUCCESSFUL_TASKS: Final = "successful-tasks"
"""The slice that averages over the tasks that succeeded only."""

_NOT_RECORDED_IN_V1: Final = frozenset({"cached_tokens", "reasoning_tokens", "cost", "elapsed_ms"})


def metric_unit(name: str, *, token_counter: str, rubric_digest: str | None = None) -> str:
    """The unit of *name* in catalog v1. The token counter is part of every token unit."""
    tokens = f"tokens({token_counter})"
    units = {
        "task_success": "fraction",
        "quality": f"rubric:{rubric_digest} points",
        "prompt_tokens": tokens,
        "output_tokens": tokens,
        "total_tokens": tokens,
        "cached_tokens": "provider-tokens",
        "reasoning_tokens": "provider-tokens",
        "cost": "currency",
        "model_latency_ms": "ms",
        "elapsed_ms": "ms",
        "turns": "count",
        "tool_calls": "count",
        "failed_tool_calls": "count",
        "refused_tool_calls": "count",
        "retries": "count",
        "context_truncations": "count",
        "peak_prompt_tokens": tokens,
        "token_budget_utilization": "fraction",
        "spend_total_tokens": tokens,
    }
    return units[name]


class MetricObservation(_Closed):
    """One metric on one task: a value, or an absence with its reason. Never both, never neither."""

    metric: str
    value: float | None = None
    absent: AbsenceReason | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> MetricObservation:
        if (self.value is None) == (self.absent is None):
            raise ValueError("an observation is a value or an absence with a reason, never both")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("an observed value is finite")
        return self


def _value(metric: str, value: float) -> MetricObservation:
    return MetricObservation(metric=metric, value=float(value))


def _absent(metric: str, reason: AbsenceReason) -> MetricObservation:
    return MetricObservation(metric=metric, absent=reason)


def success_observation(
    result: HarnessRunResult | None, reference: str | None, rule: SuccessRule
) -> MetricObservation:
    """``task_success`` for one task, by its termination (V6).

    ``final-answer``/``no-tool-call``: the success rule decides. A budget stop,
    a context overflow or a model error: 0, an attested outcome of the
    harness's own rules. ``cancelled``: absent, not the candidate's outcome.
    No attested result: absent.
    """
    if result is None:
        return _absent("task_success", "no-attested-result")
    if result.termination == "cancelled":
        return _absent("task_success", "unscored-cancelled")
    if result.termination in COMPLETIONS:
        if result.answer is None or reference is None:  # pragma: no cover - checked upstream
            raise ValueError("a completed task with a reference is scored on its answer")
        return _value("task_success", 1.0 if rule.matches(result.answer, reference) else 0.0)
    return _value("task_success", 0.0)


def _model_calls(events: Iterable[Any]) -> list[Any]:
    return [e for e in events if getattr(e, "kind", None) == "model-call-finished"]


def _replies(events: Iterable[Any]) -> list[Any]:
    return [e for e in _model_calls(events) if e.outcome == "replied"]


def _unfinished_model_calls(events: Sequence[Any]) -> bool:
    finished = {e.parent for e in _model_calls(events)}
    return any(
        getattr(e, "kind", None) == "model-call-started" and e.sequence not in finished
        for e in events
    )


def execution_observations(
    result: HarnessRunResult | None,
    outcome_events: Sequence[Any],
    attempts: Sequence[tuple[Abandonment | None, Sequence[Any]]],
    token_budget: int | None,
) -> tuple[tuple[MetricObservation, ...], int]:
    """Every execution metric for one task, and the spend's known lower bound (V6).

    *result* and *outcome_events* are the outcome attempt's (``None`` and
    ``()`` with no attested result). *attempts* is every attempt -- outcome
    and retries -- as (abandonment, recorded events). Execution measurements
    read the evidence alone and reveal nothing about a reference.
    """
    found: list[MetricObservation] = []
    attested = result is not None
    for name in ("prompt_tokens", "output_tokens", "total_tokens"):
        found.append(
            _value(name, getattr(result, name)) if attested else _absent(name, "no-attested-result")
        )
    for name in ("cached_tokens", "reasoning_tokens", "cost"):
        found.append(_absent(name, "not-recorded"))
    if not attested:
        found.append(_absent("model_latency_ms", "no-attested-result"))
    else:
        latencies = [e.latency_ms for e in _model_calls(outcome_events)]
        if any(latency is None for latency in latencies):
            found.append(_absent("model_latency_ms", "not-recorded"))
        else:
            found.append(_value("model_latency_ms", math.fsum(latencies)))
    found.append(_absent("elapsed_ms", "not-recorded"))
    for name in ("turns", "tool_calls", "failed_tool_calls", "refused_tool_calls"):
        found.append(
            _value(name, getattr(result, name)) if attested else _absent(name, "no-attested-result")
        )
    found.append(_value("retries", len(attempts) - 1))
    if attested:
        assert result is not None
        found.append(_value("context_truncations", result.context_truncations))
        peaks = [e.prompt_tokens for e in _replies(outcome_events)]
        found.append(_value("peak_prompt_tokens", max(peaks, default=0)))
    else:
        found.append(_absent("context_truncations", "no-attested-result"))
        found.append(_absent("peak_prompt_tokens", "no-attested-result"))
    if token_budget is None:
        found.append(_absent("token_budget_utilization", "not-applicable"))
    elif not attested:
        found.append(_absent("token_budget_utilization", "no-attested-result"))
    else:
        assert result is not None
        found.append(_value("token_budget_utilization", result.total_tokens / token_budget))
    spent = sum(
        e.prompt_tokens + e.output_tokens for _, events in attempts for e in _replies(events)
    )
    unaccounted = any(
        abandonment == "post-commit-evidence-lost" or _unfinished_model_calls(events)
        for abandonment, events in attempts
    )
    found.append(
        _absent("spend_total_tokens", "spend-unaccounted")
        if unaccounted
        else _value("spend_total_tokens", spent)
    )
    return tuple(found), spent


class TaskObservations(_Closed):
    """Every observation for one assigned task, in catalog order."""

    task_id: str
    position: int = Field(strict=True, ge=0)
    observations: tuple[MetricObservation, ...]

    def get(self, metric: str) -> MetricObservation | None:
        return next((o for o in self.observations if o.metric == metric), None)


class RunValue(_Closed):
    """A run-level value: the mean over the tasks it covers (``math.fsum``)."""

    metric: str
    slice: str | None = None
    value: float
    sample_count: int = Field(strict=True, ge=1)
    unit: str


class AbsenceCount(_Closed):
    """How many tasks a metric was absent on, and why."""

    metric: str
    reason: AbsenceReason
    tasks: int = Field(strict=True, ge=1)


def aggregate_run(
    tasks: Sequence[TaskObservations], metrics: Sequence[str], units: Mapping[str, str]
) -> tuple[tuple[RunValue, ...], tuple[AbsenceCount, ...]]:
    """Run values and absence counts over *tasks*, in assignment order (V6).

    A metric absent on **any** task has no run value: no imputation, and no
    denominator shrunk to the tasks that happened to be observed. The
    ``successful-tasks`` slice averages over the tasks whose ``task_success``
    is 1, only when every task was scored and at least one succeeded.
    """
    values: list[RunValue] = []
    counts: list[AbsenceCount] = []
    succeeded: list[TaskObservations] | None = None
    scored = [t.get("task_success") for t in tasks]
    if scored and all(o is not None and o.value is not None for o in scored):
        succeeded = [t for t, o in zip(tasks, scored) if o is not None and o.value == 1.0]
    for metric in metrics:
        observed = [t.get(metric) for t in tasks]
        reasons: dict[str, int] = {}
        for o in observed:
            if o is not None and o.absent is not None:
                reasons[o.absent] = reasons.get(o.absent, 0) + 1
        counts.extend(
            AbsenceCount(metric=metric, reason=reason, tasks=n)  # type: ignore[arg-type]
            for reason, n in sorted(reasons.items())
        )
        if reasons or not observed or any(o is None for o in observed):
            continue
        numbers = [o.value for o in observed if o is not None and o.value is not None]
        values.append(
            RunValue(
                metric=metric,
                value=math.fsum(numbers) / len(numbers),
                sample_count=len(numbers),
                unit=units[metric],
            )
        )
        if succeeded and metric not in REFERENCE_DERIVED:
            subset = [t.get(metric) for t in succeeded]
            if all(o is not None and o.value is not None for o in subset):
                picked = [o.value for o in subset if o is not None and o.value is not None]
                values.append(
                    RunValue(
                        metric=metric,
                        slice=SUCCESSFUL_TASKS,
                        value=math.fsum(picked) / len(picked),
                        sample_count=len(picked),
                        unit=units[metric],
                    )
                )
    return tuple(values), tuple(counts)


# ---- the report: released, and restricted (V5, V6) -------------------------------------------

EVALUATION_REPORT_SCHEMA: Final = "xaytune.harness-evaluation-report/v1alpha1"


class JudgeExchange(_Closed):
    """What one task's judge was asked, and answered. Restricted: it may hold the reference."""

    task_id: str
    request_fingerprint: Digest
    system: str
    message: str
    reply: FrozenDict | None
    failure: str | None = None


class HarnessEvaluationReport(_Closed):
    """What one evaluation run measured, as released under its disclosure policy (V5, V6).

    Attached to the ``EvaluationResult`` as an ``evaluation_report``. Holds
    every run value, every absence with its reason, the spend lower bound,
    and the per-task observations a reader may see: execution measurements
    always; reference-derived ones only under ``per-task`` disclosure. A
    run-level reference-derived value over a single task equals that task's
    score, so under ``aggregate`` it is withheld and named in ``withheld``.
    """

    schema_version: Literal["xaytune.harness-evaluation-report/v1alpha1"] = (
        "xaytune.harness-evaluation-report/v1alpha1"
    )
    evaluator: str
    evaluator_version: str
    evaluation_fingerprint: Digest
    evidence_digest: Digest
    seed: int | None
    replicate: int | None
    catalog: Literal["xaytune.harness-metrics/v1"] = "xaytune.harness-metrics/v1"
    disclosure: Disclosure
    grant: ReferenceEgressGrant | None = None
    run_values: tuple[RunValue, ...]
    absences: tuple[AbsenceCount, ...]
    withheld: tuple[str, ...] = ()
    """Run values a disclosure rule kept out of release, as ``metric`` or ``metric/slice``."""
    tasks: tuple[TaskObservations, ...]
    spend_lower_bound: int = Field(strict=True, ge=0)
    """Tokens every attempt is known to have spent: a lower bound, never a ``MetricResult``."""
    failure: str | None = None
    """Why the run produced no ``EvaluationResult``, when it produced none."""
    claim: Literal["measurement"] = "measurement"


class RestrictedEvaluationReport(_Closed):
    """What only the restricted tier may hold (V5, ADR-021 T8).

    Every per-task reference-derived observation, every run value before
    disclosure (including what the released report withheld), and every
    judge exchange. Never copied into a summary, a metric, a comparison
    report or a provider's input.
    """

    schema_version: Literal["xaytune.harness-evaluation-restricted/v1alpha1"] = (
        "xaytune.harness-evaluation-restricted/v1alpha1"
    )
    evaluation_fingerprint: Digest
    evidence_digest: Digest
    run_values: tuple[RunValue, ...] = ()
    """Every run value, before the disclosure policy."""
    tasks: tuple[TaskObservations, ...]
    judge: tuple[JudgeExchange, ...] = ()
