"""Evaluating harness runs: two evaluators and a local driver (H06, ADR-022 V4).

```text
HarnessRunEvidence ─publish_evidence→ subject ArtifactRef (agent_run_evidence)
EvaluationSpec(harness-outcome | harness-judge) ─bind_evaluation (supports→resolve→supports)→ bound
run_harness_evaluation: EvaluationRun → EvaluationAttempt → prepare → execute → EvaluationResult
  execute: verify_run_evidence FIRST, then measure, then report (released + restricted)
```

**No new evaluation engine.** Both evaluators implement the existing
:class:`~xaytune.evaluation.Evaluator` protocol (and
:class:`~xaytune.evaluation.ResolvableEvaluator`), and the driver walks the
existing ``EvaluationRun``/``EvaluationAttempt`` state machines and refuses a
result ``result_provenance_problems`` objects to. Until the durable harness
path exists (ADR-019 M1–M6), execution is local and in-process and writes no
repository rows.

- ``harness-outcome`` is ``DETERMINISTIC``: task success by the declared
  rule, and every execution metric of ``xaytune.harness-metrics/v1``.
- ``harness-judge`` is ``STOCHASTIC``: quality, by a pinned judge model
  against a pinned rubric, asked through
  :func:`~xaytune.agent.invoke_agent_model` for one integer on the scale.

**Every run verifies its subject itself**, at the start of execution and
never in ``prepare()``, which must stay deterministic: it lists the store
afresh and never relies on assembly, or another run, having verified the
same digest.

**Held-out references stay on the evaluation side** (V5). Only ``execute``
resolves them. Per-task reference-derived scores and every judge exchange go
to the restricted report only; the released report carries them per task
only under ``per-task`` disclosure.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import Field, ValidationError

from xaytune._version import __version__
from xaytune.agent import (
    AgentMessage,
    AgentModel,
    AgentModelError,
    AgentModelIdentity,
    AgentModelRequest,
    invoke_agent_model,
)
from xaytune.benchmarks import (
    BenchmarkSource,
    UnverifiedBenchmarkError,
    resolve_benchmark_suite,
    resolve_reference,
)
from xaytune.core.agent_execution import COMPLETIONS, HarnessRunResult
from xaytune.core.capabilities import PLUGIN_API_VERSIONS, CapabilityDocument, PluginDescriptor
from xaytune.core.domain.benchmark import (
    BenchmarkSuiteRef,
    ResolvedBenchmarkSuite,
    agent_visible,
)
from xaytune.core.domain.evaluation import (
    EvaluationAttempt,
    EvaluationResult,
    EvaluationRun,
    EvaluationSpec,
    EvaluatorDeterminism,
    EvaluatorSpec,
    MetricResult,
    result_provenance_problems,
)
from xaytune.core.execution import (
    ArtifactOutput,
    EvaluationExecutionSpec,
    EvaluatorIdentity,
    PythonModuleEntrypoint,
    ResourceRequirements,
)
from xaytune.core.harness_evaluation import (
    CATALOG,
    EXECUTION_METRICS,
    REFERENCE_DERIVED,
    Disclosure,
    HarnessEvaluationReport,
    HarnessJudgeConfig,
    HarnessOutcomeConfig,
    HarnessRunEvidence,
    JudgeExchange,
    MetricObservation,
    ProviderFeedback,
    ReferenceEgressGrant,
    RestrictedEvaluationReport,
    RunValue,
    TaskObservations,
    aggregate_run,
    canonical_config_refusals,
    execution_observations,
    metric_unit,
    parse_evaluation_config,
    success_observation,
)
from xaytune.core.ids import (
    ArtifactId,
    EvaluationAttemptId,
    EvaluationId,
    EvaluationRunId,
    ExperimentId,
    ExperimentNodeId,
)
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw
from xaytune.core.refs import ArtifactRef
from xaytune.core.state.status import EvaluationAttemptStatus, EvaluationRunStatus
from xaytune.core.support import SupportResult
from xaytune.evaluation import (
    EvaluationContext,
    ResolvableEvaluator,
    UnsupportedEvaluationError,
)
from xaytune.harness.evidence import (
    EvidenceRefusedError,
    HarnessEvidenceStore,
    ListedAttempt,
    verify_run_evidence,
)
from xaytune.trajectory._files import make_directory, read_bytes, write_new
from xaytune.trajectory.verify import load_events

__all__ = [
    "EVIDENCE_FILE",
    "HARNESS_JUDGE",
    "HARNESS_OUTCOME",
    "HarnessEvaluation",
    "HarnessJudgeEvaluator",
    "HarnessOutcomeEvaluator",
    "bind_evaluation",
    "execute_harness_evaluation",
    "provider_feedback",
    "publish_evidence",
    "run_harness_evaluation",
]

HARNESS_OUTCOME: Final = "harness-outcome"
HARNESS_JUDGE: Final = "harness-judge"
EVIDENCE_FILE: Final = "run-evidence.json"
"""Where :func:`publish_evidence` puts a replicate's evidence: at its store's root."""
_WORKER_API: Final = "xaytune.harness-evaluation-worker/v1alpha1"
_ENTRYPOINT: Final = PythonModuleEntrypoint(
    module="xaytune.evaluation.harness", function="execute_harness_evaluation"
)


# ---- the subject ---------------------------------------------------------------------------


def publish_evidence(evidence: HarnessRunEvidence, store: HarnessEvidenceStore) -> ArtifactRef:
    """Write *evidence* at its store's root, once, and return the subject that names it.

    The subject's digest is the evidence digest: the subject half of every
    reuse key (ADR-022 V11). Publishing again writes nothing if the bytes are
    identical, and refuses anything else.
    """
    location = store.root / EVIDENCE_FILE
    data = evidence.model_dump_json().encode("utf-8")
    if not write_new(location, data) and read_bytes(location) != data:
        raise FileExistsError("other evidence is already published at this store's root")
    return ArtifactRef(
        id=ArtifactId.generate(),
        kind="agent_run_evidence",
        uri=str(location),
        digest=evidence.evidence_digest,
    )


# ---- the evaluators ------------------------------------------------------------------------


class _WorkerConfig(FrozenDomainModel):
    """What execution needs beyond the subject: everything ``prepare()`` decided."""

    api_version: Literal["xaytune.harness-evaluation-worker/v1alpha1"]
    evaluator: str
    evaluator_version: str
    config: FrozenDict
    disclosure: Literal["aggregate", "per-task"]
    grant: ReferenceEgressGrant | None = None
    evidence_uri: str
    output_dir: str
    seed: int | None = None
    replicate: int | None = Field(default=None, ge=0)


def _disclosure(spec: EvaluationSpec) -> tuple[Disclosure | None, str | None]:
    value = spec.metadata.get("disclosure", "aggregate")
    if value in ("aggregate", "per-task"):
        return value, None
    return None, "metadata.disclosure is 'aggregate' (the default) or 'per-task'"


class _HarnessEvaluator:
    name: str
    determinism: EvaluatorDeterminism
    descriptor: PluginDescriptor

    def __init__(self, source: BenchmarkSource) -> None:
        self._source = source

    def capabilities(self) -> CapabilityDocument:
        """One local process; nothing beyond it."""
        return CapabilityDocument()

    def supports(self, spec: EvaluationSpec) -> SupportResult:
        """Whether *spec* can be measured exactly as declared. Every reason (G17)."""
        reasons = tuple(self._refusals(spec))
        return SupportResult(supported=not reasons, reasons=reasons)

    def resolve(self, spec: EvaluationSpec) -> EvaluationSpec:
        """*spec* with its suite alias pinned to a ``BenchmarkSuiteRef``.

        Raises:
            UnsupportedEvaluationError: If the suite cannot be resolved.
        """
        config = parse_evaluation_config(thaw(spec.evaluator.config))
        if isinstance(config.suite, BenchmarkSuiteRef):
            return spec
        try:
            resolved = resolve_benchmark_suite(config.suite, self._source)
        except UnverifiedBenchmarkError as error:
            raise UnsupportedEvaluationError(self.name, error.reasons) from None
        pinned = config.model_copy(update={"suite": resolved.ref})
        evaluator = spec.evaluator.model_copy(
            update={"config": FrozenDict(pinned.model_dump(mode="json"))}
        )
        return spec.model_copy(update={"evaluator": evaluator})

    def _refusals(self, spec: EvaluationSpec) -> list[str]:
        reasons: list[str] = []
        if spec.evaluator.name != self.name:
            reasons.append(f"evaluator.name is {spec.evaluator.name!r}, not {self.name!r}")
        if spec.dataset is not None:
            reasons.append("dataset is declared; a harness evaluation measures run evidence")
        if spec.slices:
            reasons.append("slices are declared; the catalog defines the only slice")
        _, problem = _disclosure(spec)
        if problem:
            reasons.append(problem)
        problems = canonical_config_refusals(spec.evaluator.config, self.name)
        reasons.extend(problems)
        if problems:
            return reasons
        config = parse_evaluation_config(thaw(spec.evaluator.config))
        reasons.extend(self._config_refusals(config))
        if isinstance(config.suite, BenchmarkSuiteRef):
            try:
                suite = resolve_benchmark_suite(config.suite, self._source)
            except UnverifiedBenchmarkError as error:
                reasons.extend(f"suite: {reason}" for reason in error.reasons)
            else:
                missing = [t.id for t in suite.tasks if t.reference is None]
                if missing:
                    reasons.append(
                        f"exact-match needs a held-out reference, and tasks {missing} have none"
                    )
        return reasons

    def _config_refusals(self, config: HarnessOutcomeConfig | HarnessJudgeConfig) -> list[str]:
        return []

    def prepare(
        self, subject: ArtifactRef, spec: EvaluationSpec, context: EvaluationContext
    ) -> EvaluationExecutionSpec:
        """How to measure *subject*. Deterministic: reads no file, verifies no evidence.

        Verification needs the store, and the store can change; so it runs at
        the start of execution, on every run.

        Raises:
            UnsupportedEvaluationError: With every reason.
        """
        reasons = self._refusals(spec)
        if spec.evaluator.version != self.descriptor.plugin_version:
            reasons.append("the spec is not bound to this evaluator's version (bind_evaluation)")
        if subject.kind != "agent_run_evidence":
            reasons.append(f"the subject is a {subject.kind!r} artifact, not agent_run_evidence")
        if subject.digest is None:
            reasons.append("the subject names no digest; evidence is measured by digest")
        location = Path(subject.uri)
        if not location.is_absolute() or ".." in location.parts or location.name != EVIDENCE_FILE:
            reasons.append(f"the subject is not a published {EVIDENCE_FILE} at a local path")
        output = Path(context.output_uri) if context.output_uri else None
        if output is None or not output.is_absolute() or ".." in output.parts:
            reasons.append("output_uri is not an absolute local directory to write reports to")
        elif output.is_relative_to(location.parent):
            reasons.append("output_uri is inside the evidence it measures, which stays read-only")
        grant: ReferenceEgressGrant | None = None
        if not reasons:
            config = parse_evaluation_config(thaw(spec.evaluator.config))
            if not isinstance(config.suite, BenchmarkSuiteRef):
                reasons.append("the suite is not pinned; resolve the spec before preparing it")
            grant, problem = self._grant(config)
            if problem:
                reasons.append(problem)
        if reasons:
            raise UnsupportedEvaluationError(self.name, tuple(dict.fromkeys(reasons)))
        disclosure, _ = _disclosure(spec)
        assert disclosure is not None and output is not None
        worker = _WorkerConfig(
            api_version=_WORKER_API,
            evaluator=self.name,
            evaluator_version=self.descriptor.plugin_version,
            config=spec.evaluator.config,
            disclosure=disclosure,
            grant=grant,
            evidence_uri=str(location),
            output_dir=str(output),
            seed=context.seed,
            replicate=context.replicate,
        )
        return EvaluationExecutionSpec(
            evaluator=EvaluatorIdentity(
                name=self.name, version=self.descriptor.plugin_version, descriptor=self.descriptor
            ),
            evaluation_fingerprint=spec.evaluation_fingerprint(),
            subject=subject,
            entrypoint=_ENTRYPOINT,
            config=FrozenDict(worker.model_dump(mode="json")),
            outputs=(
                ArtifactOutput(
                    name="report", uri=str(output / "report.json"), kind="evaluation_report"
                ),
            ),
            resources=ResourceRequirements(workers=1),
        )

    def _grant(
        self, config: HarnessOutcomeConfig | HarnessJudgeConfig
    ) -> tuple[ReferenceEgressGrant | None, str | None]:
        return None, None

    @property
    def grants(self) -> tuple[ReferenceEgressGrant, ...]:
        """The host's grants: what execution checks the prepared one against."""
        return ()


def _descriptor(name: str) -> PluginDescriptor:
    return PluginDescriptor(
        api_version=PLUGIN_API_VERSIONS[0],
        name=name,
        plugin_version="0.1.0",
        provider="xaytune",
        xaytune_version=__version__,
    )


class HarnessOutcomeEvaluator(_HarnessEvaluator):
    """Task success and every execution metric of one replicate. ``DETERMINISTIC``."""

    name = HARNESS_OUTCOME
    determinism = EvaluatorDeterminism.DETERMINISTIC
    descriptor = _descriptor(HARNESS_OUTCOME)


class HarnessJudgeEvaluator(_HarnessEvaluator):
    """Quality, by a pinned judge model against a pinned rubric. ``STOCHASTIC``.

    *grants* are the host's :class:`ReferenceEgressGrant` s. A judge that
    uses the reference is prepared only with a grant naming its suite and its
    pinned model: nothing is sent otherwise (G2a).
    """

    name = HARNESS_JUDGE
    determinism = EvaluatorDeterminism.STOCHASTIC
    descriptor = _descriptor(HARNESS_JUDGE)

    def __init__(
        self, source: BenchmarkSource, *, grants: Sequence[ReferenceEgressGrant] = ()
    ) -> None:
        super().__init__(source)
        self._grants = tuple(grants)

    @property
    def grants(self) -> tuple[ReferenceEgressGrant, ...]:
        return self._grants

    def _config_refusals(self, config: HarnessOutcomeConfig | HarnessJudgeConfig) -> list[str]:
        assert isinstance(config, HarnessJudgeConfig)
        if config.judge_model.revision is None:
            return [
                "the judge model is not pinned to a revision: a judge is asked through a model "
                "identity, and an artifact digest alone cannot name what answers"
            ]
        return []

    def _grant(
        self, config: HarnessOutcomeConfig | HarnessJudgeConfig
    ) -> tuple[ReferenceEgressGrant | None, str | None]:
        assert isinstance(config, HarnessJudgeConfig)
        if config.reference_access == "none":
            return None, None
        grant = next((g for g in self._grants if g.covers(config)), None)
        if grant is None:
            return None, (
                "reference_access is 'reference' and no ReferenceEgressGrant names this suite "
                "and this pinned judge model; nothing is sent"
            )
        return grant, None


def bind_evaluation(evaluator: _HarnessEvaluator, spec: EvaluationSpec) -> EvaluationSpec:
    """*spec* as a host records it: ``supports`` → ``resolve`` → ``supports``, then bound.

    The local counterpart of the experiment host's submission step: the
    recorded spec names the evaluator's version and declared determinism.

    Raises:
        UnsupportedEvaluationError: With every reason.
    """
    support = evaluator.supports(spec)
    if not support:
        raise UnsupportedEvaluationError(evaluator.name, support.reasons)
    resolved = evaluator.resolve(spec) if isinstance(evaluator, ResolvableEvaluator) else spec
    support = evaluator.supports(resolved)
    if not support:
        raise UnsupportedEvaluationError(
            evaluator.name, tuple(f"as resolved: {reason}" for reason in support.reasons)
        )
    bound = EvaluatorSpec(
        name=resolved.evaluator.name,
        version=evaluator.descriptor.plugin_version,
        determinism=evaluator.determinism,
        config=resolved.evaluator.config,
    )
    return resolved.model_copy(update={"evaluator": bound})


# ---- execution ----------------------------------------------------------------------------


class EvaluationRefusedError(RuntimeError):
    """Execution measured nothing: the subject or the suite did not verify. Every reason."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("the evaluation measured nothing: " + "; ".join(reasons))


@dataclass(frozen=True)
class _Task:
    """One assigned task, read from the store: its outcome and every attempt's events."""

    task_id: str
    position: int
    result: HarnessRunResult | None
    outcome_events: tuple[Any, ...]
    attempts: tuple[tuple[Any, tuple[Any, ...]], ...]
    reference: str | None
    variables: FrozenDict


@dataclass(frozen=True)
class _Measured:
    report: HarnessEvaluationReport
    restricted: RestrictedEvaluationReport
    released: tuple[RunValue, ...]


def _read_tasks(
    evidence: HarnessRunEvidence,
    suite: ResolvedBenchmarkSuite,
    source: BenchmarkSource,
    store: HarnessEvidenceStore,
    *,
    references: bool,
) -> list[_Task]:
    """Every assigned task. References are resolved only when *references* is true."""
    tasks: list[_Task] = []
    for entry in evidence.entries:
        listed, _ = store.attempts_for(entry.request_digest)
        attempts: list[tuple[Any, tuple[Any, ...]]] = []
        outcome: ListedAttempt | None = None
        for attempt in listed:
            manifest = attempt.manifest
            assert manifest is not None  # verified: no attempt is open
            events, _ = load_events(attempt.store, attempt.paths, manifest)
            attempts.append((manifest.abandonment, events))
            if entry.outcome is not None and attempt.trajectory_id == entry.outcome.trajectory_id:
                outcome = attempt
        result = None
        outcome_events: tuple[Any, ...] = ()
        if outcome is not None and outcome.manifest is not None:
            result = HarnessRunResult.model_validate_json(read_bytes(outcome.paths.result))
            outcome_events, _ = load_events(outcome.store, outcome.paths, outcome.manifest)
        task = suite.task(entry.task_id)
        tasks.append(
            _Task(
                task_id=entry.task_id,
                position=entry.position,
                result=result,
                outcome_events=outcome_events,
                attempts=tuple(attempts),
                reference=resolve_reference(task, source) if references else None,
                variables=agent_visible(task).variables,
            )
        )
    return tasks


def execute_harness_evaluation(
    execution: EvaluationExecutionSpec,
    *,
    source: BenchmarkSource,
    judge_model: AgentModel | None = None,
    grants: Sequence[ReferenceEgressGrant] = (),
) -> _Measured:
    """Measure the subject *execution* names. Verifies it first, on every call.

    The execution config is data, so its grant is checked again here, first,
    against the host's *grants*: before the evidence, the suite or any
    reference is read, and before anything is sent. It must be one of them,
    and it must cover this suite and this pinned judge (V5, G2a).

    Raises:
        EvaluationRefusedError: If the subject, the suite, the grant or the
            judge cannot be trusted; nothing was measured or sent.
    """
    try:
        worker = _WorkerConfig.model_validate(thaw(execution.config))
        config = parse_evaluation_config(thaw(worker.config))
    except ValidationError:
        raise EvaluationRefusedError(
            ("the execution config is not one prepare() writes",)
        ) from None
    refused = _grant_refusals(config, worker.grant, grants)
    if refused:
        raise EvaluationRefusedError(tuple(refused))
    subject = execution.subject
    location = Path(worker.evidence_uri)
    try:
        evidence = HarnessRunEvidence.model_validate_json(read_bytes(location))
    except (OSError, ValidationError):
        raise EvaluationRefusedError(("the subject's evidence cannot be read",)) from None
    reasons: list[str] = []
    if evidence.evidence_digest != subject.digest:
        reasons.append("the published evidence is not the subject's digest")
    if evidence.suite != config.suite or evidence.sampling != config.sampling:
        reasons.append("the evidence ran another suite or sampling than this evaluation measures")
    assert isinstance(config.suite, BenchmarkSuiteRef)
    try:
        suite = resolve_benchmark_suite(config.suite, source)
    except UnverifiedBenchmarkError as error:
        raise EvaluationRefusedError(tuple(f"suite: {r}" for r in error.reasons)) from None
    store = HarnessEvidenceStore(location.parent)
    try:
        verify_run_evidence(evidence, suite, source, store)
    except EvidenceRefusedError as error:
        reasons.extend(error.reasons)
    if isinstance(config, HarnessJudgeConfig):
        reasons.extend(_judge_refusals(config, judge_model))
    if reasons:
        raise EvaluationRefusedError(tuple(dict.fromkeys(reasons)))

    sends_reference = (
        isinstance(config, HarnessJudgeConfig) and config.reference_access == "reference"
    )
    tasks = _read_tasks(
        evidence,
        suite,
        source,
        store,
        references=isinstance(config, HarnessOutcomeConfig) or sends_reference,
    )
    token_counter = evidence.facets.token_counter
    exchanges: list[JudgeExchange] = []
    observed: list[TaskObservations] = []
    spend = 0
    if isinstance(config, HarnessOutcomeConfig):
        metrics: tuple[str, ...] = ("task_success", *EXECUTION_METRICS)
        units = {m: metric_unit(m, token_counter=token_counter) for m in metrics}
        for task in tasks:
            execution_found, spent = execution_observations(
                task.result, task.outcome_events, task.attempts, evidence.token_budget
            )
            spend += spent
            success = success_observation(task.result, task.reference, config.success)
            observed.append(
                TaskObservations(
                    task_id=task.task_id,
                    position=task.position,
                    observations=(success, *execution_found),
                )
            )
    else:
        assert judge_model is not None
        metrics = ("quality",)
        units = {
            "quality": metric_unit(
                "quality", token_counter=token_counter, rubric_digest=config.rubric.digest
            )
        }
        for task in tasks:
            quality, exchange = _judge(config, judge_model, task)
            if exchange is not None:
                exchanges.append(exchange)
            observed.append(
                TaskObservations(
                    task_id=task.task_id, position=task.position, observations=(quality,)
                )
            )

    run_values, absences = aggregate_run(observed, metrics, units)
    released, withheld = _disclose(run_values, worker.disclosure)
    failure = None
    if all(
        o.absent == "unscored-cancelled"
        for t in observed
        for o in t.observations
        if o.metric in REFERENCE_DERIVED
    ):
        failure = "every task was cancelled: no task has a scored outcome"
    elif not run_values:
        failure = "every metric is absent on at least one task: there is no run value"
    elif not released:
        failure = "every run value is withheld by the disclosure policy"
    report = HarnessEvaluationReport(
        evaluator=worker.evaluator,
        evaluator_version=worker.evaluator_version,
        evaluation_fingerprint=execution.evaluation_fingerprint,
        evidence_digest=evidence.evidence_digest,
        seed=worker.seed,
        replicate=worker.replicate,
        disclosure=worker.disclosure,
        grant=worker.grant,
        run_values=released,
        absences=absences,
        withheld=withheld,
        tasks=tuple(_released_tasks(observed, worker.disclosure)),
        spend_lower_bound=spend,
        failure=failure,
    )
    restricted = RestrictedEvaluationReport(
        evaluation_fingerprint=execution.evaluation_fingerprint,
        evidence_digest=evidence.evidence_digest,
        run_values=run_values,
        tasks=tuple(observed),
        judge=tuple(exchanges),
    )
    return _Measured(report=report, restricted=restricted, released=released if not failure else ())


def _disclose(
    values: tuple[RunValue, ...], disclosure: Disclosure
) -> tuple[tuple[RunValue, ...], tuple[str, ...]]:
    """Run values that may be released.

    Under ``aggregate``, neither a reference-derived value over one task (it
    is that task's score) nor any ``successful-tasks`` slice: a slice is
    chosen by per-task success, and next to the released per-task execution
    measurements its mean can tell which tasks succeeded, at any sample count.
    """
    if disclosure == "per-task":
        return values, ()
    released: list[RunValue] = []
    withheld: list[str] = []
    for value in values:
        if value.slice is not None or (
            value.metric in REFERENCE_DERIVED and value.sample_count == 1
        ):
            withheld.append(value.metric + (f"/{value.slice}" if value.slice else ""))
        else:
            released.append(value)
    return tuple(released), tuple(withheld)


def _released_tasks(
    observed: Sequence[TaskObservations], disclosure: Disclosure
) -> list[TaskObservations]:
    if disclosure == "per-task":
        return list(observed)
    return [
        task.model_copy(
            update={
                "observations": tuple(
                    o for o in task.observations if o.metric not in REFERENCE_DERIVED
                )
            }
        )
        for task in observed
    ]


# ---- the judge ----------------------------------------------------------------------------


def _judge_identity(config: HarnessJudgeConfig) -> AgentModelIdentity:
    binding = config.judge_model
    return AgentModelIdentity(
        provider=binding.provider, name=binding.model, revision=binding.revision
    )


def _grant_refusals(
    config: HarnessOutcomeConfig | HarnessJudgeConfig,
    grant: ReferenceEgressGrant | None,
    grants: Sequence[ReferenceEgressGrant],
) -> list[str]:
    """Execution's own check of the prepared grant: nothing upstream is trusted (V5, G2a)."""
    if not isinstance(config, HarnessJudgeConfig) or config.reference_access == "none":
        if grant is not None:
            return ["the execution carries a grant for an evaluation that sends no reference"]
        return []
    if grant is None:
        return [
            "reference_access is 'reference' and the execution carries no "
            "ReferenceEgressGrant; nothing is sent"
        ]
    reasons: list[str] = []
    if not grant.covers(config):
        reasons.append(
            "the execution's ReferenceEgressGrant does not name this suite and this pinned "
            "judge model; nothing is sent"
        )
    if grant not in grants:
        reasons.append(
            "the execution's ReferenceEgressGrant is not one the host supplied; nothing is sent"
        )
    return reasons


def _judge_refusals(config: HarnessJudgeConfig, model: AgentModel | None) -> list[str]:
    if model is None:
        return ["a judge evaluation needs a judge model"]
    if model.descriptor.model != _judge_identity(config):
        return ["the judge model offered is not the one the config pins; nothing is sent"]
    return []


def _judge(
    config: HarnessJudgeConfig, model: AgentModel, task: _Task
) -> tuple[MetricObservation, JudgeExchange | None]:
    """``quality`` for one task, asking the judge only about an attested answer."""
    result = task.result
    if result is None:
        return MetricObservation(metric="quality", absent="no-attested-result"), None
    if result.termination == "cancelled":
        return MetricObservation(metric="quality", absent="unscored-cancelled"), None
    if result.termination not in COMPLETIONS or result.answer is None:
        # An attested stop of the harness's own rules has no answer to judge.
        # task_success scores it 0 (V6); quality was not measured, and no score
        # is imputed. A failure penalty would be its own versioned scoring policy.
        return MetricObservation(metric="quality", absent="not-applicable"), None
    sections = [
        "Task inputs:\n" + json.dumps(dict(task.variables), sort_keys=True),
        "Answer:\n" + result.answer,
        "Rubric:\n" + config.rubric.text,
    ]
    if config.reference_access == "reference" and task.reference is not None:
        sections.append("Reference answer:\n" + task.reference)
    message = "\n\n".join(sections)
    generation = config.judge_model.generation
    request = AgentModelRequest(
        system=config.prompt.text,
        messages=(AgentMessage(role="user", content=message),),
        response_schema=FrozenDict(
            {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "integer",
                        "minimum": config.scale.minimum,
                        "maximum": config.scale.maximum,
                    }
                },
                "required": ["score"],
                "additionalProperties": False,
            }
        ),
        temperature=generation.temperature,
        max_output_tokens=generation.max_output_tokens,
    )
    fingerprint = request.fingerprint(model.descriptor.model)
    try:
        response = asyncio.run(invoke_agent_model(model, request))
    except AgentModelError as error:
        exchange = JudgeExchange(
            task_id=task.task_id,
            request_fingerprint=fingerprint,
            system=request.system,
            message=message,
            reply=None,
            failure=type(error).__name__,
        )
        return MetricObservation(metric="quality", absent="judge-failed"), exchange
    exchange = JudgeExchange(
        task_id=task.task_id,
        request_fingerprint=fingerprint,
        system=request.system,
        message=message,
        reply=response.content,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    return MetricObservation(metric="quality", value=float(response.content["score"])), exchange


# ---- the local driver ----------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessEvaluation:
    """One evaluation run, driven locally: its lifecycle records, result and reports."""

    run: EvaluationRun
    attempt: EvaluationAttempt
    result: EvaluationResult | None
    report: HarnessEvaluationReport
    restricted: RestrictedEvaluationReport | None
    """``None`` when nothing was measured."""
    report_ref: ArtifactRef | None = None
    restricted_ref: ArtifactRef | None = None


def run_harness_evaluation(
    evaluator: _HarnessEvaluator,
    spec: EvaluationSpec,
    subject: ArtifactRef,
    *,
    source: BenchmarkSource,
    output_uri: str,
    seed: int | None = None,
    replicate: int | None = None,
    judge_model: AgentModel | None = None,
    experiment_id: ExperimentId | None = None,
    node_id: ExperimentNodeId | None = None,
    evaluation_cycle: int = 1,
) -> HarnessEvaluation:
    """Run one evaluation of *subject* locally, through the existing lifecycle (V4).

    Mints an ``EvaluationRun`` and an ``EvaluationAttempt``, walks them
    through ``EVALUATION_RUN_MACHINE`` and ``EVALUATION_ATTEMPT_MACHINE``,
    executes the prepared spec in-process, and builds the
    ``EvaluationResult`` only if ``result_provenance_problems`` reports
    nothing. Every call is a new run: nothing is reused (V11).

    *spec* must already be bound (:func:`bind_evaluation`).

    Raises:
        UnsupportedEvaluationError: If ``prepare()`` refuses; no run executed.
    """
    run = EvaluationRun(
        id=EvaluationRunId.generate(),
        experiment_id=experiment_id or ExperimentId.generate(),
        node_id=node_id or ExperimentNodeId.generate(),
        evaluation_cycle=evaluation_cycle,
        spec=spec,
        subject=subject,
        evaluation_fingerprint=spec.evaluation_fingerprint(),
        seed=seed,
        replicate=replicate,
    )
    attempt = EvaluationAttempt(
        id=EvaluationAttemptId.generate(), evaluation_run_id=run.id, attempt_number=1
    )
    context = EvaluationContext(
        experiment_id=str(run.experiment_id),
        node_id=str(run.node_id),
        evaluation_run_id=str(run.id),
        seed=seed,
        replicate=replicate,
        output_uri=output_uri,
    )
    execution = evaluator.prepare(subject, spec, context)
    run = run.with_status(EvaluationRunStatus.ACTIVE)
    for status in (
        EvaluationAttemptStatus.QUEUED,
        EvaluationAttemptStatus.STARTING,
        EvaluationAttemptStatus.RUNNING,
    ):
        attempt = attempt.with_status(status)
    try:
        measured = execute_harness_evaluation(
            execution, source=source, judge_model=judge_model, grants=evaluator.grants
        )
    except EvaluationRefusedError as error:
        report = HarnessEvaluationReport(
            evaluator=evaluator.name,
            evaluator_version=evaluator.descriptor.plugin_version,
            evaluation_fingerprint=run.evaluation_fingerprint,
            evidence_digest=subject.digest or "sha256:" + "0" * 64,
            seed=seed,
            replicate=replicate,
            disclosure=_disclosure(spec)[0] or "aggregate",
            run_values=(),
            absences=(),
            tasks=(),
            spend_lower_bound=0,
            failure="; ".join(error.reasons),
        )
        return HarnessEvaluation(
            run=run.with_status(EvaluationRunStatus.FAILED),
            attempt=attempt.with_status(EvaluationAttemptStatus.FAILED),
            result=None,
            report=report,
            restricted=None,
        )

    output = Path(output_uri)
    result_id = EvaluationId.generate()
    report_ref, restricted_ref = _write_reports(output, measured, result_id)
    if measured.report.failure is not None:
        return HarnessEvaluation(
            run=run.with_status(EvaluationRunStatus.FAILED),
            attempt=attempt.with_status(EvaluationAttemptStatus.FAILED),
            result=None,
            report=measured.report,
            restricted=measured.restricted,
            report_ref=report_ref,
            restricted_ref=restricted_ref,
        )
    result = EvaluationResult(
        id=result_id,
        evaluation_run_id=run.id,
        node_id=run.node_id,
        subject=subject,
        evaluation_fingerprint=run.evaluation_fingerprint,
        metrics=tuple(_metric(value, run) for value in measured.released),
        artifacts=(report_ref,),
    )
    problems = result_provenance_problems(result, run)
    if problems:
        return HarnessEvaluation(
            run=run.with_status(EvaluationRunStatus.FAILED),
            attempt=attempt.with_status(EvaluationAttemptStatus.FAILED),
            result=None,
            report=measured.report.model_copy(update={"failure": "; ".join(problems)}),
            restricted=measured.restricted,
            report_ref=report_ref,
            restricted_ref=restricted_ref,
        )
    return HarnessEvaluation(
        run=run.with_status(EvaluationRunStatus.SUCCEEDED),
        attempt=attempt.with_status(EvaluationAttemptStatus.SUCCEEDED),
        result=result,
        report=measured.report,
        restricted=measured.restricted,
        report_ref=report_ref,
        restricted_ref=restricted_ref,
    )


def _metric(value: RunValue, run: EvaluationRun) -> MetricResult:
    evaluator = run.spec.evaluator
    return MetricResult(
        name=value.metric,
        value=value.value,
        sample_count=value.sample_count,
        slice=value.slice,
        evaluator_name=evaluator.name,
        evaluator_version=evaluator.version,
        seed=run.seed,
        metadata=FrozenDict({"unit": value.unit, "catalog": CATALOG}),
    )


def _write_reports(
    output: Path, measured: _Measured, result_id: EvaluationId
) -> tuple[ArtifactRef, ArtifactRef]:
    """The released report beside the result; the restricted one in its own owner-only tier."""
    output.mkdir(parents=True, exist_ok=True)
    restricted_dir = output / "restricted"
    make_directory(restricted_dir)
    released = output / "report.json"
    restricted = restricted_dir / "report.json"
    write_new(released, measured.report.model_dump_json().encode("utf-8"))
    write_new(restricted, measured.restricted.model_dump_json().encode("utf-8"))
    return (
        ArtifactRef(
            id=ArtifactId.generate(),
            kind="evaluation_report",
            uri=str(released),
            producer_evaluation_id=result_id,
            metadata=FrozenDict({"tier": "ordinary"}),
        ),
        ArtifactRef(
            id=ArtifactId.generate(),
            kind="evaluation_report",
            uri=str(restricted),
            producer_evaluation_id=result_id,
            metadata=FrozenDict({"tier": "restricted"}),
        ),
    )


# ---- what a provider may receive (V5, G2c) -------------------------------------------------


def _provider_values(
    values: tuple[RunValue, ...],
) -> tuple[tuple[RunValue, ...], tuple[str, ...]]:
    """Run values a provider may see, whatever the report's disclosure.

    - A reference-derived value over one task is that task's score.
    - A ``successful-tasks`` slice is chosen by per-task success. Next to the
      per-task execution measurements a provider receives, its mean can tell
      which tasks succeeded, at any sample count.
    """
    kept: list[RunValue] = []
    withheld: list[str] = []
    for value in values:
        if value.slice is not None or (
            value.metric in REFERENCE_DERIVED and value.sample_count == 1
        ):
            withheld.append(value.metric + (f"/{value.slice}" if value.slice else ""))
        else:
            kept.append(value)
    return tuple(kept), tuple(withheld)


def provider_feedback(report: HarnessEvaluationReport) -> ProviderFeedback:
    """The provider's view of *report*: no per-task reference-derived score, at any level."""
    run_values, withheld = _provider_values(report.run_values)
    return ProviderFeedback(
        evaluation_fingerprint=report.evaluation_fingerprint,
        evidence_digest=report.evidence_digest,
        run_values=run_values,
        withheld=withheld,
        tasks=tuple(_released_tasks(report.tasks, "aggregate")),
    )
