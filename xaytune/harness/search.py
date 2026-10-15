"""Harness search and mutation: providers, the planner and the local loop (H07, ADR-024 §P12).

```text
HarnessSearchProvider.select(HarnessSearchContext)     → SearchChoice | NoChoice   where
HarnessMutationProvider.mutate(MutationRequest, 1)     → (HarnessPatch,) | ()      what
HarnessPlanner.plan: select → request → mutate → apply_patch → verify_harness_proposal → dedupe
run_local_harness_search: compile → run → evidence → evaluate → provider feedback → plan → …
```

**Providers decide nothing else.** They mint no id, write nothing, run
nothing and never edit a candidate; what they return is data that Xaytune
validates, applies, verifies and deduplicates. They see only what the
contracts in :mod:`xaytune.core.domain.harness_search` carry: provider
feedback and provider-safe comparisons, never a released report (§P6).

**The local loop is local.** It runs candidates in-process, under a
spending envelope that reserves each unit of work before it starts, and
writes a report. It touches no control plane and claims no durable
orchestration: a crashed search is re-run from its report
(:func:`replay_local_harness_search`), not resumed. Branching a harness
proposal into an experiment is refused until ADR-019 M1–M6.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, Protocol, runtime_checkable

from pydantic import ConfigDict, Field, ValidationError
from typing_extensions import Self

from xaytune.agent import AgentModel
from xaytune.benchmarks import BenchmarkSource
from xaytune.core.capabilities import PluginDescriptor, require_supported_plugin
from xaytune.core.domain.benchmark import (
    AllTasks,
    FirstTasks,
    ResolvedBenchmarkSuite,
    SeededSample,
    record_assignment,
)
from xaytune.core.domain.evaluation import EvaluationResult, EvaluationSpec
from xaytune.core.domain.harness import AgentHarnessSpec, harness_fingerprint
from xaytune.core.domain.harness_search import (
    FeedbackRef,
    HarnessCandidateProposal,
    HarnessFeedback,
    HarnessMutationCapabilities,
    HarnessNodeView,
    HarnessOperation,
    HarnessPatch,
    HarnessProposalRefusedError,
    HarnessProviderSpec,
    HarnessSearchCapabilities,
    HarnessSearchContext,
    HarnessSearchPolicy,
    MutationRequest,
    NoChoice,
    ObjectiveView,
    PatchRefusedError,
    ProviderComparison,
    ProviderMetricComparison,
    ProviderProvenance,
    Reservation,
    SearchChoice,
    Spend,
    SpendingExceededError,
    SpendingLedger,
    SpendingLimitError,
    SpendingUnenforceableError,
    apply_patch,
    attempt_reservation,
    envelope_refusals,
    verify_harness_proposal,
)
from xaytune.core.domain.planning import EvidenceRef, ProposalProvenance
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.errors import XaytuneError
from xaytune.core.fingerprint import fingerprint
from xaytune.core.harness_comparison import (
    CandidateRuns,
    ComparisonSpec,
    ReplicateRuns,
    SummaryRefusedError,
    compare,
)
from xaytune.core.harness_comparison import ObjectiveMetric as ComparisonObjective
from xaytune.core.harness_evaluation import (
    REFERENCE_DERIVED,
    HarnessRunEvidence,
    ProviderFeedback,
    RunValue,
    bind_task,
)
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw
from xaytune.core.trajectory import TrajectoryRecordingSpec
from xaytune.evaluation.harness import (
    HARNESS_JUDGE,
    provider_feedback,
    publish_evidence,
    run_harness_evaluation,
)
from xaytune.harness.compiler import (
    TOKEN_COUNTER,
    HarnessCompilationContext,
    ReferenceHarnessCompiler,
)
from xaytune.harness.content import ContentSource, resolve_harness_content
from xaytune.harness.evidence import EvidenceRefusedError, HarnessEvidenceStore, assemble_evidence
from xaytune.harness.runner import (
    HarnessRunRefusedError,
    HarnessToolbox,
    ScriptedHarnessModel,
    ScriptedHarnessRunner,
)
from xaytune.planning import _descriptor

__all__ = [
    "AGENT_TOKEN_UNIT",
    "BestFirstSearchProvider",
    "EnumeratedMutationConfig",
    "EnumeratedMutationProvider",
    "EvaluatedReplicate",
    "HarnessMutationProvider",
    "HarnessPlanner",
    "HarnessProviderConfigurationError",
    "HarnessSearchError",
    "HarnessSearchProvider",
    "LocalBenchmark",
    "LocalEvaluation",
    "LocalSearchReport",
    "PlanOutcome",
    "ReplayMismatchError",
    "bind_harness_provider",
    "explain_local_search",
    "provider_comparison",
    "provider_provenance",
    "replay_local_harness_search",
    "run_local_harness_search",
]

AGENT_TOKEN_UNIT: Final = f"tokens({TOKEN_COUNTER})"
"""The unit every task attempt is charged in: the runner's own token counter."""


class HarnessProviderConfigurationError(ValueError):
    """A harness provider cannot be bound: wrong kind, version or configuration."""

    def __init__(self, kind: str, reasons: Sequence[str]) -> None:
        self.kind = kind
        self.reasons = tuple(reasons)
        super().__init__(f"harness provider {kind!r} cannot be bound: " + "; ".join(self.reasons))


class HarnessSearchError(XaytuneError):
    """A provider broke its contract; nothing was proposed."""


# ---- §P4: the provider contracts ------------------------------------------------------------


@runtime_checkable
class HarnessSearchProvider(Protocol):
    """Chooses where to search next. Plugin kind ``harness-search``, API v1."""

    descriptor: PluginDescriptor
    spec: HarnessProviderSpec
    capabilities: HarnessSearchCapabilities

    def select(self, context: HarnessSearchContext) -> SearchChoice | NoChoice:
        """A base node to explore from, or why there is none. Depends on *context* only."""
        ...


@runtime_checkable
class HarnessMutationProvider(Protocol):
    """Proposes what to change. Plugin kind ``harness-mutation``, API v1."""

    descriptor: PluginDescriptor
    spec: HarnessProviderSpec
    capabilities: HarnessMutationCapabilities

    def mutate(self, request: MutationRequest, count: int = 1) -> tuple[HarnessPatch, ...]:
        """At most *count* patches of ``request.base``; ``()`` when it has nothing more."""
        ...


def bind_harness_provider(
    spec: HarnessProviderSpec, providers: Mapping[str, Callable[[HarnessProviderSpec], Any]]
) -> Any:
    """Resolve *spec* to a provider bound under it. Providers are registered explicitly.

    Raises:
        HarnessProviderConfigurationError: If no provider has the kind, or it refuses the spec.
    """
    factory = providers.get(spec.kind)
    if factory is None:
        raise HarnessProviderConfigurationError(
            spec.kind, (f"no harness provider of kind {spec.kind!r}; known: {sorted(providers)}",)
        )
    return factory(spec)


def _bind(
    descriptor: PluginDescriptor, spec: HarnessProviderSpec, config: FrozenDict
) -> HarnessProviderSpec:
    require_supported_plugin(descriptor)
    reasons = []
    if spec.kind != descriptor.name:
        reasons.append(f"the spec names {spec.kind!r}, not {descriptor.name!r}")
    if spec.version is not None and spec.version != descriptor.plugin_version:
        reasons.append(
            f"the spec names version {spec.version}, but this provider is "
            f"{descriptor.plugin_version}"
        )
    if reasons:
        raise HarnessProviderConfigurationError(spec.kind, reasons)
    return HarnessProviderSpec(
        kind=descriptor.name, version=descriptor.plugin_version, config=config
    )


def _config(kind: str, config_type: type[FrozenDomainModel], config: FrozenDict) -> Any:
    try:
        return config_type.model_validate(thaw(config))
    except ValidationError as invalid:
        raise HarnessProviderConfigurationError(
            kind,
            tuple(
                f"{'.'.join(str(part) for part in error['loc']) or 'config'}: {error['msg']}"
                for error in invalid.errors()
            ),
        ) from None


def provider_provenance(
    provider: HarnessSearchProvider | HarnessMutationProvider,
) -> ProviderProvenance:
    """*provider* as bound (§P8): descriptor, configuration, capabilities, engine and seed."""
    descriptor, spec = provider.descriptor, provider.spec
    if spec.version is None:
        raise HarnessSearchError("a provider always runs under a bound spec")
    config = thaw(spec.config)
    return ProviderProvenance(
        provider=descriptor.provider,
        name=descriptor.name,
        plugin_version=descriptor.plugin_version,
        api_version=descriptor.api_version,
        spec_kind=spec.kind,
        spec_version=spec.version,
        spec_fingerprint=fingerprint(
            {
                "type": "harness-provider-spec",
                "version": 1,
                "spec": {"kind": spec.kind, "version": spec.version, "config": config},
                "descriptor": {
                    "provider": descriptor.provider,
                    "name": descriptor.name,
                    "plugin_version": descriptor.plugin_version,
                    "api_version": descriptor.api_version,
                },
            }
        ),
        capabilities_fingerprint=fingerprint(
            {
                "type": "harness-provider-capabilities",
                "version": 1,
                "content": provider.capabilities.model_dump(mode="json"),
            }
        ),
        engine=FrozenDict(config.get("engine") or {}),
        seed=config.get("seed"),
    )


# ---- reference providers --------------------------------------------------------------------


class _Config(FrozenDomainModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class EnumeratedMutationConfig(_Config):
    """A declared list of single-operation alternatives, tried in order (§P12)."""

    alternatives: tuple[HarnessOperation, ...] = Field(min_length=1)
    seed: int = 0


class EnumeratedMutationProvider:
    """Proposes the first declared alternative not yet tried on the base. No model, no dependency.

    Deterministic: the same request gives the same patch. An alternative is
    skipped when it is outside the allowed surfaces, was refused earlier in
    the round, does not apply, or gives a harness the search already has.
    """

    descriptor: ClassVar[PluginDescriptor] = _descriptor("enumerated-mutation", "1.0.0")
    capabilities: HarnessMutationCapabilities
    spec: HarnessProviderSpec

    def __init__(self, config: EnumeratedMutationConfig) -> None:
        self.config = config
        self.capabilities = HarnessMutationCapabilities(
            operations=tuple(sorted({alternative.op for alternative in config.alternatives})),
            determinism="deterministic",
            feedback="none",
        )
        self.spec = _bind(
            self.descriptor,
            HarnessProviderSpec(kind=self.descriptor.name),
            FrozenDict(config.model_dump(mode="json")),
        )

    @classmethod
    def from_spec(cls, spec: HarnessProviderSpec) -> Self:
        provider = cls(_config(spec.kind, EnumeratedMutationConfig, spec.config))
        provider.spec = _bind(cls.descriptor, spec, provider.spec.config)
        return provider

    def mutate(self, request: MutationRequest, count: int = 1) -> tuple[HarnessPatch, ...]:
        if count != 1:
            raise HarnessSearchError("count is 1 in v1: parallel proposals are deferred")
        for index, alternative in enumerate(self.config.alternatives):
            if alternative.op not in request.allowed_operations:
                continue
            patch = HarnessPatch(
                base_fingerprint=request.base_fingerprint,
                operations=(alternative,),
                rationale=f"declared alternative {index + 1} of {len(self.config.alternatives)} "
                f"({alternative.op})",
            )
            if patch.patch_fingerprint() in request.refused:
                continue
            try:
                child = apply_patch(
                    request.base,
                    patch,
                    allowed=request.allowed_operations,
                    ceilings=request.ceilings,
                )
            except PatchRefusedError:
                continue
            if harness_fingerprint(child) in request.existing:
                continue
            return (patch,)
        return ()


class BestFirstSearchConfig(_Config):
    seed: int = 0


class BestFirstSearchProvider:
    """Explores from the best node: by the primary objective, or Pareto over a vector (§P12).

    Ranks only nodes evaluated under the context's evaluation fingerprint,
    by their provider-visible values; a node whose value was withheld or is
    absent is not ranked on it. For a vector objective, a node is dominated
    only when a provider-safe comparison *favors* another node over it:
    ``not-comparable``, ``inconclusive`` and ``withheld-from-provider`` are
    never a loss or a tie. Ties go to the lowest node id.
    """

    descriptor: ClassVar[PluginDescriptor] = _descriptor("best-first-search", "1.0.0")
    capabilities = HarnessSearchCapabilities(
        objectives="vector", determinism="deterministic", needs="evaluated-nodes"
    )
    spec: HarnessProviderSpec

    def __init__(self, config: BestFirstSearchConfig | None = None) -> None:
        self.config = config or BestFirstSearchConfig()
        self.spec = _bind(
            self.descriptor,
            HarnessProviderSpec(kind=self.descriptor.name),
            FrozenDict(self.config.model_dump(mode="json")),
        )

    @classmethod
    def from_spec(cls, spec: HarnessProviderSpec) -> Self:
        provider = cls(_config(spec.kind, BestFirstSearchConfig, spec.config))
        provider.spec = _bind(cls.descriptor, spec, provider.spec.config)
        return provider

    def select(self, context: HarnessSearchContext) -> SearchChoice | NoChoice:
        if context.scalarization is not None:
            raise HarnessSearchError("best-first search never scalarizes; none may be declared")
        evaluated = [
            node
            for node in context.nodes
            if node.status == "evaluated"
            and node.node_id not in context.exhausted
            and any(
                f.ref.evaluation_fingerprint == context.evaluation_fingerprint
                for f in node.feedback
            )
        ]
        if not evaluated:
            waiting = any(node.status == "unevaluated" for node in context.nodes)
            return NoChoice(
                reason="waiting-for-evaluation" if waiting else "exhausted",
                detail="no evaluated node is left to explore from",
            )
        if len(context.objectives) == 1:
            return self._best(context, evaluated)
        return self._pareto(context, evaluated)

    def _best(
        self, context: HarnessSearchContext, nodes: list[HarnessNodeView]
    ) -> SearchChoice | NoChoice:
        objective = context.objectives[0]
        ranked = [
            (value, node.node_id)
            for node in nodes
            if (value := _value(node, context.evaluation_fingerprint, objective)) is not None
        ]
        if not ranked:
            return NoChoice(
                reason="nothing-comparable",
                detail=f"no node has a provider-visible {_key(objective)}",
            )
        sign = -1.0 if objective.direction == "maximize" else 1.0
        value, node_id = min(ranked, key=lambda pair: (sign * pair[0], pair[1]))
        return SearchChoice(
            base_node_id=node_id,
            rationale=(
                f"best {_key(objective)} ({objective.direction}): {value:.6g} on {node_id}; "
                f"{len(ranked)} node(s) ranked, {len(nodes) - len(ranked)} not ranked "
                "(withheld from providers or absent)"
            ),
            objective_view=FrozenDict(
                {
                    "objective": _key(objective),
                    "values": {nid: v for v, nid in sorted(ranked, key=lambda p: p[1])},
                }
            ),
        )

    def _pareto(self, context: HarnessSearchContext, nodes: list[HarnessNodeView]) -> SearchChoice:
        candidates = {node.node_id for node in nodes}
        dominated: dict[str, str] = {}
        for comparison in context.comparisons:
            if comparison.evaluation_fingerprint != context.evaluation_fingerprint:
                continue
            if comparison.relation == "favors-a" and comparison.b in candidates:
                dominated.setdefault(comparison.b, comparison.a)
            elif comparison.relation == "favors-b" and comparison.a in candidates:
                dominated.setdefault(comparison.a, comparison.b)
        front = sorted(candidates - set(dominated))
        chosen = front[0] if front else sorted(candidates)[0]
        return SearchChoice(
            base_node_id=chosen,
            rationale=(
                f"Pareto-nondominated under {', '.join(_key(o) for o in context.objectives)}: "
                f"{', '.join(front) or 'none'}; ties go to the lowest node id"
            ),
            objective_view=FrozenDict(
                {"front": front, "dominated": dict(sorted(dominated.items()))}
            ),
        )


def _key(objective: ObjectiveView) -> str:
    return objective.metric + (f"/{objective.slice}" if objective.slice else "")


def _value(
    node: HarnessNodeView, evaluation_fingerprint: str, objective: ObjectiveView
) -> float | None:
    """The mean provider-visible value of *objective* over the node's replicates, or ``None``."""
    values = [
        value.value
        for feedback in node.feedback
        if feedback.ref.evaluation_fingerprint == evaluation_fingerprint
        for value in feedback.feedback.run_values
        if (value.metric, value.slice) == (objective.metric, objective.slice)
    ]
    replicates = sum(
        1 for f in node.feedback if f.ref.evaluation_fingerprint == evaluation_fingerprint
    )
    if not values or len(values) != replicates:
        return None
    return sum(values) / len(values)


# ---- §P6: provider-safe comparisons ---------------------------------------------------------


@dataclass(frozen=True)
class EvaluatedReplicate:
    """One execution replicate as Xaytune holds it: evidence, result and the provider's view."""

    evidence: HarnessRunEvidence
    result: EvaluationResult
    feedback: ProviderFeedback


def _visible(replicate: EvaluatedReplicate) -> EvaluationResult | None:
    """The result with only what the provider was shown; ``None`` if nothing is left."""
    shown = {(value.metric, value.slice) for value in replicate.feedback.run_values}
    metrics = tuple(m for m in replicate.result.metrics if (m.name, m.slice) in shown)
    if not metrics:
        return None
    return replicate.result.model_copy(update={"metrics": metrics})


def provider_comparison(
    a: tuple[str, Sequence[EvaluatedReplicate]],
    b: tuple[str, Sequence[EvaluatedReplicate]],
    *,
    evaluation: EvaluationSpec,
    objectives: Sequence[ObjectiveView],
    suite: ResolvedBenchmarkSuite,
    k: float = 2.0,
) -> ProviderComparison:
    """H06's ``compare()`` of A and B, restricted to what providers were shown (§P6).

    Results are cut down to the provider-visible run values and compared
    with no restricted report, so neither a withheld value nor per-task
    outcome counts can reach a verdict. An objective withheld on either side
    is ``withheld-from-provider``: no number, no verdict.
    """
    (a_id, a_runs), (b_id, b_runs) = a, b
    withheld = {name for r in (*a_runs, *b_runs) for name in r.feedback.withheld}
    # The same rule provider_feedback() applies, whatever the release withheld first:
    # a slice is chosen by per-task success, and a reference-derived value over one
    # task is that task's score. Both disclosure modes give the same answer.
    single_task = any(len(r.evidence.assignment.task_ids) == 1 for r in (*a_runs, *b_runs))
    withheld |= {
        _key(o)
        for o in objectives
        if o.slice is not None or (o.metric in REFERENCE_DERIVED and single_task)
    }

    def runs(replicates: Sequence[EvaluatedReplicate]) -> CandidateRuns:
        return CandidateRuns(
            evaluations=(evaluation,),
            replicates=tuple(
                ReplicateRuns(
                    evidence=r.evidence,
                    results=tuple(result for result in (_visible(r),) if result is not None),
                )
                for r in replicates
            ),
        )

    spec = ComparisonSpec(
        objectives=tuple(
            ComparisonObjective(
                metric=o.metric, slice=o.slice, direction=o.direction, min_effect=o.min_effect
            )
            for o in objectives
        ),
        k=k,
    )
    fingerprint_ = evaluation.evaluation_fingerprint()
    try:
        report = compare(runs(a_runs), runs(b_runs), spec, suite)
    except SummaryRefusedError as refused:
        return ProviderComparison(
            a=a_id,
            b=b_id,
            evaluation_fingerprint=fingerprint_,
            comparable=False,
            reasons=refused.reasons,
            metrics=tuple(
                ProviderMetricComparison(
                    metric=o.metric,
                    slice=o.slice,
                    direction=o.direction,
                    verdict="withheld-from-provider" if _key(o) in withheld else "not-comparable",
                    reasons=("the design is unbalanced",),
                )
                for o in objectives
            ),
            relation="withheld-from-provider"
            if any(_key(o) in withheld for o in objectives)
            else "not-comparable",
        )
    metrics = []
    for objective, measured in zip(objectives, report.metrics, strict=True):
        if _key(objective) in withheld:
            metrics.append(
                ProviderMetricComparison(
                    metric=objective.metric,
                    slice=objective.slice,
                    direction=objective.direction,
                    verdict="withheld-from-provider",
                    reasons=("withheld from providers on a side (ADR-022 V5, G2c)",),
                )
            )
        else:
            metrics.append(
                ProviderMetricComparison(
                    metric=measured.metric,
                    slice=measured.slice,
                    direction=measured.direction,
                    verdict=measured.verdict,
                    mean_difference=measured.mean_difference,
                    standard_error=measured.standard_error,
                    threshold=measured.threshold,
                    replicates=len(measured.differences),
                    reasons=measured.reasons,
                )
            )
    relation = (
        "withheld-from-provider"
        if any(m.verdict == "withheld-from-provider" for m in metrics)
        else report.relation
    )
    return ProviderComparison(
        a=a_id,
        b=b_id,
        evaluation_fingerprint=fingerprint_,
        comparable=report.comparability.comparable,
        reasons=report.comparability.reasons,
        metrics=tuple(metrics),
        relation=relation,
    )


# ---- the planner ----------------------------------------------------------------------------


class PatchRefusal(FrozenDomainModel):
    patch_fingerprint: str
    operations: tuple[str, ...]
    reasons: tuple[str, ...]


class PlanOutcome(FrozenDomainModel):
    """One planning round as recorded: the choice, every request and refusal, the result."""

    context_fingerprint: str
    choice: SearchChoice | NoChoice
    requests: tuple[MutationRequest, ...] = ()
    refusals: tuple[PatchRefusal, ...] = ()
    proposal: HarnessCandidateProposal | None = None
    exhausted: str | None = None
    """The base the mutation provider had nothing more for, if that is why nothing was proposed."""
    reason: str


class HarnessPlanner:
    """Orchestrates a search provider and a mutation provider into at most one proposal (§P1).

    A Xaytune-owned planner: it builds the mutation request, applies and
    verifies every patch (§P7), deduplicates, and retries a refused patch at
    most ``policy.max_retries`` times. It holds no state between rounds.
    """

    descriptor: ClassVar[PluginDescriptor] = _descriptor("harness-search", "1.0.0")

    def __init__(
        self,
        search: HarnessSearchProvider,
        mutation: HarnessMutationProvider,
        policy: HarnessSearchPolicy,
    ) -> None:
        operations = set(policy.allowed_operations) & set(mutation.capabilities.operations)
        if not operations:
            raise HarnessProviderConfigurationError(
                mutation.spec.kind,
                ("its operations do not intersect the experiment's allowed surfaces",),
            )
        if mutation.capabilities.determinism == "stochastic":
            raise HarnessProviderConfigurationError(
                mutation.spec.kind,
                ("stochastic mutation providers are deferred in v1 (ADR-024 §P13)",),
            )
        self.search, self.mutation, self.policy = search, mutation, policy
        self.operations = tuple(sorted(operations))
        self.search_provenance = provider_provenance(search)
        self.mutation_provenance = provider_provenance(mutation)
        self.spec = PlannerSpec(
            kind=self.descriptor.name,
            version=self.descriptor.plugin_version,
            config=FrozenDict(
                {
                    "search": self.search_provenance.spec_fingerprint,
                    "mutation": self.mutation_provenance.spec_fingerprint,
                    "policy": policy.policy_fingerprint(),
                }
            ),
        )

    def _provenance(self, context_fingerprint: str) -> ProposalProvenance:
        descriptor, spec = self.descriptor, self.spec
        assert spec.version is not None
        return ProposalProvenance(
            planner_provider=descriptor.provider,
            planner_name=descriptor.name,
            planner_version=descriptor.plugin_version,
            planner_api_version=descriptor.api_version,
            planner_spec_kind=spec.kind,
            planner_spec_version=spec.version,
            planner_spec_identity_version=1,
            planner_spec_fingerprint=fingerprint(
                {
                    "type": "harness-planner-spec",
                    "version": 1,
                    "content": spec.model_dump(mode="json"),
                }
            ),
            context_identity_version=1,
            context_fingerprint=context_fingerprint,
        )

    def plan(
        self,
        context: HarnessSearchContext,
        harnesses: Mapping[str, AgentHarnessSpec],
        *,
        seed: int,
        round: int,
    ) -> PlanOutcome:
        """At most one verified proposal from *context*; *harnesses* are the recorded nodes'."""
        context_fingerprint = context.context_fingerprint()
        choice = self.search.select(context)
        if isinstance(choice, NoChoice):
            return PlanOutcome(
                context_fingerprint=context_fingerprint,
                choice=choice,
                reason=f"the search provider chose nothing: {choice.reason}",
            )
        base_view = context.node(choice.base_node_id)
        if base_view is None or base_view.node_id not in harnesses:
            raise HarnessSearchError("the search provider chose a node the context does not have")
        if base_view.node_id in context.exhausted:
            # The mutation provider already had nothing more for it: asking again
            # would answer the same, round after round, and the search never ends.
            raise HarnessSearchError(
                f"the search provider chose {base_view.node_id}, which the context lists as "
                "exhausted"
            )
        base = harnesses[base_view.node_id]
        feedback = _lineage_feedback(context, base_view)
        existing = tuple(sorted({node.harness_fingerprint for node in context.nodes}))
        requests: list[MutationRequest] = []
        refusals: list[PatchRefusal] = []
        refused: list[str] = []
        for attempt in range(self.policy.max_retries + 1):
            request = MutationRequest(
                base=base,
                base_fingerprint=base_view.harness_fingerprint,
                base_node_id=base_view.node_id,
                allowed_operations=self.operations,
                ceilings=self.policy.ceilings,
                feedback=feedback,
                existing=existing,
                refused=tuple(sorted(refused)),
                seed=seed,
                round=round,
                attempt=attempt,
            )
            requests.append(request)
            patches = self.mutation.mutate(request, 1)
            if len(patches) > 1:
                raise HarnessSearchError("the mutation provider returned more than count patches")
            if not patches:
                return PlanOutcome(
                    context_fingerprint=context_fingerprint,
                    choice=choice,
                    requests=tuple(requests),
                    refusals=tuple(refusals),
                    exhausted=base_view.node_id,
                    reason=f"the mutation provider has nothing more for {base_view.node_id}",
                )
            (patch,) = patches
            try:
                child = apply_patch(
                    base,
                    patch,
                    allowed=self.operations,
                    authorized_tools=self.policy.authorized_tools,
                    ceilings=self.policy.ceilings,
                )
                proposal = HarnessCandidateProposal(
                    harness=child,
                    harness_fingerprint=harness_fingerprint(child),
                    parent_ids=(base_view.node_id,),
                    patch=patch,
                    hypothesis=(
                        f"{', '.join(op.op for op in patch.operations)} on {base_view.node_id} "
                        f"moves {', '.join(_key(o) for o in context.objectives)} "
                        "in the declared direction"
                    ),
                    reason=f"{choice.rationale}. {patch.rationale}".strip(),
                    evidence_refs=tuple(
                        EvidenceRef(kind="evaluation-result", id=f.ref.evaluation_result_id)
                        for f in base_view.feedback
                    ),
                    provenance=self._provenance(context_fingerprint),
                    search=self.search_provenance,
                    mutation=self.mutation_provenance,
                    feedback=_feedback_refs(context, feedback),
                )
                verify_harness_proposal(
                    proposal,
                    base,
                    self.policy,
                    existing,
                    provider_operations=self.mutation.capabilities.operations,
                )
            except (PatchRefusedError, HarnessProposalRefusedError) as refusal:
                refusals.append(
                    PatchRefusal(
                        patch_fingerprint=patch.patch_fingerprint(),
                        operations=tuple(op.op for op in patch.operations),
                        reasons=refusal.reasons,
                    )
                )
                refused.append(patch.patch_fingerprint())
                continue
            return PlanOutcome(
                context_fingerprint=context_fingerprint,
                choice=choice,
                requests=tuple(requests),
                refusals=tuple(refusals),
                proposal=proposal,
                reason="proposed, verified (§P7) and new",
            )
        return PlanOutcome(
            context_fingerprint=context_fingerprint,
            choice=choice,
            requests=tuple(requests),
            refusals=tuple(refusals),
            reason=f"every patch was refused within {self.policy.max_retries} retries",
        )


def _lineage_feedback(
    context: HarnessSearchContext, base: HarnessNodeView
) -> tuple[HarnessFeedback, ...]:
    """The base's feedback, and its ancestors' (§P4)."""
    found: list[HarnessFeedback] = []
    seen: set[str] = set()
    queue = [base.node_id]
    while queue:
        node_id = queue.pop(0)
        if node_id in seen:
            continue
        seen.add(node_id)
        node = context.node(node_id)
        if node is None:
            continue
        found.extend(node.feedback)
        queue.extend(node.parent_ids)
    return tuple(found)


def _feedback_refs(
    context: HarnessSearchContext, request_feedback: Sequence[HarnessFeedback]
) -> tuple[FeedbackRef, ...]:
    """Every feedback either provider received: the whole context's, and the request's."""
    refs = {f.ref for node in context.nodes for f in node.feedback} | {
        f.ref for f in request_feedback
    }
    return tuple(
        sorted(refs, key=lambda ref: (ref.evaluation_fingerprint, ref.evaluation_result_id))
    )


# ---- the local loop -------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalBenchmark:
    """The pinned suite every candidate runs on, and the seeds of its execution replicates."""

    suite: ResolvedBenchmarkSuite
    source: BenchmarkSource
    sampling: AllTasks | FirstTasks | SeededSample
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class LocalEvaluation:
    """One bound evaluation every replicate is measured by (one evaluation fingerprint)."""

    evaluator: Any
    spec: EvaluationSpec
    judge_model: AgentModel | None = None


class ReplicateRecord(FrozenDomainModel):
    seed: int
    evidence_digest: str | None = None
    attempts: int = 0
    attempt_tokens: int = 0
    agent_calls: int = 0
    evaluation_result_id: str | None = None
    released_run_values: tuple[RunValue, ...] = ()
    feedback: HarnessFeedback | None = None
    failure: str | None = None
    attempt_errors: tuple[str, ...] = ()
    """``task: ErrorType`` for each task attempt that raised; its whole reservation stays
    charged. The type only: an exception's message can carry what it was handed."""


class RoundRecord(FrozenDomainModel):
    round: int
    node_id: str
    harness: AgentHarnessSpec
    harness_fingerprint: str
    parent_ids: tuple[str, ...] = ()
    created_by: str | None = None
    """The proposal fingerprint that created this node; ``None`` for the root."""
    replicates: tuple[ReplicateRecord, ...] = ()
    comparisons: tuple[ProviderComparison, ...] = ()
    plans: tuple[tuple[HarnessSearchContext, PlanOutcome], ...] = ()
    spending: tuple[FrozenDict, ...] = ()


class LocalSearchReport(FrozenDomainModel):
    """Everything a local search did and why: what a reviewer and a replay read."""

    schema_version: Literal["xaytune.local-harness-search-report/v1alpha1"] = (
        "xaytune.local-harness-search-report/v1alpha1"
    )
    claim: Literal["local-exploration"] = "local-exploration"
    """Not durable orchestration: nothing here is in a control plane (Q4)."""
    policy: HarnessSearchPolicy
    policy_fingerprint: str
    planner: PlannerSpec
    search: ProviderProvenance
    mutation: ProviderProvenance
    objectives: tuple[ObjectiveView, ...]
    evaluation_fingerprint: str
    seeds: tuple[int, ...]
    rounds: tuple[RoundRecord, ...]
    stop_reason: str
    stop_detail: str = ""
    spent: Spend
    limits: Spend


class ReplayMismatchError(XaytuneError):
    """Re-planning from a report did not reproduce a round. Every mismatch."""

    def __init__(self, mismatches: Sequence[str]) -> None:
        self.mismatches = tuple(mismatches)
        super().__init__("the report does not replay: " + "; ".join(self.mismatches))


@dataclass
class _Node:
    node_id: str
    harness: AgentHarnessSpec
    fingerprint: str
    parent_ids: tuple[str, ...]
    created_by: str | None
    replicates: list[EvaluatedReplicate]
    feedback: list[HarnessFeedback]
    status: Literal["evaluated", "unevaluated", "failed"] = "unevaluated"


def _spend_of_attempt(result: Any) -> Spend:
    return Spend(
        attempts=1,
        model_calls=FrozenDict({"agent": result.turns}),
        tokens=FrozenDict({AGENT_TOKEN_UNIT: result.total_tokens}),
    )


def run_local_harness_search(
    *,
    root: AgentHarnessSpec,
    planner: HarnessPlanner,
    objectives: Sequence[ObjectiveView],
    benchmark: LocalBenchmark,
    evaluation: LocalEvaluation,
    content: ContentSource,
    model: Callable[[AgentHarnessSpec], ScriptedHarnessModel],
    toolbox: Callable[[], HarnessToolbox],
    recording: TrajectoryRecordingSpec,
    workspace: Path,
    comparison_k: float = 2.0,
) -> LocalSearchReport:
    """Run a harness search locally, round by round, under the policy's spending envelope.

    Each round runs the current candidate on every seed (compile, run,
    assemble and verify evidence), evaluates each replicate, gives the
    planner provider feedback and provider-safe comparisons only, and takes
    the verified proposal as the next round's candidate. The report is
    written to ``workspace/search-report.json`` whatever ends the search.

    Raises:
        SpendingUnenforceableError: If the envelope cannot be enforced; nothing runs.
        SpendingExceededError: If a unit measured more than its reservation
            (after the report is written).
        HarnessSearchError: If a provider broke its contract (after the report
            is written, with ``stop_reason="provider-error"``).
        HarnessRunRefusedError: If the runner refused a task run (after the
            report is written, with ``stop_reason="run-refused"``).
        Exception: Whatever else ended a round -- an evaluator or evidence
            store that raised -- after the interrupted unit is charged in full
            and the report is written, with ``stop_reason="error"`` naming the
            unit and the error type.
    """
    policy = planner.policy
    judge = evaluation.spec.evaluator.name == HARNESS_JUDGE
    judge_unit = _judge_unit(evaluation) if judge else None
    refusals = envelope_refusals(
        policy, root, token_unit=AGENT_TOKEN_UNIT, judge_unit=judge_unit, provider_calls=False
    )
    if refusals:
        raise SpendingUnenforceableError(refusals)
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    ledger = SpendingLedger(policy.envelope)
    evaluation_fingerprint = evaluation.spec.evaluation_fingerprint()
    objectives = tuple(objectives)
    nodes: dict[str, _Node] = {}
    rounds: list[RoundRecord] = []
    exhausted: set[str] = set()
    history: list[str] = []
    stop_reason, stop_detail = "max-rounds", ""
    failure: Exception | None = None

    def add(harness: AgentHarnessSpec, parents: tuple[str, ...], created_by: str | None) -> _Node:
        node = _Node(
            node_id=f"node-{len(nodes):03d}",
            harness=harness,
            fingerprint=harness_fingerprint(harness),
            parent_ids=parents,
            created_by=created_by,
            replicates=[],
            feedback=[],
        )
        nodes[node.node_id] = node
        return node

    current = add(root, (), None)
    round_number = 0
    while True:
        round_number += 1
        mark = len(ledger.entries)
        try:
            ledger.settle(
                ledger.reserve(
                    Reservation(
                        unit="round", label=f"round {round_number}", worst_case=Spend(rounds=1)
                    )
                ),
                Spend(rounds=1),
            )
        except SpendingLimitError as limit:
            stop_reason, stop_detail = "spending-limit", str(limit)
            break
        replicates: list[ReplicateRecord] = []
        record_node = current
        try:
            for seed in benchmark.seeds:
                replicates.append(
                    _evaluate_replicate(
                        current,
                        seed,
                        benchmark=benchmark,
                        evaluation=evaluation,
                        judge_unit=judge_unit,
                        policy=policy,
                        content=content,
                        model=model,
                        toolbox=toolbox,
                        recording=recording,
                        workspace=workspace,
                        ledger=ledger,
                    )
                )
        except SpendingLimitError as limit:
            stop_reason, stop_detail = "spending-limit", str(limit)
        except SpendingExceededError as error:
            stop_reason, stop_detail, failure = "spending-exceeded", str(error), error
        except HarnessRunRefusedError as error:
            ledger.abandon(note=type(error).__name__)
            stop_reason, stop_detail, failure = "run-refused", str(error), error
        except Exception as error:
            # Anything else -- an evaluator, the evidence store, a compiler. The
            # unit it interrupted is charged in full, the report keeps what ran
            # and names the unit, and the error is raised once it is written.
            abandoned = ledger.abandon(note=type(error).__name__)
            stop_reason, stop_detail, failure = (
                "error",
                f"{type(error).__name__}"
                + ("" if abandoned is None else f" during {abandoned.unit} {abandoned.label}"),
                error,
            )
        current.status = (
            "evaluated"
            if current.replicates and len(current.replicates) == len(benchmark.seeds)
            else "failed"
        )
        comparisons: tuple[ProviderComparison, ...] = ()
        plans: list[tuple[HarnessSearchContext, PlanOutcome]] = []
        proposal: HarnessCandidateProposal | None = None
        phase = "comparing candidates"
        try:
            comparisons = tuple(
                provider_comparison(
                    (current.node_id, current.replicates),
                    (other.node_id, other.replicates),
                    evaluation=evaluation.spec,
                    objectives=objectives,
                    suite=benchmark.suite,
                    k=comparison_k,
                )
                for other in nodes.values()
                if other is not current
                and other.status == "evaluated"
                and current.status == "evaluated"
            )
            all_comparisons = tuple(c for r in rounds for c in r.comparisons) + comparisons
            planning = failure is None and stop_reason != "spending-limit"
            if planning and ledger.spent.rounds >= policy.envelope.max_rounds:
                stop_reason = "max-rounds"
                stop_detail = f"{policy.envelope.max_rounds} round(s) run"
                planning = False
            while planning and proposal is None:
                phase = "building the planning context"
                context = _context(
                    nodes,
                    objectives,
                    evaluation_fingerprint,
                    all_comparisons,
                    exhausted,
                    ledger,
                    history,
                )
                phase = "planning"
                try:
                    outcome = planner.plan(
                        context,
                        {node_id: node.harness for node_id, node in nodes.items()},
                        seed=_policy_seed(planner),
                        round=round_number,
                    )
                except HarnessSearchError as error:
                    stop_reason, stop_detail, failure = "provider-error", str(error), error
                    break
                except Exception as error:
                    # A provider that raised: its message is not ours to repeat.
                    stop_reason, stop_detail, failure = (
                        "provider-error",
                        f"a provider raised {type(error).__name__} while planning",
                        error,
                    )
                    break
                plans.append((context, outcome))
                proposal = outcome.proposal
                if proposal is None:
                    if outcome.exhausted is not None:
                        exhausted.add(outcome.exhausted)
                        continue
                    stop_reason, stop_detail = "no-proposal", outcome.reason
                    break
            if proposal is not None:
                phase = "adding the proposed candidate"
                history.append(proposal.proposal_fingerprint())
                current = add(
                    proposal.harness, proposal.parent_ids, proposal.proposal_fingerprint()
                )
        except Exception as error:
            # Everything a round does after its replicates -- comparing, building
            # the context, adding the next candidate -- ends the search the same
            # way: the report keeps every round and what it spent, and says what
            # failed by its type, never its message.
            stop_reason, stop_detail, failure = (
                "error",
                f"{type(error).__name__} while {phase} in round {round_number}",
                error,
            )
            proposal = None
        rounds.append(
            RoundRecord(
                round=round_number,
                node_id=record_node.node_id,
                harness=record_node.harness,
                harness_fingerprint=record_node.fingerprint,
                parent_ids=record_node.parent_ids,
                created_by=record_node.created_by,
                replicates=tuple(replicates),
                comparisons=comparisons,
                plans=tuple(plans),
                spending=_entries(ledger, mark),
            )
        )
        if proposal is None:
            break

    report = LocalSearchReport(
        policy=policy,
        policy_fingerprint=policy.policy_fingerprint(),
        planner=planner.spec,
        search=planner.search_provenance,
        mutation=planner.mutation_provenance,
        objectives=objectives,
        evaluation_fingerprint=evaluation_fingerprint,
        seeds=benchmark.seeds,
        rounds=tuple(rounds),
        stop_reason=stop_reason,
        stop_detail=stop_detail,
        spent=ledger.spent,
        limits=ledger.limits,
    )
    (workspace / "search-report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    if failure is not None:
        raise failure
    return report


def _policy_seed(planner: HarnessPlanner) -> int:
    """The seed every mutation request carries: the mutation provider's bound seed, or 0."""
    return planner.mutation_provenance.seed or 0


def _entries(ledger: SpendingLedger, mark: int) -> tuple[FrozenDict, ...]:
    return tuple(FrozenDict(entry) for entry in ledger.entries[mark:])


def _judge_unit(evaluation: LocalEvaluation) -> str:
    config = evaluation.spec.evaluator.config
    model = config.get("judge_model") or {}
    return f"provider-tokens({model.get('provider')}/{model.get('model')})"


def _context(
    nodes: Mapping[str, _Node],
    objectives: tuple[ObjectiveView, ...],
    evaluation_fingerprint: str,
    comparisons: tuple[ProviderComparison, ...],
    exhausted: Collection[str],
    ledger: SpendingLedger,
    history: Sequence[str],
) -> HarnessSearchContext:
    return HarnessSearchContext(
        objectives=objectives,
        evaluation_fingerprint=evaluation_fingerprint,
        nodes=tuple(
            HarnessNodeView(
                node_id=node.node_id,
                harness_fingerprint=node.fingerprint,
                parent_ids=node.parent_ids,
                status=node.status,
                feedback=tuple(node.feedback),
            )
            for node in nodes.values()
        ),
        comparisons=comparisons,
        exhausted=tuple(exhausted),
        remaining=ledger.remaining(),
        history=tuple(history),
    )


def _evaluate_replicate(
    node: _Node,
    seed: int,
    *,
    benchmark: LocalBenchmark,
    evaluation: LocalEvaluation,
    judge_unit: str | None,
    policy: HarnessSearchPolicy,
    content: ContentSource,
    model: Callable[[AgentHarnessSpec], ScriptedHarnessModel],
    toolbox: Callable[[], HarnessToolbox],
    recording: TrajectoryRecordingSpec,
    workspace: Path,
    ledger: SpendingLedger,
) -> ReplicateRecord:
    """Run one replicate of *node* on every assigned task, then evaluate it; each unit reserved."""
    suite = benchmark.suite
    assignment = record_assignment(suite, benchmark.sampling, seed)
    evaluation_reservation = _evaluation_reservation(
        evaluation,
        policy,
        judge_unit,
        tasks=len(assignment.task_ids),
        label=f"{node.node_id} seed {seed}",
    )
    # Admitted before any task runs: a replicate whose evaluation cannot start
    # would spend its attempts for nothing. The evaluation touches only its own
    # limits (evaluations, judge calls and judge tokens), which the task
    # attempts never change, so admission still holds after they run.
    ledger.admit(evaluation_reservation)
    store = HarnessEvidenceStore(workspace / node.node_id / f"seed-{seed}")
    resolved = resolve_harness_content(node.harness, content)
    compiler = _ReplicateCompiler(node.harness, policy, resolved, recording, suite)
    specs = []
    errors: list[str] = []
    attempts = tokens = calls = 0
    for task_id in assignment.task_ids:
        task_input, binding = bind_task(suite, assignment, task_id)
        spec = compiler.compile(task_input, binding, seed, store.task_output(binding.position))
        specs.append(spec)
        reservation = ledger.reserve(
            attempt_reservation(
                node.harness, AGENT_TOKEN_UNIT, f"{node.node_id} seed {seed} {task_id}"
            )
        )
        try:
            result = ScriptedHarnessRunner().run(
                spec,
                model=model(node.harness),
                toolbox=toolbox(),
                authorized_tools=policy.authorized_tools,
            )
        except HarnessRunRefusedError:
            ledger.settle(reservation, None)
            raise
        except Exception as error:
            # Unaccounted: the whole reservation stays charged, and the report says why.
            ledger.settle(reservation, None, note=type(error).__name__)
            errors.append(f"{task_id}: {type(error).__name__}")
            continue
        charged = ledger.settle(reservation, _spend_of_attempt(result))
        attempts += 1
        tokens += charged.tokens.get(AGENT_TOKEN_UNIT, 0)
        calls += charged.model_calls.get("agent", 0)
    try:
        evidence = assemble_evidence(
            specs, suite, benchmark.sampling, assignment, benchmark.source, store
        )
    except EvidenceRefusedError as refused:
        return ReplicateRecord(
            seed=seed,
            attempts=attempts,
            attempt_tokens=tokens,
            agent_calls=calls,
            failure=f"evidence refused: {'; '.join(refused.reasons)}",
            attempt_errors=tuple(errors),
        )
    subject = publish_evidence(evidence, store)
    reservation = ledger.reserve(evaluation_reservation)
    evaluated = run_harness_evaluation(
        evaluation.evaluator,
        evaluation.spec,
        subject,
        source=benchmark.source,
        output_uri=str(workspace / node.node_id / f"evaluation-seed-{seed}"),
        seed=seed,
        replicate=0,
        judge_model=evaluation.judge_model,
    )
    # A judge's spend is what its provider reported; a call it reported nothing for
    # leaves the whole reservation charged, and a report above it stops the search.
    ledger.settle(
        reservation,
        Spend(evaluations=1) if judge_unit is None else _judge_spend(evaluated, judge_unit),
    )
    if evaluated.result is None:
        return ReplicateRecord(
            seed=seed,
            evidence_digest=evidence.evidence_digest,
            attempts=attempts,
            attempt_tokens=tokens,
            agent_calls=calls,
            failure=evaluated.report.failure or "the evaluation measured nothing",
            attempt_errors=tuple(errors),
        )
    feedback = provider_feedback(evaluated.report)
    harness_feedback = HarnessFeedback(
        ref=FeedbackRef(
            evaluation_fingerprint=evaluated.report.evaluation_fingerprint,
            evidence_digest=evaluated.report.evidence_digest,
            evaluation_result_id=str(evaluated.result.id),
        ),
        feedback=feedback,
    )
    node.replicates.append(
        EvaluatedReplicate(evidence=evidence, result=evaluated.result, feedback=feedback)
    )
    node.feedback.append(harness_feedback)
    return ReplicateRecord(
        seed=seed,
        evidence_digest=evidence.evidence_digest,
        attempts=attempts,
        attempt_tokens=tokens,
        agent_calls=calls,
        evaluation_result_id=str(evaluated.result.id),
        released_run_values=evaluated.report.run_values,
        feedback=harness_feedback,
        attempt_errors=tuple(errors),
    )


def _evaluation_reservation(
    evaluation: LocalEvaluation,
    policy: HarnessSearchPolicy,
    judge_unit: str | None,
    *,
    tasks: int,
    label: str,
) -> Reservation:
    """One replicate's evaluation and its worst case: a judge call per task, at its bound."""
    worst = Spend(evaluations=1)
    if judge_unit is not None:
        per_call = (policy.envelope.judge_prompt_tokens or 0) + int(
            evaluation.spec.evaluator.config["judge_model"]["generation"]["max_output_tokens"]
        )
        worst = worst.plus(
            Spend(
                model_calls=FrozenDict({"judge": tasks}),
                tokens=FrozenDict({judge_unit: tasks * per_call}),
            )
        )
    return Reservation(unit="evaluation", label=label, worst_case=worst)


def _judge_spend(evaluated: Any, judge_unit: str) -> Spend | None:
    """The judge calls an evaluation made and the tokens its provider reported for them.

    ``None`` (unaccounted, charged in full) when nothing was measured or any
    call has no reported token counts: a count is never estimated.
    """
    if evaluated.restricted is None:
        return None
    reported = [(e.input_tokens, e.output_tokens) for e in evaluated.restricted.judge]
    counts = [
        prompt + reply for prompt, reply in reported if prompt is not None and reply is not None
    ]
    if len(counts) != len(reported):
        return None
    return Spend(
        evaluations=1,
        model_calls=FrozenDict({"judge": len(counts)}),
        tokens=FrozenDict({judge_unit: sum(counts)}),
    )


class _ReplicateCompiler:
    """Compiles one candidate's task-bound, recording specs with the policy's authorized tools."""

    def __init__(
        self,
        harness: AgentHarnessSpec,
        policy: HarnessSearchPolicy,
        resolved: Any,
        recording: TrajectoryRecordingSpec,
        suite: ResolvedBenchmarkSuite,
    ) -> None:
        self.harness, self.policy, self.resolved = harness, policy, resolved
        self.recording, self.suite = recording, suite

    def compile(self, task_input: Any, binding: Any, seed: int, output: Path) -> Any:
        return ReferenceHarnessCompiler().compile(
            self.harness,
            HarnessCompilationContext(
                task=task_input,
                seed=seed,
                replicate=0,
                authorized_tools=self.policy.authorized_tools,
                content=self.resolved,
                output_uri=str(output),
                trajectory=self.recording,
                task_binding=binding,
                suite=self.suite,
            ),
        )


# ---- replay and explanation -----------------------------------------------------------------


def replay_local_harness_search(report: LocalSearchReport, planner: HarnessPlanner) -> int:
    """Planning replay: re-plan every recorded round from its recorded context.

    It replays *decisions* only. Nothing is re-executed: no model is called,
    no task runs and no evaluation is measured again. The recorded contexts
    are its input, so it shows that the same evidence, given to the same
    bound providers under the same policy, gives the same proposals -- not
    that the evidence itself would be reproduced. Returns how many plans were
    replayed.

    The planner must be bound exactly as recorded, and every round must give
    the same choice, requests, refusals and proposal.

    Raises:
        ReplayMismatchError: Naming every round that does not reproduce.
    """
    mismatches: list[str] = []
    if provider_provenance(planner.search) != report.search:
        mismatches.append("the search provider is not bound as recorded")
    if provider_provenance(planner.mutation) != report.mutation:
        mismatches.append("the mutation provider is not bound as recorded")
    if planner.policy.policy_fingerprint() != report.policy_fingerprint:
        mismatches.append("the policy is not the recorded policy")
    if mismatches:
        raise ReplayMismatchError(mismatches)
    harnesses = {r.node_id: r.harness for r in report.rounds}
    replayed = 0
    for record in report.rounds:
        for index, (context, recorded) in enumerate(record.plans):
            if context.context_fingerprint() != recorded.context_fingerprint:
                mismatches.append(f"round {record.round} plan {index + 1}: the context changed")
                continue
            known = {node.node_id for node in context.nodes}
            outcome = planner.plan(
                context,
                {node_id: harness for node_id, harness in harnesses.items() if node_id in known},
                seed=_policy_seed(planner),
                round=record.round,
            )
            if outcome.model_dump(mode="json") != recorded.model_dump(mode="json"):
                mismatches.append(f"round {record.round} plan {index + 1}: a different outcome")
            replayed += 1
    if mismatches:
        raise ReplayMismatchError(mismatches)
    return replayed


def _spend_text(spend: Mapping[str, Any]) -> str:
    parts = []
    for name in ("rounds", "evaluations", "attempts"):
        if spend.get(name):
            parts.append(f"{spend[name]} {name}")
    parts += [
        f"{calls} {role} call(s)"
        for role, calls in sorted((spend.get("model_calls") or {}).items())
        if calls
    ]
    parts += [
        f"{tokens} {unit}" for unit, tokens in sorted((spend.get("tokens") or {}).items()) if tokens
    ]
    return ", ".join(parts) or "nothing"


def explain_local_search(report: LocalSearchReport) -> str:
    """The report as text: every round, every decision and why, and what each one spent."""
    lines = [
        "Local harness search (local exploration: no control plane, no durable orchestration)",
        f"  policy {report.policy_fingerprint[:19]}…  autonomous search: "
        f"{report.policy.autonomous_search}; operations: "
        f"{', '.join(report.policy.allowed_operations)}",
        f"  search: {report.search.name} {report.search.plugin_version}; "
        f"mutation: {report.mutation.name} {report.mutation.plugin_version}",
        f"  objectives: {', '.join(f'{_key(o)} ({o.direction})' for o in report.objectives)}; "
        f"evaluation {report.evaluation_fingerprint[:19]}…; seeds {list(report.seeds)}",
    ]
    for record in report.rounds:
        origin = "root" if record.created_by is None else f"from proposal {record.created_by[:19]}…"
        lines.append("")
        lines.append(
            f"Round {record.round}: {record.node_id} ({origin}), "
            f"harness {record.harness_fingerprint[:19]}…"
        )
        for replicate in record.replicates:
            for error in replicate.attempt_errors:
                lines.append(f"  seed {replicate.seed}: attempt raised {error}; charged in full")
            if replicate.failure:
                lines.append(f"  seed {replicate.seed}: failed: {replicate.failure}")
                continue
            released = ", ".join(
                f"{v.metric}{'/' + v.slice if v.slice else ''}={v.value:.4g} (n={v.sample_count})"
                for v in replicate.released_run_values
                if v.metric in ("task_success", "quality", "total_tokens", "turns")
            )
            shown = replicate.feedback.feedback if replicate.feedback else None
            lines.append(
                f"  seed {replicate.seed}: {replicate.attempts} attempt(s), "
                f"{replicate.agent_calls} agent "
                f"call(s), {replicate.attempt_tokens} {AGENT_TOKEN_UNIT}; evidence "
                f"{(replicate.evidence_digest or '')[:19]}…; "
                f"result {replicate.evaluation_result_id}"
            )
            lines.append(f"    released: {released or 'nothing'}")
            if shown is not None:
                lines.append(
                    f"    providers were shown {len(shown.run_values)} run value(s); withheld from "
                    f"them: {', '.join(shown.withheld) or 'nothing'}"
                )
        for comparison in record.comparisons:
            verdicts = "; ".join(
                f"{_key_of(m)} {m.verdict}"
                + (
                    f" (mean {m.mean_difference:+.4g}, se {m.standard_error:.3g})"
                    if m.mean_difference is not None and m.standard_error is not None
                    else ""
                )
                for m in comparison.metrics
            )
            lines.append(
                f"  provider-safe comparison {comparison.a} vs {comparison.b}: "
                f"{comparison.relation}; {verdicts}"
            )
        for context, outcome in record.plans:
            lines.append(f"  plan from context {outcome.context_fingerprint[:19]}…")
            if isinstance(outcome.choice, SearchChoice):
                lines.append(
                    f"    search chose {outcome.choice.base_node_id}: {outcome.choice.rationale}"
                )
            else:
                lines.append(
                    f"    search chose nothing: {outcome.choice.reason} {outcome.choice.detail}"
                )
            for refusal in outcome.refusals:
                lines.append(
                    f"    refused patch {refusal.patch_fingerprint[:19]}… "
                    f"({', '.join(refusal.operations)}): " + "; ".join(refusal.reasons)
                )
            if outcome.proposal is not None:
                patch = outcome.proposal.patch
                lines.append(
                    f"    proposed {outcome.proposal.proposal_fingerprint()[:19]}…: "
                    + json.dumps(
                        [op.model_dump(mode="json") for op in patch.operations], sort_keys=True
                    )
                )
                lines.append(
                    "      verified (§P7: patch integrity, no escalation, ceilings, surfaces, "
                    f"not a "
                    f"duplicate); learned from {len(outcome.proposal.feedback)} feedback ref(s)"
                )
            else:
                lines.append(f"    no proposal: {outcome.reason}")
        for entry in record.spending:
            charged = entry.get("charged") or entry.get("reserved") or {}
            lines.append(
                f"  spent [{entry['unit']}] {entry['label']}: {entry['status']}"
                + (f" ({entry['note']})" if entry.get("note") else "")
                + f", {_spend_text(thaw(charged))}"
            )
    lines.append("")
    lines.append(f"Stopped: {report.stop_reason}. {report.stop_detail}".rstrip())
    lines.append(
        f"Total spent: {_spend_text(report.spent.model_dump(mode='json'))} "
        f"(limits: {_spend_text(report.limits.model_dump(mode='json'))})"
    )
    return "\n".join(lines)


def _key_of(metric: ProviderMetricComparison) -> str:
    return metric.metric + (f"/{metric.slice}" if metric.slice else "")
