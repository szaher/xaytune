"""Planners: what scientific work an experiment should try next (spec 09, PR-024).

```text
PlanningContext ──Planner.propose()──▶ (CandidateProposal | ActionProposal, ...)
```

A planner is the sibling of a decision engine, held to the same rule: **it
proposes; it applies nothing.** It reads only its context -- no clock, no id,
no database, no environment, no runtime -- so the same bound planner and the
same context give identical proposals. Accepting a proposal (policy, budget,
branching) is somebody else's work, later.

Ownership, so a planner never becomes a second decision engine or a second
recovery planner:

```text
objective / constraint verdicts on a candidate   DecisionEngine (PR-015, PR-015b)
an attempt that failed, ran out of memory, NaN   RecoveryCoordinator (PR-019–021)
what candidate to try next, once candidates      a Planner (here)
  are decided on their merits
turning a CandidateProposal into a node          branching (PR-025)
plateau detection                                deferred: needs multi-candidate history
```

A planner acts only at the planning stage: the experiment ``ACTIVE`` and every
candidate decided on its merits (``REJECTED``, or ``COMPLETED`` after a
``BRANCH``). Anything else is not its turn, and it proposes nothing.

**No reuse** (ADR-017). A planner skips a candidate the experiment already
has -- that is lineage deduplication, comparing current candidate identities
-- but never substitutes an existing artifact for work: every run a proposal
leads to executes.

Planners are bound from a durable :class:`~xaytune.core.domain.specs.PlannerSpec`
(ADR-016) through :data:`PLANNERS`; the bound spec, with its resolved version,
is what an experiment records.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field, StrictInt, ValidationError

from xaytune._version import __version__
from xaytune.core.capabilities import (
    PLUGIN_API_VERSIONS,
    PluginDescriptor,
    require_supported_plugin,
)
from xaytune.core.domain.candidate import CandidateSpec
from xaytune.core.domain.decision import DecisionOutcome
from xaytune.core.domain.planning import (
    PLANNING_CONTEXT_IDENTITY_VERSION,
    CandidateProposal,
    DecisionSummary,
    NodeSummary,
    PlanningContext,
    Proposal,
    ProposalProvenance,
)
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus

__all__ = [
    "PLANNERS",
    "IncreaseLoRARank",
    "Mutation",
    "MutationRule",
    "NoOpPlanner",
    "Planner",
    "PlannerConfigurationError",
    "RuleBasedPlanner",
    "RuleBasedPlannerConfig",
    "bind_planner",
]


class PlannerConfigurationError(ValueError):
    """A planner spec cannot be bound: wrong kind, version or configuration. Carries why."""

    def __init__(self, kind: str, reasons: tuple[str, ...]) -> None:
        self.kind = kind
        self.reasons = reasons
        super().__init__(f"planner {kind!r} cannot be bound: " + "; ".join(reasons))


@runtime_checkable
class Planner(Protocol):
    """Proposes an experiment's next work from its planning context alone."""

    descriptor: PluginDescriptor
    spec: PlannerSpec
    """The bound spec this planner runs under: ``kind``, resolved ``version``, canonical config."""

    async def propose(self, context: PlanningContext) -> tuple[Proposal, ...]:
        """What *context* suggests doing next; ``()`` when nothing is proposed.

        Pure: nothing but *context* and the bound configuration is read, and
        nothing is minted -- no id, no timestamp, no record.
        """
        ...


def _descriptor(name: str, version: str) -> PluginDescriptor:
    return PluginDescriptor(
        api_version=PLUGIN_API_VERSIONS[0],
        name=name,
        plugin_version=version,
        provider="xaytune",
        xaytune_version=__version__,
    )


def _bound_spec(descriptor: PluginDescriptor, spec: PlannerSpec, config: FrozenDict) -> PlannerSpec:
    """*spec* bound to *descriptor*, or refused naming every reason."""
    require_supported_plugin(descriptor)
    reasons = []
    if spec.kind != descriptor.name:
        reasons.append(f"the spec names {spec.kind!r}, not {descriptor.name!r}")
    if spec.version is not None and spec.version != descriptor.plugin_version:
        reasons.append(
            f"the spec names version {spec.version}, but this planner is "
            f"{descriptor.plugin_version}"
        )
    if reasons:
        raise PlannerConfigurationError(spec.kind, tuple(reasons))
    return PlannerSpec(kind=descriptor.name, version=descriptor.plugin_version, config=config)


def _provenance(planner: Planner, context: PlanningContext) -> ProposalProvenance:
    descriptor, spec = planner.descriptor, planner.spec
    assert spec.version is not None, "a planner always runs under a bound spec"
    return ProposalProvenance(
        planner_provider=descriptor.provider,
        planner_name=descriptor.name,
        planner_version=descriptor.plugin_version,
        planner_spec_kind=spec.kind,
        planner_spec_version=spec.version,
        context_identity_version=PLANNING_CONTEXT_IDENTITY_VERSION,
        context_fingerprint=context.input_fingerprint(),
    )


# ---- the baseline ------------------------------------------------------------------------


class NoOpPlanner:
    """Proposes nothing, ever: the baseline, and an experiment that plans by hand."""

    descriptor = _descriptor("no-op", "1.0.0")

    def __init__(self) -> None:
        self.spec = _bound_spec(self.descriptor, PlannerSpec(kind="no-op"), FrozenDict())

    @classmethod
    def from_spec(cls, spec: PlannerSpec) -> NoOpPlanner:
        """Bind *spec*. It takes no configuration, so any is refused."""
        if spec.config:
            raise PlannerConfigurationError(spec.kind, ("the no-op planner takes no config",))
        planner = cls()
        planner.spec = _bound_spec(cls.descriptor, spec, FrozenDict())
        return planner

    async def propose(self, context: PlanningContext) -> tuple[Proposal, ...]:
        return ()


# ---- typed mutation rules ----------------------------------------------------------------


@dataclass(frozen=True)
class Mutation:
    """One rule applied to one candidate: the new candidate, and why, as data."""

    candidate: CandidateSpec
    hypothesis: str
    description: FrozenDict


class IncreaseLoRARank(FrozenDomainModel):
    """Grow a LoRA adapter's rank: ``min(rank × factor, max_rank)``.

    Applies only to a candidate whose adapter is ``lora`` with an explicitly
    declared, positive rank, and only when the rank actually grows -- a rank
    already at or above ``max_rank`` is not changed, and never shrunk. Every
    other field of the candidate is kept exactly; the new candidate is built
    through validated copies, never patched in place.
    """

    kind: Literal["increase-lora-rank"] = "increase-lora-rank"
    factor: StrictInt = Field(ge=2)
    max_rank: StrictInt = Field(ge=1)

    def mutate(self, candidate: CandidateSpec) -> Mutation | None:
        training = candidate.training
        adapter = training.adapter
        if adapter is None or adapter.type != "lora" or adapter.rank is None:
            return None
        rank = adapter.rank
        if rank <= 0:
            return None
        grown = min(rank * self.factor, self.max_rank)
        if grown <= rank:
            return None
        changed = candidate.model_copy(
            update={
                "training": training.model_copy(
                    update={"adapter": adapter.model_copy(update={"rank": grown})}
                )
            }
        )
        return Mutation(
            candidate=changed,
            hypothesis="Adapter capacity may be limiting task performance.",
            description=FrozenDict(
                {
                    "rule": self.kind,
                    "field": "training.adapter.rank",
                    "from": rank,
                    "to": grown,
                }
            ),
        )


MutationRule = IncreaseLoRARank
"""Every built-in mutation rule. A union, discriminated by ``kind``, once there are more."""


class RuleBasedPlannerConfig(FrozenDomainModel):
    """The rule-based planner's configuration: an ordered list of typed mutation rules.

    Tried in declared order; the first that yields a candidate the experiment
    does not already have is proposed. Unknown keys and unknown rule kinds are
    refused.
    """

    rules: tuple[MutationRule, ...] = Field(min_length=1)


# ---- the rule-based planner --------------------------------------------------------------

_SETTLED = frozenset({ExperimentNodeStatus.REJECTED, ExperimentNodeStatus.COMPLETED})


@dataclass(frozen=True)
class _Parent:
    node: NodeSummary
    decision: DecisionSummary
    value: float
    evaluation_result_id: str


class RuleBasedPlanner:
    """Proposes the next candidate by applying declared mutations to the best one so far.

    ```text
    not the planning stage, or a quota exhausted      → nothing
    parent = best COMPLETED candidate by its recorded,
             unsliced primary metric (a BRANCH);
             ties broken by node id                   → none eligible: nothing
    for each rule, in declared order:
        not applicable to the parent                  → next rule
        the result already exists in the experiment   → next rule
        otherwise                                     → propose it, and stop
    no rule produced a novel candidate                → nothing
    ```

    **Parents are valid candidates only.** A ``REJECTED`` node violated a
    constraint and is never built on; a ``COMPLETED`` node under an open
    experiment is one the adaptive decision engine finished short of the
    target. Its value is read from the evaluation results its latest decision
    decided on, and must be measured exactly once there: a missing or repeated
    measurement makes the node ineligible rather than guessed at.

    **Budget is read, never reserved.** A quota with nothing left means there
    is no point proposing new work; anything finer is the branching path's to
    reserve when a proposal is accepted.
    """

    descriptor = _descriptor("rule-based", "1.0.0")

    def __init__(self, config: RuleBasedPlannerConfig) -> None:
        self.config = config
        self.spec = _bound_spec(
            self.descriptor,
            PlannerSpec(kind="rule-based"),
            FrozenDict(config.model_dump(mode="json")),
        )

    @classmethod
    def from_spec(cls, spec: PlannerSpec) -> RuleBasedPlanner:
        """Bind *spec*, validating its config into typed rules.

        Raises:
            PlannerConfigurationError: Naming every problem with the spec.
        """
        try:
            config = RuleBasedPlannerConfig.model_validate(thaw(spec.config))
        except ValidationError as invalid:
            raise PlannerConfigurationError(
                spec.kind,
                tuple(
                    f"{'.'.join(str(part) for part in error['loc']) or 'config'}: {error['msg']}"
                    for error in invalid.errors()
                ),
            ) from None
        planner = cls(config)
        planner.spec = _bound_spec(cls.descriptor, spec, planner.spec.config)
        return planner

    async def propose(self, context: PlanningContext) -> tuple[Proposal, ...]:
        if not _planning_stage(context):
            return ()
        if context.budget is not None and context.budget.exhausted:
            return ()
        parent = _best_parent(context)
        if parent is None:
            return ()
        existing = context.candidate_fingerprints
        for rule in self.config.rules:
            mutation = rule.mutate(parent.node.candidate)
            if mutation is None:
                continue
            identity = mutation.candidate.candidate_fingerprint()
            if identity in existing:
                continue
            primary = context.objective.primary
            return (
                CandidateProposal(
                    candidate=mutation.candidate,
                    candidate_fingerprint=identity,
                    parent_ids=(parent.node.node_id,),
                    hypothesis=mutation.hypothesis,
                    reason=(
                        f"node {parent.node.node_id} is the best completed candidate "
                        f"({primary.name} = {parent.value!r}, decided "
                        f"{parent.decision.outcome.value}); {rule.kind} changes "
                        f"{mutation.description['field']} from {mutation.description['from']} "
                        f"to {mutation.description['to']}"
                    ),
                    mutation=mutation.description,
                    evidence_refs=(
                        f"decision:{parent.decision.decision_id}",
                        f"evaluation-result:{parent.evaluation_result_id}",
                    ),
                    provenance=_provenance(self, context),
                ),
            )
        return ()


def _planning_stage(context: PlanningContext) -> bool:
    """The experiment is open and every candidate was decided on its merits."""
    return (
        context.experiment_status is ExperimentStatus.ACTIVE
        and bool(context.nodes)
        and all(node.status in _SETTLED for node in context.nodes)
    )


def _best_parent(context: PlanningContext) -> _Parent | None:
    primary = context.objective.primary
    eligible: list[_Parent] = []
    for node in context.nodes:
        decision = node.latest_decision
        if (
            node.status is not ExperimentNodeStatus.COMPLETED
            or decision is None
            or decision.outcome is not DecisionOutcome.BRANCH
        ):
            continue
        decided = set(decision.evaluation_result_ids)
        measured = [
            (metric.value, str(evaluation.evaluation_result_id))
            for evaluation in node.evaluations
            if evaluation.evaluation_result_id in decided
            for metric in evaluation.metrics
            if metric.name == primary.name and metric.slice is None
        ]
        if len(measured) != 1:
            continue
        ((value, result_id),) = measured
        eligible.append(_Parent(node, decision, value, result_id))
    if not eligible:
        return None
    sign = -1 if primary.direction == "maximize" else 1
    return min(eligible, key=lambda parent: (sign * parent.value, str(parent.node.node_id)))


# ---- binding -----------------------------------------------------------------------------

PLANNERS: Mapping[str, Callable[[PlannerSpec], Planner]] = {
    "rule-based": RuleBasedPlanner.from_spec,
    "no-op": NoOpPlanner.from_spec,
}
"""The built-in planners by kind: how a recorded ``PlannerSpec`` becomes a planner."""


def bind_planner(
    spec: PlannerSpec, planners: Mapping[str, Callable[[PlannerSpec], Planner]] = PLANNERS
) -> Planner:
    """Resolve *spec* to a planner bound under it; its ``spec`` is what to record.

    Raises:
        PlannerConfigurationError: If no planner has the kind, or it refuses the spec.
    """
    factory = planners.get(spec.kind)
    if factory is None:
        raise PlannerConfigurationError(
            spec.kind, (f"no planner of kind {spec.kind!r}; known: {sorted(planners)}",)
        )
    return factory(spec)
