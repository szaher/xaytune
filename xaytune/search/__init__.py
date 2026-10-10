"""Search providers: an algorithm proposes the next candidate; Xaytune owns the rest (PR-034).

```text
durable record ──▶ PlanningContext ──SearchPlanner──▶ SearchContext
                                         │  observe(CandidateObservation) for each ended trial
                                         ▼
                                  SearchProvider.suggest(context, count=1)
                                         │
                                         ▼
                            CandidateProposal ──▶ branching (PR-025) ──▶ ExperimentNode
```

**A search provider searches.** Xaytune owns experiments, candidate identity,
lineage, execution, decisions and durability. A provider never creates a
node, launches training or talks to a runtime; it returns full, validated
:class:`~xaytune.core.domain.planning.CandidateProposal` s, which reach the
experiment through the existing planner path -- :class:`SearchPlanner` is a
:class:`~xaytune.planning.Planner` -- so branching's provenance, staleness,
budget and duplicate checks govern them exactly as they govern any planner's.
The same search therefore runs whatever executes its candidates: local, Ray
Jobs or Ray Train are the experiment's runtime, and appear nowhere here.

**Restart: the history is the searcher's own sequence.** A provider holds no
durable state of its own, and nothing it suggests is remembered anywhere but
in the record. :class:`SequentialSearchProvider` rebuilds its algorithm for
every suggestion, from its bound configuration (space, algorithm, seed)
alone, and replays it:

```text
for suggestion 0, 1, 2, ... of a freshly seeded algorithm:
    the candidate it makes is the base or one it already made   → a duplicate:
        told what is known of that candidate, and asked again (bounded)
    the experiment has that candidate                            → a trial:
        observed     → told the observation; next suggestion
        not yet      → nothing to suggest: one trial at a time
    the experiment has not                                       → the next proposal,
        once every other candidate of the experiment was matched; otherwise the
        record is not this search's history, and the provider refuses
```

**A trial is proven, not matched.** A candidate equal to the algorithm's
suggestion is its trial only if the node's durable branch origin (PR-025)
says so: this planner and provider as bound, the engine, the space, the
suggestion index, the values and the history before it. A candidate made by
hand or by another planner -- even exactly the one the search would suggest
-- is refused. And the **engine** -- the exact releases of the libraries that
implement the algorithm -- is part of the bound spec: replay cannot catch a
release that reproduces the history and differs on the next suggestion.

So the same configuration and the same durable outcomes give the same next
proposal -- before a restart or after one, and after a suggestion that was
never branched (the record is unchanged, so the same candidate is suggested
again, and branching it twice is the same node). An already-existing
candidate is never proposed as new: it is either matched as a trial or
skipped as a duplicate. A record the algorithm no longer reproduces (another
library version, a hand-made node) is refused, never searched around.

**Observations are idempotent.** :meth:`SearchProvider.observe` records what
became of a trial; the same observation again changes nothing, and a
different one for the same candidate is an :class:`ObservationConflictError`.
The search planner derives every observation from the record
(:func:`~xaytune.core.domain.search.observation_of`) and replays them all
before each suggestion. Objective direction is the experiment's
:class:`~xaytune.core.domain.objective.Objective`'s, never guessed, and a
missing measurement is ``unmeasured``, never zero.

**One at a time.** ``count`` is 1. Parallel suggestions need admission and
scheduling semantics that do not exist yet, so a provider refuses more
(:class:`SearchRefusedError`).

Adapters live below this boundary: Ray Tune in :mod:`xaytune.ray.search`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import Field, StrictInt, ValidationError, field_validator
from typing_extensions import Self

from xaytune.core.capabilities import PluginDescriptor, require_supported_plugin
from xaytune.core.domain.objective import Objective
from xaytune.core.domain.planning import (
    CandidateProposal,
    NodeSummary,
    PlanningContext,
    Proposal,
    provenance_identity,
)
from xaytune.core.domain.search import (
    CANDIDATE_OBSERVATION_IDENTITY_VERSION,
    SEARCH_CONTEXT_IDENTITY_VERSION,
    SEARCH_PROVIDER_SPEC_IDENTITY_VERSION,
    SEARCH_SPACE_IDENTITY_VERSION,
    CandidateObservation,
    SearchCandidate,
    SearchContext,
    SearchProviderSpec,
    SearchSpace,
    apply_parameters,
    candidate_observation_identity_v1,
    observation_of,
    search_provider_spec_identity_v1,
)
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.errors import XaytuneError
from xaytune.core.fingerprint import canonical_encode, fingerprint
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw
from xaytune.planning import (
    PlannerConfigurationError,
    _bound_spec,
    _descriptor,
    _planning_stage,
    _provenance,
)

__all__ = [
    "SEARCH_HISTORY_IDENTITY_VERSION",
    "ObservationConflictError",
    "SearchError",
    "SearchHistoryError",
    "SearchPlanner",
    "SearchPlannerConfig",
    "SearchProvider",
    "SearchProviderConfigurationError",
    "SearchRefusedError",
    "Searcher",
    "SequentialSearchConfig",
    "SequentialSearchProvider",
    "bind_search_provider",
    "search_planner_factory",
]


class SearchProviderConfigurationError(ValueError):
    """A search provider spec cannot be bound: wrong kind, version or configuration."""

    def __init__(self, kind: str, reasons: tuple[str, ...]) -> None:
        self.kind = kind
        self.reasons = reasons
        super().__init__(f"search provider {kind!r} cannot be bound: " + "; ".join(reasons))


class SearchError(XaytuneError):
    """A search cannot go on safely; nothing was proposed."""


class SearchRefusedError(SearchError):
    """The provider does not do what was asked -- more than one suggestion at a time."""


class SearchHistoryError(SearchError):
    """The record is not this search's history: the algorithm does not reproduce it."""


class ObservationConflictError(SearchError):
    """A candidate was observed twice, differently. The first observation stands."""


@runtime_checkable
class SearchProvider(Protocol):
    """Proposes the experiment's next candidates from a search context and what it observed."""

    descriptor: PluginDescriptor
    spec: SearchProviderSpec
    """The bound spec it runs under: ``kind``, resolved ``version``, canonical config."""

    async def suggest(
        self, context: SearchContext, count: int = 1
    ) -> tuple[CandidateProposal, ...]:
        """Up to *count* new candidates, each a child of ``context.base``; ``()`` for none.

        Depends on the bound configuration, *context* and the observations
        made so far -- nothing else -- and mints nothing: every proposal
        carries ``context.provenance``. Never proposes a candidate in
        ``context.candidates``.
        """
        ...

    async def observe(self, observation: CandidateObservation) -> None:
        """Record what became of a candidate. Idempotent; a different one conflicts."""
        ...


def _bound_provider_spec(
    descriptor: PluginDescriptor, spec: SearchProviderSpec, config: FrozenDict
) -> SearchProviderSpec:
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
        raise SearchProviderConfigurationError(spec.kind, tuple(reasons))
    return SearchProviderSpec(
        kind=descriptor.name, version=descriptor.plugin_version, config=config
    )


def bind_search_provider(
    spec: SearchProviderSpec,
    providers: Mapping[str, Callable[[SearchProviderSpec], SearchProvider]],
) -> SearchProvider:
    """Resolve *spec* to a provider bound under it; its ``spec`` is what to record.

    No provider is built in: each is registered explicitly, by whoever has
    its dependencies (``{"ray-tune": RayTuneSearchProvider.from_spec}``).

    Raises:
        SearchProviderConfigurationError: If no provider has the kind, or it refuses the spec.
    """
    factory = providers.get(spec.kind)
    if factory is None:
        raise SearchProviderConfigurationError(
            spec.kind, (f"no search provider of kind {spec.kind!r}; known: {sorted(providers)}",)
        )
    return factory(spec)


def _provider_record(provider: SearchProvider) -> dict[str, Any]:
    """Which provider, bound how: what a proposal's ``mutation`` names it by."""
    descriptor, spec = provider.descriptor, provider.spec
    assert spec.version is not None, "a provider always runs under a bound spec"
    return {
        "provider": descriptor.provider,
        "name": descriptor.name,
        "version": descriptor.plugin_version,
        "api_version": descriptor.api_version,
        "engine": thaw(spec.config.get("engine")),
        "spec_identity_version": SEARCH_PROVIDER_SPEC_IDENTITY_VERSION,
        "spec_fingerprint": fingerprint(
            search_provider_spec_identity_v1(
                spec,
                provider=descriptor.provider,
                name=descriptor.name,
                plugin_version=descriptor.plugin_version,
                api_version=descriptor.api_version,
            )
        ),
    }


# ---- sequential, replayed search -----------------------------------------------------------


class Searcher(Protocol):
    """One run of a search algorithm, asked and told in order, from a fresh seed.

    A :class:`SequentialSearchProvider` makes a new one for every suggestion
    and replays the record into it, so it needs no state of its own beyond
    this run -- and must be deterministic: the same seed and the same calls
    give the same suggestions, in any process.
    """

    def suggest(self, index: int) -> Mapping[str, Any] | None:
        """Parameter values for suggestion *index* (0, 1, ...), by name; ``None`` when exhausted."""
        ...

    def complete(self, index: int, observation: CandidateObservation | None) -> None:
        """What became of suggestion *index*; ``None``: nothing is known (it was the base)."""
        ...


class SequentialSearchConfig(FrozenDomainModel):
    """What every sequential search is configured with.

    ``max_consecutive_duplicates`` bounds how many suggestions in a row may
    repeat a candidate the search already has before it gives up proposing
    -- a small discrete space runs out.
    """

    search_space: SearchSpace
    seed: StrictInt = Field(ge=0)
    max_consecutive_duplicates: StrictInt = Field(default=16, ge=0)
    engine: FrozenDict | None = None
    """The exact releases of what implements the algorithm, by distribution name.

    ``None`` as a request; binding records what is installed, and a recorded
    engine that is not what is installed is refused. Replay alone cannot
    catch a release that reproduces every recorded suggestion and differs on
    the next -- the lost one a restart must suggest again -- so the engine is
    part of the search's identity, and of every proposal's record.
    """

    @field_validator("engine")
    @classmethod
    def _names_and_versions(cls, engine: FrozenDict | None) -> FrozenDict | None:
        if engine is not None and not all(
            isinstance(name, str) and isinstance(version, str) and name and version
            for name, version in engine.items()
        ):
            raise ValueError("an engine names each distribution and its exact version")
        return engine


SEARCH_HISTORY_IDENTITY_VERSION = 1


class SequentialSearchProvider:
    """A search provider over a deterministic :class:`Searcher`, replayed for every suggestion.

    Subclasses name a ``descriptor`` and a ``config_type`` and build a fresh
    :meth:`searcher`; everything else -- binding, the observation ledger,
    replay, the duplicate rule, the proposal and its record -- is here, the
    same for every algorithm. See the module docstring for the replay.
    """

    descriptor: PluginDescriptor
    config_type: ClassVar[type[SequentialSearchConfig]] = SequentialSearchConfig

    def __init__(self, config: SequentialSearchConfig) -> None:
        installed = dict(self.engine_versions())
        if config.engine is None:
            config = config.model_copy(update={"engine": installed})
        elif dict(config.engine) != installed:
            recorded = dict(config.engine)
            raise SearchProviderConfigurationError(
                self.descriptor.name,
                tuple(
                    f"engine {name}: the search was recorded under {recorded.get(name)!r}, "
                    f"but {installed.get(name)!r} is installed"
                    for name in sorted(set(recorded) | set(installed))
                    if recorded.get(name) != installed.get(name)
                ),
            )
        self.config = config
        self.spec = _bound_provider_spec(
            self.descriptor,
            SearchProviderSpec(kind=self.descriptor.name),
            FrozenDict(config.model_dump(mode="json")),
        )
        self._observations: dict[str, CandidateObservation] = {}

    @classmethod
    def from_spec(cls, spec: SearchProviderSpec) -> Self:
        """Bind *spec*, validating its config into the provider's typed configuration.

        Raises:
            SearchProviderConfigurationError: Naming every problem with the spec.
        """
        try:
            config = cls.config_type.model_validate(thaw(spec.config))
        except ValidationError as invalid:
            raise SearchProviderConfigurationError(
                spec.kind,
                tuple(
                    f"{'.'.join(str(part) for part in error['loc']) or 'config'}: {error['msg']}"
                    for error in invalid.errors()
                ),
            ) from None
        provider = cls(config)
        provider.spec = _bound_provider_spec(cls.descriptor, spec, provider.spec.config)
        return provider

    @classmethod
    def engine_versions(cls) -> Mapping[str, str]:
        """The exact installed releases that implement the algorithm; ``{}`` for none.

        Raises:
            SearchProviderConfigurationError: If one is not installed.
        """
        return {}

    def searcher(self, objective: Objective) -> Searcher:
        """A fresh run of the algorithm, seeded from the configuration, optimizing *objective*."""
        raise NotImplementedError

    async def observe(self, observation: CandidateObservation) -> None:
        recorded = self._observations.get(observation.candidate_fingerprint)
        if recorded is None:
            self._observations[observation.candidate_fingerprint] = observation
            return
        if recorded != observation:
            raise ObservationConflictError(
                f"candidate {observation.candidate_fingerprint} was observed as "
                f"{recorded.outcome} ({recorded.fingerprint()}), and now as "
                f"{observation.outcome} ({observation.fingerprint()}); the first stands"
            )

    async def suggest(
        self, context: SearchContext, count: int = 1
    ) -> tuple[CandidateProposal, ...]:
        if count != 1:
            raise SearchRefusedError(
                f"{self.descriptor.name} suggests one candidate at a time, not {count}: "
                f"parallel suggestions have no admission semantics yet"
            )
        self._require_observations_of(context)
        return self._replay(context)

    # ---- the replay ------------------------------------------------------------------------

    def _require_observations_of(self, context: SearchContext) -> None:
        nodes = context.nodes_by_fingerprint
        problems = []
        for candidate, observation in sorted(self._observations.items()):
            if candidate == context.base_fingerprint:
                problems.append(f"the base candidate {candidate} was observed; it is no trial")
            elif nodes.get(candidate) != observation.node_id:
                problems.append(
                    f"candidate {candidate} was observed as node {observation.node_id}, which "
                    f"the experiment does not have"
                )
            if observation.metric != context.objective.primary.name:
                problems.append(
                    f"candidate {candidate} was observed on {observation.metric!r}, not the "
                    f"objective's {context.objective.primary.name!r}"
                )
        if problems:
            raise SearchHistoryError("; ".join(problems))

    def _replay(self, context: SearchContext) -> tuple[CandidateProposal, ...]:
        space = self.config.search_space
        searcher = self.searcher(context.objective)
        nodes = context.nodes_by_fingerprint
        trials = set(nodes) - {context.base_fingerprint}
        known: dict[str, CandidateObservation | None] = {context.base_fingerprint: None}
        history: list[dict[str, Any]] = []
        duplicates = 0
        index = 0
        while True:
            values = searcher.suggest(index)
            if values is None:
                self._require_all_matched(trials, known, "the algorithm is exhausted")
                return ()
            candidate = apply_parameters(context.base, space, values)
            identity = candidate.candidate_fingerprint()
            if identity in known:
                # Already the base, or already a trial: never new. The
                # algorithm is told what is known of it, and asked again.
                searcher.complete(index, known[identity])
                history.append(_entry(index, identity, "duplicate", known[identity]))
                duplicates += 1
                if duplicates > self.config.max_consecutive_duplicates:
                    self._require_all_matched(
                        trials, known, f"{duplicates} suggestions in a row were not new"
                    )
                    return ()
                index += 1
                continue
            duplicates = 0
            if identity in trials:
                self._require_proposed_here(context, identity, index, values, history)
                observation = self._observations.get(identity)
                if observation is None:
                    # This trial has not ended: one at a time.
                    self._require_all_matched(trials, known | {identity: None}, "it waits")
                    return ()
                searcher.complete(index, observation)
                known[identity] = observation
                history.append(_entry(index, identity, "trial", observation))
                index += 1
                continue
            self._require_all_matched(trials, known, "it proposes a new candidate")
            return (self._proposal(context, candidate, values, index, history),)

    def _require_proposed_here(
        self,
        context: SearchContext,
        candidate: str,
        index: int,
        values: Mapping[str, Any],
        history: list[dict[str, Any]],
    ) -> None:
        """Refuse a trial unless its durable branch origin proves this search proposed it here.

        Equal candidates are not equal histories: a candidate made by hand, or
        by another planner, may be exactly what the search would suggest. So
        the node's branch origin (PR-025) must name this planner as bound, and
        its ``mutation["search"]`` must name this provider as bound (engine
        included), this search space, this suggestion index and these values,
        after exactly this history.
        """
        node = context.nodes_by_fingerprint[candidate]
        origin = context.origins_by_fingerprint[candidate]
        expected = _record_of(self, index, values, history)
        problems = []
        if origin is None:
            problems.append("it has no branch origin: no planner proposed it")
        else:
            provenance = provenance_identity(origin.provenance)
            ours = provenance_identity(context.provenance)
            differing = sorted(
                name
                for name in provenance
                if not name.startswith("context_") and provenance[name] != ours[name]
            )
            if differing:
                problems.append(
                    f"another planner, or this one configured differently, proposed it "
                    f"({', '.join(differing)})"
                )
            recorded = thaw(origin.mutation).get("search")
            if not isinstance(recorded, dict):
                problems.append("its proposal carries no search record")
            else:
                problems.extend(
                    f"its search record's {name} is {recorded.get(name)!r}, not {value!r}"
                    for name, value in expected.items()
                    if canonical_encode(recorded.get(name)) != canonical_encode(value)
                )
        if problems:
            raise SearchHistoryError(
                f"node {node} has candidate {candidate}, which is suggestion {index} of this "
                f"search, but the record does not show this search proposed it: "
                + "; ".join(problems)
            )

    @staticmethod
    def _require_all_matched(
        trials: set[str], known: Mapping[str, CandidateObservation | None], when: str
    ) -> None:
        unmatched = sorted(trials - set(known))
        if unmatched:
            raise SearchHistoryError(
                f"the search does not reproduce the experiment's candidates {unmatched} before "
                f"{when}: the record is not this search's history (another algorithm version, "
                f"or candidates made some other way)"
            )

    def _proposal(
        self,
        context: SearchContext,
        candidate: Any,
        values: Mapping[str, Any],
        index: int,
        history: list[dict[str, Any]],
    ) -> CandidateProposal:
        space = self.config.search_space
        primary = context.objective.primary
        parameters = {name: values[name] for name in space.names}
        observed = sum(1 for entry in history if entry["feedback"] == "trial")
        return CandidateProposal(
            candidate=candidate,
            candidate_fingerprint=candidate.candidate_fingerprint(),
            parent_ids=(context.base_node_id,),
            hypothesis=(
                f"The {self.descriptor.name} search's next point may "
                f"{primary.direction} {primary.name}."
            ),
            reason=(
                f"suggestion {index} of the {self.descriptor.name} search over "
                f"{', '.join(space.names)} (seed {self.config.seed}), after {observed} observed "
                f"trial(s): " + ", ".join(f"{name}={value!r}" for name, value in parameters.items())
            ),
            mutation=FrozenDict(
                {
                    "search": {
                        **_record_of(self, index, values, history),
                        "context": {
                            "identity_version": SEARCH_CONTEXT_IDENTITY_VERSION,
                            "fingerprint": context.input_fingerprint(),
                        },
                    }
                }
            ),
            provenance=context.provenance,
        )


def _record_of(
    provider: SequentialSearchProvider,
    index: int,
    values: Mapping[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """What a proposal's search record says, apart from the context it was made in.

    Exactly what a later replay checks a trial against: the provider as
    bound, the space, the suggestion index, the values, and the history
    before it (its fingerprint, and how many trials it observed).
    """
    space = provider.config.search_space
    return {
        "provider": _provider_record(provider),
        "search_space": {
            "identity_version": SEARCH_SPACE_IDENTITY_VERSION,
            "fingerprint": space.fingerprint(),
        },
        "history": {
            "identity_version": SEARCH_HISTORY_IDENTITY_VERSION,
            "fingerprint": fingerprint(
                {
                    "kind": "search-history",
                    "identity_version": SEARCH_HISTORY_IDENTITY_VERSION,
                    "entries": history,
                }
            ),
            "observed": sum(1 for entry in history if entry["feedback"] == "trial"),
        },
        "suggestion": index,
        "parameters": {name: values[name] for name in space.names},
    }


def _entry(
    index: int, candidate: str, feedback: str, observation: CandidateObservation | None
) -> dict[str, Any]:
    return {
        "suggestion": index,
        "candidate_fingerprint": candidate,
        "feedback": feedback,
        "observation": None
        if observation is None
        else {
            "identity_version": CANDIDATE_OBSERVATION_IDENTITY_VERSION,
            "identity": candidate_observation_identity_v1(observation),
        },
    }


# ---- the planner -------------------------------------------------------------------------


class SearchPlannerConfig(FrozenDomainModel):
    """The search planner's configuration: the search provider, as a spec."""

    provider: SearchProviderSpec


class SearchPlanner:
    """A :class:`~xaytune.planning.Planner` whose proposals come from a search provider.

    ```text
    not the planning stage, or a quota exhausted      → nothing
    not exactly one root node                         → SearchHistoryError
    a non-root node that is not the root's only child → SearchHistoryError
    each other node that ended → provider.observe(observation_of(node))
    provider.suggest(SearchContext(base = the root, ...), count=1)
    ```

    The root is the base every candidate varies, and the parent of each; the
    root's own result is the experiment's starting point, not an observation
    of the search. Every proposal is checked before it is returned -- one at
    most, this planner's provenance, a child of the root, a candidate the
    experiment does not have -- and branching checks it all again.

    The provider is recorded inside the planner's spec, so the planner's
    spec fingerprint, on every proposal's provenance, covers the provider,
    its version and engine, search space, algorithm and seed.
    """

    descriptor = _descriptor("search", "1.0.0")

    def __init__(self, provider: SearchProvider) -> None:
        self.provider = provider
        self.spec = _bound_spec(
            self.descriptor,
            PlannerSpec(kind="search"),
            FrozenDict({"provider": provider.spec.model_dump(mode="json")}),
        )

    @classmethod
    def from_spec(
        cls,
        spec: PlannerSpec,
        providers: Mapping[str, Callable[[SearchProviderSpec], SearchProvider]],
    ) -> SearchPlanner:
        """Bind *spec* and the provider it names.

        Raises:
            PlannerConfigurationError: Naming every problem, the provider's included.
        """
        try:
            config = SearchPlannerConfig.model_validate(thaw(spec.config))
        except ValidationError as invalid:
            raise PlannerConfigurationError(
                spec.kind,
                tuple(
                    f"{'.'.join(str(part) for part in error['loc']) or 'config'}: {error['msg']}"
                    for error in invalid.errors()
                ),
            ) from None
        try:
            provider = bind_search_provider(config.provider, providers)
        except SearchProviderConfigurationError as refused:
            raise PlannerConfigurationError(
                spec.kind, tuple(f"provider: {reason}" for reason in refused.reasons)
            ) from None
        planner = cls(provider)
        planner.spec = _bound_spec(cls.descriptor, spec, planner.spec.config)
        return planner

    async def propose(self, context: PlanningContext) -> tuple[Proposal, ...]:
        if not _planning_stage(context):
            return ()
        if context.budget is not None and context.budget.exhausted:
            return ()
        base = _base(context)
        for node in context.nodes:
            if node is base:
                continue
            observation = observation_of(node, context.objective)
            if observation is not None:
                await self.provider.observe(observation)
        search = SearchContext(
            experiment_id=context.experiment_id,
            objective=context.objective,
            base_node_id=base.node_id,
            base=base.candidate,
            candidates=tuple(
                SearchCandidate(
                    node_id=node.node_id,
                    candidate_fingerprint=node.candidate_fingerprint,
                    branch_origin=node.branch_origin,
                )
                for node in context.nodes
            ),
            provenance=_provenance(self, context),
        )
        proposals = await self.provider.suggest(search, count=1)
        _require_proposable(proposals, search)
        return proposals


def _base(context: PlanningContext) -> NodeSummary:
    roots = [node for node in context.nodes if not node.parent_ids]
    if len(roots) != 1:
        raise SearchHistoryError(
            f"a search varies the experiment's one root candidate; experiment "
            f"{context.experiment_id} has {len(roots)} roots"
        )
    (base,) = roots
    strays = [
        str(node.node_id)
        for node in context.nodes
        if node is not base and node.parent_ids != (base.node_id,)
    ]
    if strays:
        raise SearchHistoryError(
            f"nodes {strays} are not children of the root {base.node_id} alone: the record is "
            f"not a search's history"
        )
    return base


def _require_proposable(proposals: tuple[Any, ...], context: SearchContext) -> None:
    """Refuse a provider's answer that is not one new child of the base, attributed to us."""
    problems = []
    if len(proposals) > 1:
        problems.append(f"{len(proposals)} proposals, for a count of 1")
    existing = context.nodes_by_fingerprint
    for proposal in proposals:
        if not isinstance(proposal, CandidateProposal):
            problems.append(f"a {type(proposal).__name__}, not a CandidateProposal")
            continue
        if proposal.provenance != context.provenance:
            problems.append("a proposal attributed to someone else")
        if proposal.parent_ids != (context.base_node_id,):
            problems.append(f"a proposal not branched from the base {context.base_node_id}")
        if proposal.candidate_fingerprint in existing:
            problems.append(
                f"candidate {proposal.candidate_fingerprint}, which the experiment already has "
                f"as node {existing[proposal.candidate_fingerprint]}"
            )
    if problems:
        raise SearchError("the search provider proposed " + "; ".join(problems))


def search_planner_factory(
    providers: Mapping[str, Callable[[SearchProviderSpec], SearchProvider]],
) -> Callable[[PlannerSpec], SearchPlanner]:
    """A planner factory for ``kind="search"``, binding providers from *providers*.

    Register it explicitly, beside the built-in planners::

        EmbeddedControllerHost(..., planners={
            **PLANNERS,
            "search": search_planner_factory({"ray-tune": RayTuneSearchProvider.from_spec}),
        })
    """

    def bind(spec: PlannerSpec) -> SearchPlanner:
        return SearchPlanner.from_spec(spec, providers)

    return bind
