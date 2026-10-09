"""A deterministic search without Ray Tune, and an experiment driven by it in memory.

``HillClimbSearch`` is the smallest algorithm that makes the contract
observable: its first suggestions are seeded random draws, and once anything
was measured it perturbs the best point so far -- so what it suggests next
depends on the seed, on every observation, on their order and on the
objective's direction. It records every call it receives, so a test can see
what the provider told it.

``Experiment`` stands in for the record and the controller: a root candidate
decided ``BRANCH``, planning contexts built from its nodes, proposals branched
into ``PLANNED`` children, children "run" to an outcome. Nothing about how a
candidate is executed exists here, as nothing about it exists in a planning
context.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from xaytune._version import __version__
from xaytune.core.capabilities import PLUGIN_API_VERSIONS, PluginDescriptor
from xaytune.core.domain.candidate import (
    AdapterSpec,
    AlgorithmSpec,
    CandidateSpec,
    DataSpec,
    ModelSpec,
    OptimizationSpec,
    TrainingKind,
    TrainingSpec,
)
from xaytune.core.domain.decision import DecisionOutcome
from xaytune.core.domain.objective import Objective, ObjectiveMetric
from xaytune.core.domain.planning import (
    CandidateBranchOrigin,
    CandidateProposal,
    DecisionSummary,
    EvaluationSummary,
    MetricSummary,
    NodeSummary,
    PlanningContext,
)
from xaytune.core.domain.search import (
    CandidateObservation,
    ChoiceParameter,
    FloatParameter,
    IntParameter,
    SearchProviderSpec,
)
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.ids import DecisionId, EvaluationId, ExperimentId, ExperimentNodeId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import DatasetRef, ModelRef
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus
from xaytune.search import (
    SearchPlanner,
    SearchProvider,
    SequentialSearchConfig,
    SequentialSearchProvider,
    search_planner_factory,
)

METRIC = "task_success"

SPACE = {
    "parameters": [
        {
            "name": "learning_rate",
            "type": "float",
            "path": "training.optimization.learning_rate",
            "low": 1e-5,
            "high": 1e-3,
            "log": True,
        },
        {"name": "rank", "type": "int", "path": "training.adapter.rank", "low": 4, "high": 64},
        {
            "name": "beta",
            "type": "choice",
            "path": "training.algorithm.params.beta",
            "values": [0.1, 0.2, 0.5],
        },
    ]
}


def base_candidate() -> CandidateSpec:
    return CandidateSpec(
        model=ModelSpec(model=ModelRef(uri="Qwen/Qwen3-8B")),
        data=DataSpec(dataset=DatasetRef(uri="./data/support-v4.jsonl", revision="sha256:v4")),
        training=TrainingSpec(
            kind=TrainingKind.SFT,
            algorithm=AlgorithmSpec(params=FrozenDict({"beta": 0.1})),
            adapter=AdapterSpec(
                type="lora",
                rank=16,
                alpha=32.0,
                target_modules=("q_proj", "v_proj"),
                metadata=FrozenDict({"note": "kept"}),
            ),
            optimization=OptimizationSpec(
                learning_rate=2e-5, micro_batch_size=4, gradient_accumulation=8, epochs=2
            ),
        ),
        metadata=FrozenDict({"ticket": "kept too"}),
    )


# ---- the algorithm -------------------------------------------------------------------------


class HillClimbSearcher:
    def __init__(self, config: SequentialSearchConfig, objective: Objective) -> None:
        self.rng = random.Random(config.seed)
        self.parameters = config.search_space.parameters
        self.maximize = objective.primary.direction == "maximize"
        self.limit = getattr(config, "limit", None)
        self.suggested: dict[int, dict[str, Any]] = {}
        self.best: tuple[float, dict[str, Any]] | None = None
        self.told: list[tuple[int, CandidateObservation | None]] = []

    def suggest(self, index: int) -> Mapping[str, Any] | None:
        if self.limit is not None and index >= self.limit:
            return None
        values = {p.name: self._draw(p) for p in self.parameters}
        self.suggested[index] = values
        return values

    def _draw(self, p: Any) -> Any:
        anchor = None if self.best is None else self.best[1][p.name]
        if isinstance(p, ChoiceParameter):
            return self.rng.choice(p.values)
        if isinstance(p, IntParameter):
            if anchor is None:
                return self.rng.randint(p.low, p.high)
            return min(p.high, max(p.low, anchor + self.rng.randint(-4, 4)))
        assert isinstance(p, FloatParameter)
        if anchor is None:
            return self.rng.uniform(p.low, p.high)
        return min(p.high, max(p.low, anchor * self.rng.uniform(0.5, 2.0)))

    def complete(self, index: int, observation: CandidateObservation | None) -> None:
        self.told.append((index, observation))
        if observation is None or observation.value is None:
            return
        value = observation.value
        if (
            self.best is None
            or (self.maximize and value > self.best[0])
            or (not self.maximize and value < self.best[0])
        ):
            self.best = (value, self.suggested[index])


class HillClimbConfig(SequentialSearchConfig):
    limit: int | None = None


class HillClimbSearch(SequentialSearchProvider):
    descriptor = PluginDescriptor(
        api_version=PLUGIN_API_VERSIONS[0],
        name="hill-climb",
        plugin_version="1.0.0",
        provider="tests",
        xaytune_version=__version__,
    )
    config_type = HillClimbConfig
    searchers: list[HillClimbSearcher] = []

    def searcher(self, objective: Objective) -> HillClimbSearcher:
        searcher = HillClimbSearcher(self.config, objective)
        type(self).searchers.append(searcher)
        return searcher


PROVIDERS: Mapping[str, Callable[[SearchProviderSpec], SearchProvider]] = {
    "hill-climb": HillClimbSearch.from_spec
}


def provider_spec(**config: Any) -> SearchProviderSpec:
    return SearchProviderSpec(
        kind="hill-climb", config=FrozenDict({"search_space": SPACE, "seed": 7, **config})
    )


def planner_spec(**config: Any) -> PlannerSpec:
    return PlannerSpec(
        kind="search", config=FrozenDict({"provider": provider_spec(**config).model_dump()})
    )


def planner(**config: Any) -> SearchPlanner:
    return search_planner_factory(PROVIDERS)(planner_spec(**config))


def run(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


# ---- the experiment ------------------------------------------------------------------------


def decided(
    candidate: CandidateSpec,
    *,
    node_id: ExperimentNodeId | None = None,
    parent: ExperimentNodeId | None = None,
    status: ExperimentNodeStatus = ExperimentNodeStatus.COMPLETED,
    outcome: DecisionOutcome | None = DecisionOutcome.BRANCH,
    metrics: tuple[MetricSummary, ...] = (),
) -> NodeSummary:
    result = EvaluationId.generate()
    return NodeSummary(
        node_id=node_id or ExperimentNodeId.generate(),
        status=status,
        parent_ids=() if parent is None else (parent,),
        candidate=candidate,
        candidate_fingerprint=candidate.candidate_fingerprint(),
        decisions=()
        if outcome is None
        else (
            DecisionSummary(
                decision_id=DecisionId.generate(),
                evaluation_cycle=1,
                outcome=outcome,
                engine_name="adaptive-threshold",
                engine_version="1.0.0",
                input_fingerprint="sha256:input",
                evaluation_result_ids=(result,),
            ),
        ),
        evaluations=(
            EvaluationSummary(evaluation_result_id=result, evaluation_cycle=1, metrics=metrics),
        ),
    )


def measured(value: float) -> tuple[MetricSummary, ...]:
    return (MetricSummary(name=METRIC, value=value, evaluator_name="support"),)


def score(candidate: CandidateSpec) -> float:
    """A smooth, deterministic objective: best near lr 3e-4, rank 32, beta 0.2."""
    training = candidate.training
    lr = training.optimization.learning_rate
    assert lr is not None and training.adapter is not None and training.adapter.rank is not None
    beta = training.algorithm.params["beta"]
    return round(
        0.9 - abs(lr - 3e-4) * 200 - abs(training.adapter.rank - 32) / 400 - abs(beta - 0.2) / 10,
        6,
    )


@dataclass
class Experiment:
    direction: str = "maximize"
    experiment_id: ExperimentId = field(default_factory=ExperimentId.generate)
    nodes: list[NodeSummary] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.nodes:
            self.nodes.append(decided(base_candidate(), metrics=measured(0.5)))

    @property
    def root(self) -> NodeSummary:
        return self.nodes[0]

    @property
    def objective(self) -> Objective:
        return Objective(primary=ObjectiveMetric(name=METRIC, direction=self.direction))  # type: ignore[arg-type]

    def context(self) -> PlanningContext:
        return PlanningContext(
            experiment_id=self.experiment_id,
            experiment_status=ExperimentStatus.ACTIVE,
            objective=self.objective,
            nodes=tuple(self.nodes),
        )

    def branch(self, proposal: CandidateProposal) -> NodeSummary:
        """Branching, as far as a planning context can tell: one PLANNED child per candidate."""
        for node in self.nodes:
            if node.candidate_fingerprint == proposal.candidate_fingerprint:
                return node
        (parent,) = proposal.parent_ids
        child = NodeSummary(
            node_id=ExperimentNodeId.generate(),
            status=ExperimentNodeStatus.PLANNED,
            parent_ids=(parent,),
            candidate=proposal.candidate,
            candidate_fingerprint=proposal.candidate_fingerprint,
            branch_origin=CandidateBranchOrigin.of(proposal),
        )
        self.nodes.append(child)
        return child

    def settle(
        self,
        node: NodeSummary,
        *,
        status: ExperimentNodeStatus = ExperimentNodeStatus.COMPLETED,
        outcome: DecisionOutcome | None = DecisionOutcome.BRANCH,
        metrics: tuple[MetricSummary, ...] | None = None,
    ) -> NodeSummary:
        settled = decided(
            node.candidate,
            node_id=node.node_id,
            parent=node.parent_ids[0],
            status=status,
            outcome=outcome,
            metrics=measured(score(node.candidate)) if metrics is None else metrics,
        ).model_copy(update={"branch_origin": node.branch_origin})
        self.nodes[self.nodes.index(node)] = settled
        return settled

    def step(self, search: SearchPlanner) -> CandidateProposal | None:
        """Plan once, branch what was proposed, and run it to its measured outcome."""
        proposals = run(search.propose(self.context()))
        if not proposals:
            return None
        (proposal,) = proposals
        self.settle(self.branch(proposal))
        return proposal
