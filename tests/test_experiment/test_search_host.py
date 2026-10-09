"""A search provider drives the adaptive loop through the existing planner path (PR-034).

```text
node root  LoRA 16  → 0.70 → BRANCH
   ↓ SearchPlanner (recorded, kind "search") → provider.suggest → CandidateProposal
   ↓ branching (PR-025) → PLANNED child → realization (PR-026) → run → evaluation → decision
node trial LoRA r   → value(r) → BRANCH ... until the target is met or no run is left
```

The host has no search-specific code: the search planner is registered like
any planner. Which runtime executes the candidates is the experiment's
runtime spec, and the search never sees it -- so the same search, configured
the same, makes the same candidates in the same order whether they run
locally, as Ray jobs or under Ray Train. The scripted runtime serves all
three kinds here; that the real Ray runtimes give the same durable history as
local is ``test_ray_controller.py``'s (PR-033).

A host that dies at any resting point -- before a suggestion, after one that
was never branched, after branching -- is succeeded by one that makes exactly
the same search.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation_fixtures import EVALUATORS
from tests.test_experiment.adaptive_fixtures import AdaptiveRuntime, LoRACompiler, adaptive_spec
from tests.test_search.search_support import PROVIDERS
from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.domain.planning import CandidateBranchOrigin
from xaytune.core.domain.policy import PolicyVerdict
from xaytune.core.domain.search import SearchProviderSpec
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.immutable import FrozenDict
from xaytune.core.state.status import ExperimentStatus
from xaytune.decision import AdaptiveThresholdDecisionEngine
from xaytune.experiment import EmbeddedControllerHost
from xaytune.planning import PLANNERS
from xaytune.policy import RulePolicyEngine
from xaytune.search import search_planner_factory

_TIMEOUT = 30
VALUES = {16: 0.70, 8: 0.61, 24: 0.76, 32: 0.83, 48: 0.79, 56: 0.77}
RANKS = {
    "parameters": [
        {
            "name": "rank",
            "type": "choice",
            "path": "training.adapter.rank",
            "values": [8, 24, 32, 48, 56],
        }
    ]
}


def _planner(seed: int = 3) -> PlannerSpec:
    provider = SearchProviderSpec(
        kind="hill-climb", config=FrozenDict({"search_space": RANKS, "seed": seed})
    )
    return PlannerSpec(kind="search", config=FrozenDict({"provider": provider.model_dump()}))


def _spec(tmp_path: Path, runtime: str, seed: int = 3) -> Any:
    from xaytune.experiment import RuntimeSpec

    spec = adaptive_spec(tmp_path, budget=BudgetSpec(max_runs=5))
    return spec.model_copy(
        update={
            "planner": _planner(seed),
            "runtime": RuntimeSpec(kind=runtime, config={"root": str(tmp_path / "runtime")}),
        }
    )


def _world(tmp_path: Path) -> dict[str, Any]:
    manager = CheckpointManager(SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles"))
    return {
        "manager": manager,
        "runtime": AdaptiveRuntime(manager, tmp_path, values=VALUES, oom_rank=None),
    }


def _host(tmp_path: Path, world: dict[str, Any]) -> EmbeddedControllerHost:
    runtime = world["runtime"]
    return EmbeddedControllerHost(
        tmp_path / "state.db",
        compilers={"native": lambda: LoRACompiler()},
        runtimes={kind: (lambda config: runtime) for kind in ("local", "ray-jobs", "ray-train")},
        evaluators=EVALUATORS,
        decision_engine=AdaptiveThresholdDecisionEngine(),
        policy=RulePolicyEngine(default=PolicyVerdict.ALLOW),
        checkpoint_manager=world["manager"],
        planners={**PLANNERS, "search": search_planner_factory(PROVIDERS)},
    )


def _search(host: EmbeddedControllerHost, experiment_id: Any) -> dict[str, Any]:
    """The search as the record holds it: each candidate's rank, value and branch origin."""
    aggregates = host.repository.aggregates
    nodes = sorted(
        aggregates.nodes_for_experiment(str(experiment_id)), key=lambda node: node.created_at
    )
    return {
        "status": aggregates.load_experiment(str(experiment_id)).status,
        "ranks": [node.candidate.candidate.training.adapter.rank for node in nodes],
        "values": [
            [
                m.value
                for r in aggregates.evaluation_results_for_node(str(node.id))
                for m in r.metrics
            ]
            for node in nodes
        ],
        "suggestions": [
            None
            if node.branch_origin is None
            else node.branch_origin.mutation["search"]["suggestion"]
            for node in nodes
        ],
        "origins": [node.branch_origin for node in nodes],
    }


async def _run(tmp_path: Path, runtime: str, seed: int = 3) -> dict[str, Any]:
    world = _world(tmp_path)
    host = _host(tmp_path, world)
    try:
        handle = await host.submit(_spec(tmp_path, runtime, seed))
        await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)
        return _search(host, handle.experiment_id)
    finally:
        await host.close()


def test_a_search_runs_the_adaptive_loop_through_branching(tmp_path: Path) -> None:
    search = asyncio.run(_run(tmp_path, "local"))

    assert search["status"] in (ExperimentStatus.SUCCEEDED, ExperimentStatus.BUDGET_EXHAUSTED)
    ranks = search["ranks"]
    assert ranks[0] == 16 and len(ranks) >= 3, "the root, then at least two searched trials"
    assert len(set(ranks)) == len(ranks), "never the same candidate twice"
    assert all(
        values == [VALUES[rank]] for rank, values in zip(ranks, search["values"], strict=True)
    )
    root, *trials = search["origins"]
    assert root is None
    for origin in trials:
        assert isinstance(origin, CandidateBranchOrigin)
        assert origin.provenance.planner_name == "search"
        record = origin.mutation["search"]
        assert record["provider"]["name"] == "hill-climb"
        assert set(record["parameters"]) == {"rank"}
    if search["status"] is ExperimentStatus.SUCCEEDED:
        assert ranks[-1] == 32, "only rank 32 meets the 0.82 target"


def test_the_same_search_whatever_runtime_executes_the_candidates(tmp_path: Path) -> None:
    searches = {
        kind: asyncio.run(_run(tmp_path / kind, kind))
        for kind in ("local", "ray-jobs", "ray-train")
    }
    local = searches["local"]
    for kind, search in searches.items():
        assert search["status"] == local["status"], kind
        assert search["ranks"] == local["ranks"], kind
        assert search["suggestions"] == local["suggestions"], kind
        assert search["values"] == local["values"], kind
    other_seed = asyncio.run(_run(tmp_path / "seed-4", "local", seed=4))
    assert other_seed["ranks"] != local["ranks"], "the seed is the search; the runtime is not"


async def _submit_and_wait(host: EmbeddedControllerHost, spec: Any) -> Any:
    handle = await host.submit(spec)
    return handle, await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)


async def _never(*args: Any, **kwargs: Any) -> None:
    return None


@pytest.mark.parametrize(
    "stopped",
    [
        "_continue_adaptive_experiment",
        "_materialize_candidate_proposal",
        "_realize_planned_candidate",
    ],
    ids=["before-suggesting", "suggested-never-branched", "branched-never-run"],
)
def test_a_new_host_makes_exactly_the_same_search(tmp_path: Path, stopped: str) -> None:
    expected = asyncio.run(_run(tmp_path / "uninterrupted", "local"))
    root = tmp_path / "interrupted"
    world = _world(root)
    lost: list[Any] = []

    async def scenario() -> dict[str, Any]:
        first = _host(root, world)
        try:
            if stopped == "_materialize_candidate_proposal":

                def suggested_then_dies(experiment_id: Any, proposal: Any, **_: Any) -> None:
                    lost.append(proposal)  # ... and the host dies before branching it

                first._materialize_candidate_proposal = suggested_then_dies  # type: ignore[method-assign]
            else:
                setattr(first, stopped, _never)
            handle, resting = await _submit_and_wait(first, _spec(root, "local"))
            assert resting.status is ExperimentStatus.ACTIVE
            experiment_id = handle.experiment_id
        finally:
            await first.close()

        second = _host(root, world)
        try:
            handle = await second.attach(experiment_id)
            await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)
            return _search(second, experiment_id)
        finally:
            await second.close()

    search = asyncio.run(scenario())
    assert search["status"] == expected["status"]
    assert search["ranks"] == expected["ranks"]
    assert search["suggestions"] == expected["suggestions"]
    if stopped == "_materialize_candidate_proposal":
        assert lost, "the first host suggested a candidate it never branched"
        branched = search["origins"][1]
        assert branched.proposal_fingerprint == lost[0].proposal_fingerprint(), (
            "the new host branched exactly the suggestion the dead one lost"
        )
