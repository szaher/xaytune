"""RayTuneSearchProvider: Ray Tune's OptunaSearch behind the generic search contract (PR-034).

Real Ray Tune and Optuna (the ``ray-tune`` extra); no Ray cluster, because the
provider never starts or contacts one. Skipped without the extra, failed
under ``XAYTUNE_REQUIRE_RAY_TUNE=1`` (CI's ``ray`` job). The boundary tests at
the end need neither.

TPE draws its first ten suggestions at random (Optuna's ``n_startup_trials``)
and models the observations after that, so the tests that show observations
and direction steering the search run past ten trials.
"""

from __future__ import annotations

import ast
import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.test_search.search_support import SPACE, Experiment, measured, run
from xaytune.core.domain.decision import DecisionOutcome
from xaytune.core.domain.search import SearchProviderSpec, SearchSpace, apply_parameters
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.immutable import FrozenDict
from xaytune.core.state.status import ExperimentNodeStatus
from xaytune.search import (
    SearchHistoryError,
    SearchPlanner,
    SearchProvider,
    SearchRefusedError,
    search_planner_factory,
)

REPO = Path(__file__).resolve().parents[2]


def _tune() -> Any:
    pytest.importorskip("optuna", reason="needs the ray-tune extra")
    pytest.importorskip("ray.tune", reason="needs the ray-tune extra")
    from xaytune.ray import search

    return search


def provider_spec(**config: Any) -> SearchProviderSpec:
    return SearchProviderSpec(
        kind="ray-tune", config=FrozenDict({"search_space": SPACE, "seed": 7, **config})
    )


def planner(**config: Any) -> SearchPlanner:
    tune = _tune()
    spec = PlannerSpec(
        kind="search", config=FrozenDict({"provider": provider_spec(**config).model_dump()})
    )
    return search_planner_factory({"ray-tune": tune.RayTuneSearchProvider.from_spec})(spec)


def _search(experiment: Experiment, rounds: int, **config: Any) -> list[str]:
    for _ in range(rounds):
        assert experiment.step(planner(**config)) is not None
    return [node.candidate_fingerprint for node in experiment.nodes]


# ---- the space, as Tune's ---------------------------------------------------------------


@pytest.mark.ray_tune
def test_the_space_becomes_tunes_domains_and_optunas_distributions() -> None:
    tune = _tune()
    import optuna
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search.sample import Categorical, Float, Integer

    space = SearchSpace.model_validate(
        {
            "parameters": [
                {"name": "lr", "type": "float", "path": "a.lr", "low": 1e-5, "high": 1e-3,
                 "log": True},
                {"name": "dropout", "type": "float", "path": "a.d", "low": 0.0, "high": 0.5},
                {"name": "rank", "type": "int", "path": "a.r", "low": 4, "high": 64},
                {"name": "layers", "type": "int", "path": "a.l", "low": 1, "high": 32,
                 "log": True},
                {"name": "kind", "type": "choice", "path": "a.k", "values": [2, 2.5, True, "x"]},
            ]
        }
    )  # fmt: skip
    domains = tune.tune_search_space(space)
    assert list(domains) == ["dropout", "kind", "layers", "lr", "rank"]
    assert isinstance(domains["lr"], Float) and isinstance(domains["rank"], Integer)
    assert isinstance(domains["kind"], Categorical)
    assert (domains["rank"].lower, domains["rank"].upper) == (4, 65), "Tune's top is exclusive"

    distributions = OptunaSearch.convert_search_space(domains)
    assert distributions["lr"] == optuna.distributions.FloatDistribution(1e-5, 1e-3, log=True)
    assert distributions["dropout"] == optuna.distributions.FloatDistribution(0.0, 0.5)
    assert distributions["rank"] == optuna.distributions.IntDistribution(4, 64)
    assert distributions["layers"] == optuna.distributions.IntDistribution(1, 32, log=True)
    assert distributions["kind"].choices == (2, 2.5, True, "x")
    assert [type(c) for c in distributions["kind"].choices] == [int, float, bool, str]


# ---- the provider -------------------------------------------------------------------------


@pytest.mark.ray_tune
def test_the_provider_is_bound_like_any_search_provider() -> None:
    tune = _tune()
    provider = tune.RayTuneSearchProvider.from_spec(provider_spec())
    assert isinstance(provider, SearchProvider)
    assert (provider.descriptor.provider, provider.descriptor.name) == ("xaytune", "ray-tune")
    assert provider.spec.version == "1.0.0"
    assert provider.spec.config["algorithm"] == "tpe", "the default is recorded"

    from xaytune.search import SearchProviderConfigurationError

    with pytest.raises(SearchProviderConfigurationError, match="algorithm"):
        tune.RayTuneSearchProvider.from_spec(provider_spec(algorithm="bohb"))


@pytest.mark.ray_tune
@pytest.mark.parametrize("algorithm", ["tpe", "random"])
def test_a_restarted_search_is_the_same_search(algorithm: str) -> None:
    """Twelve rounds -- past TPE's random start -- with a kept planner and a new one each time."""
    kept, restarted = Experiment(), Experiment()
    restarted.nodes[0] = kept.nodes[0]
    search = planner(algorithm=algorithm)
    for _ in range(12):
        a = kept.step(search)
        b = restarted.step(planner(algorithm=algorithm))
        assert a is not None and b is not None
        assert a.candidate == b.candidate
        assert a.mutation["search"]["parameters"] == b.mutation["search"]["parameters"]
    assert len({node.candidate_fingerprint for node in kept.nodes}) == 13


@pytest.mark.ray_tune
def test_a_suggestion_never_branched_is_suggested_again() -> None:
    experiment = Experiment()
    _search(experiment, 3)
    (lost,) = run(planner().propose(experiment.context()))
    (again,) = run(planner().propose(experiment.context()))
    assert again.proposal_fingerprint() == lost.proposal_fingerprint()


@pytest.mark.ray_tune
def test_tpe_is_steered_by_the_observations_and_the_direction() -> None:
    base = Experiment()
    climbs = _search(base, 12)

    flat = Experiment()
    flat.nodes[0] = base.nodes[0]
    for _ in range(12):
        (proposal,) = run(planner().propose(flat.context()))
        flat.settle(flat.branch(proposal), metrics=measured(0.5))
    assert [n.candidate_fingerprint for n in flat.nodes[:11]] == climbs[:11], "random start"
    assert [n.candidate_fingerprint for n in flat.nodes[11:]] != climbs[11:], "then modelled"

    down = Experiment(direction="minimize")
    down.nodes[0] = base.nodes[0]
    assert _search(down, 12)[11:] != climbs[11:]


@pytest.mark.ray_tune
def test_what_tune_is_told_for_each_outcome() -> None:
    """Measured → COMPLETE with the value; anything else → FAIL, never a value."""
    tune = _tune()
    import optuna

    searchers: list[Any] = []

    class Watched(tune.RayTuneSearchProvider):
        def searcher(self, objective):
            searcher = super().searcher(objective)
            searchers.append(searcher)
            return searcher

    experiment = Experiment()
    outcomes = [
        (ExperimentNodeStatus.COMPLETED, DecisionOutcome.BRANCH, None),
        (ExperimentNodeStatus.COMPLETED, DecisionOutcome.BRANCH, ()),
        (ExperimentNodeStatus.REJECTED, DecisionOutcome.REJECT, None),
    ]
    bind = search_planner_factory({"ray-tune": Watched.from_spec})
    spec = PlannerSpec(kind="search", config=FrozenDict({"provider": provider_spec().model_dump()}))
    for status, outcome, metrics in outcomes:
        (proposal,) = run(bind(spec).propose(experiment.context()))
        experiment.settle(
            experiment.branch(proposal), status=status, outcome=outcome, metrics=metrics
        )
    searchers.clear()
    run(bind(spec).propose(experiment.context()))
    (searcher,) = searchers
    trials = searcher._search._ot_study.trials
    states = [trial.state for trial in trials[:3]]
    assert states == [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.FAIL,
    ]
    first = experiment.nodes[1]
    assert trials[0].value == first.evaluations[0].metrics[0].value
    assert trials[1].value is None and trials[2].value is None


@pytest.mark.ray_tune
def test_only_new_valid_candidates_are_proposed() -> None:
    tiny = {
        "parameters": [
            {"name": "beta", "type": "choice", "path": "training.algorithm.params.beta",
             "values": [0.1, 0.2, 0.5]},
        ]
    }  # fmt: skip
    experiment = Experiment()
    proposed = []
    while (proposal := experiment.step(planner(search_space=tiny))) is not None:
        proposed.append(proposal.candidate.training.algorithm.params["beta"])
        assert len(proposed) <= 2
    assert sorted(proposed) == [0.2, 0.5], "0.1 is the base; each other point once"


@pytest.mark.ray_tune
def test_every_suggestion_is_a_full_validated_candidate() -> None:
    experiment = Experiment()
    space = SearchSpace.model_validate(SPACE)
    for _ in range(4):
        proposal = experiment.step(planner())
        assert proposal is not None
        parameters = proposal.mutation["search"]["parameters"]
        assert apply_parameters(experiment.root.candidate, space, parameters) == proposal.candidate
        assert isinstance(parameters["learning_rate"], float)
        assert isinstance(parameters["rank"], int) and 4 <= parameters["rank"] <= 64
        assert parameters["beta"] in (0.1, 0.2, 0.5)


@pytest.mark.ray_tune
def test_a_record_from_another_seed_is_refused() -> None:
    experiment = Experiment()
    _search(experiment, 2)
    with pytest.raises(SearchHistoryError):
        run(planner(seed=8).propose(experiment.context()))


@pytest.mark.ray_tune
def test_more_than_one_suggestion_is_refused() -> None:
    tune = _tune()
    from tests.test_search.test_search_contract import _search_context

    provider = tune.RayTuneSearchProvider.from_spec(provider_spec())
    with pytest.raises(SearchRefusedError):
        run(provider.suggest(_search_context(Experiment()), count=2))


@pytest.mark.ray_tune
def test_the_same_ray_tune_search_whatever_runtime_executes_the_candidates(
    tmp_path: Path,
) -> None:
    """The controller, the search planner and Ray Tune -- under local, ray-jobs and ray-train."""
    tune = _tune()
    from tests.test_experiment import test_search_host as host_test

    providers = {"ray-tune": tune.RayTuneSearchProvider.from_spec}

    def planner_spec(seed: int = 3) -> PlannerSpec:
        provider = SearchProviderSpec(
            kind="ray-tune", config=FrozenDict({"search_space": host_test.RANKS, "seed": seed})
        )
        return PlannerSpec(kind="search", config=FrozenDict({"provider": provider.model_dump()}))

    searches = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(host_test, "PROVIDERS", providers)
        patch.setattr(host_test, "_planner", planner_spec)
        for kind in ("local", "ray-jobs", "ray-train"):
            searches[kind] = asyncio.run(host_test._run(tmp_path / kind, kind))
    local = searches["local"]
    assert len(local["ranks"]) >= 2 and len(set(local["ranks"])) == len(local["ranks"])
    assert all(
        origin.mutation["search"]["provider"]["name"] == "ray-tune"
        for origin in local["origins"][1:]
    )
    for kind, search in searches.items():
        assert {k: search[k] for k in ("status", "ranks", "suggestions", "values")} == {
            k: local[k] for k in ("status", "ranks", "suggestions", "values")
        }, kind


# ---- review: choices Optuna cannot tell apart, and the bound engine ------------------------


@pytest.mark.ray_tune
@pytest.mark.parametrize(
    "values", [[1, 1.0], [1, True], [0, False], [1.0, True], ["a", 0, 0.0]], ids=str
)
def test_choices_python_equality_merges_are_refused(values: list) -> None:
    """Distinct to Xaytune's identity, one value to Optuna: refused, not searched wrongly."""
    tune = _tune()
    from xaytune.search import SearchProviderConfigurationError

    space = {"parameters": [{"name": "c", "type": "choice", "path": "a.c", "values": values}]}
    assert SearchSpace.model_validate(space), "the generic contract keeps them distinct"
    with pytest.raises(SearchProviderConfigurationError, match="cannot tell apart"):
        tune.RayTuneSearchProvider.from_spec(provider_spec(search_space=space))


@pytest.mark.ray_tune
def test_choices_optuna_can_tell_apart_are_searched() -> None:
    tune = _tune()
    space = {
        "parameters": [
            {"name": "dtype", "type": "choice", "path": "training.precision.dtype",
             "values": ["bf16", "fp16"]},
        ]
    }  # fmt: skip
    provider = tune.RayTuneSearchProvider.from_spec(provider_spec(search_space=space))
    assert provider.config.search_space.parameters[0].values == ("bf16", "fp16")


@pytest.mark.ray_tune
def test_the_exact_ray_and_optuna_releases_are_the_searchs_engine() -> None:
    from importlib import metadata

    search = planner()
    engine = dict(search.spec.config["provider"]["config"]["engine"])
    assert engine == {"ray": metadata.version("ray"), "optuna": metadata.version("optuna")}
    (proposal,) = run(search.propose(Experiment().context()))
    assert dict(proposal.mutation["search"]["provider"]["engine"]) == engine


@pytest.mark.ray_tune
@pytest.mark.parametrize("distribution", ["ray", "optuna"])
def test_a_search_recorded_under_other_releases_is_refused(distribution: str) -> None:
    """A patch release inside the pinned minor is still another engine."""
    tune = _tune()
    from xaytune.planning import PlannerConfigurationError

    recorded = planner().spec
    provider = dict(recorded.config["provider"])
    config = dict(provider["config"])
    config["engine"] = {**config["engine"], distribution: "0.0.0-recorded"}
    other = recorded.model_copy(
        update={
            "version": None,
            "config": FrozenDict({"provider": {**provider, "config": config}}),
        }
    )
    bind = search_planner_factory({"ray-tune": tune.RayTuneSearchProvider.from_spec})
    with pytest.raises(PlannerConfigurationError, match=f"engine {distribution}"):
        bind(other)


@pytest.mark.ray_tune
def test_without_the_extra_binding_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    tune = _tune()
    from importlib import metadata

    from xaytune.search import SearchProviderConfigurationError

    real = metadata.version

    def without_optuna(name: str) -> str:
        if name == "optuna":
            raise metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(tune.metadata, "version", without_optuna)
    with pytest.raises(SearchProviderConfigurationError, match="optuna not installed"):
        tune.RayTuneSearchProvider.from_spec(provider_spec())


# ---- the boundary: needs neither Ray Tune nor Optuna ---------------------------------------


def test_ray_tune_and_optuna_are_imported_only_by_the_adapter() -> None:
    importers = []
    for path in sorted((REPO / "xaytune").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            if any(n == "optuna" or n.startswith(("optuna.", "ray.tune")) for n in names):
                importers.append(str(path.relative_to(REPO)))
    assert sorted(set(importers)) == ["xaytune/ray/search.py"]


def test_importing_xaytune_imports_neither() -> None:
    probe = (
        "import sys\n"
        "import xaytune.search, xaytune.ray, xaytune.ray.search, xaytune.experiment, "
        "xaytune.planning\n"
        "print(sorted(m for m in sys.modules if m == 'optuna' or m.startswith(('optuna.', "
        "'ray.tune'))))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=REPO
    )
    assert out.stdout.strip() == "[]"
