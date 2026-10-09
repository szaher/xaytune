"""The generic search contract: search space, context, observations, replay, the planner.

Pure tests over a deterministic in-memory search (``search_support``); no Ray
Tune. The Ray Tune adapter is ``test_ray_tune_search.py``; a search driving a
real controller on each runtime is ``tests/test_experiment/test_search_host.py``.
"""

from __future__ import annotations

import pytest

from tests.test_search.search_support import (
    METRIC,
    PROVIDERS,
    SPACE,
    Experiment,
    HillClimbSearch,
    base_candidate,
    decided,
    measured,
    planner,
    planner_spec,
    provider_spec,
    run,
)
from xaytune.core.domain.decision import DecisionOutcome
from xaytune.core.domain.objective import Objective, ObjectiveMetric
from xaytune.core.domain.planning import (
    CandidateBranchOrigin,
    CandidateProposal,
    EvidenceRef,
    MetricSummary,
    ProposalProvenance,
    planning_candidate_projection_v1,
)
from xaytune.core.domain.search import (
    CandidateObservation,
    SearchCandidate,
    SearchContext,
    SearchSpace,
    SearchSpaceError,
    apply_parameters,
    observation_of,
    search_context_identity_v1,
    search_space_identity_v1,
)
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.ids import ExperimentNodeId
from xaytune.core.immutable import FrozenDict
from xaytune.core.state.status import ExperimentNodeStatus
from xaytune.planning import Planner, PlannerConfigurationError
from xaytune.search import (
    ObservationConflictError,
    SearchError,
    SearchHistoryError,
    SearchPlanner,
    SearchProvider,
    SearchRefusedError,
    search_planner_factory,
)

COMPLETED, REJECTED = ExperimentNodeStatus.COMPLETED, ExperimentNodeStatus.REJECTED
FAILED, CANCELLED = ExperimentNodeStatus.FAILED, ExperimentNodeStatus.CANCELLED


def space(*parameters: dict) -> SearchSpace:
    return SearchSpace.model_validate({"parameters": list(parameters)})


LR = {
    "name": "lr",
    "type": "float",
    "path": "training.optimization.learning_rate",
    "low": 1e-5,
    "high": 1e-3,
}
RANK = {"name": "rank", "type": "int", "path": "training.adapter.rank", "low": 4, "high": 64}


# ---- the search space ---------------------------------------------------------------------


def test_parameters_are_kept_in_name_order_so_declaration_order_is_not_identity() -> None:
    a, b = space(RANK, LR), space(LR, RANK)
    assert a == b
    assert a.names == ("lr", "rank")
    assert a.fingerprint() == b.fingerprint()
    assert search_space_identity_v1(a) == {
        "kind": "search-space",
        "identity_version": 1,
        "parameters": [
            {
                "name": "lr",
                "path": "training.optimization.learning_rate",
                "type": "float",
                "low": 1e-5,
                "high": 1e-3,
                "log": False,
            },
            {
                "name": "rank",
                "path": "training.adapter.rank",
                "type": "int",
                "low": 4,
                "high": 64,
                "log": False,
            },
        ],
    }
    assert space(LR, {**RANK, "high": 32}).fingerprint() != a.fingerprint()


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ((LR, {**RANK, "name": "lr"}), "more than once"),
        ((LR, {**RANK, "path": LR["path"]}), "overlapping"),
        ((LR, {**RANK, "path": "training.optimization"}), "overlapping"),
        (({**LR, "path": "training.metadata.lr"},), "metadata"),
        (({**LR, "path": "training.Optimization"},), "not field names"),
        (({**LR, "low": 1e-3},), "below high"),
        (({**LR, "low": 0.0, "log": True},), "positive"),
        (({**RANK, "low": 1.5},), "valid integer"),
        (({**LR, "high": float("inf")},), "finite"),
        (({"name": "b", "type": "choice", "path": "a.b", "values": [1, 1]},), "more than once"),
        (({"name": "b", "type": "choice", "path": "a.b", "values": [1]},), "at least 2"),
        ((), "at least 1"),
    ],
)
def test_a_malformed_space_is_refused(parameters: tuple, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        space(*parameters)


def test_choices_keep_their_types() -> None:
    choice = space({"name": "b", "type": "choice", "path": "a.b", "values": [1, 1.0, True, "1"]})
    (parameter,) = choice.parameters
    assert [type(value) for value in parameter.values] == [int, float, bool, str]  # type: ignore[union-attr]
    assert parameter.contains(1.0) and not parameter.contains(2)  # type: ignore[union-attr]


# ---- applying values ----------------------------------------------------------------------


def test_values_set_exactly_their_fields_and_keep_everything_else() -> None:
    base = base_candidate()
    full = SearchSpace.model_validate(SPACE)
    candidate = apply_parameters(base, full, {"learning_rate": 3e-4, "rank": 32, "beta": 0.2})

    assert candidate.training.optimization.learning_rate == 3e-4
    assert candidate.training.adapter is not None and candidate.training.adapter.rank == 32
    assert candidate.training.algorithm.params["beta"] == 0.2
    assert candidate.candidate_fingerprint() != base.candidate_fingerprint()
    projection = planning_candidate_projection_v1(candidate)
    assert (
        projection["beyond_identity"] == planning_candidate_projection_v1(base)["beyond_identity"]
    )


def test_the_bases_own_values_make_the_base() -> None:
    base = base_candidate()
    same = apply_parameters(
        base, SearchSpace.model_validate(SPACE), {"learning_rate": 2e-5, "rank": 16, "beta": 0.1}
    )
    assert same.candidate_fingerprint() == base.candidate_fingerprint()
    assert planning_candidate_projection_v1(same) == planning_candidate_projection_v1(base)


@pytest.mark.parametrize(
    ("parameters", "values", "message"),
    [
        ((LR,), {"lr": 1.0}, "outside the parameter's domain"),
        ((LR,), {"lr": 1}, "outside the parameter's domain"),
        ((RANK,), {"rank": 8.0}, "outside the parameter's domain"),
        ((RANK,), {"rank": True}, "outside the parameter's domain"),
        ((LR,), {"lr": 1e-4, "rank": 8}, "the space's parameters are"),
        ((LR,), {}, "the space's parameters are"),
        (({**RANK, "path": "training.adapter.size"},), {"rank": 8}, "has no field"),
        (({**RANK, "path": "training.algorithm.params.k"},), {"rank": 8}, "has no field"),
        (({**RANK, "path": "reward.graders"},), {"rank": 8}, "has no field"),
        (
            ({**LR, "path": "training.optimization.max_grad_norm", "low": -1.0},),
            {"lr": -0.5},
            "valid candidate",
        ),
    ],
)
def test_values_that_do_not_make_a_valid_candidate_are_refused(
    parameters: tuple, values: dict, message: str
) -> None:
    with pytest.raises(SearchSpaceError, match=message):
        apply_parameters(base_candidate(), space(*parameters), values)


# ---- observations -------------------------------------------------------------------------


OBJECTIVE = Objective(primary=ObjectiveMetric(name=METRIC, direction="maximize"))
EXPERIMENT = Experiment().experiment_id


def test_a_measured_candidate_is_observed_with_its_value_and_evidence() -> None:
    node = decided(base_candidate(), metrics=measured(0.8))
    observation = observation_of(node, OBJECTIVE)
    assert observation is not None
    assert (observation.outcome, observation.value, observation.metric) == ("measured", 0.8, METRIC)
    (decision,) = node.decisions
    (evaluation,) = node.evaluations
    assert observation.evidence_refs == (
        EvidenceRef(kind="decision", id=str(decision.decision_id)),
        EvidenceRef(kind="evaluation-result", id=str(evaluation.evaluation_result_id)),
    )


@pytest.mark.parametrize(
    "metrics",
    [
        (),
        (MetricSummary(name="other", value=0.8, evaluator_name="e"),),
        (MetricSummary(name=METRIC, value=0.8, slice="hard", evaluator_name="e"),),
        tuple(measured(0.8)) + tuple(measured(0.7)),
    ],
    ids=["nothing", "another-metric", "sliced-only", "measured-twice"],
)
def test_a_missing_or_ambiguous_measurement_is_unmeasured_never_zero(metrics: tuple) -> None:
    observation = observation_of(decided(base_candidate(), metrics=metrics), OBJECTIVE)
    assert observation is not None
    assert (observation.outcome, observation.value) == ("unmeasured", None)


def test_a_completed_candidate_with_no_decision_is_unmeasured() -> None:
    observation = observation_of(
        decided(base_candidate(), outcome=None, metrics=measured(0.8)), OBJECTIVE
    )
    assert observation is not None and observation.outcome == "unmeasured"


@pytest.mark.parametrize(
    ("status", "outcome"),
    [(REJECTED, "rejected"), (FAILED, "failed"), (CANCELLED, "cancelled")],
)
def test_ended_candidates_have_explicit_outcomes_and_no_value(status, outcome) -> None:
    observation = observation_of(
        decided(base_candidate(), status=status, metrics=measured(0.95)), OBJECTIVE
    )
    assert observation is not None
    assert (observation.outcome, observation.value) == (outcome, None)


@pytest.mark.parametrize(
    "status",
    [
        ExperimentNodeStatus.CREATED,
        ExperimentNodeStatus.PLANNED,
        ExperimentNodeStatus.READY,
        ExperimentNodeStatus.ACTIVE,
        ExperimentNodeStatus.EVALUATING,
        ExperimentNodeStatus.DECIDING,
    ],
)
def test_a_candidate_in_flight_is_not_observed(status: ExperimentNodeStatus) -> None:
    assert observation_of(decided(base_candidate(), status=status, outcome=None), OBJECTIVE) is None


def test_an_observation_carries_a_value_exactly_when_measured() -> None:
    node = ExperimentNodeId.generate()
    with pytest.raises(ValueError, match="exactly when"):
        CandidateObservation(
            node_id=node, candidate_fingerprint="sha256:x", outcome="measured", metric=METRIC
        )
    with pytest.raises(ValueError, match="exactly when"):
        CandidateObservation(
            node_id=node,
            candidate_fingerprint="sha256:x",
            outcome="failed",
            metric=METRIC,
            value=0.0,
        )


# ---- the context --------------------------------------------------------------------------


def _provenance() -> ProposalProvenance:
    return ProposalProvenance(
        planner_provider="xaytune",
        planner_name="search",
        planner_version="1.0.0",
        planner_api_version="xaytune.plugins/v1alpha1",
        planner_spec_kind="search",
        planner_spec_version="1.0.0",
        planner_spec_identity_version=1,
        planner_spec_fingerprint="sha256:spec",
        context_identity_version=1,
        context_fingerprint="sha256:context",
    )


def _context(*candidates: SearchCandidate, base=None, base_id=None) -> SearchContext:
    base = base or base_candidate()
    base_id = base_id or ExperimentNodeId.generate()
    root = SearchCandidate(node_id=base_id, candidate_fingerprint=base.candidate_fingerprint())
    return SearchContext(
        experiment_id=EXPERIMENT,
        objective=OBJECTIVE,
        base_node_id=base_id,
        base=base,
        candidates=(root, *candidates),
        provenance=_provenance(),
    )


def test_the_base_must_be_one_of_the_experiments_candidates() -> None:
    base_id = ExperimentNodeId.generate()
    with pytest.raises(ValueError, match="not among"):
        SearchContext(
            experiment_id=EXPERIMENT,
            objective=OBJECTIVE,
            base_node_id=base_id,
            base=base_candidate(),
            candidates=(SearchCandidate(node_id=base_id, candidate_fingerprint="sha256:other"),),
            provenance=_provenance(),
        )


def test_a_candidate_or_node_appears_once() -> None:
    other = SearchCandidate(node_id=ExperimentNodeId.generate(), candidate_fingerprint="sha256:a")
    with pytest.raises(ValueError, match="more than once"):
        _context(other, other)
    with pytest.raises(ValueError, match="more than once"):
        _context(other, other.model_copy(update={"node_id": ExperimentNodeId.generate()}))


def test_the_context_identity_is_explicit_ordered_and_covers_every_field() -> None:
    a = SearchCandidate(node_id=ExperimentNodeId.generate(), candidate_fingerprint="sha256:a")
    b = SearchCandidate(node_id=ExperimentNodeId.generate(), candidate_fingerprint="sha256:b")
    base_id = ExperimentNodeId.generate()
    one, two = _context(a, b, base_id=base_id), _context(b, a, base_id=base_id)
    assert one.input_fingerprint() == two.input_fingerprint()
    identity = search_context_identity_v1(one)
    assert (identity["kind"], identity["identity_version"]) == ("search-context", 1)
    assert set(identity) == {
        "kind",
        "identity_version",
        "experiment_id",
        "objective",
        "base",
        "candidates",
        "provenance",
    }
    assert set(SearchContext.model_fields) == {
        "experiment_id",
        "objective",
        "base_node_id",
        "base",
        "candidates",
        "provenance",
    }, "a field added to the context must be added to its identity"
    assert set(identity["provenance"]) == set(ProposalProvenance.model_fields)
    assert set(CandidateObservation.model_fields) == {
        "node_id",
        "candidate_fingerprint",
        "outcome",
        "metric",
        "value",
        "evidence_refs",
    }, "a field added to an observation must be added to its identity"


# ---- the provider: same state, same suggestion --------------------------------------------


def test_the_providers_satisfy_the_protocols() -> None:
    assert isinstance(HillClimbSearch.from_spec(provider_spec()), SearchProvider)
    assert isinstance(planner(), Planner)
    assert SearchPlanner.descriptor.name == "search"


def test_the_same_configuration_and_record_give_the_same_proposal() -> None:
    experiment = Experiment()
    (first,) = run(planner().propose(experiment.context()))
    (again,) = run(planner().propose(experiment.context()))
    assert isinstance(first, CandidateProposal)
    assert first == again
    assert first.proposal_fingerprint() == again.proposal_fingerprint()


def test_a_search_is_reproduced_by_a_restart_before_every_suggestion() -> None:
    """One planner kept for the whole search, and a new one -- a restarted host -- each round."""
    kept, restarted = Experiment(), Experiment()
    kept.nodes[0] = restarted.nodes[0]
    search = planner()
    for _ in range(6):
        a = kept.step(search)
        b = restarted.step(planner())
        assert a is not None and b is not None
        assert a.candidate_fingerprint == b.candidate_fingerprint
        assert a.mutation["search"]["parameters"] == b.mutation["search"]["parameters"]
        assert a.mutation["search"]["suggestion"] == b.mutation["search"]["suggestion"]


def test_a_restart_after_a_suggestion_that_was_never_branched_suggests_it_again() -> None:
    experiment = Experiment()
    experiment.step(planner())
    (lost,) = run(planner().propose(experiment.context()))
    # The host dies before branching it. The record is unchanged, so:
    (again,) = run(planner().propose(experiment.context()))
    assert again.proposal_fingerprint() == lost.proposal_fingerprint()
    # ... and branching it twice is the same node -- no second candidate invented.
    assert experiment.branch(lost) is experiment.branch(again)
    assert len(experiment.nodes) == 3


def test_a_branched_trial_that_has_not_ended_holds_the_search() -> None:
    experiment = Experiment()
    (proposal,) = run(planner().propose(experiment.context()))
    child = experiment.branch(proposal)
    assert run(planner().propose(experiment.context())) == (), "not the planning stage"

    provider = HillClimbSearch.from_spec(provider_spec())
    search = _search_context(experiment)
    assert run(provider.suggest(search)) == (), "the provider waits for the trial too"
    experiment.settle(child)
    assert run(planner().propose(experiment.context())) != ()


def _search_context(experiment: Experiment) -> SearchContext:
    root = experiment.root
    return SearchContext(
        experiment_id=experiment.experiment_id,
        objective=experiment.objective,
        base_node_id=root.node_id,
        base=root.candidate,
        candidates=tuple(
            SearchCandidate(
                node_id=n.node_id,
                candidate_fingerprint=n.candidate_fingerprint,
                branch_origin=n.branch_origin,
            )
            for n in experiment.nodes
        ),
        provenance=_planner_provenance(experiment),
    )


def _planner_provenance(experiment: Experiment) -> ProposalProvenance:
    """What the search planner, as configured by ``planner()``, attributes proposals to."""
    from xaytune.planning import _provenance

    return _provenance(planner(), experiment.context())


def test_what_comes_next_depends_on_the_observations() -> None:
    a, b = Experiment(), Experiment()
    b.nodes[0] = a.nodes[0]
    for _ in range(3):
        a.step(planner())
    for _ in range(3):
        proposal = run(planner().propose(b.context()))[0]
        b.settle(b.branch(proposal), metrics=measured(0.1))  # every trial measures the same
    assert [n.candidate_fingerprint for n in a.nodes[:2]] == [
        n.candidate_fingerprint for n in b.nodes[:2]
    ], "the first suggestion depends on the seed alone"
    (next_a,) = run(planner().propose(a.context()))
    (next_b,) = run(planner().propose(b.context()))
    assert next_a.candidate_fingerprint != next_b.candidate_fingerprint


def test_the_direction_is_the_objectives() -> None:
    up, down = Experiment(direction="maximize"), Experiment(direction="minimize")
    down.nodes[0] = up.nodes[0]
    proposals = {}
    for experiment in (up, down):
        for _ in range(3):
            experiment.step(planner())
        HillClimbSearch.searchers.clear()
        (proposals[experiment.direction],) = run(planner().propose(experiment.context()))
        (searcher,) = HillClimbSearch.searchers
        assert searcher.maximize is (experiment.direction == "maximize")
    assert [n.candidate_fingerprint for n in up.nodes[:3]] == [
        n.candidate_fingerprint for n in down.nodes[:3]
    ], "one measurement is the best either way"
    assert up.nodes[3].candidate_fingerprint != down.nodes[3].candidate_fingerprint, (
        "after two, each climbs its own way"
    )


def test_the_seed_and_the_space_are_the_search() -> None:
    experiment = Experiment()
    (seven,) = run(planner().propose(experiment.context()))
    (eight,) = run(planner(seed=8).propose(experiment.context()))
    assert seven.candidate_fingerprint != eight.candidate_fingerprint
    assert seven.provenance.planner_spec_fingerprint != eight.provenance.planner_spec_fingerprint
    record7, record8 = seven.mutation["search"], eight.mutation["search"]
    assert record7["provider"]["spec_fingerprint"] != record8["provider"]["spec_fingerprint"]
    assert record7["search_space"] == record8["search_space"]


# ---- observations: idempotent, conflicts fail closed --------------------------------------


def _observation(node: ExperimentNodeId, value: float | None = 0.8) -> CandidateObservation:
    return CandidateObservation(
        node_id=node,
        candidate_fingerprint="sha256:trial",
        outcome="measured" if value is not None else "failed",
        metric=METRIC,
        value=value,
    )


def test_replaying_an_observation_changes_nothing() -> None:
    provider = HillClimbSearch.from_spec(provider_spec())
    node = ExperimentNodeId.generate()
    run(provider.observe(_observation(node)))
    run(provider.observe(_observation(node)))
    assert list(provider._observations.values()) == [_observation(node)]


@pytest.mark.parametrize("value", [0.7, None])
def test_a_conflicting_observation_fails_closed_and_the_first_stands(value) -> None:
    provider = HillClimbSearch.from_spec(provider_spec())
    node = ExperimentNodeId.generate()
    run(provider.observe(_observation(node)))
    with pytest.raises(ObservationConflictError, match="the first stands"):
        run(provider.observe(_observation(node, value)))
    assert list(provider._observations.values()) == [_observation(node)]


def test_a_changed_outcome_in_the_record_is_a_conflict_for_a_kept_planner() -> None:
    experiment = Experiment()
    search = planner()
    experiment.step(search)
    run(search.propose(experiment.context()))  # observes the trial
    trial = experiment.nodes[1]
    experiment.settle(trial, metrics=measured(0.123))  # re-decided differently
    with pytest.raises(ObservationConflictError):
        run(search.propose(experiment.context()))


def test_an_observation_the_record_does_not_explain_is_refused() -> None:
    experiment = Experiment()
    provider = HillClimbSearch.from_spec(provider_spec())
    run(provider.observe(_observation(ExperimentNodeId.generate())))
    with pytest.raises(SearchHistoryError, match="does not have"):
        run(provider.suggest(_search_context(experiment)))

    provider = HillClimbSearch.from_spec(provider_spec())
    root = experiment.root
    run(
        provider.observe(
            CandidateObservation(
                node_id=root.node_id,
                candidate_fingerprint=root.candidate_fingerprint,
                outcome="measured",
                metric=METRIC,
                value=0.5,
            )
        )
    )
    with pytest.raises(SearchHistoryError, match="base candidate"):
        run(provider.suggest(_search_context(experiment)))


def test_an_observation_on_another_metric_is_refused() -> None:
    experiment = Experiment()
    experiment.step(planner())
    trial = experiment.nodes[1]
    provider = HillClimbSearch.from_spec(provider_spec())
    run(
        provider.observe(
            CandidateObservation(
                node_id=trial.node_id,
                candidate_fingerprint=trial.candidate_fingerprint,
                outcome="measured",
                metric="loss",
                value=0.5,
            )
        )
    )
    with pytest.raises(SearchHistoryError, match="'loss'"):
        run(provider.suggest(_search_context(experiment)))


@pytest.mark.parametrize(
    ("status", "outcome", "metrics"),
    [
        (COMPLETED, DecisionOutcome.BRANCH, ()),
        (REJECTED, DecisionOutcome.REJECT, None),
    ],
    ids=["unmeasured", "rejected"],
)
def test_the_algorithm_is_told_the_outcome_and_never_a_made_up_value(
    status, outcome, metrics
) -> None:
    experiment = Experiment()
    (proposal,) = run(planner().propose(experiment.context()))
    experiment.settle(experiment.branch(proposal), status=status, outcome=outcome, metrics=metrics)
    HillClimbSearch.searchers.clear()
    run(planner().propose(experiment.context()))
    (searcher,) = HillClimbSearch.searchers
    ((index, told),) = [(i, o) for i, o in searcher.told if o is not None]
    assert index == 0
    assert told.outcome == ("unmeasured" if status is COMPLETED else "rejected")
    assert told.value is None


@pytest.mark.parametrize(("status", "outcome"), [(FAILED, "failed"), (CANCELLED, "cancelled")])
def test_failed_and_cancelled_trials_are_observed_as_such(status, outcome) -> None:
    """Not the planning stage for the controller -- but a provider still knows what they mean."""
    experiment = Experiment()
    (proposal,) = run(planner().propose(experiment.context()))
    trial = experiment.settle(experiment.branch(proposal), status=status, outcome=None)
    assert run(planner().propose(experiment.context())) == ()

    provider = HillClimbSearch.from_spec(provider_spec())
    observation = observation_of(trial, experiment.objective)
    assert observation is not None and observation.outcome == outcome
    run(provider.observe(observation))
    HillClimbSearch.searchers.clear()
    (next_one,) = run(provider.suggest(_search_context(experiment)))
    (searcher,) = HillClimbSearch.searchers
    assert searcher.told == [(0, observation)]
    assert next_one.mutation["search"]["history"]["observed"] == 1


# ---- never an existing candidate ----------------------------------------------------------


def test_a_suggestion_repeating_a_candidate_is_skipped_never_proposed() -> None:
    """A two-point space: the base is one point, so there is exactly one new candidate."""
    tiny = {
        "parameters": [
            {"name": "beta", "type": "choice", "path": "training.algorithm.params.beta",
             "values": [0.1, 0.2]},
        ]
    }  # fmt: skip
    experiment = Experiment()
    first = experiment.step(planner(search_space=tiny, max_consecutive_duplicates=8))
    assert first is not None
    assert first.candidate.training.algorithm.params["beta"] == 0.2
    again = planner(search_space=tiny, max_consecutive_duplicates=8)
    assert run(again.propose(experiment.context())) == ()
    fingerprints = [n.candidate_fingerprint for n in experiment.nodes]
    assert len(set(fingerprints)) == len(fingerprints) == 2


def test_a_finished_algorithm_proposes_nothing() -> None:
    experiment = Experiment()
    assert experiment.step(planner(limit=2)) is not None
    assert experiment.step(planner(limit=2)) is not None
    assert run(planner(limit=2).propose(experiment.context())) == ()


def test_a_record_the_algorithm_does_not_reproduce_is_refused() -> None:
    experiment = Experiment()
    experiment.step(planner())
    # The same experiment, searched with another seed: its trial is not this search's.
    with pytest.raises(SearchHistoryError, match="not this search's history"):
        run(planner(seed=8).propose(experiment.context()))


def test_a_hand_made_child_is_not_searched_around() -> None:
    experiment = Experiment()
    candidate = apply_parameters(
        experiment.root.candidate,
        SearchSpace.model_validate(SPACE),
        {"learning_rate": 1e-4, "rank": 8, "beta": 0.5},
    )
    experiment.nodes.append(decided(candidate, parent=experiment.root.node_id, metrics=measured(1)))
    with pytest.raises(SearchHistoryError, match="not this search's history"):
        run(planner().propose(experiment.context()))


def test_only_the_roots_children_make_a_search() -> None:
    experiment = Experiment()
    experiment.step(planner())
    grandchild = apply_parameters(
        experiment.root.candidate,
        SearchSpace.model_validate(SPACE),
        {"learning_rate": 1e-4, "rank": 8, "beta": 0.5},
    )
    experiment.nodes.append(
        decided(grandchild, parent=experiment.nodes[1].node_id, metrics=measured(1))
    )
    with pytest.raises(SearchHistoryError, match="children of the root"):
        run(planner().propose(experiment.context()))

    two_roots = Experiment()
    two_roots.nodes.append(decided(grandchild, metrics=measured(1)))
    with pytest.raises(SearchHistoryError, match="2 roots"):
        run(planner().propose(two_roots.context()))


# ---- one at a time -------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 2])
def test_one_suggestion_at_a_time(count: int) -> None:
    provider = HillClimbSearch.from_spec(provider_spec())
    with pytest.raises(SearchRefusedError, match="one candidate at a time"):
        run(provider.suggest(_search_context(Experiment()), count=count))


# ---- the proposal --------------------------------------------------------------------------


def test_the_proposal_is_a_full_child_of_the_base_with_the_searchs_record() -> None:
    experiment = Experiment()
    experiment.step(planner())
    context = experiment.context()
    search = planner()
    (proposal,) = run(search.propose(context))

    assert proposal.parent_ids == (experiment.root.node_id,)
    assert proposal.candidate_fingerprint not in context.candidate_fingerprints
    assert proposal.evidence_refs == ()
    provenance = proposal.provenance
    assert (provenance.planner_name, provenance.planner_spec_kind) == ("search", "search")
    assert provenance.context_fingerprint == context.input_fingerprint()

    record = proposal.mutation["search"]
    assert set(record) == {
        "provider",
        "search_space",
        "context",
        "history",
        "suggestion",
        "parameters",
    }
    descriptor = HillClimbSearch.descriptor
    assert {k: record["provider"][k] for k in ("provider", "name", "version", "api_version")} == {
        "provider": descriptor.provider,
        "name": descriptor.name,
        "version": descriptor.plugin_version,
        "api_version": descriptor.api_version,
    }
    assert record["search_space"] == {
        "identity_version": 1,
        "fingerprint": SearchSpace.model_validate(SPACE).fingerprint(),
    }
    assert record["context"]["identity_version"] == 1
    assert record["history"]["identity_version"] == 1
    assert record["history"]["observed"] == 1
    assert record["suggestion"] == 1
    assert set(record["parameters"]) == {"beta", "learning_rate", "rank"}
    rebuilt = apply_parameters(
        experiment.root.candidate, SearchSpace.model_validate(SPACE), record["parameters"]
    )
    assert rebuilt == proposal.candidate, "the record names exactly the candidate proposed"


def test_a_provider_that_proposes_what_it_must_not_is_refused() -> None:
    class Rogue(HillClimbSearch):
        mode = "existing"

        async def suggest(self, context, count=1):
            (proposal,) = await super().suggest(context, count)
            if self.mode == "two":
                return (proposal, proposal)
            if self.mode == "parent":
                return (proposal.model_copy(update={"parent_ids": (ExperimentNodeId.generate(),)}),)
            if self.mode == "provenance":
                other = proposal.provenance.model_copy(update={"context_fingerprint": "sha256:x"})
                return (proposal.model_copy(update={"provenance": other}),)
            return (
                proposal.model_copy(
                    update={
                        "candidate": context.base,
                        "candidate_fingerprint": context.base.candidate_fingerprint(),
                    }
                ),
            )

    experiment = Experiment()
    for mode, message in [
        ("existing", "already has"),
        ("two", "2 proposals"),
        ("parent", "not branched from the base"),
        ("provenance", "someone else"),
    ]:
        Rogue.mode = mode
        search = search_planner_factory({"hill-climb": Rogue.from_spec})(planner_spec())
        with pytest.raises(SearchError, match=message):
            run(search.propose(experiment.context()))


# ---- binding -------------------------------------------------------------------------------


def test_the_planner_records_the_bound_provider_and_rebinds_to_itself() -> None:
    search = planner()
    assert search.spec.kind == "search" and search.spec.version == "1.0.0"
    recorded = search.spec.config["provider"]
    assert (recorded["kind"], recorded["version"]) == ("hill-climb", "1.0.0")
    assert recorded["config"]["search_space"]["parameters"][0]["name"] == "beta", "canonical"
    # What a host does with the recorded spec: bind it again, and require the same.
    again = search_planner_factory(PROVIDERS)(search.spec.model_copy(update={"version": None}))
    assert again.spec == search.spec


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (PlannerSpec(kind="search", config=FrozenDict({})), "provider"),
        (
            PlannerSpec(
                kind="search",
                config=FrozenDict({"provider": {"kind": "optuna", "config": {}}}),
            ),
            "no search provider of kind 'optuna'",
        ),
        (
            PlannerSpec(
                kind="search",
                config=FrozenDict(
                    {"provider": {"kind": "hill-climb", "version": "9.9.9", "config": {}}}
                ),
            ),
            "search_space",
        ),
        (planner_spec(seed=-1), "seed"),
        (planner_spec(algorithm="tpe"), "algorithm"),
        (
            PlannerSpec(
                kind="search", config=FrozenDict({"provider": provider_spec().model_dump(), "x": 1})
            ),
            "x",
        ),
    ],
)
def test_a_planner_that_cannot_be_bound_is_refused(spec: PlannerSpec, message: str) -> None:
    with pytest.raises(PlannerConfigurationError, match=message):
        search_planner_factory(PROVIDERS)(spec)


def test_a_provider_version_that_moved_is_refused() -> None:
    spec = provider_spec().model_copy(update={"version": "2.0.0"})
    planner_config = FrozenDict({"provider": spec.model_dump()})
    with pytest.raises(PlannerConfigurationError, match="names version 2.0.0"):
        search_planner_factory(PROVIDERS)(PlannerSpec(kind="search", config=planner_config))


def test_the_search_planner_is_registered_explicitly_never_built_in() -> None:
    from xaytune.planning import PLANNERS

    assert "search" not in PLANNERS


# ---- planner gating -------------------------------------------------------------------------


def test_the_planner_acts_only_at_the_planning_stage() -> None:
    from xaytune.core.state.status import ExperimentStatus

    experiment = Experiment()
    paused = experiment.context().model_copy(update={"experiment_status": ExperimentStatus.PAUSED})
    assert run(planner().propose(paused)) == ()
    stopped = Experiment()
    stopped.nodes[0] = decided(
        base_candidate(), outcome=DecisionOutcome.STOP_SUCCEEDED, metrics=measured(0.9)
    )
    assert run(planner().propose(stopped.context())) == ()


def test_an_exhausted_quota_proposes_nothing() -> None:
    from decimal import Decimal

    from xaytune.core.domain.budget import BudgetDimension, BudgetStatus, DimensionStatus

    budget = BudgetStatus(
        dimensions=(
            DimensionStatus(
                dimension=BudgetDimension.RUNS,
                kind="quota",
                limit=Decimal(2),
                reserved=Decimal(0),
                committed=Decimal(0),
                consumed=Decimal(2),
                released=Decimal(0),
                outstanding=Decimal(0),
                remaining=Decimal(0),
            ),
        )
    )
    context = Experiment().context().model_copy(update={"budget": budget})
    assert run(planner().propose(context)) == ()


# ---- review: the engine is bound into the search's identity ---------------------------------


class Versioned(HillClimbSearch):
    """A search whose engine is a library at an exact release."""

    installed = {"hill-climb-lib": "1.2.3"}

    @classmethod
    def engine_versions(cls):
        return dict(cls.installed)


VERSIONED = {"hill-climb": Versioned.from_spec}


def test_binding_records_the_installed_engine_and_every_proposal_names_it() -> None:
    search = search_planner_factory(VERSIONED)(planner_spec())
    recorded = search.spec.config["provider"]["config"]["engine"]
    assert dict(recorded) == {"hill-climb-lib": "1.2.3"}
    (proposal,) = run(search.propose(Experiment().context()))
    assert dict(proposal.mutation["search"]["provider"]["engine"]) == {"hill-climb-lib": "1.2.3"}
    again = search_planner_factory(VERSIONED)(search.spec.model_copy(update={"version": None}))
    assert again.spec == search.spec, "a recorded engine rebinds to itself"


def test_a_search_recorded_under_another_engine_release_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """history H; 1.2.3 suggests X; the host dies before branching; an upgrade; refused, not Y."""
    experiment = Experiment()
    search = search_planner_factory(VERSIONED)(planner_spec())
    experiment.step(search)
    run(search.propose(experiment.context()))  # suggests X ... never branched
    recorded = search.spec

    monkeypatch.setattr(Versioned, "installed", {"hill-climb-lib": "1.2.4"})
    with pytest.raises(PlannerConfigurationError, match="recorded under '1.2.3'.*'1.2.4'"):
        search_planner_factory(VERSIONED)(recorded.model_copy(update={"version": None}))
    monkeypatch.setattr(Versioned, "installed", {"hill-climb-lib": "1.2.3", "extra": "1"})
    with pytest.raises(PlannerConfigurationError, match="engine extra"):
        search_planner_factory(VERSIONED)(recorded.model_copy(update={"version": None}))


def test_an_engine_spelled_wrong_is_refused() -> None:
    with pytest.raises(PlannerConfigurationError, match="engine"):
        search_planner_factory(PROVIDERS)(planner_spec(engine={"lib": 1}))


# ---- review: a trial is proven by its branch origin, not by its candidate --------------------


def _next(experiment: Experiment) -> CandidateProposal:
    (proposal,) = run(planner().propose(experiment.context()))
    return proposal


def _add_child(experiment: Experiment, proposal: CandidateProposal, origin, **settled) -> None:
    from xaytune.core.domain.planning import NodeSummary

    child = NodeSummary(
        node_id=ExperimentNodeId.generate(),
        status=ExperimentNodeStatus.PLANNED,
        parent_ids=(experiment.root.node_id,),
        candidate=proposal.candidate,
        candidate_fingerprint=proposal.candidate_fingerprint,
        branch_origin=origin,
    )
    experiment.nodes.append(child)
    if settled.get("settle", True):
        experiment.settle(child)


@pytest.mark.parametrize("settle", [True, False], ids=["ended", "in-flight"])
def test_a_hand_made_candidate_equal_to_the_next_suggestion_is_not_its_history(
    settle: bool,
) -> None:
    experiment = Experiment()
    experiment.step(planner())
    expected = _next(experiment)
    _add_child(experiment, expected, None, settle=settle)  # same candidate, made by hand
    if settle:
        with pytest.raises(SearchHistoryError, match="no branch origin"):
            run(planner().propose(experiment.context()))
    provider = HillClimbSearch.from_spec(provider_spec())
    for node in experiment.nodes[1:]:
        observation = observation_of(node, experiment.objective)
        if observation is not None:
            run(provider.observe(observation))
    with pytest.raises(SearchHistoryError, match="no branch origin"):
        run(provider.suggest(_search_context(experiment)))


def _forged(proposal: CandidateProposal, **search: object) -> CandidateBranchOrigin:
    from xaytune.core.immutable import thaw

    mutation = thaw(proposal.mutation)
    for path, value in search.items():
        target = mutation["search"]
        *parents, last = path.split("__")
        for key in parents:
            target = target[key]
        target[last] = value
    return CandidateBranchOrigin.of(proposal).model_copy(update={"mutation": mutation})


@pytest.mark.parametrize(
    ("forgery", "message"),
    [
        ({"suggestion": 7}, "suggestion"),
        ({"parameters__rank": 9}, "parameters"),
        ({"history__fingerprint": "sha256:another-past"}, "history"),
        ({"provider__spec_fingerprint": "sha256:other"}, "provider"),
        ({"provider__engine": {"lib": "0"}}, "provider"),
        ({"search_space__fingerprint": "sha256:other"}, "search_space"),
    ],
)
def test_a_trial_whose_search_record_is_not_this_searchs_is_refused(
    forgery: dict, message: str
) -> None:
    experiment = Experiment()
    experiment.step(planner())
    expected = _next(experiment)
    _add_child(experiment, expected, _forged(expected, **forgery))
    with pytest.raises(SearchHistoryError, match=f"search record's {message}"):
        run(planner().propose(experiment.context()))


def test_a_trial_branched_by_another_planner_is_refused() -> None:
    experiment = Experiment()
    experiment.step(planner())
    expected = _next(experiment)
    other = expected.provenance.model_copy(
        update={"planner_name": "rule-based", "planner_spec_kind": "rule-based"}
    )
    _add_child(
        experiment,
        expected,
        CandidateBranchOrigin.of(expected.model_copy(update={"provenance": other})),
    )
    with pytest.raises(SearchHistoryError, match="another planner.*planner_name"):
        run(planner().propose(experiment.context()))


def test_a_trial_with_no_search_record_is_refused() -> None:
    experiment = Experiment()
    experiment.step(planner())
    expected = _next(experiment)
    origin = CandidateBranchOrigin.of(expected).model_copy(update={"mutation": FrozenDict()})
    _add_child(experiment, expected, origin)
    with pytest.raises(SearchHistoryError, match="no search record"):
        run(planner().propose(experiment.context()))


def test_the_genuine_trial_is_accepted_and_only_its_planning_context_may_differ() -> None:
    experiment = Experiment()
    experiment.step(planner())
    proposal = _next(experiment)
    _add_child(experiment, proposal, CandidateBranchOrigin.of(proposal))
    (after,) = run(planner().propose(experiment.context()))
    assert after.mutation["search"]["history"]["observed"] == 2
    trial_origin = experiment.nodes[2].branch_origin
    assert trial_origin is not None
    assert trial_origin.provenance.context_fingerprint != after.provenance.context_fingerprint


# ---- review: training.api_version is not identity -------------------------------------------


def test_a_path_that_is_not_candidate_identity_is_refused() -> None:
    with pytest.raises(ValueError, match="not candidate identity"):
        space({"name": "v", "type": "choice", "path": "training.api_version",
               "values": ["xaytune.ai/v1alpha1", "xaytune.ai/v1alpha2"]})  # fmt: skip
    base = base_candidate()
    other = base.model_copy(
        update={"training": base.training.model_copy(update={"api_version": "xaytune.ai/v9"})}
    )
    assert other.candidate_fingerprint() == base.candidate_fingerprint(), "why it is refused"
