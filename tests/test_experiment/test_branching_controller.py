"""The controller's half of branching: only the bound planner's own proposals (PR-025).

The repository verifies the recorded spec; only the host holding the bound
planner knows its real descriptor, so a proposal claiming another provider,
API version or configuration is refused before the repository is asked.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.test_storage.test_branching import GROW_2, GROW_4, _bound
from tests.test_storage.test_planning_context import _evaluated, _rows
from xaytune.core.refs import Actor
from xaytune.core.state.status import ExperimentNodeStatus
from xaytune.experiment import EmbeddedControllerHost
from xaytune.planning import PlannerConfigurationError, _provenance_for, require_proposed_by
from xaytune.storage.control_plane import ProvenanceError

ACTOR = Actor(type="rule", id="rule-based-planner")


def _scenario(tmp_path, check):
    async def run():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            planner = _bound(GROW_2)
            experiment, node, *_ = _evaluated(
                host.repository, rank=16, value=0.79, planner=planner.spec
            )
            context = host.repository.planning_context(experiment.id)
            (proposal,) = await planner.propose(context)
            check(host, experiment, node, planner, context, proposal)
        finally:
            await host.close()

    asyncio.run(run())


def test_the_host_branches_a_proposal_of_its_recorded_planner(tmp_path) -> None:
    def check(host, experiment, node, planner, context, proposal):
        child = host._materialize_candidate_proposal(experiment.id, proposal, actor=ACTOR)
        assert child.status is ExperimentNodeStatus.PLANNED
        assert child.parent_ids == (node.id,)
        assert host.repository.aggregates.runs_for_node(str(child.id)) == ()

    _scenario(tmp_path, check)


@pytest.mark.parametrize(
    "change",
    [{"planner_provider": "someone-else"}, {"planner_api_version": "xaytune.plugins/v9"}],
    ids=["provider", "api-version"],
)
def test_a_proposal_claiming_another_descriptor_is_refused_before_storage(tmp_path, change) -> None:
    def check(host, experiment, node, planner, context, proposal):
        forged = proposal.model_copy(
            update={"provenance": proposal.provenance.model_copy(update=change)}
        )
        before = _rows(host.repository)
        with pytest.raises(PlannerConfigurationError):
            host._materialize_candidate_proposal(experiment.id, forged, actor=ACTOR)
        assert _rows(host.repository) == before

    _scenario(tmp_path, check)


def test_a_proposal_from_another_configuration_is_refused_by_the_host(tmp_path) -> None:
    def check(host, experiment, node, planner, context, proposal):
        other = _bound(GROW_4)
        forged = proposal.model_copy(
            update={"provenance": _provenance_for(other, context.input_fingerprint())}
        )
        with pytest.raises(PlannerConfigurationError, match="planner_spec_fingerprint"):
            require_proposed_by(planner, forged)
        with pytest.raises(PlannerConfigurationError):
            host._materialize_candidate_proposal(experiment.id, forged, actor=ACTOR)

    _scenario(tmp_path, check)


def test_an_experiment_with_no_planner_is_refused_by_the_host(tmp_path) -> None:
    async def run():
        host = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            planner = _bound(GROW_2)
            experiment, *_ = _evaluated(host.repository, rank=16, value=0.79)
            (proposal,) = await planner.propose(host.repository.planning_context(experiment.id))
            with pytest.raises(ProvenanceError, match="records no planner"):
                host._materialize_candidate_proposal(experiment.id, proposal, actor=ACTOR)
        finally:
            await host.close()

    asyncio.run(run())
