"""Proposal envelopes: typed intent, provenance attached, nothing minted."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.test_planning.test_rule_based_planner import candidate
from xaytune.core.domain.action import ActionTarget
from xaytune.core.domain.actions import ActionSpec, RejectCandidate
from xaytune.core.domain.planning import ActionProposal, CandidateProposal, ProposalProvenance
from xaytune.core.ids import ExperimentNodeId

PROVENANCE = ProposalProvenance(
    planner_provider="xaytune",
    planner_name="rule-based",
    planner_version="1.0.0",
    planner_api_version="xaytune.plugins/v1alpha1",
    planner_spec_kind="rule-based",
    planner_spec_version="1.0.0",
    planner_spec_identity_version=1,
    planner_spec_fingerprint="sha256:spec",
    context_identity_version=1,
    context_fingerprint="sha256:context",
)


def test_an_action_proposal_carries_a_typed_action_and_serializes_it_whole() -> None:
    node = ExperimentNodeId.generate()
    spec = RejectCandidate(target=ActionTarget(kind="node", id=str(node)))
    proposal = ActionProposal(action=spec, reason="r", provenance=PROVENANCE)
    assert proposal.action is spec
    dumped = proposal.model_dump(mode="json")
    assert dumped["action"]["type"] == "reject-candidate"
    assert dumped["action"]["target"] == {"kind": "node", "id": str(node)}


@pytest.mark.parametrize(
    "action",
    [
        {"type": "reject-candidate", "target": {"kind": "node", "id": "x"}},
        {"type": "stop-experiment", "target": {"kind": "experiment", "id": "x"}},
    ],
    ids=["mapping", "nonexistent-type"],
)
def test_an_action_proposal_refuses_an_untyped_action(action) -> None:
    with pytest.raises(ValidationError, match="typed ActionSpec"):
        ActionProposal(action=action, reason="r", provenance=PROVENANCE)


def test_an_action_proposal_refuses_the_bare_base_class() -> None:
    with pytest.raises(ValidationError, match="typed ActionSpec"):
        ActionProposal(
            action=ActionSpec.model_construct(target=ActionTarget(kind="node", id="x")),
            reason="r",
            provenance=PROVENANCE,
        )


def test_a_candidate_proposal_refuses_a_fingerprint_that_is_not_its_candidates() -> None:
    spec = candidate(32)
    with pytest.raises(ValidationError, match="does not describe the candidate"):
        CandidateProposal(
            candidate=spec,
            candidate_fingerprint=candidate(16).candidate_fingerprint(),
            parent_ids=(ExperimentNodeId.generate(),),
            hypothesis="h",
            reason="r",
            provenance=PROVENANCE,
        )


def test_a_candidate_proposal_needs_a_parent() -> None:
    spec = candidate(32)
    with pytest.raises(ValidationError):
        CandidateProposal(
            candidate=spec,
            candidate_fingerprint=spec.candidate_fingerprint(),
            parent_ids=(),
            hypothesis="h",
            reason="r",
            provenance=PROVENANCE,
        )


def test_an_action_proposal_refuses_an_unregistered_action_type() -> None:
    from typing import ClassVar, Literal

    from xaytune.core.domain.actions import MutationClass

    class Unregistered(ActionSpec):
        mutation_class: ClassVar[MutationClass] = MutationClass.EXPERIMENT
        target_kinds: ClassVar[tuple[str, ...]] = ("node",)  # type: ignore[assignment]
        type: Literal["planner-invented"] = "planner-invented"

    spec = Unregistered(target=ActionTarget(kind="node", id="x"))
    with pytest.raises(ValidationError, match="not a registered action type"):
        ActionProposal(action=spec, reason="r", provenance=PROVENANCE)


def test_an_action_proposal_refuses_a_subclass_posing_as_a_registered_type() -> None:
    """Same ``type`` literal, different schema: the registered class must be the one used."""

    class Impostor(RejectCandidate):
        pass

    spec = Impostor(target=ActionTarget(kind="node", id="x"))
    with pytest.raises(ValidationError, match="not the schema registered"):
        ActionProposal(action=spec, reason="r", provenance=PROVENANCE)
