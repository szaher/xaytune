"""LLMPlanner: an agent model proposes at most one typed, explicitly allowed action.

Everything runs against ``ScriptedAgentModel``: no network, no keys, and the
answers a real model gets wrong are scripted on purpose.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any, Literal

import pytest
from pydantic import Field, StrictInt, field_validator

from tests.test_planning.test_rule_based_planner import context, node
from xaytune.agent import (
    AgentModelIdentity,
    AgentModelInvocationError,
    AgentModelOutputError,
    AgentModelRequest,
    AgentModelResponse,
    ScriptedAgentModel,
)
from xaytune.core.capabilities import PLUGIN_API_VERSIONS, PluginDescriptor
from xaytune.core.domain.actions import contract
from xaytune.core.domain.actions.builtin import RejectCandidate
from xaytune.core.domain.actions.contract import ActionDescriptor, ActionSpec, MutationClass
from xaytune.core.domain.agent_invocation import InMemoryAgentInvocationJournal
from xaytune.core.domain.budget import BudgetDimension, BudgetStatus, DimensionStatus
from xaytune.core.domain.decision import DecisionOutcome
from xaytune.core.domain.planning import ActionProposal, PlanningContext
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.immutable import FrozenDict
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus
from xaytune.planning import (
    PLANNERS,
    PlannerConfigurationError,
    bind_planner,
    require_provenance_of,
)
from xaytune.planning import llm as llm_module
from xaytune.planning.llm import (
    LLM_PLANNER_PROMPT_VERSION,
    LLMPlanner,
    llm_planner_factory,
)

PROMPT_V1_FINGERPRINT = "sha256:f985d5f0bb5aef022fa72e9a53aa8ccbb8164891156da48e1650eb7802e5f02f"

MODEL = AgentModelIdentity(provider="vendor", name="planner-large", revision="2026-10-01")
ACME = PluginDescriptor(
    api_version=PLUGIN_API_VERSIONS[0],
    name="acme-actions",
    plugin_version="1.0.0",
    provider="acme",
    xaytune_version="1.0.0",
)


class RetuneNode(ActionSpec):
    """A plugin action on a node, with parameters: what an LLM planner may propose."""

    mutation_class = MutationClass.EXPERIMENT
    target_kinds = ("node",)
    type: Literal["acme/retune-node"] = "acme/retune-node"
    rank: StrictInt = Field(ge=1, le=256)
    note: str | None = None

    @field_validator("note")
    @classmethod
    def _trimmed(cls, note: str | None) -> str | None:
        if note is not None and note != note.strip():
            raise ValueError("a note has no surrounding whitespace")
        return note


class Freeform(ActionSpec):
    """Parameters no closed schema can describe."""

    mutation_class = MutationClass.EXPERIMENT
    target_kinds = ("node",)
    type: Literal["acme/freeform"] = "acme/freeform"
    anything: FrozenDict = Field(default_factory=FrozenDict)


class LateArrival(ActionSpec):
    """Registered after a planner was bound."""

    mutation_class = MutationClass.EXPERIMENT
    target_kinds = ("node",)
    type: Literal["acme/late-arrival"] = "acme/late-arrival"


@pytest.fixture(autouse=True)
def registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract, "_DESCRIPTORS", dict(contract._DESCRIPTORS))
    contract.register_action(ActionDescriptor.for_spec(RetuneNode, provider=ACME))
    contract.register_action(ActionDescriptor.for_spec(Freeform, provider=ACME))


def config(**changes: Any) -> dict[str, Any]:
    return {
        "model": MODEL.model_dump(),
        "prompt_version": LLM_PLANNER_PROMPT_VERSION,
        "allowed_actions": [{"type": "acme/retune-node"}, {"type": "reject-candidate"}],
        "temperature": 0.0,
        "max_output_tokens": 512,
        **changes,
    }


def bind(script: list[Any], **changes: Any) -> tuple[LLMPlanner, ScriptedAgentModel]:
    model = ScriptedAgentModel(script, model=MODEL)
    spec = PlannerSpec(kind="llm", config=FrozenDict(config(**changes)))
    return LLMPlanner.from_spec(spec, model).with_journal(InMemoryAgentInvocationJournal()), model


def propose(planner: LLMPlanner, ctx: PlanningContext) -> tuple[Any, ...]:
    return asyncio.run(planner.propose(ctx))


def answer(
    target: str,
    *,
    kind: str = "node",
    action: str = "acme/retune-node",
    parameters: dict[str, Any] | None = None,
    evidence: list[dict[str, str]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "proposal": {
            "action": {
                "type": action,
                "version": "1",
                "target": {"kind": kind, "id": target},
                "parameters": {"rank": 32} if parameters is None else parameters,
            },
            "reason": "rank 16 plateaued below target",
            "evidence_refs": [] if evidence is None else evidence,
            **extra,
        }
    }


def refs(n: Any) -> list[dict[str, str]]:
    return [
        {"kind": "decision", "id": str(n.decisions[0].decision_id)},
        {"kind": "evaluation-result", "id": str(n.evaluations[0].evaluation_result_id)},
    ]


# ---- a typed proposal, or nothing ---------------------------------------------------------


def test_a_valid_answer_is_exactly_one_typed_action_proposal() -> None:
    parent = node()
    ctx = context(parent)
    planner, _ = bind([answer(str(parent.node_id), evidence=refs(parent))])
    (proposal,) = propose(planner, ctx)
    assert isinstance(proposal, ActionProposal)
    assert isinstance(proposal.action, RetuneNode)
    assert proposal.action.rank == 32
    assert proposal.action.target.id == str(parent.node_id)
    assert [(r.kind, r.id) for r in proposal.evidence_refs] == [
        (r["kind"], r["id"]) for r in refs(parent)
    ]
    assert proposal.reason == "rank 16 plateaued below target"


def test_a_built_in_action_is_proposed_as_its_own_type() -> None:
    parent = node()
    planner, _ = bind([answer(str(parent.node_id), action="reject-candidate", parameters={})])
    (proposal,) = propose(planner, context(parent))
    assert type(proposal.action) is RejectCandidate


def test_no_proposal_is_nothing() -> None:
    planner, model = bind([{"proposal": None}])
    assert propose(planner, context(node())) == ()
    assert len(model.requests) == 1


def test_one_proposal_at_most_by_construction() -> None:
    planner, _ = bind([])
    proposal = planner.response_schema["properties"]["proposal"]
    assert [alt.get("type") for alt in proposal["anyOf"]] == ["null", "object"]
    parent = node()
    many = {"proposals": [answer(str(parent.node_id))["proposal"]] * 2}
    planner, _ = bind([many])
    with pytest.raises(AgentModelOutputError, match="unexpected 'proposals'"):
        propose(planner, context(parent))


# ---- the model call goes through the boundary ---------------------------------------------


def test_a_malformed_answer_fails_closed_through_invoke_agent_model() -> None:
    parent = node()
    planner, _ = bind([{"proposal": {"action": "retune everything"}}])
    with pytest.raises(AgentModelOutputError):
        propose(planner, context(parent))


def test_the_planner_asks_only_through_invoke_agent_model(monkeypatch) -> None:
    calls: list[AgentModelRequest] = []
    real = llm_module.invoke_agent_model

    async def spy(model: Any, request: AgentModelRequest) -> AgentModelResponse:
        calls.append(request)
        return await real(model, request)

    monkeypatch.setattr(llm_module, "invoke_agent_model", spy)
    planner, model = bind([{"proposal": None}])
    propose(planner, context(node()))
    assert calls == list(model.requests)


def test_model_failures_propagate_and_are_never_a_proposal() -> None:
    planner, _ = bind([AgentModelInvocationError("rate limited")])
    with pytest.raises(AgentModelInvocationError, match="rate limited"):
        propose(planner, context(node()))
    planner, _ = bind([ConnectionError("down")])
    with pytest.raises(AgentModelInvocationError, match="ConnectionError"):
        propose(planner, context(node()))


def test_a_more_specific_served_model_name_does_not_change_parsing() -> None:
    parent = node()
    served = AgentModelResponse(
        content=FrozenDict(answer(str(parent.node_id))),
        model="planner-large-2026-10-01-preview",
        model_revision=MODEL.revision,
    )
    planner, _ = bind([served])
    (proposal,) = propose(planner, context(parent))
    assert proposal.action.rank == 32


# ---- what the model may choose, checked against the allowlist and the context -------------


@pytest.mark.parametrize(
    ("action", "parameters"),
    [
        ("promote-candidate", {}),  # registered, not allowed
        ("acme/never-registered", {}),
    ],
)
def test_an_action_outside_the_allowlist_is_refused(action: str, parameters: dict) -> None:
    parent = node()
    planner, _ = bind([answer(str(parent.node_id), action=action, parameters=parameters)])
    with pytest.raises(AgentModelOutputError):
        propose(planner, context(parent))


@pytest.mark.parametrize(
    ("parameters", "reason"),
    [
        ({"rank": 0}, "must be >= 1"),
        ({"rank": 512}, "must be <= 256"),
        ({"rank": "32"}, "expected integer"),
        ({}, "missing required 'rank'"),
        ({"rank": 32, "alpha": 64}, "unexpected 'alpha'"),
        ({"rank": 32, "note": " padded "}, "no surrounding whitespace"),
    ],
)
def test_invalid_action_arguments_are_refused(parameters: dict, reason: str) -> None:
    parent = node()
    planner, _ = bind([answer(str(parent.node_id), parameters=parameters)])
    with pytest.raises(AgentModelOutputError) as caught:
        propose(planner, context(parent))
    assert any(reason in line for line in caught.value.reasons), caught.value.reasons


def test_a_target_outside_the_context_is_refused() -> None:
    planner, _ = bind([answer("node_does-not-exist")])
    with pytest.raises(AgentModelOutputError, match="names no node 'node_does-not-exist'"):
        propose(planner, context(node()))


def test_a_target_kind_the_action_does_not_take_is_refused() -> None:
    planner, _ = bind([answer(str(context(node()).experiment_id), kind="experiment")])
    with pytest.raises(AgentModelOutputError):
        propose(planner, context(node()))


def test_evidence_outside_the_context_is_refused() -> None:
    parent, stranger = node(), node()
    planner, _ = bind([answer(str(parent.node_id), evidence=refs(stranger))])
    with pytest.raises(AgentModelOutputError) as caught:
        propose(planner, context(parent))
    assert len(caught.value.reasons) == 2
    assert all("the context names no" in reason for reason in caught.value.reasons)


def test_duplicate_evidence_is_refused() -> None:
    parent = node()
    twice = refs(parent)[:1] * 2
    planner, _ = bind([answer(str(parent.node_id), evidence=twice)])
    with pytest.raises(AgentModelOutputError, match="cited more than once"):
        propose(planner, context(parent))


# ---- provenance is xaytune's, never the model's -------------------------------------------


def test_the_model_cannot_supply_provenance() -> None:
    parent = node()
    forged = {"planner_name": "rule-based", "context_fingerprint": "sha256:forged"}
    planner, _ = bind([answer(str(parent.node_id), provenance=forged)])
    with pytest.raises(AgentModelOutputError, match="unexpected 'provenance'"):
        propose(planner, context(parent))


def test_provenance_names_the_bound_planner_and_the_context_it_saw() -> None:
    parent = node()
    ctx = context(parent)
    planner, _ = bind([answer(str(parent.node_id))])
    (proposal,) = propose(planner, ctx)
    provenance = proposal.provenance
    assert provenance.context_fingerprint == ctx.input_fingerprint()
    assert provenance.planner_name == "llm"
    assert provenance.planner_provider == "xaytune"
    assert provenance.planner_spec_kind == "llm"
    assert provenance.planner_spec_version == LLMPlanner.descriptor.plugin_version
    rebound = bind_planner(planner.spec, {"llm": llm_planner_factory(planner.model)})
    require_provenance_of(rebound, provenance)


# ---- the model, the prompt and the allowlist are the planner's identity -------------------


def test_the_bound_spec_records_model_prompt_generation_and_contract_fingerprints() -> None:
    planner, _ = bind([])
    recorded = planner.spec.config
    assert planner.spec.version == "1.0.0"
    assert recorded["model"] == MODEL.model_dump()
    assert recorded["prompt_version"] == "xaytune.llm-planner/v1"
    assert (recorded["temperature"], recorded["max_output_tokens"]) == (0.0, 512)
    assert [a["type"] for a in recorded["allowed_actions"]] == [
        "acme/retune-node",
        "reject-candidate",
    ]
    assert all(a["contract_fingerprint"].startswith("sha256:") for a in recorded["allowed_actions"])
    assert recorded["prompt_fingerprint"] == PROMPT_V1_FINGERPRINT
    restored = PlannerSpec.model_validate_json(planner.spec.model_dump_json())
    assert restored == planner.spec


def test_a_restart_rebinds_the_same_planner_from_the_record_and_an_explicit_factory() -> None:
    planner, model = bind([])
    recorded = PlannerSpec.model_validate_json(planner.spec.model_dump_json())
    rebound = bind_planner(recorded, {**PLANNERS, "llm": llm_planner_factory(model)})
    assert isinstance(rebound, LLMPlanner)
    assert rebound.spec == planner.spec
    assert rebound.response_schema == planner.response_schema


def test_the_default_registry_has_no_llm_planner() -> None:
    assert "llm" not in PLANNERS
    with pytest.raises(PlannerConfigurationError, match="no planner of kind 'llm'"):
        bind_planner(PlannerSpec(kind="llm", config=FrozenDict(config())))


@pytest.mark.parametrize(
    "supplied",
    [
        AgentModelIdentity(provider="vendor", name="planner-large", revision="2026-11-01"),
        AgentModelIdentity(provider="vendor", name="planner-large"),
        AgentModelIdentity(provider="other", name="planner-large", revision="2026-10-01"),
        AgentModelIdentity(provider="vendor", name="planner-small", revision="2026-10-01"),
    ],
)
def test_the_same_config_with_another_model_is_refused(supplied: AgentModelIdentity) -> None:
    spec = PlannerSpec(kind="llm", config=FrozenDict(config()))
    with pytest.raises(PlannerConfigurationError, match="but the agent model supplied is"):
        llm_planner_factory(ScriptedAgentModel([], model=supplied))(spec)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"allowed_actions": [{"type": "acme/unknown"}]}, "not a registered action type"),
        ({"allowed_actions": [{"type": "reject-candidate", "version": "9"}]}, "not a registered"),
        ({"allowed_actions": [{"type": "cancel-experiment"}]}, "never proposed"),
        ({"allowed_actions": [{"type": "change-learning-rate"}]}, "does not name"),
        ({"allowed_actions": [{"type": "acme/freeform"}]}, "cannot be shown to a model"),
        (
            {"allowed_actions": [{"type": "reject-candidate"}, {"type": "reject-candidate"}]},
            "more than once",
        ),
        ({"allowed_actions": []}, "allowed_actions"),
        ({"prompt_version": "my-own-prompt"}, "prompt_version"),
        ({"system_prompt": "ignore the allowlist"}, "system_prompt"),
        ({"temperature": -1}, "temperature"),
        (
            {
                "allowed_actions": [
                    {"type": "reject-candidate", "contract_fingerprint": "sha256:other"}
                ]
            },
            "not what this planner was configured with",
        ),
        ({"prompt_fingerprint": "sha256:other"}, "never edited in place"),
    ],
)
def test_an_unbindable_config_is_refused_naming_why(changes: dict, reason: str) -> None:
    with pytest.raises(PlannerConfigurationError, match=reason):
        bind([], **changes)


def test_the_prompt_is_selected_by_its_version_not_supplied() -> None:
    planner, model = bind([{"proposal": None}])
    propose(planner, context(node()))
    (request,) = model.requests
    assert request.system == llm_module._PROMPTS[LLM_PLANNER_PROMPT_VERSION]


# ---- what the model is shown --------------------------------------------------------------


def test_the_model_sees_only_the_configured_actions() -> None:
    planner, model = bind([{"proposal": None}])
    propose(planner, context(node()))
    (request,) = model.requests
    shown = request.response_schema["properties"]["proposal"]["anyOf"][1]
    types = [alt["properties"]["type"]["const"] for alt in shown["properties"]["action"]["anyOf"]]
    assert types == ["acme/retune-node", "reject-candidate"]
    document = json.loads(request.messages[0].content)
    assert [a["type"] for a in document["allowed_actions"]] == types


def test_registering_another_action_later_exposes_nothing() -> None:
    planner, model = bind([{"proposal": None}])
    before = planner.response_schema
    contract.register_action(ActionDescriptor.for_spec(LateArrival, provider=ACME))
    propose(planner, context(node()))
    assert model.requests[0].response_schema == before
    assert "acme/late-arrival" not in model.requests[0].messages[0].content


def test_the_model_is_given_the_context_projection_and_nothing_else() -> None:
    ctx = context(node())
    planner, model = bind([{"proposal": None}])
    propose(planner, ctx)
    (request,) = model.requests
    document = json.loads(request.messages[0].content)
    assert set(document) == {"planning_context", "allowed_actions"}
    assert document["planning_context"]["experiment_id"] == str(ctx.experiment_id)
    assert (request.temperature, request.max_output_tokens) == (0.0, 512)


def test_the_same_context_is_the_same_request_and_is_asked_once() -> None:
    ctx = context(node())
    planner, model = bind([{"proposal": None}, {"proposal": None}])
    assert planner.request(ctx).fingerprint(MODEL) == planner.request(ctx).fingerprint(MODEL)
    propose(planner, ctx)
    propose(planner, ctx)  # replayed from the record (PR-032)
    assert len(model.requests) == 1


def test_only_the_user_role_is_used() -> None:
    planner, model = bind([{"proposal": None}])
    propose(planner, context(node()))
    assert {message.role for message in model.requests[0].messages} == {"user"}


# ---- review 1: bound means bound ----------------------------------------------------------


def test_a_model_whose_identity_changed_after_binding_is_never_asked() -> None:
    parent = node()
    planner, model = bind([answer(str(parent.node_id))])
    drifted = MODEL.model_copy(update={"revision": "2026-11-01"})
    model.descriptor = model.descriptor.model_copy(update={"model": drifted})
    with pytest.raises(PlannerConfigurationError, match="it is not asked"):
        propose(planner, context(parent))
    assert model.requests == ()
    assert model.remaining == 1


def _exhausted() -> BudgetStatus:
    return BudgetStatus(
        dimensions=(
            DimensionStatus(
                dimension=BudgetDimension.RUNS,
                kind="quota",
                limit=Decimal(4),
                reserved=Decimal(4),
                committed=Decimal(0),
                consumed=Decimal(0),
                released=Decimal(0),
                outstanding=Decimal(4),
                remaining=Decimal(0),
            ),
        )
    )


@pytest.mark.parametrize(
    "ctx",
    [
        pytest.param(lambda: context(node(), status=ExperimentStatus.PAUSED), id="paused"),
        pytest.param(lambda: context(node(), status=ExperimentStatus.SUCCEEDED), id="terminal"),
        pytest.param(lambda: context(), id="no-candidates"),
        pytest.param(
            lambda: context(node(status=ExperimentNodeStatus.ACTIVE, outcome=None)),
            id="a-candidate-still-training",
        ),
        pytest.param(
            lambda: context(node(outcome=DecisionOutcome.STOP_SUCCEEDED)), id="stop-decided"
        ),
        pytest.param(lambda: context(node(), budget=_exhausted()), id="quota-exhausted"),
    ],
)
def test_outside_the_planning_stage_nothing_is_proposed_and_the_model_is_not_asked(
    ctx: Any,
) -> None:
    planner, model = bind([answer("anything")])
    assert propose(planner, ctx()) == ()
    assert model.requests == ()


def _rebind_after(monkeypatch: pytest.MonkeyPatch, *specs: type[ActionSpec]) -> PlannerSpec:
    """A planner bound now, and the registry a later process might have instead."""
    planner, _ = bind([], allowed_actions=[{"type": "acme/retune-node"}])
    monkeypatch.setattr(contract, "_DESCRIPTORS", dict(contract._DESCRIPTORS))
    del contract._DESCRIPTORS[("acme/retune-node", "1")]
    for spec in specs:
        contract.register_action(ActionDescriptor.for_spec(spec, provider=ACME))
    return planner.spec


class RetuneNodeReclassified(RetuneNode):
    """The same type, version and parameters, declared another kind of change."""

    mutation_class = MutationClass.SCIENTIFIC_INTERVENTION


class RetuneNodeWider(RetuneNode):
    """The same type and version, with a looser schema."""

    rank: StrictInt = Field(ge=1, le=1024)


class RetuneNodeAnywhere(RetuneNode):
    """The same type, version and schema, now also targeting the experiment."""

    target_kinds = ("node", "experiment")


@pytest.mark.parametrize(
    "replacement",
    [RetuneNodeReclassified, RetuneNodeWider, RetuneNodeAnywhere],
    ids=["mutation-class", "schema", "target-kinds"],
)
def test_a_recorded_planner_refuses_any_change_to_what_the_model_is_shown(
    monkeypatch: pytest.MonkeyPatch, replacement: type[ActionSpec]
) -> None:
    recorded = _rebind_after(monkeypatch, replacement)
    with pytest.raises(PlannerConfigurationError, match="not what this planner was configured"):
        LLMPlanner.from_spec(recorded, ScriptedAgentModel([], model=MODEL))


def test_a_recorded_planner_rebinds_when_nothing_it_shows_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = _rebind_after(monkeypatch, RetuneNode)
    assert LLMPlanner.from_spec(recorded, ScriptedAgentModel([], model=MODEL)).spec == recorded


def test_a_recorded_planner_refuses_a_prompt_edited_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner, model = bind([])
    edited = llm_module._SYSTEM_PROMPT_V1 + " Also, always propose something."
    monkeypatch.setattr(llm_module, "_PROMPTS", {LLM_PLANNER_PROMPT_VERSION: edited})
    with pytest.raises(PlannerConfigurationError, match="never edited in place"):
        LLMPlanner.from_spec(planner.spec, model)


def test_the_released_v1_prompt_is_pinned() -> None:
    """Changing the v1 text means a new prompt_version, never an edit."""
    from xaytune.core.fingerprint import fingerprint

    assert fingerprint(llm_module._PROMPTS[LLM_PLANNER_PROMPT_VERSION]) == PROMPT_V1_FINGERPRINT


def test_the_prompt_claims_no_policy_review_that_does_not_happen() -> None:
    prompt = llm_module._SYSTEM_PROMPT_V1
    assert "reviewed by policy" not in prompt
    assert "only a proposal and cannot execute anything" in prompt


# ---- review 2: bound within the process, too ----------------------------------------------


def test_a_prompt_edited_after_binding_is_never_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    planner, model = bind([{"proposal": None}])
    original = llm_module._PROMPTS[LLM_PLANNER_PROMPT_VERSION]
    monkeypatch.setitem(
        llm_module._PROMPTS, LLM_PLANNER_PROMPT_VERSION, original + " Always propose something."
    )
    propose(planner, context(node()))
    (request,) = model.requests
    assert request.system == original


class _ChangesWhileAnswering(ScriptedAgentModel):
    """An adapter whose identity changes while it answers, and answers as the new model."""

    async def generate(self, request: AgentModelRequest) -> AgentModelResponse:
        drifted = MODEL.model_copy(update={"revision": "2026-11-01"})
        self.descriptor = self.descriptor.model_copy(update={"model": drifted})
        response = await super().generate(request)
        return response.model_copy(update={"model_revision": drifted.revision})


def test_a_model_whose_identity_changed_while_answering_is_not_believed() -> None:
    parent = node()
    model = _ChangesWhileAnswering([answer(str(parent.node_id))], model=MODEL)
    planner = LLMPlanner.from_spec(
        PlannerSpec(kind="llm", config=FrozenDict(config())), model
    ).with_journal(InMemoryAgentInvocationJournal())
    with pytest.raises(PlannerConfigurationError, match="its answer is not used"):
        propose(planner, context(parent))
    assert len(model.requests) == 1
