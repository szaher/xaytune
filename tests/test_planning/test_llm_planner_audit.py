"""LLMPlanner records every invocation: intent before the call, the answer, then the proposal.

PR-032. Against ``ScriptedAgentModel`` and the in-memory journal; the durable
journal and a restarted host are ``tests/test_experiment/test_llm_planner_host.py``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tests.test_planning.test_llm_planner import (
    MODEL,
    answer,
    config,
    context,
    node,
    refs,
    registry,  # noqa: F401 -- the autouse registry fixture, for the acme action
)
from xaytune.agent import (
    AgentModelInvocationError,
    AgentModelOutputError,
    AgentModelRequest,
    AgentModelResponse,
    ScriptedAgentModel,
)
from xaytune.core.domain.agent_invocation import (
    AgentInvocationJournal,
    AgentInvocationStatus,
    InMemoryAgentInvocationJournal,
    require_derived_from,
)
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.immutable import FrozenDict, thaw
from xaytune.planning import PlannerConfigurationError
from xaytune.planning.llm import LLMPlanner

S = AgentInvocationStatus
SECRET = "Authorization: Bearer sk-live-0123456789abcdef"


def planner_for(
    model: Any, journal: InMemoryAgentInvocationJournal | None = None
) -> tuple[LLMPlanner, InMemoryAgentInvocationJournal]:
    journal = journal or InMemoryAgentInvocationJournal()
    bound = LLMPlanner.from_spec(PlannerSpec(kind="llm", config=FrozenDict(config())), model)
    return bound.with_journal(journal), journal


def propose(planner: LLMPlanner, ctx: Any) -> Any:
    return asyncio.run(planner.propose(ctx))


def only(journal: InMemoryAgentInvocationJournal) -> Any:
    (invocation,) = journal.invocations
    return invocation


# ---- what a completed invocation records ---------------------------------------------------


def test_a_proposal_is_derived_from_and_bound_to_its_recorded_invocation() -> None:
    parent = node()
    ctx = context(parent)
    model = ScriptedAgentModel([answer(str(parent.node_id), evidence=refs(parent))], model=MODEL)
    planner, journal = planner_for(model)
    (made,) = propose(planner, ctx)
    invocation = only(journal)
    assert invocation.status is S.COMPLETED and invocation.attempt == 1
    assert made.agent_invocation_id == invocation.id
    require_derived_from(made, invocation)
    assert invocation.proposal_fingerprint == made.proposal_fingerprint()


def test_the_record_answers_who_asked_what_of_which_model_and_what_came_back() -> None:
    parent = node()
    ctx = context(parent)
    served = AgentModelResponse(
        content=FrozenDict(answer(str(parent.node_id))),
        model="planner-large-2026-10-01",
        model_revision=MODEL.revision,
        finish_reason="stop",
        usage={"input_tokens": 1200, "output_tokens": 80},  # type: ignore[arg-type]
        provider_request_id="req_123",
        latency_seconds=1.5,
    )
    model = ScriptedAgentModel([served], model=MODEL)
    planner, journal = planner_for(model)
    propose(planner, ctx)
    (request,) = model.requests
    intent = only(journal).intent
    assert intent.experiment_id == ctx.experiment_id
    assert (intent.planner_kind, intent.planner_version) == ("llm", "1.0.0")
    _recorded_as_before_pr_034(only(journal), ctx)
    assert intent.prompt_version == "xaytune.llm-planner/v1"
    assert intent.prompt_fingerprint == planner.config.prompt_fingerprint
    assert intent.request_fingerprint == request.fingerprint(MODEL)
    assert AgentModelRequest.model_validate(thaw(intent.request)) == request
    assert thaw(intent.agent_model) == model.descriptor.model_dump(mode="json")
    assert AgentModelResponse.model_validate(thaw(only(journal).response)) == served


def test_no_proposal_is_recorded_as_completed_with_none() -> None:
    planner, journal = planner_for(ScriptedAgentModel([{"proposal": None}], model=MODEL))
    assert propose(planner, context(node())) == ()
    invocation = only(journal)
    assert invocation.status is S.COMPLETED and invocation.proposal is None


def test_nothing_is_recorded_when_the_model_is_not_asked() -> None:
    from xaytune.core.state.status import ExperimentStatus

    planner, journal = planner_for(ScriptedAgentModel([], model=MODEL))
    assert propose(planner, context(node(), status=ExperimentStatus.PAUSED)) == ()
    assert journal.invocations == ()


def test_a_planner_without_a_journal_asks_nothing() -> None:
    model = ScriptedAgentModel([{"proposal": None}], model=MODEL)
    bare = LLMPlanner.from_spec(PlannerSpec(kind="llm", config=FrozenDict(config())), model)
    with pytest.raises(PlannerConfigurationError, match="records every invocation"):
        propose(bare, context(node()))
    assert model.requests == ()


# ---- refusals and failures are recorded, sanitized ----------------------------------------


def test_a_schema_invalid_answer_is_recorded_refused_with_what_was_refused() -> None:
    bad = {"proposal": {"action": "retune everything"}}
    planner, journal = planner_for(ScriptedAgentModel([bad], model=MODEL))
    with pytest.raises(AgentModelOutputError):
        propose(planner, context(node()))
    invocation = only(journal)
    assert invocation.status is S.REFUSED
    assert invocation.failure.kind == "output-refused"
    assert invocation.failure.reasons
    assert thaw(invocation.response)["content"] == bad
    assert invocation.proposal is None


def test_an_answer_naming_what_the_context_lacks_is_recorded_refused_after_its_answer() -> None:
    parent = node()
    planner, journal = planner_for(ScriptedAgentModel([answer("node_elsewhere")], model=MODEL))
    with pytest.raises(AgentModelOutputError):
        propose(planner, context(parent))
    invocation = only(journal)
    assert invocation.status is S.REFUSED
    assert invocation.response is not None, "the answer was recorded before it was refused"
    assert any("names no node" in reason for reason in invocation.failure.reasons)


def test_a_failed_call_records_its_classification_and_never_the_providers_text() -> None:
    leaked = RuntimeError(f"POST https://user:pw@api.vendor.test/v1 failed; {SECRET}")
    planner, journal = planner_for(ScriptedAgentModel([leaked], model=MODEL))
    with pytest.raises(AgentModelInvocationError):
        propose(planner, context(node()))
    invocation = only(journal)
    assert invocation.status is S.FAILED
    assert invocation.failure.model_dump() == {
        "kind": "invocation-failed",
        "error_type": "AgentModelInvocationError",
        "reasons": (),
    }
    recorded = json.dumps(invocation.model_dump(mode="json"))
    for fragment in ("sk-live", "api.vendor.test", "user:pw", "Traceback", "RuntimeError: POST"):
        assert fragment not in recorded


def test_an_adapters_own_message_is_not_recorded_either() -> None:
    planner, journal = planner_for(
        ScriptedAgentModel([AgentModelInvocationError(f"rate limited; {SECRET}")], model=MODEL)
    )
    with pytest.raises(AgentModelInvocationError):
        propose(planner, context(node()))
    assert "sk-live" not in json.dumps(only(journal).model_dump(mode="json"))


class _ChangesWhileAnswering(ScriptedAgentModel):
    async def generate(self, request: AgentModelRequest) -> AgentModelResponse:
        drifted = MODEL.model_copy(update={"revision": "2026-11-01"})
        self.descriptor = self.descriptor.model_copy(update={"model": drifted})
        response = await super().generate(request)
        return response.model_copy(update={"model_revision": drifted.revision})


def test_an_answer_from_a_model_that_changed_mid_call_is_recorded_failed_and_unused() -> None:
    parent = node()
    planner, journal = planner_for(
        _ChangesWhileAnswering([answer(str(parent.node_id))], model=MODEL)
    )
    with pytest.raises(PlannerConfigurationError, match="its answer is not used"):
        propose(planner, context(parent))
    invocation = only(journal)
    assert invocation.status is S.FAILED
    assert invocation.failure.kind == "model-identity-changed"
    assert invocation.response is not None and invocation.proposal is None


# ---- replay and restart ---------------------------------------------------------------------


def test_a_completed_round_is_replayed_and_the_model_is_not_asked_again() -> None:
    parent = node()
    ctx = context(parent)
    model = ScriptedAgentModel([answer(str(parent.node_id))], model=MODEL)
    planner, journal = planner_for(model)
    first = propose(planner, ctx)
    again = propose(planner, ctx)
    assert again == first
    assert len(model.requests) == 1 and len(journal.invocations) == 1


def test_a_round_answered_before_a_crash_is_finished_from_the_recorded_answer() -> None:
    parent = node()
    ctx = context(parent)
    model = ScriptedAgentModel([answer(str(parent.node_id))], model=MODEL)
    planner, journal = planner_for(model)

    def crash(*_: Any) -> Any:
        raise SystemExit("the process stops after the answer is recorded")

    journal.completed = crash  # type: ignore[method-assign]
    with pytest.raises(SystemExit):
        propose(planner, ctx)
    assert only(journal).status is S.ANSWERED
    del journal.completed  # the restarted process has the real method
    restarted, _ = planner_for(ScriptedAgentModel([], model=MODEL), journal)
    (made,) = propose(restarted, ctx)
    assert only(journal).status is S.COMPLETED
    require_derived_from(made, only(journal))


def test_a_refused_round_is_refused_again_without_asking() -> None:
    planner, journal = planner_for(ScriptedAgentModel([{"proposal": {"x": 1}}], model=MODEL))
    ctx = context(node())
    with pytest.raises(AgentModelOutputError):
        propose(planner, ctx)
    replay, _ = planner_for(ScriptedAgentModel([], model=MODEL), journal)
    with pytest.raises(AgentModelOutputError):
        propose(replay, ctx)
    assert len(journal.invocations) == 1


class _Hangs(ScriptedAgentModel):
    async def generate(self, request: AgentModelRequest) -> AgentModelResponse:
        self._requests.append(request)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def test_a_call_interrupted_mid_flight_is_unknown_and_the_round_asks_again() -> None:
    parent = node()
    ctx = context(parent)
    planner, journal = planner_for(_Hangs([], model=MODEL))

    async def interrupted() -> None:
        task = asyncio.ensure_future(planner.propose(ctx))
        while not journal.invocations:
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(interrupted())
    assert only(journal).status is S.INTENDED, "nothing pretends the call did not happen"
    retry, _ = planner_for(ScriptedAgentModel([answer(str(parent.node_id))], model=MODEL), journal)
    (made,) = propose(retry, ctx)
    first, second = journal.invocations
    assert first.status is S.OUTCOME_UNKNOWN
    assert (second.attempt, second.status) == (2, S.COMPLETED)
    assert made.agent_invocation_id == second.id


def test_a_failed_round_asks_again_as_a_new_attempt() -> None:
    parent = node()
    ctx = context(parent)
    planner, journal = planner_for(ScriptedAgentModel([ConnectionError("down")], model=MODEL))
    with pytest.raises(AgentModelInvocationError):
        propose(planner, ctx)
    retry, _ = planner_for(ScriptedAgentModel([answer(str(parent.node_id))], model=MODEL), journal)
    propose(retry, ctx)
    assert [(i.attempt, i.status) for i in journal.invocations] == [
        (1, S.FAILED),
        (2, S.COMPLETED),
    ]


def test_the_journal_is_the_protocol_the_planner_records_into() -> None:
    assert isinstance(InMemoryAgentInvocationJournal(), AgentInvocationJournal)


def test_a_derivation_bug_after_the_answer_never_asks_the_model_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review 1 (S1): the answer is durable; a local error leaves it ANSWERED, not FAILED."""
    parent = node()
    ctx = context(parent)
    model = ScriptedAgentModel([answer(str(parent.node_id))], model=MODEL)
    planner, journal = planner_for(model)

    def broken(*_: Any, **__: Any) -> Any:
        raise RuntimeError("a bug in xaytune's derivation")

    fixed = LLMPlanner._derive
    monkeypatch.setattr(LLMPlanner, "_derive", broken)
    with pytest.raises(RuntimeError):
        propose(planner, ctx)
    assert only(journal).status is S.ANSWERED
    monkeypatch.setattr(LLMPlanner, "_derive", fixed)  # fixed, and the controller restarts

    restarted, _ = planner_for(ScriptedAgentModel([], model=MODEL), journal)
    (made,) = propose(restarted, ctx)
    assert only(journal).status is S.COMPLETED
    require_derived_from(made, only(journal))
    assert len(model.requests) == 1, "the model was asked once, ever"


# ---- PR-034: the planning context's identity moved to v2; the model's did not -----------------


def _with_branch_origin(ctx: Any) -> Any:
    """*ctx* after PR-034: a node now carries a branch origin the model is never shown."""
    from xaytune.core.domain.planning import CandidateBranchOrigin, ProposalProvenance

    origin = CandidateBranchOrigin(
        proposal_identity_version=1,
        proposal_fingerprint="sha256:a-branched-proposal",
        provenance=ProposalProvenance(
            planner_provider="xaytune",
            planner_name="search",
            planner_version="1.0.0",
            planner_api_version="xaytune.plugins/v1alpha1",
            planner_spec_kind="search",
            planner_spec_version="1.0.0",
            planner_spec_identity_version=1,
            planner_spec_fingerprint="sha256:search-spec",
            context_identity_version=2,
            context_fingerprint="sha256:earlier",
        ),
        mutation=FrozenDict({"search": {"suggestion": 0}}),
    )
    (first, *rest) = ctx.nodes
    return ctx.model_copy(
        update={"nodes": (first.model_copy(update={"branch_origin": origin}), *rest)}
    )


def _recorded_as_before_pr_034(invocation: Any, ctx: Any) -> None:
    """What the planner recorded before PR-034: context identity v1, and its fingerprint."""
    from xaytune.core.domain.planning import planning_context_identity_v1
    from xaytune.core.fingerprint import fingerprint

    assert invocation.intent.context_identity_version == 1
    assert invocation.intent.context_fingerprint == fingerprint(planning_context_identity_v1(ctx))


def test_the_model_is_shown_the_same_request_whatever_the_model_is_not_shown() -> None:
    parent = node()
    before = context(parent)
    after = _with_branch_origin(before)
    assert after.input_fingerprint() != before.input_fingerprint(), "the context identity moved"
    planner, _ = planner_for(ScriptedAgentModel([], model=MODEL))
    assert planner.request(after) == planner.request(before)
    assert planner.request(after).fingerprint(MODEL) == planner.request(before).fingerprint(MODEL)


def test_an_answered_v1_round_is_finished_after_pr_034_without_asking_again() -> None:
    parent = node()
    before = context(parent)
    model = ScriptedAgentModel([answer(str(parent.node_id), evidence=refs(parent))], model=MODEL)
    planner, journal = planner_for(model)

    def crash(*_: Any) -> Any:
        raise SystemExit("the process stops after the answer is recorded")

    journal.completed = crash  # type: ignore[method-assign]
    with pytest.raises(SystemExit):
        propose(planner, before)
    recorded = only(journal)
    assert recorded.status is S.ANSWERED
    _recorded_as_before_pr_034(recorded, before)
    del journal.completed

    # Upgraded: the same experiment's context now carries a branch origin.
    upgraded = ScriptedAgentModel([], model=MODEL)
    restarted, _ = planner_for(upgraded, journal)
    (made,) = propose(restarted, _with_branch_origin(before))
    assert upgraded.requests == () and len(model.requests) == 1, "the model was asked once, ever"
    assert len(journal.invocations) == 1 and only(journal).status is S.COMPLETED
    require_derived_from(made, only(journal))
    assert made.provenance.context_identity_version == 1
    assert made.provenance.context_fingerprint == recorded.intent.context_fingerprint


def test_a_completed_v1_round_replays_its_exact_proposal_after_pr_034() -> None:
    parent = node()
    before = context(parent)
    model = ScriptedAgentModel([answer(str(parent.node_id), evidence=refs(parent))], model=MODEL)
    planner, journal = planner_for(model)
    (first,) = propose(planner, before)
    _recorded_as_before_pr_034(only(journal), before)

    upgraded = ScriptedAgentModel([], model=MODEL)
    restarted, _ = planner_for(upgraded, journal)
    (again,) = propose(restarted, _with_branch_origin(before))
    assert again == first
    assert again.proposal_fingerprint() == only(journal).proposal_fingerprint
    assert upgraded.requests == () and len(journal.invocations) == 1
    require_derived_from(again, only(journal))
