"""An adaptive experiment planned by an LLM: every invocation is in its record (PR-032).

The host attaches its durable journal to the planner. The model proposes an
action; the host verifies the proposal against the recorded invocation,
creates no action, and escalates. A restarted host replays the round from the
record and does not ask the model again; a host stopped mid-call leaves an
invocation whose outcome is unknown, and the next one asks again.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_experiment.adaptive_fixtures import adaptive_spec
from tests.test_experiment.test_adaptive_mvp import _TIMEOUT, _host, _record, _world
from xaytune.agent import AgentModelIdentity, AgentModelRequest, AgentModelResponse
from xaytune.agent.scripted import ScriptedAgentModel
from xaytune.core.domain.agent_invocation import AgentInvocationStatus as S
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.immutable import FrozenDict
from xaytune.experiment import ReconciliationEscalatedError
from xaytune.planning import PLANNERS
from xaytune.planning.llm import llm_planner_factory

IDENTITY = AgentModelIdentity(provider="vendor", name="planner-large", revision="2026-10-01")


class RejectsTheFirstNode(ScriptedAgentModel):
    """Reads the planning context it is sent, and proposes rejecting the first candidate."""

    def __init__(self, *, hang: bool = False) -> None:
        super().__init__([], model=IDENTITY)
        self.hang = hang

    async def generate(self, request: AgentModelRequest) -> AgentModelResponse:
        self._requests.append(request)
        if self.hang:
            await asyncio.Event().wait()
        document = json.loads(request.messages[0].content)
        node = document["planning_context"]["nodes"][0]
        return AgentModelResponse(
            content=FrozenDict(
                {
                    "proposal": {
                        "action": {
                            "type": "reject-candidate",
                            "version": "1",
                            "target": {"kind": "node", "id": node["node_id"]},
                            "parameters": {},
                        },
                        "reason": "below target, and the evidence says it will not get there",
                        "evidence_refs": [
                            {"kind": "decision", "id": node["decisions"][0]["decision_id"]}
                        ],
                    }
                }
            ),
            model=IDENTITY.name,
            model_revision=IDENTITY.revision,
            usage={"input_tokens": 900, "output_tokens": 60},  # type: ignore[arg-type]
        )


def _spec(tmp_path: Path) -> Any:
    llm = PlannerSpec(
        kind="llm",
        config=FrozenDict(
            {
                "model": IDENTITY.model_dump(),
                "prompt_version": "xaytune.llm-planner/v1",
                "allowed_actions": [{"type": "reject-candidate"}],
                "temperature": 0.0,
            }
        ),
    )
    return adaptive_spec(tmp_path).model_copy(update={"planner": llm})


def _planners(model: Any) -> dict[str, Any]:
    return {**PLANNERS, "llm": llm_planner_factory(model)}


def test_an_llm_proposal_is_recorded_verified_escalated_and_replayed_after_restart(
    tmp_path: Path,
) -> None:
    world = _world(tmp_path)
    model = RejectsTheFirstNode()

    async def scenario() -> None:
        first = _host(tmp_path, world, planners=_planners(model))
        try:
            handle = await first.submit(_spec(tmp_path))
            with pytest.raises(ReconciliationEscalatedError, match="derived from agent invocation"):
                await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)
            experiment_id = handle.experiment_id
            (invocation,) = first.repository.agent_invocations.for_experiment(str(experiment_id))
            assert invocation.status is S.COMPLETED and invocation.proposal is not None
            assert invocation.intent.agent_model["model"] == IDENTITY.model_dump()
            assert invocation.intent.agent_model["plugin"]["name"] == "scripted-agent-model"
            assert invocation.response["usage"] == {"input_tokens": 900, "output_tokens": 60}
            record = _record(first, experiment_id)
            assert first.repository.actions.for_target("node", str(record["nodes"][0].id)) == ()
        finally:
            await first.close()

        second = _host(tmp_path, world, planners=_planners(model))
        try:
            handle = await second.attach(experiment_id)
            with pytest.raises(ReconciliationEscalatedError, match=str(invocation.id)):
                await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)
            assert second.repository.agent_invocations.for_experiment(str(experiment_id)) == (
                invocation,
            )
        finally:
            await second.close()
        assert len(model.requests) == 1, "the restarted host replayed; it did not ask again"

    asyncio.run(scenario())


def test_a_host_stopped_mid_call_leaves_an_unknown_outcome_and_the_next_asks_again(
    tmp_path: Path,
) -> None:
    world = _world(tmp_path)
    hanging = RejectsTheFirstNode(hang=True)

    async def scenario() -> None:
        first = _host(tmp_path, world, planners=_planners(hanging))
        try:
            handle = await first.submit(_spec(tmp_path))
            experiment_id = handle.experiment_id
            invocations = first.repository.agent_invocations

            async def asked() -> None:
                while not invocations.for_experiment(str(experiment_id)):
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(asked(), timeout=_TIMEOUT)
            (orphan,) = invocations.for_experiment(str(experiment_id))
            assert orphan.status is S.INTENDED
        finally:
            await first.close()

        answering = RejectsTheFirstNode()
        second = _host(tmp_path, world, planners=_planners(answering))
        try:
            handle = await second.attach(experiment_id)
            with pytest.raises(ReconciliationEscalatedError, match="derived from agent invocation"):
                await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)
            first_attempt, second_attempt = second.repository.agent_invocations.for_experiment(
                str(experiment_id)
            )
            assert (first_attempt.id, first_attempt.status) == (orphan.id, S.OUTCOME_UNKNOWN)
            assert (second_attempt.attempt, second_attempt.status) == (2, S.COMPLETED)
            assert len(answering.requests) == 1
        finally:
            await second.close()

    asyncio.run(scenario())


class _UnboundActionPlanner:
    """A planner that records invocations, yet proposes an action naming none."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.descriptor, self.spec = inner.descriptor, inner.spec
        self.journals: list[Any] = []

    def with_journal(self, journal: Any) -> Any:
        self.journals.append(journal)
        return self

    async def propose(self, context: Any) -> Any:
        from tests.test_experiment.test_adaptive_mvp import _action

        return _action(await self.inner.propose(context), context)


def test_an_unbound_action_from_a_model_backed_planner_is_refused(tmp_path: Path) -> None:
    """Review 1 (S1): verification is unconditional for a planner that records invocations."""
    from tests.test_experiment.adaptive_fixtures import adaptive_spec as rule_based_spec

    world = _world(tmp_path)
    wrapped: list[_UnboundActionPlanner] = []

    def factory(spec: Any) -> Any:
        wrapped.append(_UnboundActionPlanner(PLANNERS["rule-based"](spec)))
        return wrapped[-1]

    async def scenario() -> None:
        host = _host(tmp_path, world, planners={**PLANNERS, "rule-based": factory})
        try:
            handle = await host.submit(rule_based_spec(tmp_path))
            with pytest.raises(ReconciliationEscalatedError, match="names no agent invocation"):
                await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)
            assert any(planner.journals for planner in wrapped), "the host attached its journal"
            record = _record(host, handle.experiment_id)
            assert host.repository.actions.for_target("node", str(record["nodes"][0].id)) == ()
        finally:
            await host.close()

    asyncio.run(scenario())
