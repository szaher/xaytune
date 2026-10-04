"""ScriptedAgentModel: deterministic answers with no network, for planners' tests."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from xaytune.agent import (
    AgentModel,
    AgentModelIdentity,
    AgentModelInvocationError,
    AgentModelOutputError,
    AgentModelRequest,
    AgentModelResponse,
    ScriptedAgentModel,
    invoke_agent_model,
)
from xaytune.core.capabilities import require_supported_plugin

REQUEST = AgentModelRequest(
    system="s",
    messages=({"role": "user", "content": "go"},),  # type: ignore[arg-type]
    response_schema={  # type: ignore[arg-type]
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
    },
)


def run(model: ScriptedAgentModel, times: int = 1) -> list[AgentModelResponse]:
    async def ask() -> list[AgentModelResponse]:
        return [await invoke_agent_model(model, REQUEST) for _ in range(times)]

    return asyncio.run(ask())


def test_it_is_an_agent_model_under_a_supported_plugin_api() -> None:
    model = ScriptedAgentModel([])
    assert isinstance(model, AgentModel)
    require_supported_plugin(model.descriptor.plugin)
    assert model.descriptor.model == AgentModelIdentity(provider="xaytune", name="scripted")


def test_it_answers_in_script_order_and_records_every_request() -> None:
    model = ScriptedAgentModel([{"n": 1}, {"n": 2}, {"n": 3}])
    answers = run(model, 2)
    assert [answer.content["n"] for answer in answers] == [1, 2]
    assert model.requests == (REQUEST, REQUEST)
    assert model.remaining == 1


def test_the_same_script_gives_the_same_answers() -> None:
    def script() -> list[Any]:
        return [{"n": 1}, {"n": 2}]

    assert run(ScriptedAgentModel(script()), 2) == run(ScriptedAgentModel(script()), 2)


def test_it_names_the_model_it_was_given() -> None:
    model = ScriptedAgentModel(
        [{"n": 1}], model=AgentModelIdentity(provider="p", name="m", revision="r")
    )
    (answer,) = run(model)
    assert (answer.model, answer.model_revision) == ("m", "r")


def test_a_whole_response_is_returned_unchanged() -> None:
    scripted = AgentModelResponse(content={"n": 7}, model="m", finish_reason="length")
    assert run(ScriptedAgentModel([scripted])) == [scripted]


def test_a_malformed_scripted_answer_is_refused_at_the_boundary() -> None:
    model = ScriptedAgentModel([{"n": "seven"}])
    with pytest.raises(AgentModelOutputError):
        run(model)
    assert len(model.requests) == 1


def test_a_scripted_exception_is_raised_and_the_request_still_recorded() -> None:
    model = ScriptedAgentModel([AgentModelInvocationError("rate limited")])
    with pytest.raises(AgentModelInvocationError, match="rate limited"):
        run(model)
    assert model.requests == (REQUEST,)


def test_running_out_of_script_is_an_invocation_error() -> None:
    model = ScriptedAgentModel([{"n": 1}])
    with pytest.raises(AgentModelInvocationError, match="ran out after 1 answers"):
        run(model, 2)
