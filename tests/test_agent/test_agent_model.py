"""The agent-model boundary: request identity, structured answers, failing closed."""

from __future__ import annotations

import asyncio
import traceback
from typing import Any

import pytest
from pydantic import ValidationError

from xaytune.agent import (
    SCRIPTED_MODEL,
    AgentModelDescriptor,
    AgentModelIdentity,
    AgentModelInvocationError,
    AgentModelOutputError,
    AgentModelRequest,
    AgentModelResponse,
    AgentModelUsage,
    ScriptedAgentModel,
    UnsupportedResponseSchemaError,
    agent_model_request_identity_v1,
    invoke_agent_model,
)
from xaytune.core.capabilities import PluginDescriptor
from xaytune.core.errors import IncompatiblePluginError

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "enum": ["widen", "deepen"]},
        "rationale": {"type": "string", "maxLength": 200},
    },
    "required": ["choice", "rationale"],
}


def request(**changes: Any) -> AgentModelRequest:
    fields: dict[str, Any] = {
        "system": "You choose the next experiment.",
        "messages": [{"role": "user", "content": "Which next?"}],
        "response_schema": SCHEMA,
        "temperature": 0.0,
        "max_output_tokens": 256,
        **changes,
    }
    return AgentModelRequest.model_validate(fields)


def invoke(model: Any, req: AgentModelRequest) -> AgentModelResponse:
    return asyncio.run(invoke_agent_model(model, req))


# -- request: canonical, serializable, identified ---------------------------------


def test_a_request_round_trips_through_json_unchanged() -> None:
    original = request(
        messages=[
            {"role": "user", "content": "Which next?"},
            {"role": "assistant", "content": '{"choice": "widen"}'},
            {"role": "user", "content": "Why?"},
        ]
    )
    restored = AgentModelRequest.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.fingerprint(SCRIPTED_MODEL) == original.fingerprint(SCRIPTED_MODEL)


def test_the_fingerprint_survives_serialization_and_key_order() -> None:
    reordered = {
        "required": ["choice", "rationale"],
        "properties": dict(reversed(list(SCHEMA["properties"].items()))),
        "type": "object",
    }
    restored = AgentModelRequest.model_validate_json(request().model_dump_json())
    assert (
        request().fingerprint(SCRIPTED_MODEL)
        == request(response_schema=reordered).fingerprint(SCRIPTED_MODEL)
        == restored.fingerprint(SCRIPTED_MODEL)
    )


def test_an_integral_temperature_is_the_same_request_as_its_float() -> None:
    assert request(temperature=1).fingerprint(SCRIPTED_MODEL) == request(
        temperature=1.0
    ).fingerprint(SCRIPTED_MODEL)


@pytest.mark.parametrize(
    "changes",
    [
        {"system": "You choose the next experiment carefully."},
        {"messages": [{"role": "user", "content": "Which next, please?"}]},
        {
            "messages": [
                {"role": "assistant", "content": "Which next?"},
                {"role": "user", "content": "Which next?"},
            ]
        },
        {"response_schema": {**SCHEMA, "required": ["choice"]}},
        {"temperature": 0.7},
        {"temperature": None},
        {"max_output_tokens": 512},
        {"max_output_tokens": None},
    ],
)
def test_every_part_of_the_request_is_identity(changes: dict[str, Any]) -> None:
    assert request(**changes).fingerprint(SCRIPTED_MODEL) != request().fingerprint(SCRIPTED_MODEL)


@pytest.mark.parametrize(
    "model",
    [
        AgentModelIdentity(provider="xaytune", name="scripted-2"),
        AgentModelIdentity(provider="other", name="scripted"),
        AgentModelIdentity(provider="xaytune", name="scripted", revision="2026-10-01"),
    ],
)
def test_the_model_asked_is_identity(model: AgentModelIdentity) -> None:
    assert request().fingerprint(model) != request().fingerprint(SCRIPTED_MODEL)


def test_the_adapter_is_not_identity() -> None:
    plugin = ScriptedAgentModel([]).descriptor.plugin
    other = plugin.model_copy(update={"plugin_version": "99.0.0", "name": "another-adapter"})
    a = AgentModelDescriptor(plugin=plugin, model=SCRIPTED_MODEL)
    b = AgentModelDescriptor(plugin=other, model=SCRIPTED_MODEL)
    assert request().fingerprint(a.model) == request().fingerprint(b.model)


def test_what_an_attempt_reports_is_not_the_requests_identity() -> None:
    req = request()
    before = req.fingerprint(SCRIPTED_MODEL)
    answers = [
        AgentModelResponse(
            content={"choice": "widen", "rationale": "r"},
            model="scripted-2026",
            model_revision=revision,
            finish_reason="stop",
            usage=AgentModelUsage(input_tokens=tokens, output_tokens=tokens),
            provider_request_id=request_id,
            latency_seconds=latency,
        )
        for revision, tokens, request_id, latency in [
            (None, None, None, None),
            ("r1", 10, "req-1", 0.5),
            ("r2", 999, "req-2", 30.0),
        ]
    ]
    model = ScriptedAgentModel(answers)
    for _ in answers:
        invoke(model, req)
    assert {asked.fingerprint(SCRIPTED_MODEL) for asked in model.requests} == {before}
    identity = agent_model_request_identity_v1(SCRIPTED_MODEL, req)
    assert set(identity) == {
        "version",
        "model",
        "system",
        "messages",
        "response_schema",
        "generation",
    }


def test_the_identity_names_every_field_of_the_request() -> None:
    """A field added to the request or the model identity must be placed in the identity."""
    assert set(AgentModelRequest.model_fields) == {
        "system",
        "messages",
        "response_schema",
        "temperature",
        "max_output_tokens",
    }
    assert set(AgentModelIdentity.model_fields) == {"provider", "name", "revision"}
    identity = agent_model_request_identity_v1(SCRIPTED_MODEL, request())
    assert set(identity["generation"]) == {"temperature", "max_output_tokens"}
    assert set(identity["model"]) == {"provider", "name", "revision"}


def test_the_identity_is_pinned() -> None:
    """Version 1 is a durable promise: these bytes must not drift."""
    assert request().fingerprint(SCRIPTED_MODEL) == (
        "sha256:5c124c9e39e73051922b968ca900c09a90aa5472a1ba24a73c6527eff0332835"
    )


# -- request: what cannot be asked ------------------------------------------------


def test_a_request_without_a_response_schema_cannot_be_built() -> None:
    with pytest.raises(ValidationError, match="response_schema"):
        AgentModelRequest.model_validate(
            {"system": "s", "messages": [{"role": "user", "content": "hi"}]}
        )


def test_a_response_schema_outside_the_subset_cannot_be_asked_for() -> None:
    with pytest.raises(ValidationError) as caught:
        request(response_schema={**SCHEMA, "patternProperties": {}})
    assert isinstance(caught.value.errors()[0]["ctx"]["error"], UnsupportedResponseSchemaError)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "{"}],
        [{"role": "system", "content": "hi"}],
        [{"role": "user", "content": ""}],
    ],
)
def test_messages_are_a_conversation_ending_with_the_user(messages: list[Any]) -> None:
    with pytest.raises(ValidationError):
        request(messages=messages)


@pytest.mark.parametrize(
    "changes",
    [
        {"temperature": -0.1},
        {"temperature": float("nan")},
        {"max_output_tokens": 0},
        {"max_output_tokens": 1.5},
        {"tools": []},
    ],
)
def test_generation_parameters_are_bounded_and_closed(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        request(**changes)


def test_a_request_is_deeply_immutable() -> None:
    req = request()
    with pytest.raises(TypeError):
        req.response_schema["properties"]["choice"]["enum"] = ["anything"]  # type: ignore[index]
    with pytest.raises(ValidationError):
        req.system = "changed"  # type: ignore[misc]


# -- response: structured, or refused whole ----------------------------------------


def test_a_conforming_answer_is_returned() -> None:
    answer = invoke(ScriptedAgentModel([{"choice": "widen", "rationale": "loss"}]), request())
    assert answer.content == {"choice": "widen", "rationale": "loss"}
    assert answer.model == "scripted"
    assert answer.finish_reason == "stop"


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ({"choice": "widen"}, "missing required 'rationale'"),
        ({"choice": "shrink", "rationale": "r"}, "must be one of"),
        ({"choice": "widen", "rationale": "r", "run": "rm -rf /"}, "unexpected 'run'"),
        ({"choice": 1, "rationale": "r"}, "expected string, got integer"),
        ({"choice": "widen", "rationale": "x" * 201}, "at most 200 characters"),
        ({}, "missing required 'choice'"),
    ],
)
def test_a_malformed_answer_is_refused_whole(content: dict[str, Any], reason: str) -> None:
    req = request()
    with pytest.raises(AgentModelOutputError) as caught:
        invoke(ScriptedAgentModel([content]), req)
    assert any(reason in line for line in caught.value.reasons), caught.value.reasons
    assert caught.value.request_fingerprint == req.fingerprint(SCRIPTED_MODEL)


def test_every_violation_is_reported_not_just_the_first() -> None:
    with pytest.raises(AgentModelOutputError) as caught:
        invoke(ScriptedAgentModel([{"choice": "shrink", "extra": True}]), request())
    assert len(caught.value.reasons) == 3


def test_an_adapter_returning_something_else_is_refused() -> None:
    class RawProvider(ScriptedAgentModel):
        async def generate(self, request: AgentModelRequest) -> Any:
            return {"choice": "widen", "rationale": "raw provider object"}

    with pytest.raises(AgentModelOutputError, match="not an AgentModelResponse"):
        invoke(RawProvider([]), request())


def test_an_adapter_failure_keeps_no_reference_to_the_adapters_exception() -> None:
    cause = ConnectionError("provider unreachable")
    with pytest.raises(AgentModelInvocationError, match="ConnectionError$") as caught:
        invoke(ScriptedAgentModel([cause]), request())
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_an_adapter_failure_traceback_never_exposes_provider_error_text() -> None:
    secret = "Authorization: Bearer sk-live-0123456789abcdef"
    cause = RuntimeError(f"POST https://user:pw@api.vendor.test/v1 failed; {secret}")
    with pytest.raises(AgentModelInvocationError) as caught:
        invoke(ScriptedAgentModel([cause]), request())
    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    assert "RuntimeError" in rendered
    assert "sk-live" not in rendered
    assert "api.vendor.test" not in rendered
    assert "user:pw" not in rendered


def test_an_adapter_failure_message_names_the_type_and_never_the_text() -> None:
    """SDK messages can carry URLs, headers and keys; the wrapper must not repeat them."""
    secret = "Authorization: Bearer sk-live-0123456789abcdef"
    cause = RuntimeError(f"POST https://user:pw@api.vendor.test/v1 failed; {secret}")
    req = request()
    with pytest.raises(AgentModelInvocationError) as caught:
        invoke(ScriptedAgentModel([cause]), req)
    message = str(caught.value)
    assert "RuntimeError" in message
    assert req.fingerprint(SCRIPTED_MODEL) in message
    assert "sk-live" not in message
    assert "api.vendor.test" not in message
    assert caught.value.args == (message,)


def test_an_adapters_own_invocation_error_keeps_its_message() -> None:
    with pytest.raises(AgentModelInvocationError, match="^rate limited, retry after 30s$"):
        invoke(
            ScriptedAgentModel([AgentModelInvocationError("rate limited, retry after 30s")]),
            request(),
        )


PINNED = AgentModelIdentity(provider="vendor", name="large", revision="2026-10-01")


def answered_by(revision: str | None) -> AgentModelResponse:
    return AgentModelResponse(
        content={"choice": "widen", "rationale": "r"}, model="large", model_revision=revision
    )


def test_a_response_from_another_revision_than_the_pinned_one_is_refused_whole() -> None:
    req = request()
    with pytest.raises(AgentModelOutputError) as caught:
        invoke(ScriptedAgentModel([answered_by("2026-10-02")], model=PINNED), req)
    assert caught.value.reasons == (
        "the model reports revision '2026-10-02', not the pinned '2026-10-01'",
    )
    assert caught.value.request_fingerprint == req.fingerprint(PINNED)


def test_the_revision_is_checked_before_the_content() -> None:
    wrong = answered_by("2026-10-02").model_copy(update={"content": {"bad": True}})
    with pytest.raises(AgentModelOutputError) as caught:
        invoke(ScriptedAgentModel([wrong], model=PINNED), request())
    assert len(caught.value.reasons) == 1
    assert "pinned" in caught.value.reasons[0]


@pytest.mark.parametrize(
    ("pinned", "reported"),
    [
        ("2026-10-01", "2026-10-01"),
        ("2026-10-01", None),  # the provider did not say: not a contradiction
        (None, "2026-10-02"),  # nothing pinned: any revision is the one served
        (None, None),
    ],
)
def test_a_revision_that_does_not_contradict_the_pin_is_accepted(
    pinned: str | None, reported: str | None
) -> None:
    model = AgentModelIdentity(provider="vendor", name="large", revision=pinned)
    answer = invoke(ScriptedAgentModel([answered_by(reported)], model=model), request())
    assert answer.model_revision == reported


def test_an_alias_resolving_to_a_more_specific_name_is_accepted() -> None:
    served = answered_by("2026-10-01").model_copy(update={"model": "large-2026-10-01"})
    answer = invoke(ScriptedAgentModel([served], model=PINNED), request())
    assert answer.model == "large-2026-10-01"


def test_an_adapter_speaking_another_plugin_api_is_refused_before_it_is_asked() -> None:
    model = ScriptedAgentModel([{"choice": "widen", "rationale": "r"}])
    model.descriptor = model.descriptor.model_copy(
        update={
            "plugin": model.descriptor.plugin.model_copy(
                update={"api_version": "xaytune.plugins/v0"}
            )
        }
    )
    with pytest.raises(IncompatiblePluginError):
        invoke(model, request())
    assert model.requests == ()


def test_unknown_usage_stays_unknown() -> None:
    answer = invoke(ScriptedAgentModel([{"choice": "widen", "rationale": "r"}]), request())
    assert answer.usage == AgentModelUsage()
    assert answer.usage.input_tokens is None
    assert answer.usage.output_tokens is None
    partial = AgentModelUsage(input_tokens=12)
    assert partial.output_tokens is None
    with pytest.raises(ValidationError):
        AgentModelUsage(input_tokens=-1)
    with pytest.raises(ValidationError):
        AgentModelUsage(output_tokens=1.0)  # type: ignore[arg-type]


def test_a_response_round_trips_and_is_closed() -> None:
    answer = AgentModelResponse(
        content={"choice": "widen", "rationale": "r"},
        model="m",
        usage=AgentModelUsage(input_tokens=3, output_tokens=4),
        provider_request_id="req-1",
        latency_seconds=0.25,
    )
    assert AgentModelResponse.model_validate_json(answer.model_dump_json()) == answer
    with pytest.raises(ValidationError):
        AgentModelResponse.model_validate({**answer.model_dump(), "reasoning": "hidden"})


# -- secrets stay in adapter configuration ------------------------------------------


def test_adapter_credentials_never_appear_in_what_is_serialized() -> None:
    secret = "sk-live-0123456789abcdef"

    class CredentialedAdapter(ScriptedAgentModel):
        def __init__(self, api_key: str) -> None:
            super().__init__(
                [{"choice": "widen", "rationale": "r"}],
                model=AgentModelIdentity(provider="vendor", name="large"),
            )
            self._api_key = api_key

    adapter = CredentialedAdapter(secret)
    req = request()
    answer = invoke(adapter, req)
    serialized = [
        req.model_dump_json(),
        answer.model_dump_json(),
        adapter.descriptor.model_dump_json(),
        repr(agent_model_request_identity_v1(adapter.descriptor.model, req)),
    ]
    assert not any(secret in text for text in serialized)
    assert set(AgentModelDescriptor.model_fields) == {"plugin", "model"}
    assert set(PluginDescriptor.model_fields) >= {"name", "plugin_version", "provider"}
