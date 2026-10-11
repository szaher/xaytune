"""Harness compilation: deterministic, side-effect free, refusing what it cannot honour.

ADR-019 E1-E4.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import socket
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tests.test_harness.fixtures import (
    CONTRACT,
    CONTRACT_DIGEST,
    CONTRACT_URI,
    compile_spec,
    context,
    harness,
    harness_record,
    source,
)
from xaytune.core.agent_execution import (
    AgentExecutionSpec,
    ResolvedContent,
    ResolvedHarnessContent,
    agent_execution_fingerprint,
    agent_execution_identity_v1,
    agent_run_request_digest,
)
from xaytune.core.capabilities import PluginDescriptor
from xaytune.core.domain.harness import AgentHarnessSpec, ContentRef
from xaytune.core.errors import IncompatiblePluginError
from xaytune.core.execution import ResourceRequirements
from xaytune.core.immutable import FrozenDict
from xaytune.harness import (
    HARNESS_COMPILER_API,
    SCRIPTED_RUNNER,
    HarnessCompiler,
    MappingContentSource,
    ReferenceHarnessCompiler,
    UnsupportedHarnessError,
    UnverifiedContentError,
    resolve_harness_content,
)

# ---- deterministic -----------------------------------------------------------------------


def test_the_same_harness_in_the_same_context_compiles_to_the_same_spec(tmp_path: Path) -> None:
    first = compile_spec(tmp_path)
    # Parsed again from JSON in reversed key order: the compiled spec must not care.
    reversed_record = json.loads(
        json.dumps(harness_record()), object_pairs_hook=lambda p: dict(reversed(p))
    )
    again = AgentHarnessSpec.model_validate(reversed_record)
    second = ReferenceHarnessCompiler().compile(again, context(again, tmp_path))
    assert first == second
    # Mappings keep their input order when dumped; their canonical form is what is identity.
    assert json.dumps(first.model_dump(mode="json"), sort_keys=True) == json.dumps(
        second.model_dump(mode="json"), sort_keys=True
    )
    assert agent_execution_fingerprint(first) == agent_execution_fingerprint(second)
    assert agent_run_request_digest(first) == agent_run_request_digest(second)


def test_a_compiled_spec_round_trips_through_json(tmp_path: Path) -> None:
    spec = compile_spec(tmp_path)
    again = AgentExecutionSpec.model_validate_json(spec.model_dump_json())
    assert again == spec
    assert agent_run_request_digest(again) == agent_run_request_digest(spec)


def test_compilation_reads_no_file_and_opens_no_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = harness()
    ctx = context(spec, tmp_path)

    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("compilation must not touch a file or the network")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(Path, "exists", refuse)
    compiled = ReferenceHarnessCompiler().compile(spec, ctx)
    assert compiled.harness == spec


def test_the_compiled_spec_carries_what_was_checked(tmp_path: Path) -> None:
    spec = compile_spec(tmp_path)
    assert spec.api_version == "xaytune.agent-execution/v1alpha1"
    assert spec.harness_fingerprint == spec.harness.harness_fingerprint()
    assert spec.effective_tools == ("kb.search",)
    assert [e.digest for e in spec.content.entries] == [CONTRACT_DIGEST]
    assert spec.content.entries[0].text == CONTRACT
    assert spec.telemetry.protocol_version == "xaytune.agent-telemetry/v1alpha1"
    assert spec.config == {
        "compiler_api": HARNESS_COMPILER_API,
        "token_counter": "xaytune.whitespace-tokens/v1",
    }
    assert spec.required_capabilities.extensions == {"agent_runner": SCRIPTED_RUNNER}
    assert [(o.name, o.uri, o.kind) for o in spec.outputs] == [
        ("result", str(tmp_path / "agent-run.json"), "xaytune.agent-run-result/v1alpha1")
    ]
    assert spec.producer.name == "reference-harness"


# ---- identities ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "path", "value"),
    [
        ("instructions", ("system", "text"), "You are terse."),
        ("context", ("max_context_tokens",), 999),
        ("context", ("overflow",), "fail"),
        ("stopping", ("max_turns",), 3),
        ("stopping", ("terminate_on",), ["no-tool-call"]),
        ("tools", ("definitions", 0, "description", "text"), "Search the KB, briefly."),
        ("tools", ("enabled",), ["kb.search", "tickets.close"]),
    ],
)
def test_a_different_harness_is_a_different_scientific_identity(
    tmp_path: Path, section: str, path: tuple[Any, ...], value: Any
) -> None:
    base = compile_spec(tmp_path)
    record = harness_record()
    node: Any = record[section]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    changed = AgentHarnessSpec.model_validate(record)
    other = ReferenceHarnessCompiler().compile(changed, context(changed, tmp_path))
    assert other.harness_fingerprint != base.harness_fingerprint
    assert agent_execution_fingerprint(other) != agent_execution_fingerprint(base)


class _NextRelease(ReferenceHarnessCompiler):
    descriptor = ReferenceHarnessCompiler.descriptor.model_copy(update={"plugin_version": "0.2.0"})


@pytest.mark.parametrize(
    "change",
    ["seed", "replicate", "output", "task", "compiler-release", "resources"],
)
def test_an_operational_change_moves_execution_identity_not_scientific(
    tmp_path: Path, change: str
) -> None:
    spec = harness()
    base = ReferenceHarnessCompiler().compile(spec, context(spec, tmp_path))
    if change == "seed":
        other = ReferenceHarnessCompiler().compile(spec, context(spec, tmp_path, seed=8))
    elif change == "replicate":
        other = ReferenceHarnessCompiler().compile(spec, context(spec, tmp_path, replicate=1))
    elif change == "output":
        other = ReferenceHarnessCompiler().compile(spec, context(spec, tmp_path / "elsewhere"))
    elif change == "task":
        task = base.task.model_copy(update={"task_id": "support/refund-002"})
        other = ReferenceHarnessCompiler().compile(spec, context(spec, tmp_path, task=task))
    elif change == "compiler-release":
        other = _NextRelease().compile(spec, context(spec, tmp_path))
    else:
        other = base.model_copy(update={"resources": ResourceRequirements(cpus=2.0)})
    assert other.harness_fingerprint == base.harness_fingerprint
    assert agent_execution_fingerprint(other) != agent_execution_fingerprint(base)
    assert agent_run_request_digest(other) != agent_run_request_digest(base)


def test_annotations_move_neither_identity(tmp_path: Path) -> None:
    base = compile_spec(tmp_path)
    annotated = base.model_copy(
        update={
            "metadata": FrozenDict({"owner": "team-a"}),
            "harness": base.harness.model_copy(update={"metadata": FrozenDict({"note": "x"})}),
        }
    )
    assert annotated.harness_fingerprint == base.harness_fingerprint
    assert agent_execution_fingerprint(annotated) == agent_execution_fingerprint(base)
    # The request is literally different, so it is a different request.
    assert agent_run_request_digest(annotated) != agent_run_request_digest(base)


def test_execution_identity_is_a_frozen_v1_projection() -> None:
    spec = compile_spec("/runs/r1")
    identity = agent_execution_identity_v1(spec)
    assert identity["type"] == "agent-execution"
    assert identity["version"] == 1
    assert identity["harness_fingerprint"] == spec.harness_fingerprint
    assert "harness" not in identity and "metadata" not in identity
    assert "observability" not in identity
    assert identity["content"] == [CONTRACT_DIGEST]
    assert set(identity) == {
        "type", "version", "harness_fingerprint", "compiler", "entrypoint", "arguments",
        "config", "environment", "secrets", "dependencies", "container", "inputs", "outputs",
        "resources", "telemetry", "required_capabilities", "task", "seed", "replicate",
        "effective_tools", "content",
    }  # fmt: skip
    assert agent_execution_fingerprint(spec) == PINNED_EXECUTION_V1


PINNED_EXECUTION_V1 = "sha256:1bd6fa828f15c04bacb18c4f474be11b5fd99531e4976e1cdf484f2177bbf67c"
"""``agent_execution_fingerprint`` of the fixture spec at ``/runs/r1``. A change here
re-identifies every recorded harness execution."""


# ---- refusals -------------------------------------------------------------------------------


def _model(**changes: Any) -> dict[str, Any]:
    model = harness_record()["model"]
    model.update(changes)
    return model


def _generation(**changes: Any) -> dict[str, Any]:
    model = harness_record()["model"]
    model["generation"].update(changes)
    return model


def _template(**changes: Any) -> dict[str, Any]:
    instructions = harness_record()["instructions"]
    instructions["task_template"].update(changes)
    return instructions


def _tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    tools = harness_record()["tools"]
    tools["definitions"][0]["input_schema"] = schema
    return tools


@pytest.mark.parametrize(
    ("sections", "reason"),
    [
        (
            {"model": _model(provider="anthropic", model="claude-sonnet-5-5", revision="20260901")},
            "scripted model",
        ),
        ({"model": _model(artifact_digest=None, revision="v1")}, "artifact_digest"),
        ({"model": _generation(provider_parameters={"thinking_budget": 1})}, "provider_parameters"),
        ({"instructions": _template(syntax="jinja2")}, "'jinja2' is not supported"),
        (
            {"instructions": {"system": {"kind": "inline", "text": "s"}, "task_template": None}},
            "task template is required",
        ),
        (
            {
                "instructions": _template(
                    content={"kind": "inline", "text": "{ticket.id} {customer}"}
                )
            },
            "plain names",
        ),
        (
            {
                "instructions": _template(
                    content={"kind": "inline", "text": "{ticket!r} {customer}"}
                )
            },
            "no format spec",
        ),
        (
            {
                "instructions": _template(
                    content={"kind": "inline", "text": "{ticket:>9} {customer}"}
                )
            },
            "no format spec",
        ),
        (
            {"instructions": _template(content={"kind": "inline", "text": "{0} {customer}"})},
            "plain names",
        ),
        ({"instructions": _template(content={"kind": "inline", "text": "{ticket"})}, "malformed"),
        (
            {"instructions": _template(content={"kind": "inline", "text": "only {ticket}"})},
            "not exactly its declared",
        ),
        (
            {
                "tools": _tool_schema(
                    {"type": "object", "properties": {"q": {"type": "string", "pattern": "x"}}}
                )
            },
            "agent-schema-subset",
        ),
        (
            {"tools": _tool_schema({"type": "object", "additionalProperties": True})},
            "agent-schema-subset",
        ),
    ],
)
def test_what_the_runner_could_not_honour_is_refused_before_compiling(
    tmp_path: Path, sections: dict[str, Any], reason: str
) -> None:
    spec = AgentHarnessSpec.model_validate(harness_record(**sections))
    with pytest.raises(UnsupportedHarnessError, match=reason) as refused:
        ReferenceHarnessCompiler().compile(spec, context(spec, tmp_path))
    assert refused.value.compiler == "reference-harness"


def test_supports_reports_every_reason_at_once() -> None:
    spec = AgentHarnessSpec.model_validate(
        harness_record(
            model=_generation(provider_parameters={"seed_bias": 1}),
            instructions=_template(syntax="jinja2"),
        )
    )
    result = ReferenceHarnessCompiler().supports(spec)
    assert not result
    assert len(result.reasons) == 2
    assert ReferenceHarnessCompiler().supports(harness()).supported


def test_a_disabled_tool_is_not_held_to_the_runner_schema_subset() -> None:
    tools = harness_record()["tools"]
    tools["definitions"][1]["input_schema"] = {
        "type": "object",
        "properties": {"x": {"type": "string", "format": "uri"}},
    }
    assert ReferenceHarnessCompiler().supports(harness(tools=tools)).supported


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"authorized_tools": ()}, "not authorized: kb.search"),
        ({"authorized_tools": ("tickets.close",)}, "cannot authorize its own tools"),
        ({"output_uri": "runs/r1"}, "absolute local directory"),
        ({"output_uri": "/runs/../etc"}, "absolute local directory"),
    ],
)
def test_the_context_is_checked_too(tmp_path: Path, overrides: dict[str, Any], reason: str) -> None:
    spec = harness()
    with pytest.raises(UnsupportedHarnessError, match=reason):
        ReferenceHarnessCompiler().compile(spec, context(spec, tmp_path, **overrides))


@pytest.mark.parametrize(
    "variables",
    [{"customer": "Ann"}, {"customer": "Ann", "ticket": "t", "extra": "x"}],
    ids=["missing", "extra"],
)
def test_the_task_must_supply_exactly_the_template_variables(
    tmp_path: Path, variables: dict[str, str]
) -> None:
    spec = harness()
    base = context(spec, tmp_path)
    task = base.task.model_copy(update={"variables": FrozenDict(variables)})
    with pytest.raises(UnsupportedHarnessError, match="exactly the template's variables"):
        ReferenceHarnessCompiler().compile(spec, base.model_copy(update={"task": task}))


def test_the_compiler_is_a_versioned_plugin() -> None:
    compiler = ReferenceHarnessCompiler()
    assert isinstance(compiler, HarnessCompiler)
    capabilities = compiler.capabilities()
    assert capabilities.api_version == HARNESS_COMPILER_API
    assert capabilities.harness_schemas == ("xaytune.agent-harness/v1alpha1",)
    assert capabilities.model_providers == ("xaytune",)
    assert capabilities.template_syntaxes == ("literal", "python-format")
    assert capabilities.provider_parameters == ()
    assert capabilities.runner == SCRIPTED_RUNNER


def test_a_compiler_speaking_an_unknown_plugin_api_is_refused(tmp_path: Path) -> None:
    class Future(ReferenceHarnessCompiler):
        descriptor = PluginDescriptor(
            api_version="xaytune.plugins/v9",
            name="future",
            plugin_version="9",
            provider="x",
            xaytune_version="9",
        )

    spec = harness()
    with pytest.raises(IncompatiblePluginError):
        Future().compile(spec, context(spec, tmp_path))


# ---- content ------------------------------------------------------------------------------------


class _CountingSource(MappingContentSource):
    def __init__(self, blobs: dict[str, bytes]) -> None:
        super().__init__(blobs)
        self.fetched: list[str] = []

    def fetch(self, ref: ContentRef) -> bytes | None:
        self.fetched.append(ref.uri)
        return super().fetch(ref)


def _contract_ref(**changes: Any) -> dict[str, Any]:
    tools = harness_record()["tools"]
    tools["definitions"][0]["contract"].update(changes)
    return tools


def test_only_reachable_references_are_fetched(tmp_path: Path) -> None:
    tools = harness_record()["tools"]
    disabled_uri = "https://contracts.example.com/tickets/close.txt"
    tools["definitions"][1]["contract"] = {
        "kind": "ref",
        "uri": disabled_uri,
        "digest": "sha256:" + "1" * 64,
        "media_type": "text/plain",
    }
    spec = harness(tools=tools)
    counting = _CountingSource({CONTRACT_URI: CONTRACT.encode()})
    resolved = resolve_harness_content(spec, counting)
    assert counting.fetched == [CONTRACT_URI]
    assert [e.digest for e in resolved.entries] == [CONTRACT_DIGEST]


@pytest.mark.parametrize(
    ("blobs", "tools", "reason"),
    [
        ({}, None, "tools.kb.search.contract: the content source does not hold it"),
        ({CONTRACT_URI: b"tampered"}, None, "do not have the pinned digest"),
        (
            {CONTRACT_URI: b"\xff\xfe"},
            _contract_ref(digest="sha256:" + hashlib.sha256(b"\xff\xfe").hexdigest()),
            "not valid UTF-8",
        ),
        (
            {CONTRACT_URI: CONTRACT.encode()},
            _contract_ref(media_type="application/pdf"),
            "not text/",
        ),
    ],
)
def test_unverified_content_is_refused_before_use(
    blobs: dict[str, bytes], tools: dict[str, Any] | None, reason: str
) -> None:
    spec = harness(tools=tools) if tools else harness()
    with pytest.raises(UnverifiedContentError, match=reason) as refused:
        resolve_harness_content(spec, MappingContentSource(blobs))
    assert CONTRACT_URI not in str(refused.value)


def test_compiling_with_unresolved_content_is_refused(tmp_path: Path) -> None:
    spec = harness()
    empty = ResolvedHarnessContent(entries=())
    with pytest.raises(UnsupportedHarnessError, match="was not resolved before compiling"):
        ReferenceHarnessCompiler().compile(spec, context(spec, tmp_path, content=empty))


def test_unreachable_resolved_content_is_not_carried(tmp_path: Path) -> None:
    spec = harness()
    extra = ResolvedContent(digest="sha256:" + hashlib.sha256(b"x").hexdigest(), text="x")
    content = resolve_harness_content(spec, source())
    padded = ResolvedHarnessContent(
        entries=tuple(sorted((*content.entries, extra), key=lambda e: e.digest))
    )
    compiled = ReferenceHarnessCompiler().compile(spec, context(spec, tmp_path, content=padded))
    assert compiled == compile_spec(tmp_path)


def test_resolved_content_must_have_its_digest() -> None:
    with pytest.raises(ValidationError, match="does not have the digest"):
        ResolvedContent(digest=CONTRACT_DIGEST, text="something else")


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (lambda r: r["content"]["entries"][0].update(text="edited"), "does not have the digest"),
        (lambda r: r.update(harness_fingerprint="sha256:" + "0" * 64), "harness_fingerprint"),
        (lambda r: r.update(effective_tools=["kb.search", "tickets.close"]), "effective_tools"),
        (lambda r: r["content"].update(entries=[]), "no reference unresolved"),
        (lambda r: r["task"]["variables"].pop("ticket"), "template's variables"),
        (lambda r: r["harness"]["tools"].update(enabled=[]), "harness_fingerprint"),
        (
            lambda r: r["telemetry"].update(protocol_version="xaytune.telemetry/v1alpha3"),
            "agent-telemetry",
        ),
    ],
)
def test_a_spec_edited_after_compilation_does_not_parse(
    tmp_path: Path, edit: Any, reason: str
) -> None:
    record = compile_spec(tmp_path).model_dump(mode="json")
    edit(record)
    with pytest.raises(ValidationError, match=reason):
        AgentExecutionSpec.model_validate(record)
