"""AgentHarnessSpec and HarnessFingerprint v1 (H02, ADR-018 D3-D9).

Written as "changing X does / does not change the fingerprint", like the
training candidate's tests, because that is the form the identity decisions
take. The pinned projection and literal at the end freeze v1.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError

from tests.test_core.test_candidate import _plain_candidate
from xaytune.core.domain.candidate import candidate_identity_v1, candidate_identity_v2
from xaytune.core.domain.harness import (
    AgentHarnessSpec,
    ContentRef,
    InlineContent,
    UnauthorizedToolsError,
    content_digest,
    harness_fingerprint,
    harness_identity_v1,
    require_tools_authorized,
)

SYSTEM = "You are a careful support agent. Answer from the knowledge base."
DIGEST = "sha256:" + "ab" * 32


def _record(**overrides: Any) -> dict[str, Any]:
    """A complete harness as plain JSON; *overrides* replace top-level sections."""
    record: dict[str, Any] = {
        "schema_version": "xaytune.agent-harness/v1alpha1",
        "model": {
            "provider": "anthropic",
            "model": "claude-sonnet-5-5",
            "revision": "20260901",
            "artifact_digest": None,
            "generation": {
                "temperature": 0.2,
                "top_p": 1.0,
                "max_output_tokens": 2048,
                "stop_sequences": [],
                "provider_parameters": {"thinking_budget": 1024},
            },
        },
        "instructions": {
            "system": {"kind": "inline", "text": SYSTEM},
            "task_template": {
                "syntax": "python-format",
                "content": {"kind": "inline", "text": "Ticket: {ticket}\nCustomer: {customer}"},
                "variables": ["customer", "ticket"],
            },
        },
        "tools": {
            "definitions": [
                {
                    "id": "kb.search",
                    "version": "1.2.0",
                    "description": {"kind": "inline", "text": "Search the knowledge base."},
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    "contract": None,
                },
                {
                    "id": "tickets.close",
                    "version": "2.0.0",
                    "description": {"kind": "inline", "text": "Close a ticket."},
                    "input_schema": {"type": "object", "properties": {}},
                    "contract": None,
                },
            ],
            "enabled": ["kb.search"],
        },
        "context": {
            "max_context_tokens": 32000,
            "selection": {"kind": "recent-turns", "turns": 12},
            "overflow": "truncate-oldest",
            "max_tool_result_tokens": 4000,
        },
        "stopping": {
            "max_turns": 20,
            "max_total_tokens": 100000,
            "max_tool_calls": 30,
            "terminate_on": ["final-answer"],
        },
    }
    record.update(overrides)
    return record


def _harness(**overrides: Any) -> AgentHarnessSpec:
    return AgentHarnessSpec.model_validate(_record(**overrides))


def _with(path: str, value: Any) -> AgentHarnessSpec:
    """The base harness with the value at dotted *path* replaced."""
    record = _record()
    *parents, last = path.split(".")
    node: Any = record
    for key in parents:
        node = node[int(key)] if isinstance(node, list) else node[key]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
    return AgentHarnessSpec.model_validate(record)


BASE = _harness().harness_fingerprint()


# ---- construction ----------------------------------------------------------------------


def test_a_harness_is_constructed_validated_and_frozen() -> None:
    harness = _harness()
    assert harness.tools.enabled == ("kb.search",)
    with pytest.raises(ValidationError):
        harness.stopping = harness.stopping  # type: ignore[misc]


def test_a_harness_round_trips_through_json_unchanged() -> None:
    harness = _harness()
    again = AgentHarnessSpec.model_validate_json(harness.model_dump_json())
    assert again == harness
    assert again.model_dump_json() == harness.model_dump_json()
    assert again.harness_fingerprint() == BASE


# ---- what identity covers -----------------------------------------------------------------


def _reordered(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _reordered(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reordered(item) for item in value]
    return value


def test_the_same_harness_serialized_in_another_order_has_the_same_fingerprint() -> None:
    shuffled = AgentHarnessSpec.model_validate(_reordered(_record()))
    assert shuffled.harness_fingerprint() == BASE
    from_json = AgentHarnessSpec.model_validate(json.loads(json.dumps(_record(), sort_keys=True)))
    assert from_json.harness_fingerprint() == BASE


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("instructions.system", {"kind": "inline", "text": SYSTEM + " Be brief."}),
        ("instructions.task_template.syntax", "jinja2"),
        ("instructions.task_template.content", {"kind": "inline", "text": "Ticket: {ticket}"}),
        ("instructions.task_template", None),
        ("tools.definitions.0.description", {"kind": "inline", "text": "Search the KB, cite."}),
        ("tools.definitions.0.version", "1.3.0"),
        ("tools.definitions.0.input_schema", {"type": "object", "properties": {}}),
        (
            "tools.definitions.0.contract",
            {"kind": "ref", "uri": "s3://t/kb", "digest": DIGEST, "media_type": "application/wasm"},
        ),
        ("tools.enabled", ["kb.search", "tickets.close"]),
        ("tools.enabled", []),
        ("context.max_context_tokens", 16000),
        ("context.selection", {"kind": "full-history"}),
        ("context.selection.turns", 6),
        ("context.overflow", "fail"),
        ("context.max_tool_result_tokens", None),
        ("model.revision", "20261001"),
        ("model.artifact_digest", DIGEST),
        ("model.model", "claude-opus-5-5"),
        ("model.provider", "bedrock"),
        ("model.generation.temperature", 0.7),
        ("model.generation.top_p", 0.9),
        ("model.generation.max_output_tokens", 4096),
        ("model.generation.stop_sequences", ["DONE"]),
        ("model.generation.provider_parameters", {"thinking_budget": 2048}),
        ("model.generation.provider_parameters", {}),
        ("stopping.max_turns", 10),
        ("stopping.max_total_tokens", None),
        ("stopping.max_tool_calls", 0),
        ("stopping.terminate_on", ["final-answer", "no-tool-call"]),
    ],
)
def test_every_behavior_changing_field_changes_the_fingerprint(path: str, value: Any) -> None:
    assert _with(path, value).harness_fingerprint() != BASE


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("metadata", {"owner": "support-team", "ticket": "XAY-12"}),
        # A tool the model never sees: defined, not enabled.
        ("tools.definitions.1.description", {"kind": "inline", "text": "Close it, politely."}),
        ("tools.definitions.1.version", "2.1.0"),
    ],
)
def test_bookkeeping_and_unreachable_definitions_leave_the_fingerprint(
    path: str, value: Any
) -> None:
    assert _with(path, value).harness_fingerprint() == BASE


def test_content_is_identified_by_its_bytes_not_its_location() -> None:
    text = "Search the knowledge base."
    digest = content_digest(InlineContent(text=text))
    by_ref = {
        "kind": "ref",
        "uri": "s3://prompts/kb-v1.txt",
        "digest": digest,
        "media_type": "text/plain",
    }
    moved = {**by_ref, "uri": "https://cdn.example.com/kb-v1.txt", "media_type": "text/markdown"}
    assert _with("tools.definitions.0.description", by_ref).harness_fingerprint() == BASE
    assert _with("tools.definitions.0.description", moved).harness_fingerprint() == BASE


def test_seeds_benchmarks_and_execution_have_nowhere_to_go() -> None:
    """Realization, evaluation and execution identity are not fields of the harness at all."""
    for field in ("seed", "replicate", "benchmark", "tasks", "scoring", "runtime", "attempt"):
        with pytest.raises(ValidationError, match="Extra inputs"):
            _harness(**{field: 1})


# ---- refusals -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("model.revision", "latest", "moving alias"),
        ("model.revision", "main", "moving alias"),
        ("model.revision", "claude-sonnet-latest", "moving alias"),
        ("model.revision", None, "pinned revision or an artifact digest"),
        ("model.artifact_digest", "md5:abc", "sha256"),
        ("model.generation.provider_parameters", {"api_key": "sk-x"}, "credential"),
        ("model.generation.provider_parameters", {"nested": {"a": 1}}, "string, number"),
        ("model.generation.temperature", 3.0, "temperature"),
        (
            "instructions.system",
            {
                "kind": "ref",
                "uri": "/home/me/prompt.txt",
                "digest": DIGEST,
                "media_type": "text/plain",
            },
            "local path",
        ),
        (
            "instructions.system",
            {
                "kind": "ref",
                "uri": "prompts/system.txt",
                "digest": DIGEST,
                "media_type": "text/plain",
            },
            "local path",
        ),
        (
            "instructions.system",
            {
                "kind": "ref",
                "uri": "file:///tmp/p.txt",
                "digest": DIGEST,
                "media_type": "text/plain",
            },
            "local path",
        ),
        (
            "instructions.system",
            {
                "kind": "ref",
                "uri": "https://u:p@host/p.txt",
                "digest": DIGEST,
                "media_type": "text/plain",
            },
            "credentials",
        ),
        (
            "instructions.system",
            {
                "kind": "ref",
                "uri": "s3://b/p.txt",
                "digest": "sha256:xyz",
                "media_type": "text/plain",
            },
            "64",
        ),
        ("instructions.system", {"kind": "inline", "text": ""}, "at least 1"),
        ("instructions.system", {"kind": "callable", "text": "x"}, "tag"),
        ("instructions.task_template.variables", ["ticket", "customer"], "sorted"),
        ("instructions.task_template.syntax", "literal", "literal template takes no"),
        ("tools.enabled", ["kb.search", "web.fetch"], "no definition: web.fetch"),
        ("tools.enabled", ["kb.search", "kb.search"], "once each"),
        ("tools.definitions.0.id", "KB Search", "lowercase dotted"),
        ("tools.definitions.0.version", "latest", "moving alias"),
        ("tools.definitions.0.input_schema", {"type": "string"}, "type: object"),
        ("context.selection", {"kind": "summarize", "model": "x"}, "tag"),
        ("context.max_context_tokens", 0, "greater than 0"),
        ("stopping.terminate_on", [], "at least 1"),
        ("stopping.terminate_on", ["final-answer", "final-answer"], "once each"),
        ("stopping.max_turns", "20", "valid integer"),
    ],
)
def test_an_invalid_or_mutable_scientific_reference_is_refused(
    path: str, value: Any, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _with(path, value)


@pytest.mark.parametrize(
    "section", ["memory", "middleware", "delegation", "routing", "permissions", "environment"]
)
def test_areas_outside_v1_are_refused_not_ignored(section: str) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        _harness(**{section: {"enabled": True}})


@pytest.mark.parametrize("version", ["xaytune.agent-harness/v1", "xaytune.agent-harness/v2alpha1"])
def test_an_unknown_schema_version_is_refused(version: str) -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        _harness(schema_version=version)


@pytest.mark.parametrize(
    "missing",
    [
        "model.generation.temperature",
        "model.generation.top_p",
        "model.generation.stop_sequences",
        "model.generation.provider_parameters",
        "context.max_tool_result_tokens",
        "context.overflow",
        "stopping.max_total_tokens",
        "stopping.max_tool_calls",
        "instructions.task_template",
        "tools.definitions.0.contract",
    ],
)
def test_behavior_is_never_defaulted(missing: str) -> None:
    """Even a ``None`` is a stated choice: leaving the field out is refused."""
    record = _record()
    *parents, last = missing.split(".")
    node: Any = record
    for key in parents:
        node = node[int(key)] if isinstance(node, list) else node[key]
    del node[last]
    with pytest.raises(ValidationError, match="Field required"):
        AgentHarnessSpec.model_validate(record)


# ---- authority ---------------------------------------------------------------------------


def test_enabling_a_tool_grants_nothing() -> None:
    harness = _with("tools.enabled", ["kb.search", "tickets.close"])
    require_tools_authorized(harness, {"kb.search", "tickets.close", "web.fetch"})
    with pytest.raises(UnauthorizedToolsError, match="tickets.close") as refused:
        require_tools_authorized(harness, {"kb.search"})
    assert refused.value.tools == ("tickets.close",)


def test_no_section_of_a_harness_can_carry_a_permission() -> None:
    def fields(model: Any) -> set[str]:
        names = set(model.model_fields)
        for info in model.model_fields.values():
            annotation = getattr(info.annotation, "model_fields", None)
            if annotation is not None:
                names |= fields(info.annotation)
        return names

    every = fields(AgentHarnessSpec)
    assert not {
        name
        for name in every
        if any(word in name for word in ("permission", "grant", "authoriz", "capabilit", "secret"))
    }


# ---- separate domains -------------------------------------------------------------------


def test_harness_and_training_identities_are_separate_domains() -> None:
    training = _plain_candidate()
    assert harness_identity_v1(_harness())["type"] == "agent-harness"
    assert (
        candidate_identity_v1(training)["type"]
        == candidate_identity_v2(training)["type"]
        == ("candidate")
    )
    assert BASE not in training.candidate_fingerprints_for_lookup()


def test_the_harness_contract_imports_no_provider_runtime_or_trainer() -> None:
    probe = (
        "import sys, xaytune.core.domain.harness, xaytune.core.domain.candidate_envelope\n"
        "heavy = ('torch', 'transformers', 'ray', 'torchft', 'anthropic', 'openai', 'optuna')\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in heavy))\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# ---- v1, frozen ---------------------------------------------------------------------------


def test_the_v1_projection_is_frozen() -> None:
    """The exact projection v1 hashes. A change here is a v2, never an edit."""
    search = content_digest(InlineContent(text="Search the knowledge base."))
    assert harness_identity_v1(_harness()) == {
        "type": "agent-harness",
        "version": 1,
        "model": {
            "provider": "anthropic",
            "model": "claude-sonnet-5-5",
            "revision": "20260901",
            "artifact_digest": None,
            "generation": {
                "temperature": 0.2,
                "top_p": 1.0,
                "max_output_tokens": 2048,
                "stop_sequences": [],
                "provider_parameters": {"thinking_budget": 1024},
            },
        },
        "instructions": {
            "system": {"sha256": content_digest(InlineContent(text=SYSTEM))},
            "task_template": {
                "syntax": "python-format",
                "content": {
                    "sha256": content_digest(
                        InlineContent(text="Ticket: {ticket}\nCustomer: {customer}")
                    )
                },
                "variables": ["customer", "ticket"],
            },
        },
        "tools": {
            "enabled": [
                {
                    "id": "kb.search",
                    "version": "1.2.0",
                    "description": {"sha256": search},
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    "contract": None,
                }
            ]
        },
        "context": {
            "max_context_tokens": 32000,
            "selection": {"kind": "recent-turns", "turns": 12},
            "overflow": "truncate-oldest",
            "max_tool_result_tokens": 4000,
        },
        "stopping": {
            "max_turns": 20,
            "max_total_tokens": 100000,
            "max_tool_calls": 30,
            "terminate_on": ["final-answer"],
        },
    }


def test_the_v1_fingerprint_is_pinned() -> None:
    """Brittle on purpose, like the training literal: it moves only with a new version."""
    assert harness_fingerprint(_harness()) == PINNED_V1


def test_inline_content_digest_is_the_sha256_of_its_utf8_bytes() -> None:
    import hashlib

    expected = "sha256:" + hashlib.sha256("é".encode()).hexdigest()
    assert content_digest(InlineContent(text="é")) == expected
    ref = ContentRef(uri="s3://b/x", digest=DIGEST, media_type="text/plain")
    assert content_digest(ref) == DIGEST


PINNED_V1 = "sha256:894550958202600a6f56473010b0be52e4b68cb64e56ee7b4558fd67601eafa7"


# ---- content locators (PR #74 review, S2) ---------------------------------------------------

SECRET = "s3cr3t-T0KEN-value"


@pytest.mark.parametrize(
    ("uri", "message"),
    [
        (f"https://storage.example.com/prompt?X-Amz-Signature={SECRET}", "no query or fragment"),
        (f"https://storage.example.com/prompt?sig={SECRET}&se=2026", "no query or fragment"),
        (f"https://storage.example.com/prompt#{SECRET}", "no query or fragment"),
        ("https://storage.example.com/prompt?", "no query or fragment"),
        (f"s3://bucket/key?versionId={SECRET}", "no query or fragment"),
        (f"https://user:{SECRET}@storage.example.com/p", "credentials"),
        (f"https://{SECRET}@storage.example.com/p", "credentials"),
        (f"data:text/plain,{SECRET}", "portable schemes"),
        ("javascript:alert(1)", "portable schemes"),
        (f"http://storage.example.com/{SECRET}", "portable schemes"),
        (f"ftp://storage.example.com/{SECRET}", "portable schemes"),
        (f"file:///tmp/{SECRET}", "portable schemes"),
        (f"/home/me/{SECRET}.txt", "portable schemes"),
        (f"prompts/{SECRET}.txt", "portable schemes"),
        (f"C:/prompts/{SECRET}.txt", "portable schemes"),
        (f"s3:///{SECRET}", "host, bucket or namespace"),
        (f"https:///{SECRET}", "host, bucket or namespace"),
        ("s3://bucket", "names an object"),
        ("https://storage.example.com/", "names an object"),
        (f"https://storage.example.com/a {SECRET}", "whitespace or control"),
        (f"https://storage.example.com:99999/{SECRET}", "well-formed"),
    ],
)
def test_a_content_locator_that_is_not_portable_or_may_carry_a_secret_is_refused(
    uri: str, message: str
) -> None:
    with pytest.raises(ValidationError, match=message) as refused:
        ContentRef(uri=uri, digest=DIGEST, media_type="text/plain")
    assert SECRET not in str(refused.value)
    # Nested in a whole harness, the error still does not echo the locator.
    nested = {"kind": "ref", "uri": uri, "digest": DIGEST, "media_type": "text/plain"}
    with pytest.raises(ValidationError) as refused:
        _with("instructions.system", nested)
    assert SECRET not in str(refused.value)
    assert SECRET not in repr(refused.value.errors(include_input=False))


@pytest.mark.parametrize(
    "uri",
    [
        "https://storage.example.com/prompts/system-v3.txt",
        "s3://prompts-bucket/support/system-v3.txt",
        "gs://prompts-bucket/support/system-v3.txt",
        "hf://datasets/acme/prompts/system-v3.txt",
        "https://storage.example.com:8443/prompts/system.txt",
    ],
)
def test_a_portable_content_locator_is_accepted(uri: str) -> None:
    assert ContentRef(uri=uri, digest=DIGEST, media_type="text/plain").uri == uri


def test_the_portable_schemes_are_a_documented_closed_set() -> None:
    from xaytune.core.domain.harness import CONTENT_URI_SCHEMES

    assert CONTENT_URI_SCHEMES == ("gs", "hf", "https", "s3")


def test_a_rejected_digest_is_not_echoed_either() -> None:
    with pytest.raises(ValidationError) as refused:
        ContentRef(uri="s3://b/k", digest=f"sha256:{SECRET}", media_type="text/plain")
    assert SECRET not in str(refused.value)


def test_a_refused_envelope_does_not_echo_the_harness_either() -> None:
    from xaytune.core.domain.candidate_envelope import CandidateEnvelopeError, read_candidate

    record = _record()
    record["instructions"]["system"] = {
        "kind": "ref",
        "uri": f"https://storage.example.com/p?sig={SECRET}",
        "digest": DIGEST,
        "media_type": "text/plain",
    }
    envelope = {
        "envelope_version": "xaytune.candidate/v1alpha1",
        "payload": {"kind": "agent-harness", "harness": record},
    }
    with pytest.raises(CandidateEnvelopeError, match="no query or fragment") as refused:
        read_candidate(envelope)
    assert SECRET not in str(refused.value)
