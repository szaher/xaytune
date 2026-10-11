"""A harness the reference compiler accepts, its script, toolbox and context."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tests.test_core.test_harness import _record
from xaytune.core.agent_execution import AgentExecutionSpec, HarnessTaskInput
from xaytune.core.domain.harness import AgentHarnessSpec
from xaytune.core.immutable import FrozenDict
from xaytune.harness import (
    HarnessCompilationContext,
    HarnessToolbox,
    MappingContentSource,
    ReferenceHarnessCompiler,
    ScriptedHarnessModel,
    resolve_harness_content,
)

SYSTEM = "You are a support agent. Answer from the knowledge base only."
CONTRACT = "kb.search returns at most five passages, most relevant first."
CONTRACT_URI = "https://contracts.example.com/kb/search-v1.txt"
CONTRACT_DIGEST = "sha256:" + hashlib.sha256(CONTRACT.encode("utf-8")).hexdigest()

SCRIPT: tuple[dict[str, Any], ...] = (
    {
        "text": "Let me look that up.",
        "tool_calls": [{"call_id": "c1", "tool": "kb.search", "arguments": {"query": "refund"}}],
    },
    {"text": "Refunds are issued within five days.", "final": True},
)


def harness_record(script: Iterable[Mapping[str, Any]] = SCRIPT, **sections: Any) -> dict[str, Any]:
    """The H02 base harness, bound to the scripted model and with a contract by reference."""
    record = _record()
    record["model"].update(
        provider="xaytune",
        model="scripted",
        revision=None,
        artifact_digest=ScriptedHarnessModel.digest_of(script),
    )
    record["model"]["generation"]["provider_parameters"] = {}
    record["instructions"]["system"] = {"kind": "inline", "text": SYSTEM}
    record["tools"]["definitions"][0]["contract"] = {
        "kind": "ref",
        "uri": CONTRACT_URI,
        "digest": CONTRACT_DIGEST,
        "media_type": "text/plain",
    }
    record.update(sections)
    return record


def harness(script: Iterable[Mapping[str, Any]] = SCRIPT, **sections: Any) -> AgentHarnessSpec:
    return AgentHarnessSpec.model_validate(harness_record(script, **sections))


def source() -> MappingContentSource:
    return MappingContentSource({CONTRACT_URI: CONTRACT.encode("utf-8")})


def context(
    spec: AgentHarnessSpec, output: Path | str, **overrides: Any
) -> HarnessCompilationContext:
    fields: dict[str, Any] = {
        "task": HarnessTaskInput(
            task_id="support/refund-001",
            variables=FrozenDict({"customer": "Ann Lee", "ticket": "Where is my refund?"}),
        ),
        "seed": 7,
        "replicate": 0,
        "authorized_tools": ("kb.search", "tickets.close"),
        "content": resolve_harness_content(spec, source()),
        "output_uri": str(output),
    }
    fields.update(overrides)
    return HarnessCompilationContext(**fields)


def compile_spec(
    output: Path | str, script: Iterable[Mapping[str, Any]] = SCRIPT, **sections: Any
) -> AgentExecutionSpec:
    spec = harness(script, **sections)
    return ReferenceHarnessCompiler().compile(spec, context(spec, output))


class RecordingTools:
    """A toolbox whose tools record each call. Deterministic."""

    def __init__(
        self, search_result: str = "Refund policy: refunds are issued in five days."
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.search_result = search_result

    def search(self, arguments: Mapping[str, Any]) -> str:
        self.calls.append(("kb.search", dict(arguments)))
        return self.search_result

    def close(self, arguments: Mapping[str, Any]) -> str:
        self.calls.append(("tickets.close", dict(arguments)))
        return "closed"

    def toolbox(self) -> HarnessToolbox:
        return HarnessToolbox(
            {("kb.search", "1.2.0"): self.search, ("tickets.close", "2.0.0"): self.close}
        )
