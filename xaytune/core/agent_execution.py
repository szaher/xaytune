"""How to run a harness candidate, and what one run reports (H03, ADR-019).

```text
AgentHarnessSpec + ResolvedHarnessContent + task, seed, replicate, authorized tools
      ↓ HarnessCompiler (xaytune.harness)
AgentExecutionSpec        xaytune.agent-execution/v1alpha1
      ↓ a harness runner
HarnessRunResult          xaytune.agent-run-result/v1alpha1
```

:class:`AgentExecutionSpec` is the sibling of training's and evaluation's
specs: the same transport (entrypoint, config, outputs, resources, telemetry
contract), with a harness where they have a candidate or a subject. It is
**not** a member of :data:`~xaytune.core.execution.ExecutionSpec`, and no
:class:`~xaytune.core.execution.ResolvedExecutionPlan` carries one: a plan
needs an operation target, and a harness target kind needs the persistence
migration ADR-019 names. Nothing here touches what a recorded training or
evaluation plan hashes.

Three identities, kept apart for the reasons ADR-013 keeps them apart::

    harness_fingerprint          is this the same harness proposition?   (scientific, H02)
    agent_execution_fingerprint  is this the same way of running it?     (operational)
    agent_run_request_digest     is this literally the same request?     (idempotency)

Like every spec, this one leaves the process that built it: it carries the
verified text of every content reference it reaches, so a runner never fetches
anything, and its validators re-derive every derived field, so a spec edited
after compilation fails to parse instead of running something else.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any, Final, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from xaytune.core.domain.harness import (
    AgentHarnessSpec,
    ContentRef,
    HarnessContent,
    InlineContent,
)
from xaytune.core.execution import (
    CompilerIdentity,
    TelemetryContract,
    _WorkloadExecutionSpec,
)
from xaytune.core.fingerprint import fingerprint
from xaytune.core.immutable import FrozenDict, FrozenDomainModel

__all__ = [
    "AGENT_EXECUTION_API",
    "AGENT_RUN_RESULT_SCHEMA",
    "AGENT_TELEMETRY_V1ALPHA1",
    "COMPLETIONS",
    "AgentExecutionSpec",
    "HarnessRunResult",
    "HarnessTaskInput",
    "ResolvedContent",
    "ResolvedHarnessContent",
    "RunnerIdentity",
    "TerminationReason",
    "agent_execution_fingerprint",
    "agent_execution_identity_v1",
    "agent_run_request_digest",
    "reachable_content",
]

AGENT_EXECUTION_API: Final = "xaytune.agent-execution/v1alpha1"
AGENT_TELEMETRY_V1ALPHA1: Final = "xaytune.agent-telemetry/v1alpha1"
"""One terminal :class:`HarnessRunResult`, published at the spec's declared
output; no streamed events. A turn-by-turn trajectory is H05's artifact."""
AGENT_RUN_RESULT_SCHEMA: Final = "xaytune.agent-run-result/v1alpha1"

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")


class _Quiet(FrozenDomainModel):
    """Frozen, closed, and silent about refused input, like the harness it carries."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


# ---- content ----------------------------------------------------------------------------


def reachable_content(harness: AgentHarnessSpec) -> tuple[tuple[str, HarnessContent], ...]:
    """Every piece of content the model can reach, by role, in a fixed order.

    The system prompt, the task template, and the description and contract of
    each **enabled** tool (ADR-018 D6). A disabled tool's content is never
    reachable, so it is never fetched or carried.
    """
    found: list[tuple[str, HarnessContent]] = [("instructions.system", harness.instructions.system)]
    if harness.instructions.task_template is not None:
        found.append(("instructions.task_template", harness.instructions.task_template.content))
    enabled = set(harness.tools.enabled)
    for tool in harness.tools.definitions:
        if tool.id in enabled:
            found.append((f"tools.{tool.id}.description", tool.description))
            if tool.contract is not None:
                found.append((f"tools.{tool.id}.contract", tool.contract))
    return tuple(found)


def _referenced_digests(harness: AgentHarnessSpec) -> tuple[str, ...]:
    return tuple(
        sorted({c.digest for _, c in reachable_content(harness) if isinstance(c, ContentRef)})
    )


class ResolvedContent(_Quiet):
    """The text a content reference pinned, verified against its digest."""

    digest: str
    text: str

    @model_validator(mode="after")
    def _verified(self) -> ResolvedContent:
        if not _DIGEST.match(self.digest):
            raise ValueError("a content digest is 'sha256:' and 64 hex digits")
        actual = "sha256:" + hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if actual != self.digest:
            raise ValueError("resolved content does not have the digest it is recorded under")
        return self


class ResolvedHarnessContent(_Quiet):
    """The verified text of a harness's reachable references, by digest.

    Verified whenever it is built or parsed, not only when it was fetched: the
    digest is what the harness fingerprint hashes, so text that does not have
    it is not the harness that was fingerprinted.
    """

    entries: tuple[ResolvedContent, ...]

    @field_validator("entries")
    @classmethod
    def _by_digest(cls, value: tuple[ResolvedContent, ...]) -> tuple[ResolvedContent, ...]:
        digests = [entry.digest for entry in value]
        if digests != sorted(set(digests)):
            raise ValueError("resolved content is listed by digest, once each, sorted")
        return value

    def text_of(self, content: InlineContent | ContentRef) -> str:
        """The text *content* stands for: inline as written, a reference as resolved.

        Raises:
            KeyError: If a reference was not resolved.
        """
        if isinstance(content, InlineContent):
            return content.text
        for entry in self.entries:
            if entry.digest == content.digest:
                return entry.text
        raise KeyError("a content reference was not resolved")


# ---- the execution context ------------------------------------------------------------------


class HarnessTaskInput(_Quiet):
    """The task one run performs: an opaque id and the template's variables.

    A placeholder for H04's task contract, which will pin the task's inputs by
    digest. Until then the variables are the task, and they are execution
    context, never harness identity.
    """

    task_id: str
    variables: FrozenDict

    @field_validator("task_id")
    @classmethod
    def _named(cls, value: str) -> str:
        if not _TASK_ID.match(value):
            raise ValueError("a task id is a short name of letters, digits and ._:/@+-")
        return value

    @field_validator("variables")
    @classmethod
    def _strings(cls, value: FrozenDict) -> FrozenDict:
        if not all(_VARIABLE.match(key) for key in value):
            raise ValueError("task variable names are identifiers")
        if not all(isinstance(text, str) for text in value.values()):
            raise ValueError("task variable values are strings")
        return value


class AgentExecutionSpec(_WorkloadExecutionSpec):
    """How to run one harness candidate on one task -- the sibling of training's spec.

    Produced by a :class:`~xaytune.harness.HarnessCompiler`, consumed by a
    harness runner. ``effective_tools`` records the authorization the compiler
    checked; it grants nothing to a runner, which checks its own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    api_version: Literal["xaytune.agent-execution/v1alpha1"] = "xaytune.agent-execution/v1alpha1"

    compiler: CompilerIdentity
    harness: AgentHarnessSpec
    harness_fingerprint: str
    content: ResolvedHarnessContent

    task: HarnessTaskInput
    seed: int = Field(strict=True)
    replicate: int = Field(ge=0, strict=True)
    effective_tools: tuple[str, ...]

    telemetry: TelemetryContract = Field(
        default_factory=lambda: TelemetryContract(protocol_version=AGENT_TELEMETRY_V1ALPHA1)
    )

    @field_validator("telemetry")
    @classmethod
    def _one_terminal_result(cls, value: TelemetryContract) -> TelemetryContract:
        if value.protocol_version != AGENT_TELEMETRY_V1ALPHA1:
            raise ValueError(
                f"a harness run reports under {AGENT_TELEMETRY_V1ALPHA1}; the training and "
                f"evaluation protocols have no payload for it"
            )
        return value

    @model_validator(mode="after")
    def _derived_fields_agree(self) -> AgentExecutionSpec:
        harness = self.harness
        if self.harness_fingerprint != harness.harness_fingerprint():
            raise ValueError("harness_fingerprint is not the fingerprint of the harness carried")
        resolved = tuple(entry.digest for entry in self.content.entries)
        if resolved != _referenced_digests(harness):
            raise ValueError(
                "the resolved content must be exactly the content references the harness "
                "reaches: no reference unresolved, nothing unreachable carried"
            )
        if self.effective_tools != harness.tools.enabled:
            raise ValueError("effective_tools must be exactly the tools the harness enables")
        template = harness.instructions.task_template
        declared = template.variables if template is not None else ()
        if tuple(sorted(self.task.variables)) != declared:
            raise ValueError("the task's variables must be exactly the template's variables")
        return self

    @property
    def producer(self) -> CompilerIdentity:
        return self.compiler

    def text_of(self, content: InlineContent | ContentRef) -> str:
        """The verified text of *content*, which this spec reaches."""
        return self.content.text_of(content)


# ---- identities -----------------------------------------------------------------------------


def agent_execution_identity_v1(spec: AgentExecutionSpec) -> Mapping[str, Any]:
    """The frozen v1 projection of how a harness is run. Frozen, like every identity.

    Includes the harness only as its fingerprint, and the content only by
    digest. Excludes ``metadata``, ``observability`` and the harness's own
    metadata: annotations do not change what runs.
    """
    return {
        "type": "agent-execution",
        "version": 1,
        "harness_fingerprint": spec.harness_fingerprint,
        "compiler": {"name": spec.compiler.name, "version": spec.compiler.version},
        "entrypoint": spec.entrypoint,
        "arguments": list(spec.arguments),
        "config": spec.config,
        "environment": spec.environment,
        "secrets": list(spec.secrets),
        "dependencies": spec.dependencies,
        "container": spec.container,
        "inputs": list(spec.inputs),
        "outputs": list(spec.outputs),
        "resources": spec.resources,
        "telemetry": spec.telemetry,
        "required_capabilities": spec.required_capabilities,
        "task": {"task_id": spec.task.task_id, "variables": spec.task.variables},
        "seed": spec.seed,
        "replicate": spec.replicate,
        "effective_tools": list(spec.effective_tools),
        "content": [entry.digest for entry in spec.content.entries],
    }


def agent_execution_fingerprint(spec: AgentExecutionSpec) -> str:
    """*Is this the same way of running the same harness?* Operational, never scientific."""
    return fingerprint(agent_execution_identity_v1(spec))


def agent_run_request_digest(spec: AgentExecutionSpec) -> str:
    """The idempotency key of a run: the whole spec, literally (ADR-013's ``request_digest``)."""
    return fingerprint({"type": "agent-run-request", "version": 1, "spec": spec})


# ---- what a run reports -----------------------------------------------------------------------


class RunnerIdentity(FrozenDomainModel):
    """Which runner executed a spec, and at what version."""

    name: str
    version: str


COMPLETIONS: Final = frozenset({"final-answer", "no-tool-call"})
"""The terminations that are completions. Every other reason is a stop."""

TerminationReason = Literal[
    "final-answer",
    "no-tool-call",
    "max-turns",
    "token-budget",
    "tool-call-budget",
    "context-overflow",
    "model-error",
    "cancelled",
]
"""Why a run ended. Only the first two are completions, and only when declared
in the harness's ``terminate_on``; every other reason is a stop."""


class HarnessRunResult(FrozenDomainModel):
    """What one run did, attributable to exactly what ran. Not a score (H06).

    ``prompt_tokens`` sums what each model call was shown, so a long context is
    counted once per call that saw it. Tokens are counted by ``token_counter``,
    which names its rule, not a provider tokenizer.

    Internally consistent whenever it is built or parsed, because a result is
    also read back from disk and replayed: a record claiming a cancelled run
    completed, or tokens that do not add up, is refused rather than believed.
    Whether it belongs to a particular spec is the replaying runner's check.
    """

    schema_version: Literal["xaytune.agent-run-result/v1alpha1"] = (
        "xaytune.agent-run-result/v1alpha1"
    )

    request_digest: str = Field(pattern=_DIGEST.pattern)
    execution_fingerprint: str = Field(pattern=_DIGEST.pattern)
    harness_fingerprint: str = Field(pattern=_DIGEST.pattern)
    compiler: CompilerIdentity
    runner: RunnerIdentity
    model_digest: str = Field(pattern=_DIGEST.pattern)

    task_id: str
    seed: int = Field(strict=True)
    replicate: int = Field(ge=0, strict=True)

    termination: TerminationReason
    completed: bool
    answer: str | None
    error: str | None

    turns: int = Field(ge=0, strict=True)
    tool_calls: int = Field(ge=0, strict=True)
    refused_tool_calls: int = Field(ge=0, strict=True)
    failed_tool_calls: int = Field(ge=0, strict=True)
    context_truncations: int = Field(ge=0, strict=True)

    prompt_tokens: int = Field(ge=0, strict=True)
    output_tokens: int = Field(ge=0, strict=True)
    total_tokens: int = Field(ge=0, strict=True)
    token_counter: str = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> HarnessRunResult:
        problems: list[str] = []
        if self.completed != (self.termination in COMPLETIONS):
            problems.append("completed must be true exactly for a completing termination")
        if (self.answer is not None) != self.completed:
            problems.append("only a completed run has an answer, and a completed run has one")
        if (self.error is not None) != (self.termination == "model-error"):
            problems.append("only a model-error run records an error, and it always does")
        if self.total_tokens != self.prompt_tokens + self.output_tokens:
            problems.append("total_tokens must be prompt_tokens + output_tokens")
        if self.refused_tool_calls + self.failed_tool_calls > self.tool_calls:
            problems.append("refused and failed tool calls are counted among tool_calls")
        if self.tool_calls and not self.turns:
            problems.append("a tool call needs a model turn that asked for it")
        if not self.turns and (self.prompt_tokens or self.output_tokens):
            problems.append("tokens are counted only for model calls that answered")
        if self.termination in COMPLETIONS and not self.turns:
            problems.append("a run completes only on a model turn")
        if problems:
            raise ValueError("inconsistent harness run result: " + "; ".join(problems))
        return self
