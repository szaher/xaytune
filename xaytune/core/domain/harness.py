"""The agent harness: how a fixed model is operated, as a scientific proposition (H02, ADR-018).

A training candidate changes weights; a harness candidate changes the system
around the model's calls -- what it is told, which tools it sees and how they
are described, what context it keeps, when it stops. Both are hypotheses, and
both need an identity that answers *has this been explored?* without being
fooled by serialization or bookkeeping.

```text
AgentHarnessSpec (xaytune.agent-harness/v1alpha1)
├── model:        HarnessModelBinding     provider, model, pinned revision/digest, generation
├── instructions: HarnessInstructions     system prompt, task template
├── tools:        HarnessTools            definitions, enabled ids
├── context:      ContextPolicy           budget, selection, overflow, per-result cap
└── stopping:     StoppingPolicy          turns, tokens, tool calls, termination
```

**A domain contract, not an executable one.** Nothing here runs a model, calls
a tool, renders a template or scores anything; nothing can submit a harness yet
(H03). The spec is data, frozen all the way down, so it can be persisted,
fingerprinted and read back by a process that did not create it.

**Identity is an explicit projection** -- :func:`harness_identity_v1`, never a
dump of the model -- under its own domain (``type="agent-harness"``), so it can
never collide with a training candidate's (``type="candidate"``). What it
covers, and why, is on the function.

**What v1 deliberately does not have**: persistent memory, middleware,
multi-agent delegation, model routing or fallbacks, a behavior-environment
contract. Unknown fields are refused rather than ignored, so a spec asking for
any of them fails at validation instead of being run without it. Each arrives
as a new schema version with its own projection.

**A harness grants nothing.** Its enabled tools are what it asks to use; which
tools it may use is decided outside it, and :func:`require_tools_authorized`
only compares the two.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection, Mapping
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, field_validator, model_validator

from xaytune.core.fingerprint import fingerprint
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw

__all__ = [
    "AGENT_HARNESS_SCHEMA",
    "AgentHarnessSpec",
    "ContentRef",
    "ContextPolicy",
    "FullHistory",
    "GenerationSettings",
    "HarnessContent",
    "HarnessInstructions",
    "HarnessModelBinding",
    "HarnessTools",
    "InlineContent",
    "PromptTemplate",
    "RecentTurns",
    "StoppingPolicy",
    "ToolDefinition",
    "UnauthorizedToolsError",
    "content_digest",
    "harness_fingerprint",
    "harness_identity_v1",
    "require_tools_authorized",
]

AGENT_HARNESS_SCHEMA = "xaytune.agent-harness/v1alpha1"

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")
_TOOL_ID = re.compile(r"^[a-z][a-z0-9_-]*(\.[a-z][a-z0-9_-]*)*$")
_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

_MUTABLE_ALIASES = frozenset(
    {"latest", "main", "master", "head", "stable", "default", "current", "newest", "trunk"}
)
"""Revision names that move. Presenting one as a pin would claim a reproducibility
the record cannot back."""

CONTENT_URI_SCHEMES: tuple[str, ...] = ("gs", "hf", "https", "s3")
"""Where a ``ContentRef`` may point: portable content stores, each naming a host or bucket.

``https`` (a host), ``s3`` and ``gs`` (a bucket), ``hf`` (a Hugging Face
namespace). Nothing local (``file:``, paths), nothing that embeds content or
code (``data:``, ``javascript:``), nothing unencrypted (``http:``). Widening the
set is a deliberate act with its own review."""

_SECRET_KEYS = re.compile(r"(api[_-]?key|token|secret|password|authorization|credential)", re.I)

_MAX_INLINE_CHARS = 200_000
"""Inline content is for prompts and descriptions; anything larger is an artifact."""

Name = Annotated[str, Field(pattern=_NAME.pattern)]


class _HarnessModel(FrozenDomainModel):
    """Frozen and closed like every domain model, and silent about refused input.

    A harness holds locators and parameters a caller may have pasted a
    credential into; a refusal says what was wrong, never what was given.
    Pydantic hides input per model, so every model a refused value can be
    nested in hides it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class UnauthorizedToolsError(ValueError):
    """A harness enables tools outside the authorized set. Nothing is granted by asking."""

    def __init__(self, tools: tuple[str, ...]) -> None:
        self.tools = tools
        super().__init__(
            f"the harness enables tools that are not authorized: {', '.join(tools)}; a "
            f"candidate cannot authorize its own tools"
        )


# ---- content -------------------------------------------------------------------------


class InlineContent(_HarnessModel):
    """Text carried in the spec itself: a prompt or a tool description."""

    kind: Literal["inline"] = "inline"
    text: str = Field(min_length=1, max_length=_MAX_INLINE_CHARS)


class ContentRef(_HarnessModel):
    """Content held elsewhere, pinned by its digest: the digest is the identity.

    ``uri`` says where the bytes can be fetched; it is a locator, not the
    content, so it does not enter identity -- the same bytes moved are the
    same science. It must still be portable and carry no credentials: a local
    path resolves only on the host that wrote it, and the spec is persisted.
    Whoever fetches the bytes verifies them against ``digest`` (H03).
    """

    kind: Literal["ref"] = "ref"
    uri: str = Field(min_length=1, max_length=2048)
    digest: str
    media_type: str = Field(min_length=1)

    @field_validator("digest")
    @classmethod
    def _sha256(cls, value: str) -> str:
        if not _DIGEST.match(value):
            raise ValueError("digest must be 'sha256:' and 64 lowercase hex digits")
        return value

    @field_validator("uri")
    @classmethod
    def _portable(cls, value: str) -> str:
        """A portable locator: an allowed scheme, an authority, a path, and nothing secret.

        No message names the value: a URI refused for carrying a credential
        would otherwise put the credential in the error, and errors are logged.
        """
        try:
            parts = urlsplit(value)
            parts.port  # noqa: B018 -- raises on a malformed port
        except ValueError:
            raise ValueError("the content reference is not a well-formed URI") from None
        if parts.scheme not in CONTENT_URI_SCHEMES:
            raise ValueError(
                f"a content reference must use one of the portable schemes "
                f"{', '.join(CONTENT_URI_SCHEMES)}; local paths, file:, data:, javascript: and "
                f"http: are refused"
            )
        if "@" in parts.netloc or parts.username is not None or parts.password is not None:
            raise ValueError("a content reference must not carry credentials")
        if parts.query or parts.fragment or "?" in value or "#" in value:
            raise ValueError(
                "a content reference carries no query or fragment: content is addressed by its "
                "digest, and query strings are where signed URLs keep their credentials"
            )
        if not parts.hostname:
            raise ValueError("a content reference names its host, bucket or namespace")
        if parts.path.strip("/") == "":
            raise ValueError("a content reference names an object, not only its host or bucket")
        if any(ord(char) < 0x21 or ord(char) == 0x7F for char in value):
            raise ValueError("a content reference contains whitespace or control characters")
        return value


HarnessContent = Annotated[InlineContent | ContentRef, Field(discriminator="kind")]


def content_digest(content: InlineContent | ContentRef) -> str:
    """The SHA-256 of *content*'s bytes: inline text as UTF-8, a reference as pinned."""
    if isinstance(content, ContentRef):
        return content.digest
    return "sha256:" + hashlib.sha256(content.text.encode("utf-8")).hexdigest()


# ---- model ----------------------------------------------------------------------------


class GenerationSettings(_HarnessModel):
    """How the model samples, stated in full: none of these has a default.

    A provider's default is a moving, unrecorded choice, so leaving any of them
    unset would make two harnesses with the same spec behave differently on
    different days. ``provider_parameters`` holds provider-specific knobs
    (reasoning effort, say) as named scalars; they enter identity. The seed is
    not here: it belongs to a run (ADR-011).
    """

    temperature: float = Field(ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(gt=0, le=1, allow_inf_nan=False)
    max_output_tokens: int = Field(gt=0, strict=True)
    stop_sequences: tuple[Annotated[str, Field(min_length=1)], ...]
    provider_parameters: FrozenDict

    @field_validator("provider_parameters")
    @classmethod
    def _named_scalars(cls, value: FrozenDict) -> FrozenDict:
        for key, item in value.items():
            if not _VARIABLE.match(key):
                raise ValueError("provider parameter names are identifiers")
            if _SECRET_KEYS.search(key):
                raise ValueError(
                    f"provider parameter {key!r} looks like a credential; credentials are "
                    f"referenced by the runner, never stored in a scientific spec"
                )
            if not isinstance(item, (str, int, float, bool)):
                raise ValueError(
                    f"provider parameter {key!r} must be a string, number or boolean, not "
                    f"{type(item).__name__}"
                )
        return value


class HarnessModelBinding(_HarnessModel):
    """The model the harness operates, pinned, and how it generates (ADR-018 D4).

    Part of harness identity even when every candidate uses the same model:
    the results depend on it. At least one of ``revision`` and
    ``artifact_digest`` pins it; a moving alias is refused rather than
    recorded as though it were a pin.
    """

    provider: Name
    model: Name
    revision: Name | None = None
    artifact_digest: str | None = None
    generation: GenerationSettings

    @field_validator("revision")
    @classmethod
    def _immutable(cls, value: str | None) -> str | None:
        if value is not None and (
            value.lower() in _MUTABLE_ALIASES or value.lower().endswith("-latest")
        ):
            raise ValueError(
                f"revision {value!r} is a moving alias, not a pin; name the immutable "
                f"revision it resolves to"
            )
        return value

    @field_validator("artifact_digest")
    @classmethod
    def _sha256(cls, value: str | None) -> str | None:
        if value is not None and not _DIGEST.match(value):
            raise ValueError("artifact_digest must be 'sha256:' and 64 hex digits")
        return value

    @model_validator(mode="after")
    def _pinned(self) -> HarnessModelBinding:
        if self.revision is None and self.artifact_digest is None:
            raise ValueError(
                "a model binding needs a pinned revision or an artifact digest; without one "
                "the record cannot say which model ran"
            )
        return self


# ---- instructions ---------------------------------------------------------------------


class PromptTemplate(_HarnessModel):
    """The per-task prompt, and the variables a task supplies to it.

    ``syntax`` is identity: the same text under two template languages renders
    differently. Nothing here renders it.
    """

    syntax: Literal["literal", "python-format", "jinja2"]
    content: HarnessContent
    variables: tuple[str, ...]

    @field_validator("variables")
    @classmethod
    def _declared(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if list(value) != sorted(set(value)) or not all(_VARIABLE.match(v) for v in value):
            raise ValueError("template variables are identifiers, listed once each, sorted")
        return value

    @model_validator(mode="after")
    def _literal_has_none(self) -> PromptTemplate:
        if self.syntax == "literal" and self.variables:
            raise ValueError("a literal template takes no variables")
        return self


class HarnessInstructions(_HarnessModel):
    """What the agent is told: a system prompt, and the task template if there is one.

    Instructions to the agent, never authority over Xaytune: a prompt that says
    "you may use any tool" grants nothing.
    """

    system: HarnessContent
    task_template: PromptTemplate | None


# ---- tools ----------------------------------------------------------------------------


class ToolDefinition(_HarnessModel):
    """One tool as the model sees it, versioned.

    ``description`` and ``input_schema`` are what the model reads, so both are
    identity. ``contract`` pins the implementation or interface the tool binds
    to, where its behavior matters to the hypothesis.
    """

    id: str
    version: Name
    description: HarnessContent
    input_schema: FrozenDict
    contract: ContentRef | None

    @field_validator("id")
    @classmethod
    def _namespaced(cls, value: str) -> str:
        if not _TOOL_ID.match(value):
            raise ValueError("tool ids are lowercase dotted names")
        return value

    @field_validator("version")
    @classmethod
    def _pinned(cls, value: str) -> str:
        if value.lower() in _MUTABLE_ALIASES:
            raise ValueError(f"tool version {value!r} is a moving alias, not a version")
        return value

    @field_validator("input_schema")
    @classmethod
    def _object_schema(cls, value: FrozenDict) -> FrozenDict:
        if value.get("type") != "object":
            raise ValueError("a tool's input schema is a JSON Schema object (type: object)")
        return value


class HarnessTools(_HarnessModel):
    """The tools the harness defines, and which of them it enables.

    Enabling is a request, not a grant (ADR-018 D9). Definitions are listed by
    id, once each, so the same tool set has one form.
    """

    definitions: tuple[ToolDefinition, ...]
    enabled: tuple[str, ...]

    @model_validator(mode="after")
    def _consistent(self) -> HarnessTools:
        ids = [tool.id for tool in self.definitions]
        if ids != sorted(set(ids)):
            raise ValueError("tool definitions are listed by id, once each, sorted")
        if list(self.enabled) != sorted(set(self.enabled)):
            raise ValueError("enabled tools are listed once each, sorted")
        undefined = sorted(set(self.enabled) - set(ids))
        if undefined:
            raise ValueError(f"enabled tools have no definition: {', '.join(undefined)}")
        return self


# ---- context and stopping -------------------------------------------------------------


class FullHistory(_HarnessModel):
    """Keep every turn, within the budget."""

    kind: Literal["full-history"] = "full-history"


class RecentTurns(_HarnessModel):
    """Keep the system prompt and the last ``turns`` turns."""

    kind: Literal["recent-turns"] = "recent-turns"
    turns: int = Field(gt=0, strict=True)


ContextSelection = Annotated[FullHistory | RecentTurns, Field(discriminator="kind")]


class ContextPolicy(_HarnessModel):
    """What the model sees of the conversation so far.

    ``overflow`` says what happens when selection still exceeds the budget:
    fail the task, or drop the oldest turns. Summarizing compaction calls a
    model and is not in v1. ``max_tool_result_tokens`` is ``None`` only as the
    stated choice of no per-result cap.
    """

    max_context_tokens: int = Field(gt=0, strict=True)
    selection: ContextSelection
    overflow: Literal["fail", "truncate-oldest"]
    max_tool_result_tokens: int | None = Field(gt=0, strict=True)


class StoppingPolicy(_HarnessModel):
    """When a task ends: scientific stopping rules, not budget enforcement.

    ``None`` limits are the stated choice of no limit. Experiment-wide spend
    stays the budget ledger's; the stricter authorized limit governs (ch. 23 §11).
    """

    max_turns: int = Field(gt=0, strict=True)
    max_total_tokens: int | None = Field(gt=0, strict=True)
    max_tool_calls: int | None = Field(ge=0, strict=True)
    terminate_on: tuple[Literal["final-answer", "no-tool-call"], ...] = Field(min_length=1)

    @field_validator("terminate_on")
    @classmethod
    def _canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if list(value) != sorted(set(value)):
            raise ValueError("termination conditions are listed once each, sorted")
        return value  # type: ignore[return-value]


# ---- the spec -------------------------------------------------------------------------


class AgentHarnessSpec(_HarnessModel):
    """One whole harness proposition: a pinned model, operated one way (ADR-018).

    Frozen: a changed harness is a new candidate, not an edited one. Every
    section is required; ``metadata`` is the only field outside identity.
    """

    schema_version: Literal["xaytune.agent-harness/v1alpha1"]
    model: HarnessModelBinding
    instructions: HarnessInstructions
    tools: HarnessTools
    context: ContextPolicy
    stopping: StoppingPolicy
    metadata: FrozenDict = Field(default_factory=FrozenDict)

    def harness_fingerprint(self) -> str:
        """This harness's scientific identity under v1: see :func:`harness_identity_v1`."""
        return harness_fingerprint(self)


# ---- identity -------------------------------------------------------------------------


def _content_identity_v1(content: InlineContent | ContentRef) -> Mapping[str, Any]:
    return {"sha256": content_digest(content)}


def _model_identity_v1(binding: HarnessModelBinding) -> Mapping[str, Any]:
    generation = binding.generation
    return {
        "provider": binding.provider,
        "model": binding.model,
        "revision": binding.revision,
        "artifact_digest": binding.artifact_digest,
        "generation": {
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "max_output_tokens": generation.max_output_tokens,
            "stop_sequences": list(generation.stop_sequences),
            "provider_parameters": thaw(generation.provider_parameters),
        },
    }


def _instructions_identity_v1(instructions: HarnessInstructions) -> Mapping[str, Any]:
    template = instructions.task_template
    return {
        "system": _content_identity_v1(instructions.system),
        "task_template": None
        if template is None
        else {
            "syntax": template.syntax,
            "content": _content_identity_v1(template.content),
            "variables": list(template.variables),
        },
    }


def _tools_identity_v1(tools: HarnessTools) -> Mapping[str, Any]:
    enabled = set(tools.enabled)
    return {
        "enabled": [
            {
                "id": tool.id,
                "version": tool.version,
                "description": _content_identity_v1(tool.description),
                "input_schema": thaw(tool.input_schema),
                "contract": None if tool.contract is None else _content_identity_v1(tool.contract),
            }
            for tool in tools.definitions
            if tool.id in enabled
        ]
    }


def _context_identity_v1(context: ContextPolicy) -> Mapping[str, Any]:
    selection = context.selection
    return {
        "max_context_tokens": context.max_context_tokens,
        "selection": {
            "kind": selection.kind,
            "turns": selection.turns if isinstance(selection, RecentTurns) else None,
        },
        "overflow": context.overflow,
        "max_tool_result_tokens": context.max_tool_result_tokens,
    }


def _stopping_identity_v1(stopping: StoppingPolicy) -> Mapping[str, Any]:
    return {
        "max_turns": stopping.max_turns,
        "max_total_tokens": stopping.max_total_tokens,
        "max_tool_calls": stopping.max_tool_calls,
        "terminate_on": list(stopping.terminate_on),
    }


def harness_identity_v1(harness: AgentHarnessSpec) -> Mapping[str, Any]:
    """The explicit, versioned projection that defines a harness's scientific identity.

    **Covered** -- everything in v1 that changes what the agent does: the model
    (provider, name, revision, artifact digest) and every generation setting;
    the system prompt and task template (syntax, content, variables); each
    *enabled* tool's id, version, description, input schema and contract;
    the context policy; the stopping policy.

    **Content by digest.** Prompts, descriptions and contracts enter as the
    SHA-256 of their bytes, so inline text and a reference to the same bytes
    are the same harness, and moving a referenced file changes nothing.

    **Excluded, deliberately** -- ``metadata`` at any level; a reference's
    location and media type; tools defined but not enabled, which the model
    never sees; and everything that is not the hypothesis: seeds and
    replicates (a run's), benchmark membership, held-out inputs and scoring (an
    evaluation's), placement, adapters and attempts (an execution's).

    Domain-separated by ``type``: a harness and a training candidate cannot
    share a fingerprint. Frozen: a change to what v1 covers is a v2, never an
    edit -- the pinned literal in the test suite is the tripwire.
    """
    return {
        "type": "agent-harness",
        "version": 1,
        "model": _model_identity_v1(harness.model),
        "instructions": _instructions_identity_v1(harness.instructions),
        "tools": _tools_identity_v1(harness.tools),
        "context": _context_identity_v1(harness.context),
        "stopping": _stopping_identity_v1(harness.stopping),
    }


def harness_fingerprint(harness: AgentHarnessSpec) -> str:
    """``HarnessFingerprint`` v1: *has this harness proposition been explored?*"""
    return fingerprint(harness_identity_v1(harness))


def require_tools_authorized(harness: AgentHarnessSpec, authorized: Collection[str]) -> None:
    """Refuse *harness* unless every tool it enables is in *authorized*.

    *authorized* is resolved outside the candidate -- by policy, connector
    trust, the runner's sandbox (H03) -- and this only compares. It grants
    nothing, and a harness has no field that could.

    Raises:
        UnauthorizedToolsError: Naming each enabled tool outside the set.
    """
    outside = tuple(sorted(set(harness.tools.enabled) - set(authorized)))
    if outside:
        raise UnauthorizedToolsError(outside)
