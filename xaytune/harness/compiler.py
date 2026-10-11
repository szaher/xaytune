"""The compile half of harness execution (ADR-019 E1).

```text
AgentHarnessSpec + HarnessCompilationContext ──compile──▶ AgentExecutionSpec
```

**A compiler compiles. It never executes.** :class:`HarnessCompiler` is the
sibling of :class:`~xaytune.compilation.TrainerCompiler`: typed capabilities,
``supports()`` with every reason, and a deterministic ``compile()``. It never
calls a model, runs a tool, reads a file or the network, or submits a
workload. Everything compilation depends on arrives in the context,
including content already resolved and verified
(:func:`~xaytune.harness.resolve_harness_content`) and the tools a host has
authorized.

:class:`ReferenceHarnessCompiler` compiles for
:class:`~xaytune.harness.ScriptedHarnessRunner`, and refuses anything that
runner would not honour exactly.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import Field, ValidationError

from xaytune._version import __version__
from xaytune.agent.schema import UnsupportedResponseSchemaError, check_response_schema
from xaytune.core.agent_execution import (
    AGENT_RUN_RESULT_SCHEMA,
    AGENT_TELEMETRY_V1ALPHA1,
    AgentExecutionSpec,
    HarnessTaskInput,
    ResolvedHarnessContent,
    reachable_content,
)
from xaytune.core.capabilities import (
    PLUGIN_API_VERSIONS,
    CapabilityRequirements,
    PluginDescriptor,
    require_supported_plugin,
)
from xaytune.core.domain.harness import (
    AGENT_HARNESS_SCHEMA,
    AgentHarnessSpec,
    ContentRef,
    InlineContent,
    UnauthorizedToolsError,
    require_tools_authorized,
)
from xaytune.core.execution import (
    ArtifactOutput,
    CompilerIdentity,
    PythonModuleEntrypoint,
    TelemetryContract,
)
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.support import SupportResult
from xaytune.harness.templates import RENDERABLE_SYNTAXES, template_refusals

__all__ = [
    "HARNESS_COMPILER_API",
    "RESULT_FILE",
    "SCRIPTED_MODEL_NAME",
    "SCRIPTED_PROVIDER",
    "SCRIPTED_RUNNER",
    "SCRIPTED_RUNNER_ENTRYPOINT",
    "TOKEN_COUNTER",
    "HarnessCompilationContext",
    "HarnessCompiler",
    "HarnessCompilerCapabilities",
    "ReferenceHarnessCompiler",
    "UnsupportedHarnessError",
    "scripted_harness_refusals",
]

HARNESS_COMPILER_API: Final = "xaytune.harness-compiler/v1alpha1"
TOKEN_COUNTER: Final = "xaytune.whitespace-tokens/v1"
"""Tokens are whitespace-separated words. A rule, named, not a provider's tokenizer."""

SCRIPTED_PROVIDER: Final = "xaytune"
SCRIPTED_MODEL_NAME: Final = "scripted"
SCRIPTED_RUNNER: Final = "xaytune.scripted-harness-runner/v1"
"""The runner requirement the reference compiler emits and the scripted runner provides."""
SCRIPTED_RUNNER_ENTRYPOINT: Final = PythonModuleEntrypoint(
    module="xaytune.harness.runner", function="run_scripted_harness"
)
"""In-process: no runtime can submit a harness workload yet (ADR-019)."""
RESULT_FILE: Final = "agent-run.json"

_TOOL_SCHEMA_DIALECT = "xaytune.agent-schema-subset/v1"


class UnsupportedHarnessError(ValueError):
    """``compile()`` was asked for a harness it cannot honour. Carries every reason."""

    def __init__(self, compiler: str, reasons: tuple[str, ...]) -> None:
        self.compiler = compiler
        self.reasons = reasons
        super().__init__(f"{compiler} cannot compile this harness: " + "; ".join(reasons))


class HarnessCompilerCapabilities(FrozenDomainModel):
    """What a harness compiler can emit, in the vocabulary of ``AgentHarnessSpec``.

    Typed rather than a free ``CapabilityDocument`` extension, so a planner or
    reviewer can compare a harness against it field by field.
    """

    api_version: Literal["xaytune.harness-compiler/v1alpha1"] = "xaytune.harness-compiler/v1alpha1"
    harness_schemas: tuple[str, ...]
    model_providers: tuple[str, ...]
    template_syntaxes: tuple[str, ...]
    context_selections: tuple[str, ...]
    overflow_modes: tuple[str, ...]
    terminate_on: tuple[str, ...]
    tool_schema_dialect: str
    provider_parameters: tuple[str, ...]
    runner: str


class HarnessCompilationContext(FrozenDomainModel):
    """Everything compilation may depend on besides the harness. Explicit, so it is deterministic.

    ``authorized_tools`` is resolved by the host -- policy, connector trust,
    the sandbox -- never by the candidate (ADR-018 D9). ``content`` is the
    harness's reachable references, already fetched and verified.
    """

    task: HarnessTaskInput
    seed: int = Field(strict=True)
    replicate: int = Field(ge=0, strict=True)
    authorized_tools: tuple[str, ...]
    content: ResolvedHarnessContent
    output_uri: str
    """An absolute local directory; the run's result is published inside it."""

    metadata: FrozenDict = Field(default_factory=FrozenDict)


@runtime_checkable
class HarnessCompiler(Protocol):
    """Turns a harness candidate into a runtime-neutral execution spec."""

    descriptor: PluginDescriptor

    def capabilities(self) -> HarnessCompilerCapabilities:
        """What this compiler can emit."""
        ...

    def supports(self, harness: AgentHarnessSpec) -> SupportResult:
        """Whether *harness* can be compiled exactly as declared. Every reason, not the first."""
        ...

    def compile(
        self, harness: AgentHarnessSpec, context: HarnessCompilationContext
    ) -> AgentExecutionSpec:
        """How to run *harness* in *context*. Deterministic; runs and submits nothing.

        Raises:
            UnsupportedHarnessError: With every reason, if the harness or the
                context cannot be honoured exactly.
        """
        ...


_CAPABILITIES = HarnessCompilerCapabilities(
    harness_schemas=(AGENT_HARNESS_SCHEMA,),
    model_providers=(SCRIPTED_PROVIDER,),
    template_syntaxes=RENDERABLE_SYNTAXES,
    context_selections=("full-history", "recent-turns"),
    overflow_modes=("fail", "truncate-oldest"),
    terminate_on=("final-answer", "no-tool-call"),
    tool_schema_dialect=_TOOL_SCHEMA_DIALECT,
    provider_parameters=(),
    runner=SCRIPTED_RUNNER,
)


def scripted_harness_refusals(harness: AgentHarnessSpec) -> tuple[str, ...]:
    """Every reason the scripted runner could not run *harness* exactly as declared.

    Shared by the compiler, which refuses before compiling, and the runner,
    which refuses a spec from anywhere else.
    """
    reasons: list[str] = []
    binding = harness.model
    if (binding.provider, binding.model) != (SCRIPTED_PROVIDER, SCRIPTED_MODEL_NAME):
        reasons.append(
            f"model provider and name must be {SCRIPTED_PROVIDER!r}/{SCRIPTED_MODEL_NAME!r}: "
            f"this path runs only the scripted model, and cannot stand in for another"
        )
    if binding.artifact_digest is None:
        reasons.append("the scripted model is pinned by its script's digest (artifact_digest)")
    if binding.generation.provider_parameters:
        reasons.append(
            "provider_parameters must be empty: the scripted model accepts none, and an "
            "unknown parameter is refused rather than ignored"
        )
    template = harness.instructions.task_template
    if template is None:
        reasons.append("a task template is required: the scripted runner starts each task from it")
    elif template.syntax not in RENDERABLE_SYNTAXES:
        reasons.extend(template_refusals(template.syntax, "", template.variables))
    for tool in harness.tools.definitions:
        if tool.id not in harness.tools.enabled:
            continue
        try:
            check_response_schema(tool.input_schema)
        except UnsupportedResponseSchemaError as error:
            reasons.append(
                f"tool {tool.id}: input schema is outside {_TOOL_SCHEMA_DIALECT} "
                f"({len(error.reasons)} problem(s)); an unchecked keyword would be ignored"
            )
    return tuple(reasons)


class ReferenceHarnessCompiler:
    """Compiles a harness for the deterministic :class:`~xaytune.harness.ScriptedHarnessRunner`."""

    descriptor = PluginDescriptor(
        api_version=PLUGIN_API_VERSIONS[0],
        name="reference-harness",
        plugin_version="0.1.0",
        provider="xaytune",
        xaytune_version=__version__,
    )

    def capabilities(self) -> HarnessCompilerCapabilities:
        return _CAPABILITIES

    def supports(self, harness: AgentHarnessSpec) -> SupportResult:
        reasons = scripted_harness_refusals(harness)
        return SupportResult(supported=not reasons, reasons=reasons)

    def compile(
        self, harness: AgentHarnessSpec, context: HarnessCompilationContext
    ) -> AgentExecutionSpec:
        require_supported_plugin(self.descriptor)
        reasons = list(scripted_harness_refusals(harness))
        try:
            require_tools_authorized(harness, context.authorized_tools)
        except UnauthorizedToolsError as error:
            reasons.append(str(error))
        reasons.extend(_content_refusals(harness, context.content))
        template = harness.instructions.task_template
        if template is not None and template.syntax in RENDERABLE_SYNTAXES:
            text = _text_or_none(context.content, template.content)
            if text is not None:
                reasons.extend(template_refusals(template.syntax, text, template.variables))
            if tuple(sorted(context.task.variables)) != template.variables:
                reasons.append("the task must supply exactly the template's variables")
        output_dir = PurePosixPath(context.output_uri)
        if not output_dir.is_absolute() or ".." in output_dir.parts:
            reasons.append("output_uri must be an absolute local directory without '..'")
        if reasons:
            raise UnsupportedHarnessError(self.descriptor.name, tuple(dict.fromkeys(reasons)))

        referenced = {c.digest for _, c in reachable_content(harness) if isinstance(c, ContentRef)}
        content = ResolvedHarnessContent(
            entries=tuple(e for e in context.content.entries if e.digest in referenced)
        )
        try:
            return AgentExecutionSpec(
                compiler=CompilerIdentity(
                    name=self.descriptor.name,
                    version=self.descriptor.plugin_version,
                    descriptor=self.descriptor,
                ),
                harness=harness,
                harness_fingerprint=harness.harness_fingerprint(),
                content=content,
                task=context.task,
                seed=context.seed,
                replicate=context.replicate,
                effective_tools=harness.tools.enabled,
                entrypoint=SCRIPTED_RUNNER_ENTRYPOINT,
                config=FrozenDict(
                    {"compiler_api": HARNESS_COMPILER_API, "token_counter": TOKEN_COUNTER}
                ),
                outputs=(
                    ArtifactOutput(
                        name="result",
                        uri=str(output_dir / RESULT_FILE),
                        kind=AGENT_RUN_RESULT_SCHEMA,
                    ),
                ),
                telemetry=TelemetryContract(protocol_version=AGENT_TELEMETRY_V1ALPHA1),
                required_capabilities=CapabilityRequirements(
                    extensions=FrozenDict({"agent_runner": SCRIPTED_RUNNER})
                ),
            )
        except ValidationError as error:  # pragma: no cover - every input was checked above
            raise UnsupportedHarnessError(
                self.descriptor.name, ("the compiled spec is inconsistent",)
            ) from error


def _text_or_none(content: ResolvedHarnessContent, item: InlineContent | ContentRef) -> str | None:
    try:
        return content.text_of(item)
    except KeyError:
        return None


def _content_refusals(
    harness: AgentHarnessSpec, content: ResolvedHarnessContent
) -> tuple[str, ...]:
    reasons: list[str] = []
    for role, item in reachable_content(harness):
        if isinstance(item, ContentRef) and _text_or_none(content, item) is None:
            reasons.append(f"{role}: the content reference was not resolved before compiling")
    return tuple(reasons)
