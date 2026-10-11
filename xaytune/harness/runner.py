"""A deterministic harness runner: executes an ``AgentExecutionSpec`` against a script (ADR-019 E6).

```text
AgentExecutionSpec + ScriptedHarnessModel + host toolbox + host-authorized tools
      ↓ ScriptedHarnessRunner.run
HarnessRunResult, published once at the spec's declared output
```

The runner exists to make harness execution **semantics** testable exactly:
every behavior field of the harness changes what it does. It renders the
system prompt and task template, shows the model only the enabled tools,
validates tool arguments against their schemas, selects and bounds context,
truncates tool results and output, applies stop sequences, enforces every
budget and stops on the declared conditions.

It is not an agent. Its model is a script, pinned by digest in the harness's
model binding; its tools are pure functions the host supplies by id and
version; it opens no connection and runs no code from the spec. It counts
tokens by a named rule (:data:`~xaytune.harness.compiler.TOKEN_COUNTER`), not a
provider tokenizer. It measures nothing: whether an answer was good is H06's
question.

**Authority comes from the host, and is checked here again** (ADR-019 E5): the
run is refused unless every enabled tool is in ``authorized_tools`` and the
toolbox implements it at the declared version. A call to any other tool is
answered with a refusal and never executed.

**The model is the trusted scripted implementation, nothing that claims its
digest.** ``run`` accepts exactly :class:`ScriptedHarnessModel` -- not a
subclass, not another object reporting the same ``artifact_digest`` -- and
recomputes the digest from the script it holds. Real adapters will need their
own runner and their own binding (H08+).

**Idempotent, restartable, cancellable** (E8), with guarantees stated exactly:

- *One published result.* The result is published with a no-overwrite link,
  so a different request at the same location is a conflict, and a replayed
  result is checked against the current spec before it is returned.
- *One execution per host.* Runs of the same output location on one host are
  serialized by an advisory lock, so a second run waits and then replays the
  first's result instead of calling the model or tools again.
- *Not exactly-once.* A run on another host, or a rerun after a crash before
  publication, executes again and repeats its model and tool calls. That is
  harmless for a deterministic script and pure tools; production adapters
  must not inherit it as an exactly-once guarantee.

Cancellation is checked before every model call and every tool call, and a
cancelled run publishes a terminal result.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field, ValidationError, model_validator

from xaytune.agent.schema import schema_violations
from xaytune.core.agent_execution import (
    AGENT_RUN_RESULT_SCHEMA,
    COMPLETIONS,
    AgentExecutionSpec,
    HarnessRunResult,
    RunnerIdentity,
    TerminationReason,
    agent_execution_fingerprint,
    agent_run_request_digest,
)
from xaytune.core.capabilities import CapabilityRequirements
from xaytune.core.domain.harness import (
    FullHistory,
    GenerationSettings,
    ToolDefinition,
    UnauthorizedToolsError,
    require_tools_authorized,
)
from xaytune.core.execution import DependencySpec, ResourceRequirements
from xaytune.core.fingerprint import fingerprint
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw
from xaytune.core.support import SupportResult
from xaytune.harness.compiler import (
    HARNESS_COMPILER_API,
    SCRIPTED_RUNNER,
    SCRIPTED_RUNNER_ENTRYPOINT,
    TOKEN_COUNTER,
    scripted_harness_refusals,
)
from xaytune.harness.templates import render_template, template_refusals

__all__ = [
    "RUNNER",
    "CancelSignal",
    "HarnessMessage",
    "HarnessModel",
    "HarnessModelReply",
    "HarnessModelRequest",
    "HarnessRunConflictError",
    "HarnessRunRefusedError",
    "HarnessToolCall",
    "HarnessToolView",
    "HarnessToolbox",
    "ScriptExhaustedError",
    "ScriptedHarnessModel",
    "ScriptedHarnessRunner",
    "run_scripted_harness",
]

RUNNER = RunnerIdentity(name="xaytune.scripted-harness-runner", version="1")


# ---- what the model sees and says -----------------------------------------------------------


class HarnessToolCall(FrozenDomainModel):
    """One tool call a model asks for. A request, never an execution."""

    call_id: str = Field(min_length=1, max_length=200)
    tool: str = Field(min_length=1, max_length=200)
    arguments: FrozenDict


class HarnessModelReply(FrozenDomainModel):
    """One model turn: text, tool calls, and whether it is the final answer."""

    text: str = ""
    tool_calls: tuple[HarnessToolCall, ...] = ()
    final: bool = False

    @model_validator(mode="after")
    def _coherent(self) -> HarnessModelReply:
        if self.final and self.tool_calls:
            raise ValueError("a final answer calls no tools")
        ids = [call.call_id for call in self.tool_calls]
        if len(ids) != len(set(ids)):
            raise ValueError("tool call ids are unique within a turn")
        return self


class HarnessMessage(FrozenDomainModel):
    """One message of the conversation the model is shown."""

    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: tuple[HarnessToolCall, ...] = ()
    tool_call_id: str | None = None


class HarnessToolView(FrozenDomainModel):
    """An enabled tool as the model sees it: never a disabled one."""

    id: str
    version: str
    description: str
    contract: str | None
    input_schema: FrozenDict


class HarnessModelRequest(FrozenDomainModel):
    """Everything one model call is given."""

    system: str
    messages: tuple[HarnessMessage, ...]
    tools: tuple[HarnessToolView, ...]
    generation: GenerationSettings


@runtime_checkable
class HarnessModel(Protocol):
    """A model a harness runner calls. ``artifact_digest`` is what the harness binding pins."""

    @property
    def artifact_digest(self) -> str: ...

    def respond(self, request: HarnessModelRequest, *, turn: int) -> HarnessModelReply: ...


class ScriptExhaustedError(RuntimeError):
    """The script has no reply for the turn asked for."""


class ScriptedHarnessModel:
    """Replies with the script's entry for each turn, and remembers every request.

    Stateless across runs: the reply depends only on the turn number, so the
    same script answers a rerun exactly as it answered the first run. Its
    ``artifact_digest`` is the digest of the script, which is how a harness
    pins which "model" it ran.
    """

    def __init__(self, replies: Iterable[HarnessModelReply | Mapping[str, Any]]) -> None:
        self._replies = tuple(HarnessModelReply.model_validate(reply) for reply in replies)
        self._requests: list[HarnessModelRequest] = []

    @staticmethod
    def digest_of(replies: Iterable[HarnessModelReply | Mapping[str, Any]]) -> str:
        """The digest a harness binding names to run *replies*."""
        script = [HarnessModelReply.model_validate(reply) for reply in replies]
        return fingerprint({"type": "scripted-harness-model", "version": 1, "replies": script})

    @property
    def replies(self) -> tuple[HarnessModelReply, ...]:
        """The script, as validated replies."""
        return self._replies

    @property
    def artifact_digest(self) -> str:
        return self.digest_of(self._replies)

    @property
    def requests(self) -> tuple[HarnessModelRequest, ...]:
        """Every request asked so far, across runs, in order."""
        return tuple(self._requests)

    def respond(self, request: HarnessModelRequest, *, turn: int) -> HarnessModelReply:
        self._requests.append(request)
        if turn >= len(self._replies):
            raise ScriptExhaustedError(f"the script has no reply for turn {turn}")
        return self._replies[turn]


ToolFunction = Callable[[Mapping[str, Any]], str]
"""A host-supplied tool: arguments in, text out. Pure and deterministic in tests."""


class HarnessToolbox:
    """The tool implementations a host supplies, by id and version. Nothing else can run."""

    def __init__(self, tools: Mapping[tuple[str, str], ToolFunction]) -> None:
        self._tools = dict(tools)

    def get(self, tool_id: str, version: str) -> ToolFunction | None:
        return self._tools.get((tool_id, version))


class CancelSignal(Protocol):
    """Anything that says whether the run should stop; ``threading.Event`` is one."""

    def is_set(self) -> bool: ...


# ---- errors -----------------------------------------------------------------------------------


class HarnessRunRefusedError(ValueError):
    """The run was refused before anything executed. Carries every reason."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("the harness run is refused: " + "; ".join(reasons))


class HarnessRunConflictError(RuntimeError):
    """A different request's result is already published where this one would go."""


# ---- the runner -------------------------------------------------------------------------------


class ScriptedHarnessRunner:
    """Executes an :class:`AgentExecutionSpec` in-process, deterministically."""

    identity = RUNNER

    def supports(self, spec: AgentExecutionSpec) -> SupportResult:
        """Whether this runner implements everything *spec* requires. Every reason."""
        reasons = _spec_refusals(spec)
        return SupportResult(supported=not reasons, reasons=reasons)

    def run(
        self,
        spec: AgentExecutionSpec,
        *,
        model: ScriptedHarnessModel,
        toolbox: HarnessToolbox,
        authorized_tools: Collection[str],
        cancel: CancelSignal | None = None,
    ) -> HarnessRunResult:
        """Run *spec* once, or return the result already published for it.

        Raises:
            HarnessRunRefusedError: Before anything executes, with every
                reason: an unsupported requirement, an unauthorized or
                unimplemented tool, a model that is not exactly
                :class:`ScriptedHarnessModel`, or a script that is not the
                one pinned.
            HarnessRunConflictError: If the output holds a different
                request's result, or a result that does not match this spec.
        """
        reasons = list(_spec_refusals(spec))
        try:
            require_tools_authorized(spec.harness, authorized_tools)
        except UnauthorizedToolsError as error:
            reasons.append(str(error))
        for tool in _enabled(spec):
            if toolbox.get(tool.id, tool.version) is None:
                reasons.append(
                    f"tool {tool.id} {tool.version} has no implementation in the toolbox"
                )
        if type(model) is not ScriptedHarnessModel:
            # Exactly the class, never a subclass: anything else can report the
            # pinned digest and answer differently.
            reasons.append(
                "this runner executes only ScriptedHarnessModel itself; another model "
                "implementation cannot prove it answers from the pinned script"
            )
        elif ScriptedHarnessModel.digest_of(model.replies) != spec.harness.model.artifact_digest:
            reasons.append("the script is not the one the harness pins (artifact_digest differs)")
        if reasons:
            raise HarnessRunRefusedError(tuple(dict.fromkeys(reasons)))

        expected = _attribution(spec)
        location = Path(spec.outputs[0].uri)
        location.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive(location):
            if location.exists():
                return _published(location, expected)
            result = _Run(spec, model, toolbox, cancel).result(expected["request_digest"])
            return _publish(location, result, expected)


def run_scripted_harness(
    spec: AgentExecutionSpec,
    *,
    model: ScriptedHarnessModel,
    toolbox: HarnessToolbox,
    authorized_tools: Collection[str],
    cancel: CancelSignal | None = None,
) -> HarnessRunResult:
    """The entrypoint the reference compiler names: :meth:`ScriptedHarnessRunner.run`."""
    return ScriptedHarnessRunner().run(
        spec, model=model, toolbox=toolbox, authorized_tools=authorized_tools, cancel=cancel
    )


def _enabled(spec: AgentExecutionSpec) -> tuple[ToolDefinition, ...]:
    enabled = set(spec.harness.tools.enabled)
    return tuple(tool for tool in spec.harness.tools.definitions if tool.id in enabled)


def _spec_refusals(spec: AgentExecutionSpec) -> tuple[str, ...]:
    """Every requirement of *spec* this runner does not implement (ADR-019 E9)."""
    reasons: list[str] = []
    expected = {"compiler_api": HARNESS_COMPILER_API, "token_counter": TOKEN_COUNTER}
    if dict(spec.config) != expected:
        reasons.append(
            f"config must be exactly {expected}: this runner speaks one compiler API and "
            f"counts tokens one way"
        )
    if spec.entrypoint != SCRIPTED_RUNNER_ENTRYPOINT:
        reasons.append("the entrypoint is not this runner's")
    if spec.arguments:
        reasons.append("arguments are not supported: the runner is in-process")
    if spec.environment:
        reasons.append("environment variables are not supported: the runner is in-process")
    if spec.secrets:
        reasons.append("secrets are not supported: the scripted runner calls no service")
    if spec.dependencies != DependencySpec():
        reasons.append("dependencies are not supported: the runner installs nothing")
    if spec.container is not None:
        reasons.append("a container is not supported: the runner is in-process")
    if spec.inputs:
        reasons.append("artifact inputs are not supported: content travels in the spec")
    if spec.resources != ResourceRequirements():
        reasons.append("resource requirements are not supported: the runner schedules nothing")
    if spec.required_capabilities != CapabilityRequirements(
        extensions=FrozenDict({"agent_runner": SCRIPTED_RUNNER})
    ):
        reasons.append(f"the only capability this runner provides is {SCRIPTED_RUNNER!r}")
    if spec.telemetry.heartbeat_seconds is not None or spec.telemetry.endpoint is not None:
        reasons.append("telemetry heartbeats and endpoints are not supported: one result only")
    if len(spec.outputs) != 1 or spec.outputs[0].kind != AGENT_RUN_RESULT_SCHEMA:
        reasons.append(f"the spec must declare exactly one {AGENT_RUN_RESULT_SCHEMA} output")
    else:
        location = PurePosixPath(spec.outputs[0].uri)
        if not location.is_absolute() or ".." in location.parts:
            reasons.append("the result output must be an absolute local path without '..'")
    reasons.extend(scripted_harness_refusals(spec.harness))
    template = spec.harness.instructions.task_template
    if template is not None:
        reasons.extend(
            template_refusals(template.syntax, spec.text_of(template.content), template.variables)
        )
    return tuple(dict.fromkeys(reasons))


# ---- one run ------------------------------------------------------------------------------------


def _tokens(text: str) -> int:
    return len(text.split())


def _call_tokens(call: HarnessToolCall) -> int:
    return _tokens(call.tool) + _tokens(json.dumps(thaw(call.arguments), sort_keys=True))


def _message_tokens(message: HarnessMessage) -> int:
    return _tokens(message.content) + sum(_call_tokens(call) for call in message.tool_calls)


def _first_words(text: str, limit: int) -> str:
    words = text.split()
    return text if len(words) <= limit else " ".join(words[:limit])


class _Run:
    """One execution of one spec. Everything it reads is in the spec, the script or the toolbox."""

    def __init__(
        self,
        spec: AgentExecutionSpec,
        model: HarnessModel,
        toolbox: HarnessToolbox,
        cancel: CancelSignal | None,
    ) -> None:
        self.spec = spec
        self.harness = spec.harness
        self.model = model
        self.toolbox = toolbox
        self.cancel = cancel

        instructions = self.harness.instructions
        template = instructions.task_template
        assert template is not None  # refused before a run otherwise
        self.system = spec.text_of(instructions.system)
        self.task = HarnessMessage(
            role="user",
            content=render_template(
                template.syntax, spec.text_of(template.content), spec.task.variables
            ),
        )
        self.enabled = {tool.id: tool for tool in _enabled(spec)}
        self.views = tuple(
            HarnessToolView(
                id=tool.id,
                version=tool.version,
                description=spec.text_of(tool.description),
                contract=spec.text_of(tool.contract) if tool.contract is not None else None,
                input_schema=tool.input_schema,
            )
            for tool in self.enabled.values()
        )
        self.fixed_tokens = _tokens(self.system) + sum(
            _tokens(view.description)
            + _tokens(view.contract or "")
            + _tokens(json.dumps(thaw(view.input_schema), sort_keys=True))
            for view in self.views
        )

        self.history: list[list[HarnessMessage]] = []
        self.answer: str | None = None
        self.error: str | None = None
        self.turns = 0
        self.tool_calls = 0
        self.refused_tool_calls = 0
        self.failed_tool_calls = 0
        self.context_truncations = 0
        self.prompt_tokens = 0
        self.output_tokens = 0

    def result(self, request_digest: str) -> HarnessRunResult:
        termination = self._execute()
        spec = self.spec
        return HarnessRunResult(
            request_digest=request_digest,
            execution_fingerprint=agent_execution_fingerprint(spec),
            harness_fingerprint=spec.harness_fingerprint,
            compiler=spec.compiler,
            runner=RUNNER,
            model_digest=self.model.artifact_digest,
            task_id=spec.task.task_id,
            seed=spec.seed,
            replicate=spec.replicate,
            termination=termination,
            completed=termination in COMPLETIONS,
            answer=self.answer,
            error=self.error,
            turns=self.turns,
            tool_calls=self.tool_calls,
            refused_tool_calls=self.refused_tool_calls,
            failed_tool_calls=self.failed_tool_calls,
            context_truncations=self.context_truncations,
            prompt_tokens=self.prompt_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.prompt_tokens + self.output_tokens,
            token_counter=TOKEN_COUNTER,
        )

    def _cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    def _execute(self) -> TerminationReason:
        stopping = self.harness.stopping
        while True:
            if self._cancelled():
                return "cancelled"
            if self.turns >= stopping.max_turns:
                return "max-turns"
            messages = self._context()
            if messages is None:
                return "context-overflow"
            prompt = self.fixed_tokens + sum(_message_tokens(m) for m in messages)
            request = HarnessModelRequest(
                system=self.system,
                messages=messages,
                tools=self.views,
                generation=self.harness.model.generation,
            )
            try:
                reply = self.model.respond(request, turn=self.turns)
                if not isinstance(reply, HarnessModelReply):
                    reply = HarnessModelReply.model_validate(reply)
            except Exception as error:
                self.error = type(error).__name__
                return "model-error"
            self.turns += 1

            text = self._bounded_output(reply.text)
            self.prompt_tokens += prompt
            self.output_tokens += _tokens(text) + sum(_call_tokens(c) for c in reply.tool_calls)
            if (
                stopping.max_total_tokens is not None
                and self.prompt_tokens + self.output_tokens > stopping.max_total_tokens
            ):
                return "token-budget"

            turn = [HarnessMessage(role="assistant", content=text, tool_calls=reply.tool_calls)]
            self.history.append(turn)
            if reply.final and "final-answer" in stopping.terminate_on:
                self.answer = text
                return "final-answer"
            if not reply.tool_calls and "no-tool-call" in stopping.terminate_on:
                self.answer = text
                return "no-tool-call"
            for call in reply.tool_calls:
                if self._cancelled():
                    return "cancelled"
                budget = stopping.max_tool_calls
                if budget is not None and self.tool_calls >= budget:
                    return "tool-call-budget"
                self.tool_calls += 1
                result = self._invoke(call)
                turn.append(HarnessMessage(role="tool", content=result, tool_call_id=call.call_id))

    def _context(self) -> tuple[HarnessMessage, ...] | None:
        """The messages this call is shown: selected, then bounded. ``None`` if they cannot fit."""
        policy = self.harness.context
        selection = policy.selection
        turns = (
            list(self.history)
            if isinstance(selection, FullHistory)
            else self.history[-selection.turns :]
        )
        while True:
            messages = (self.task, *(message for turn in turns for message in turn))
            size = self.fixed_tokens + sum(_message_tokens(m) for m in messages)
            if size <= policy.max_context_tokens:
                return messages
            if policy.overflow == "fail" or not turns:
                return None
            turns = turns[1:]
            self.context_truncations += 1

    def _bounded_output(self, text: str) -> str:
        """*text* cut at the first stop sequence, then to ``max_output_tokens``."""
        generation = self.harness.model.generation
        cuts = [i for i in (text.find(s) for s in generation.stop_sequences) if i >= 0]
        if cuts:
            text = text[: min(cuts)]
        return _first_words(text, generation.max_output_tokens)

    def _invoke(self, call: HarnessToolCall) -> str:
        """Run one tool call if it may run; otherwise say why not. The model sees the text."""
        tool = self.enabled.get(call.tool)
        if tool is None:
            self.refused_tool_calls += 1
            return "error: that tool is not enabled for this harness"
        arguments = thaw(call.arguments)
        violations = schema_violations(tool.input_schema, arguments)
        if violations:
            self.refused_tool_calls += 1
            return "error: invalid arguments: " + "; ".join(violations)
        function = self.toolbox.get(tool.id, tool.version)
        assert function is not None  # refused before a run otherwise
        try:
            output = function(arguments)
        except Exception as error:
            self.failed_tool_calls += 1
            return f"error: the tool failed ({type(error).__name__})"
        if not isinstance(output, str):
            self.failed_tool_calls += 1
            return "error: the tool returned something other than text"
        limit = self.harness.context.max_tool_result_tokens
        return output if limit is None else _first_words(output, limit)


# ---- publication ---------------------------------------------------------------------------


def _attribution(spec: AgentExecutionSpec) -> dict[str, Any]:
    """Every field of a result that is derivable from *spec*, as this runner derives it."""
    return {
        "request_digest": agent_run_request_digest(spec),
        "execution_fingerprint": agent_execution_fingerprint(spec),
        "harness_fingerprint": spec.harness_fingerprint,
        "compiler": spec.compiler,
        "runner": RUNNER,
        "model_digest": spec.harness.model.artifact_digest,
        "task_id": spec.task.task_id,
        "seed": spec.seed,
        "replicate": spec.replicate,
        "token_counter": TOKEN_COUNTER,
    }


@contextmanager
def _exclusive(location: Path) -> Iterator[None]:
    """Hold an advisory lock for *location* on this host: one run of it at a time."""
    descriptor = os.open(location.parent / f".{location.name}.lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)  # closing releases the lock


def _published(location: Path, expected: Mapping[str, Any]) -> HarnessRunResult:
    """The result at *location*, if it is this request's and agrees with it; refused otherwise.

    The request digest says which request a record claims to answer; every
    other derivable field must then agree with what this spec derives. A
    record that matches the digest but not its attribution is inconsistent,
    and is refused rather than returned.
    """
    try:
        existing = HarnessRunResult.model_validate_json(location.read_bytes())
    except (OSError, ValidationError) as error:
        raise HarnessRunConflictError(
            "the output location holds something that is not a valid harness run result"
        ) from error
    if existing.request_digest != expected["request_digest"]:
        raise HarnessRunConflictError(
            "the output location holds the result of a different request; a result is "
            "never overwritten"
        )
    disagree = sorted(name for name, value in expected.items() if getattr(existing, name) != value)
    if disagree:
        raise HarnessRunConflictError(
            "the published result claims this request but disagrees with it on: "
            + ", ".join(disagree)
        )
    return existing


def _publish(
    location: Path, result: HarnessRunResult, expected: Mapping[str, Any]
) -> HarnessRunResult:
    """Publish *result* at *location* exactly once, atomically, never overwriting.

    Written to a private temporary file, synced, then hard-linked into place:
    the link fails if anything is already there, so a concurrent run of the
    same request publishes once and the other returns what it published.
    """
    fd, temporary = tempfile.mkstemp(dir=location.parent, prefix=".agent-run.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(result.model_dump_json(indent=2).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, location)
        except FileExistsError:
            return _published(location, expected)
    finally:
        Path(temporary).unlink(missing_ok=True)
    directory = os.open(location.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return result
