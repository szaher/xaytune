"""The scripted runner: runs the declared behavior, enforces authority, is idempotent.

ADR-019 E5-E9.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tests.test_harness.fixtures import (
    CONTRACT,
    SCRIPT,
    SYSTEM,
    RecordingTools,
    compile_spec,
    context,
    harness_record,
)
from xaytune.core.agent_execution import (
    AgentExecutionSpec,
    HarnessRunResult,
    HarnessTaskInput,
    agent_execution_fingerprint,
    agent_run_request_digest,
)
from xaytune.core.capabilities import CapabilityRequirements
from xaytune.core.domain.harness import AgentHarnessSpec
from xaytune.core.execution import (
    ArtifactInput,
    CommandEntrypoint,
    ContainerSpec,
    DependencySpec,
    ResourceRequirements,
    SecretRef,
    TelemetryContract,
)
from xaytune.core.immutable import FrozenDict, thaw
from xaytune.harness import (
    RUNNER,
    HarnessModelReply,
    HarnessRunConflictError,
    HarnessRunRefusedError,
    HarnessToolbox,
    ReferenceHarnessCompiler,
    ScriptedHarnessModel,
    ScriptedHarnessRunner,
)

AUTHORIZED = frozenset({"kb.search", "tickets.close"})


def _run(
    spec: AgentExecutionSpec,
    script: Any = SCRIPT,
    tools: RecordingTools | None = None,
    **options: Any,
) -> tuple[HarnessRunResult, ScriptedHarnessModel, RecordingTools]:
    model = ScriptedHarnessModel(script)
    tools = tools or RecordingTools()
    options.setdefault("authorized_tools", AUTHORIZED)
    result = ScriptedHarnessRunner().run(spec, model=model, toolbox=tools.toolbox(), **options)
    return result, model, tools


def _search(call_id: str, query: Any = "refund") -> dict[str, Any]:
    return {"call_id": call_id, "tool": "kb.search", "arguments": {"query": query}}


def _say(text: str, *calls: dict[str, Any], final: bool = False) -> dict[str, Any]:
    return {"text": text, "tool_calls": list(calls), "final": final}


def _with(section: str, script: Any = SCRIPT, **changes: Any) -> dict[str, Any]:
    record = harness_record(script)
    record[section].update(changes)
    return {section: record[section]}


def _spec(tmp_path: Path, script: Any = SCRIPT, **sections: Any) -> AgentExecutionSpec:
    return compile_spec(tmp_path, script, **sections)


# ---- the declared behavior is what runs ------------------------------------------------------


def test_a_run_executes_the_declared_harness_and_attributes_its_result(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    result, model, tools = _run(spec)

    assert result.termination == "final-answer" and result.completed
    assert result.answer == "Refunds are issued within five days."
    assert (result.turns, result.tool_calls, result.refused_tool_calls) == (2, 1, 0)
    assert tools.calls == [("kb.search", {"query": "refund"})]

    first, second = model.requests
    assert first.system == SYSTEM
    assert [m.content for m in first.messages] == ["Ticket: Where is my refund?\nCustomer: Ann Lee"]
    assert [(t.id, t.version, t.contract) for t in first.tools] == [
        ("kb.search", "1.2.0", CONTRACT)
    ]
    assert first.tools[0].description == "Search the knowledge base."
    assert first.generation == spec.harness.model.generation
    assert [m.role for m in second.messages] == ["user", "assistant", "tool"]
    assert second.messages[2].content == "Refund policy: refunds are issued in five days."
    assert second.messages[2].tool_call_id == "c1"

    assert result.request_digest == agent_run_request_digest(spec)
    assert result.execution_fingerprint == agent_execution_fingerprint(spec)
    assert result.harness_fingerprint == spec.harness_fingerprint
    assert result.compiler == spec.compiler and result.runner == RUNNER
    assert result.model_digest == spec.harness.model.artifact_digest
    assert (result.task_id, result.seed, result.replicate) == ("support/refund-001", 7, 0)
    assert result.token_counter == "xaytune.whitespace-tokens/v1"
    assert result.total_tokens == result.prompt_tokens + result.output_tokens > 0

    published = HarnessRunResult.model_validate_json((tmp_path / "agent-run.json").read_bytes())
    assert published == result


def test_a_disabled_tool_is_invisible_and_never_executed(tmp_path: Path) -> None:
    close = {"call_id": "c2", "tool": "tickets.close", "arguments": {}}
    script = [_say("closing", close), _say("done", final=True)]
    result, model, tools = _run(_spec(tmp_path, script), script)
    assert [t.id for t in model.requests[0].tools] == ["kb.search"]
    assert tools.calls == []
    assert result.refused_tool_calls == 1
    assert model.requests[1].messages[2].content.startswith("error: that tool is not enabled")


def test_enabling_a_tool_shows_and_runs_it(tmp_path: Path) -> None:
    close = {"call_id": "c2", "tool": "tickets.close", "arguments": {}}
    script = [_say("closing", close), _say("done", final=True)]
    sections = _with("tools", script, enabled=["kb.search", "tickets.close"])
    result, model, tools = _run(_spec(tmp_path, script, **sections), script)
    assert [t.id for t in model.requests[0].tools] == ["kb.search", "tickets.close"]
    assert tools.calls == [("tickets.close", {})]
    assert result.refused_tool_calls == 0


def test_arguments_are_checked_against_the_tool_schema(tmp_path: Path) -> None:
    script = [_say("search", _search("c1", query=5)), _say("done", final=True)]
    result, model, tools = _run(_spec(tmp_path, script), script)
    assert tools.calls == []
    assert result.refused_tool_calls == 1
    assert model.requests[1].messages[2].content.startswith("error: invalid arguments")


def test_a_failing_tool_is_reported_to_the_model_and_the_run_continues(tmp_path: Path) -> None:
    class Failing(RecordingTools):
        def search(self, arguments: Any) -> str:
            raise LookupError("index offline")

    result, model, _ = _run(_spec(tmp_path), tools=Failing())
    assert result.failed_tool_calls == 1 and result.completed
    assert model.requests[1].messages[2].content == "error: the tool failed (LookupError)"


def test_the_system_prompt_and_template_are_rendered_as_declared(tmp_path: Path) -> None:
    instructions = harness_record()["instructions"]
    instructions["system"] = {"kind": "inline", "text": "Be brief."}
    instructions["task_template"] = {
        "syntax": "python-format",
        "content": {"kind": "inline", "text": "{customer} asks: {ticket} {{verbatim}}"},
        "variables": ["customer", "ticket"],
    }
    _, model, _ = _run(_spec(tmp_path, instructions=instructions))
    assert model.requests[0].system == "Be brief."
    assert model.requests[0].messages[0].content == "Ann Lee asks: Where is my refund? {verbatim}"


def test_a_literal_template_is_sent_as_written(tmp_path: Path) -> None:
    instructions = harness_record()["instructions"]
    instructions["task_template"] = {
        "syntax": "literal",
        "content": {"kind": "inline", "text": "Handle {this} ticket."},
        "variables": [],
    }
    spec = AgentHarnessSpec.model_validate(harness_record(instructions=instructions))
    task = HarnessTaskInput(task_id="t", variables=FrozenDict())
    compiled = ReferenceHarnessCompiler().compile(spec, context(spec, tmp_path, task=task))
    _, model, _ = _run(compiled)
    assert model.requests[0].messages[0].content == "Handle {this} ticket."


# ---- context ------------------------------------------------------------------------------------


def _searching(turns: int) -> list[dict[str, Any]]:
    return [_say(f"step {i}", _search(f"c{i}")) for i in range(turns)] + [_say("done", final=True)]


def test_recent_turns_shows_only_the_last_turns(tmp_path: Path) -> None:
    script = _searching(4)
    sections = _with("context", script, selection={"kind": "recent-turns", "turns": 2})
    _, model, _ = _run(_spec(tmp_path, script, **sections), script)
    last = model.requests[-1].messages
    assert last[0].role == "user"  # the task is always shown
    assert [m.content for m in last if m.role == "assistant"] == ["step 2", "step 3"]


def test_full_history_shows_every_turn(tmp_path: Path) -> None:
    script = _searching(4)
    sections = _with("context", script, selection={"kind": "full-history"})
    _, model, _ = _run(_spec(tmp_path, script, **sections), script)
    assistant = [m.content for m in model.requests[-1].messages if m.role == "assistant"]
    assert assistant == ["step 0", "step 1", "step 2", "step 3"]


def _fixed_and_task_tokens(tmp_path: Path) -> int:
    """What every call is shown before any turn: system, tools and the task, in words."""
    _, model, _ = _run(_spec(tmp_path / "probe"))
    request = model.requests[0]
    tools = sum(
        len(t.description.split())
        + len((t.contract or "").split())
        + len(json.dumps(thaw(t.input_schema), sort_keys=True).split())
        for t in request.tools
    )
    return len(request.system.split()) + tools + len(request.messages[0].content.split())


def test_truncate_oldest_drops_whole_turns_to_fit(tmp_path: Path) -> None:
    base = _fixed_and_task_tokens(tmp_path)
    script = _searching(4)
    # Room for the task and about one searched turn, never the whole history.
    sections = _with(
        "context",
        script,
        selection={"kind": "full-history"},
        max_context_tokens=base + 20,
        overflow="truncate-oldest",
    )
    result, model, _ = _run(_spec(tmp_path / "run", script, **sections), script)
    assert result.completed and result.context_truncations > 0
    for request in model.requests:
        assert request.messages[0].role == "user"
        assert all(m.role != "tool" or m.tool_call_id for m in request.messages)
    assert len(model.requests[-1].messages) < 1 + 4 * 2


def test_overflow_fail_stops_the_run(tmp_path: Path) -> None:
    base = _fixed_and_task_tokens(tmp_path)
    script = _searching(4)
    sections = _with(
        "context",
        script,
        selection={"kind": "full-history"},
        max_context_tokens=base + 20,
        overflow="fail",
    )
    result, _, _ = _run(_spec(tmp_path / "run", script, **sections), script)
    assert result.termination == "context-overflow" and not result.completed
    assert result.answer is None


def test_a_context_that_cannot_hold_the_task_overflows_immediately(tmp_path: Path) -> None:
    sections = _with("context", max_context_tokens=1)
    result, model, _ = _run(_spec(tmp_path, **sections))
    assert result.termination == "context-overflow"
    assert model.requests == () and result.turns == 0


def test_tool_results_are_truncated_to_their_limit(tmp_path: Path) -> None:
    sections = _with("context", max_tool_result_tokens=3)
    _, model, _ = _run(_spec(tmp_path, **sections))
    assert model.requests[1].messages[2].content == "Refund policy: refunds"


# ---- output ---------------------------------------------------------------------------------


def test_output_is_cut_at_a_stop_sequence_then_to_max_output_tokens(tmp_path: Path) -> None:
    script = [_say("Refunds take five days. END internal notes follow", final=True)]
    model = harness_record(script)["model"]
    model["generation"]["stop_sequences"] = [" END"]
    result, _, _ = _run(_spec(tmp_path / "a", script, model=model), script)
    assert result.answer == "Refunds take five days."

    model["generation"].update(stop_sequences=[], max_output_tokens=2)
    result, _, _ = _run(_spec(tmp_path / "b", script, model=model), script)
    assert result.answer == "Refunds take"


# ---- stopping ---------------------------------------------------------------------------------


def test_max_turns_stops_the_run(tmp_path: Path) -> None:
    script = _searching(5)
    result, model, _ = _run(
        _spec(tmp_path, script, **_with("stopping", script, max_turns=2)), script
    )
    assert result.termination == "max-turns" and not result.completed
    assert result.turns == 2 and len(model.requests) == 2


def test_max_tool_calls_stops_the_run(tmp_path: Path) -> None:
    script = _searching(5)
    sections = _with("stopping", script, max_tool_calls=2)
    result, _, tools = _run(_spec(tmp_path, script, **sections), script)
    assert result.termination == "tool-call-budget"
    assert len(tools.calls) == 2 == result.tool_calls


def test_max_total_tokens_stops_the_run(tmp_path: Path) -> None:
    script = _searching(5)
    sections = _with("stopping", script, max_total_tokens=150)
    result, _, _ = _run(_spec(tmp_path, script, **sections), script)
    assert result.termination == "token-budget"
    assert result.total_tokens > 150


def test_terminate_on_no_tool_call_ends_at_the_first_turn_without_one(tmp_path: Path) -> None:
    script = [_say("thinking"), _say("answer", final=True)]
    sections = _with("stopping", script, terminate_on=["no-tool-call"])
    result, _, _ = _run(_spec(tmp_path, script, **sections), script)
    assert (result.termination, result.answer, result.turns) == ("no-tool-call", "thinking", 1)


def test_terminate_on_final_answer_keeps_going_past_a_turn_without_tools(tmp_path: Path) -> None:
    script = [_say("thinking"), _say("answer", final=True)]
    result, _, _ = _run(_spec(tmp_path, script), script)
    assert (result.termination, result.answer, result.turns) == ("final-answer", "answer", 2)


def test_a_model_that_fails_ends_the_run_as_a_model_error(tmp_path: Path) -> None:
    script = [_say("searching", _search("c1"))]  # no second reply
    result, _, _ = _run(_spec(tmp_path, script), script)
    assert result.termination == "model-error"
    assert result.error == "ScriptExhaustedError"
    assert not result.completed


# ---- authority is the host's ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("authorized", "toolbox", "reason"),
    [
        (frozenset(), None, "not authorized: kb.search"),
        (frozenset({"tickets.close"}), None, "not authorized"),
        (AUTHORIZED, HarnessToolbox({}), "kb.search 1.2.0 has no implementation"),
        (AUTHORIZED, HarnessToolbox({("kb.search", "1.1.0"): lambda a: "x"}), "no implementation"),
    ],
)
def test_a_run_is_refused_without_the_hosts_authority(
    tmp_path: Path, authorized: frozenset[str], toolbox: HarnessToolbox | None, reason: str
) -> None:
    spec = _spec(tmp_path)
    model = ScriptedHarnessModel(SCRIPT)
    with pytest.raises(HarnessRunRefusedError, match=reason):
        ScriptedHarnessRunner().run(
            spec,
            model=model,
            toolbox=toolbox or RecordingTools().toolbox(),
            authorized_tools=authorized,
        )
    assert model.requests == ()
    assert not (tmp_path / "agent-run.json").exists()


def test_a_model_other_than_the_pinned_one_is_refused(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    other = ScriptedHarnessModel([HarnessModelReply(text="hi", final=True)])
    with pytest.raises(HarnessRunRefusedError, match="not the one the harness pins"):
        ScriptedHarnessRunner().run(
            spec, model=other, toolbox=RecordingTools().toolbox(), authorized_tools=AUTHORIZED
        )
    assert other.requests == ()


class _Impostor:
    """Reports the pinned digest, answers something else."""

    artifact_digest = ScriptedHarnessModel.digest_of(SCRIPT)

    def __init__(self) -> None:
        self.calls = 0

    def respond(self, request: Any, *, turn: int) -> HarnessModelReply:
        self.calls += 1
        return HarnessModelReply(text="not the script", final=True)


class _Subclass(ScriptedHarnessModel):
    def respond(self, request: Any, *, turn: int) -> HarnessModelReply:
        raise AssertionError("a subclass must never be called")


@pytest.mark.parametrize(
    "make", [_Impostor, lambda: _Subclass(SCRIPT)], ids=["impostor", "subclass"]
)
def test_only_the_trusted_scripted_model_runs_even_with_the_pinned_digest(
    tmp_path: Path, make: Any
) -> None:
    spec = _spec(tmp_path)
    model = make()
    assert model.artifact_digest == spec.harness.model.artifact_digest
    tools = RecordingTools()
    with pytest.raises(HarnessRunRefusedError, match="only ScriptedHarnessModel itself"):
        ScriptedHarnessRunner().run(
            spec, model=model, toolbox=tools.toolbox(), authorized_tools=AUTHORIZED
        )
    assert getattr(model, "calls", 0) == 0 and tools.calls == []
    assert not (tmp_path / "agent-run.json").exists()


@pytest.mark.parametrize(
    ("update", "reason"),
    [
        ({"container": ContainerSpec(image="python", digest="sha256:" + "a" * 64)}, "container"),
        ({"secrets": (SecretRef(name="KEY", source="env"),)}, "secrets"),
        ({"resources": ResourceRequirements(gpus=1)}, "resource requirements"),
        ({"resources": ResourceRequirements(max_runtime_seconds=60)}, "resource requirements"),
        ({"environment": FrozenDict({"A": "1"})}, "environment variables"),
        ({"arguments": ("--fast",)}, "arguments"),
        ({"inputs": (ArtifactInput(name="x", uri="/x"),)}, "artifact inputs"),
        ({"dependencies": DependencySpec(requirements=("requests",))}, "dependencies"),
        ({"entrypoint": CommandEntrypoint(argv=("agent",))}, "entrypoint"),
        ({"config": FrozenDict({"compiler_api": "xaytune.harness-compiler/v2"})}, "config"),
        (
            {
                "required_capabilities": CapabilityRequirements(
                    extensions=FrozenDict({"agent_runner": "other"})
                )
            },
            "only capability",
        ),
        ({"required_capabilities": CapabilityRequirements(precision="bf16")}, "only capability"),
        (
            {
                "telemetry": TelemetryContract(
                    protocol_version="xaytune.agent-telemetry/v1alpha1", heartbeat_seconds=5
                )
            },
            "heartbeats",
        ),
        ({"outputs": ()}, "exactly one"),
    ],
)
def test_what_the_runner_does_not_implement_fails_closed(
    tmp_path: Path, update: dict[str, Any], reason: str
) -> None:
    spec = _spec(tmp_path).model_copy(update=update)
    support = ScriptedHarnessRunner().supports(spec)
    assert not support and any(reason in r for r in support.reasons)
    model = ScriptedHarnessModel(SCRIPT)
    with pytest.raises(HarnessRunRefusedError, match=reason):
        ScriptedHarnessRunner().run(
            spec, model=model, toolbox=RecordingTools().toolbox(), authorized_tools=AUTHORIZED
        )
    assert model.requests == ()


def test_a_compiled_spec_is_supported(tmp_path: Path) -> None:
    assert ScriptedHarnessRunner().supports(_spec(tmp_path)).supported


# ---- idempotent, restartable, cancellable ------------------------------------------------


def test_the_same_request_runs_once(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    first, model, tools = _run(spec)
    again = ScriptedHarnessRunner().run(
        spec, model=model, toolbox=tools.toolbox(), authorized_tools=AUTHORIZED
    )
    assert again == first
    assert len(tools.calls) == 1 and len(model.requests) == 2  # nothing re-executed


def test_a_different_request_at_the_same_output_is_a_conflict(tmp_path: Path) -> None:
    _run(_spec(tmp_path))
    other = _spec(tmp_path).model_copy(update={"seed": 8})
    model = ScriptedHarnessModel(SCRIPT)
    with pytest.raises(HarnessRunConflictError, match="different request"):
        ScriptedHarnessRunner().run(
            other, model=model, toolbox=RecordingTools().toolbox(), authorized_tools=AUTHORIZED
        )
    assert model.requests == ()


def test_something_else_at_the_output_is_a_conflict(tmp_path: Path) -> None:
    (tmp_path / "agent-run.json").write_text("not a result")
    with pytest.raises(HarnessRunConflictError, match="not a valid harness run result"):
        _run(_spec(tmp_path))


class _Crash(BaseException):
    """A process dying mid-run: not an error the run can handle."""


def test_a_crash_before_publication_leaves_nothing_and_the_rerun_is_identical(
    tmp_path: Path,
) -> None:
    class CrashOnce(RecordingTools):
        crashed = False

        def search(self, arguments: Any) -> str:
            if not CrashOnce.crashed:
                CrashOnce.crashed = True
                raise _Crash
            return super().search(arguments)

    spec = _spec(tmp_path / "restarted")
    with pytest.raises(_Crash):
        _run(spec, tools=CrashOnce())
    assert [p.name for p in (tmp_path / "restarted").iterdir()] == [".agent-run.json.lock"]

    restarted, _, _ = _run(spec, tools=CrashOnce())
    clean, _, _ = _run(_spec(tmp_path / "clean"))
    differs = {"request_digest", "execution_fingerprint"}  # the output location differs
    assert restarted.model_dump(exclude=differs) == clean.model_dump(exclude=differs)
    left = sorted(p.name for p in (tmp_path / "restarted").iterdir())
    assert left == [".agent-run.json.lock", "agent-run.json"]  # no temporary file survives


def test_a_run_cancelled_before_it_starts_calls_nothing(tmp_path: Path) -> None:
    cancel = threading.Event()
    cancel.set()
    result, model, tools = _run(_spec(tmp_path), cancel=cancel)
    assert result.termination == "cancelled" and not result.completed
    assert model.requests == () and tools.calls == []


def test_a_run_cancelled_midway_stops_before_the_next_call_and_stays_cancelled(
    tmp_path: Path,
) -> None:
    cancel = threading.Event()

    class CancelsAfterSearch(RecordingTools):
        def search(self, arguments: Any) -> str:
            cancel.set()
            return super().search(arguments)

    script = [_say("two searches", _search("c1"), _search("c2", "fees")), _say("done", final=True)]
    spec = _spec(tmp_path, script)
    result, model, tools = _run(spec, script, tools=CancelsAfterSearch(), cancel=cancel)
    assert result.termination == "cancelled"
    assert len(tools.calls) == 1 and len(model.requests) == 1

    # Terminal: the same request returns the cancelled result, uncancelled or not.
    replay, replay_model, replay_tools = _run(spec, script)
    assert replay == result
    assert replay_model.requests == () and replay_tools.calls == []


def test_concurrent_runs_on_one_host_execute_once_and_publish_once(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    model = ScriptedHarnessModel(SCRIPT)
    tools = RecordingTools()
    results: list[HarnessRunResult] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(
                ScriptedHarnessRunner().run(
                    spec, model=model, toolbox=tools.toolbox(), authorized_tools=AUTHORIZED
                )
            )
        except BaseException as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(results) == 4 and all(r == results[0] for r in results)
    # Serialized on this host: one execution, the others replay its result.
    assert len(tools.calls) == 1 and len(model.requests) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == [".agent-run.json.lock", "agent-run.json"]


# ---- replayed results are checked, not believed ------------------------------------------


@pytest.mark.parametrize(
    "update",
    [
        {"task_id": "support/other"},
        {"seed": 99},
        {"replicate": 3},
        {"harness_fingerprint": "sha256:" + "1" * 64},
        {"execution_fingerprint": "sha256:" + "2" * 64},
        {"model_digest": "sha256:" + "3" * 64},
        {"runner": {"name": "another-runner", "version": "1"}},
        {"compiler": {"name": "reference-harness", "version": "9.9.9"}},
        {"token_counter": "provider-tokens/v1"},
    ],
    ids=lambda u: next(iter(u)),
)
def test_a_replayed_result_must_agree_with_the_spec_on_every_attribution(
    tmp_path: Path, update: dict[str, Any]
) -> None:
    spec = _spec(tmp_path)
    honest, _, _ = _run(spec)
    forged = honest.model_copy(update=update)
    (tmp_path / "agent-run.json").write_text(forged.model_dump_json())
    model, tools = ScriptedHarnessModel(SCRIPT), RecordingTools()
    with pytest.raises(
        HarnessRunConflictError, match=f"disagrees with it on: {next(iter(update))}"
    ):
        ScriptedHarnessRunner().run(
            spec, model=model, toolbox=tools.toolbox(), authorized_tools=AUTHORIZED
        )
    assert model.requests == () and tools.calls == []


def test_an_internally_inconsistent_published_result_is_refused(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    honest, _, _ = _run(spec)
    record = honest.model_dump(mode="json")
    record.update(termination="cancelled")  # but still completed, with an answer
    (tmp_path / "agent-run.json").write_text(json.dumps(record))
    with pytest.raises(HarnessRunConflictError, match="not a valid harness run result"):
        _run(spec)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"termination": "cancelled"}, "completed must be true exactly"),
        ({"completed": False}, "completed must be true exactly"),
        ({"answer": None}, "only a completed run has an answer"),
        ({"error": "Boom"}, "only a model-error run records an error"),
        ({"total_tokens": 1}, "total_tokens must be prompt_tokens"),
        ({"refused_tool_calls": 2}, "counted among tool_calls"),
        ({"turns": 0}, "needs a model turn"),
        ({"seed": "7"}, "seed"),
        ({"turns": True}, "turns"),
        ({"request_digest": "not-a-digest"}, "request_digest"),
        ({"token_counter": ""}, "token_counter"),
    ],
    ids=lambda v: v if isinstance(v, str) else next(iter(v)),
)
def test_a_result_is_internally_consistent_or_it_does_not_parse(
    tmp_path: Path, changes: dict[str, Any], reason: str
) -> None:
    honest, _, _ = _run(_spec(tmp_path))
    record = json.loads(honest.model_dump_json())
    record.update(changes)
    with pytest.raises(ValidationError, match=reason):
        HarnessRunResult.model_validate(record)


def test_every_way_a_run_ends_produces_a_consistent_result(tmp_path: Path) -> None:
    """Each termination the runner reaches round-trips through the invariants."""
    cancel = threading.Event()
    cancel.set()
    ends = {
        _run(_spec(tmp_path / "done"))[0].termination,
        _run(_spec(tmp_path / "cancelled"), cancel=cancel)[0].termination,
        _run(_spec(tmp_path / "error", SCRIPT[:1]), SCRIPT[:1])[0].termination,
        _run(_spec(tmp_path / "overflow", **_with("context", max_context_tokens=1)))[0].termination,
    }
    assert ends == {"final-answer", "cancelled", "model-error", "context-overflow"}
    for directory in ("done", "cancelled", "error", "overflow"):
        raw = (tmp_path / directory / "agent-run.json").read_bytes()
        HarnessRunResult.model_validate_json(raw)
