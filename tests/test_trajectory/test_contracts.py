"""The pure trajectory contracts: schema, identity, structure, bounds, agreement (ADR-021).

Acceptance criteria 1 (schema), 3 (ordering and causality), 5 (manifest
structure, the first of the four operations), 6 (agreement), 11 (chunk
bounds) and 12 (purity).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tests.test_trajectory.fixtures import only_attempt, recorded_spec, recording, run
from xaytune.core.agent_execution import HarnessRunResult
from xaytune.core.trajectory import (
    TRAJECTORY_EVENT_KINDS,
    CapturedContent,
    RedactedContent,
    TrajectoryRecordingSpec,
    event_id_for,
    parse_event,
    result_disagreements,
    seal,
    structural_refusals,
    trajectory_identity_v1,
    validate_manifest,
)
from xaytune.trajectory import WRITER

TOOL_SCRIPT = (
    {
        "text": "Let me look that up.",
        "tool_calls": [
            {"call_id": "c1", "tool": "kb.search", "arguments": {"query": "refund"}},
            {"call_id": "c2", "tool": "kb.unknown", "arguments": {}},
        ],
    },
    {"text": "Refunds are issued within five days.", "final": True},
)


@pytest.fixture
def recorded(tmp_path: Path) -> Any:
    spec = recorded_spec(tmp_path / "out", TOOL_SCRIPT)
    result = run(spec, TOOL_SCRIPT)
    attempt = only_attempt(spec)
    return attempt, result


def _dump(event: Any) -> dict[str, Any]:
    return json.loads(event.model_dump_json())


# ---- schema (AC 1) -----------------------------------------------------------------------------


def test_every_recorded_event_round_trips_and_every_kind_is_closed(recorded: Any) -> None:
    attempt, _ = recorded
    events = attempt.events
    kinds = {event.kind for event in events}
    assert kinds <= TRAJECTORY_EVENT_KINDS
    assert {"run-started", "model-call-started", "tool-call-finished", "run-terminated"} <= kinds
    for event in events:
        assert parse_event(event.model_dump_json()) == event


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(lambda d: d.update(kind="memory-updated"), "kind", id="unknown-kind"),
        pytest.param(lambda d: d.update(surprise=1), "surprise", id="unknown-field"),
        pytest.param(
            lambda d: d.update(request={"kind": "inline", "text": "hello"}),
            "request",
            id="inline-content",
        ),
        pytest.param(lambda d: d["request"].update(digest="sha256:abc"), "digest", id="bad-digest"),
        pytest.param(lambda d: d.update(event_id="tev_0"), "event_id", id="underived-id"),
        pytest.param(lambda d: d.update(parent=d["sequence"]), "earlier", id="parent-not-earlier"),
    ],
)
def test_an_event_that_is_not_the_closed_schema_is_refused(
    recorded: Any, mutate: Any, match: str
) -> None:
    attempt, _ = recorded
    started = next(e for e in attempt.events if e.kind == "model-call-started")
    data = _dump(started)
    mutate(data)
    with pytest.raises(ValidationError, match=match):
        parse_event(data)


def test_an_event_says_exactly_what_its_outcome_allows(recorded: Any) -> None:
    attempt, _ = recorded
    finished = next(e for e in attempt.events if e.kind == "model-call-finished")
    data = _dump(finished)
    with pytest.raises(ValidationError, match="reply is recorded exactly"):
        parse_event({**data, "outcome": "failed", "error_class": "TimeoutError"})
    tool = next(e for e in attempt.events if e.kind == "tool-call-finished")
    with pytest.raises(ValidationError, match="refused tool call names its refusal"):
        parse_event({**_dump(tool), "outcome": "refused", "output": None, "reason": None})


def test_a_redacted_secret_cannot_carry_a_digest() -> None:
    with pytest.raises(ValidationError, match="no digest"):
        RedactedContent(rule="api-key", secret=True, removed_digest="sha256:" + "0" * 64)
    assert RedactedContent(rule="pii", secret=False, removed_digest="sha256:" + "0" * 64)


# ---- ordering and causality (AC 3) ---------------------------------------------------------


def test_event_ids_are_derived_from_the_trajectory_and_sequence(recorded: Any) -> None:
    attempt, _ = recorded
    for event in attempt.events:
        assert event.event_id == event_id_for(event.trajectory_id, event.sequence)
    assert event_id_for("ainst_x", 1) == event_id_for("ainst_x", 1) != event_id_for("ainst_x", 2)


def test_a_recorded_run_is_one_gapless_causal_history(recorded: Any) -> None:
    attempt, _ = recorded
    events = attempt.events
    assert structural_refusals(str(attempt.manifest.trajectory_id), events) == ()
    assert [e.sequence for e in events] == list(range(len(events)))
    for event in events[1:]:
        assert event.parent is not None and event.parent < event.sequence


def _rebuild(events: list[Any], position: int, **changes: Any) -> list[Any]:
    data = {**_dump(events[position]), **changes}
    return [*events[:position], parse_event(data), *events[position + 1 :]]


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        pytest.param(lambda es: es[:2] + es[3:], "sequence gap", id="gap"),
        pytest.param(lambda es: _rebuild(es, 2, parent=0), "cannot follow", id="wrong-parent-kind"),
        pytest.param(lambda es: es + [es[-1]], "sequence gap", id="duplicate"),
        pytest.param(
            lambda es: [*es[:3], parse_event({**_dump(es[3]), "call_id": "zz"}), *es[4:]],
            "did not request",
            id="unrequested-tool-call",
        ),
    ],
)
def test_a_broken_history_is_refused_with_its_reason(
    recorded: Any, damage: Any, reason: str
) -> None:
    attempt, _ = recorded
    problems = structural_refusals(
        str(attempt.manifest.trajectory_id), damage(list(attempt.events))
    )
    assert any(reason in problem for problem in problems), problems


def test_a_start_is_finished_once_and_nothing_follows_termination(recorded: Any) -> None:
    attempt, _ = recorded
    events = list(attempt.events)
    trajectory_id = str(attempt.manifest.trajectory_id)
    finished = next(e for e in events if e.kind == "model-call-finished")
    again = parse_event(
        {
            **_dump(finished),
            "sequence": len(events),
            "event_id": event_id_for(trajectory_id, len(events)),
        }
    )
    problems = structural_refusals(trajectory_id, [*events, again])
    assert any("follows run-terminated" in p for p in problems)
    assert any("already finished" in p for p in problems)


# ---- agreement with the result (AC 6) ------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "turns",
        "tool_calls",
        "refused_tool_calls",
        "failed_tool_calls",
        "context_truncations",
        "prompt_tokens",
        "output_tokens",
    ],
)
def test_a_result_that_disagrees_by_one_on_any_count_is_refused(recorded: Any, field: str) -> None:
    attempt, result = recorded
    assert result_disagreements(attempt.events, result) == ()
    data = json.loads(result.model_dump_json())
    data[field] += 1
    if field in ("prompt_tokens", "output_tokens"):
        data["total_tokens"] += 1
    if field == "tool_calls":
        pass
    changed = HarnessRunResult.model_validate(data)
    assert any(p.startswith(field) for p in result_disagreements(attempt.events, changed))


def test_a_result_with_another_answer_is_refused(recorded: Any) -> None:
    attempt, result = recorded
    changed = HarnessRunResult.model_validate(
        {**json.loads(result.model_dump_json()), "answer": "Something else."}
    )
    assert "answer: its digest differs from the captured answer" in result_disagreements(
        attempt.events, changed
    )


# ---- the manifest alone (AC 5, validate_manifest) -------------------------------------------


def test_validate_manifest_proves_structure_only(recorded: Any, tmp_path: Path) -> None:
    attempt, _ = recorded
    manifest = attempt.manifest
    # Every chunk gone: the manifest alone still validates.
    for chunk in attempt.paths.chunks.iterdir():
        chunk.unlink()
    assert validate_manifest(attempt.paths.manifest.read_bytes()) == manifest


def _body(manifest: Any, **changes: Any) -> dict[str, Any]:
    data = json.loads(manifest.model_dump_json())
    data.update(changes)
    return data


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        pytest.param({"result_digest": None}, "attests a result", id="complete-without-result"),
        pytest.param(
            {"completeness": "partial", "abandonment": "writer-lost"},
            "attests no result",
            id="partial-with-result",
        ),
        pytest.param({"abandonment": "writer-lost"}, "no abandonment", id="complete-abandoned"),
        pytest.param(
            {"unattested_result": "sha256:" + "1" * 64},
            "unattested result",
            id="unattested-without-reason",
        ),
        pytest.param({"trajectory_fingerprint": "sha256:" + "2" * 64}, "fingerprint", id="fp"),
    ],
)
def test_a_manifest_whose_bindings_disagree_is_refused(
    recorded: Any, changes: dict[str, Any], reason: str
) -> None:
    attempt, _ = recorded
    with pytest.raises(ValidationError, match=reason):
        validate_manifest(_body(attempt.manifest, **changes))


def test_a_manifest_whose_declared_chain_does_not_link_is_refused(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out", max_events=2)
    run(spec)
    manifest = only_attempt(spec).manifest
    assert len(manifest.chunks) > 1
    data = json.loads(manifest.model_dump_json())
    data["chunks"][1]["previous"] = "sha256:" + "3" * 64
    with pytest.raises(ValidationError, match="previous chunk's digest"):
        validate_manifest(data)
    data = json.loads(manifest.model_dump_json())
    data["chunks"][1]["first_sequence"] += 1
    data["chunks"][1]["last_sequence"] += 1
    with pytest.raises(ValidationError, match="continue the previous chunk's range"):
        validate_manifest(data)


def test_the_fingerprint_excludes_timestamps_and_locators(recorded: Any) -> None:
    attempt, _ = recorded
    manifest = attempt.manifest
    later = validate_manifest(
        {**json.loads(manifest.model_dump_json()), "sealed_at": "2030-01-01T00:00:00+00:00"}
        | {"trajectory_fingerprint": manifest.trajectory_fingerprint}
    )
    assert later.trajectory_fingerprint == manifest.trajectory_fingerprint
    assert set(trajectory_identity_v1(manifest)) == {
        "type",
        "version",
        "trajectory_id",
        "attribution",
        "writer",
        "runner",
        "recording",
        "chunks",
        "content",
        "completeness",
        "abandonment",
        "termination",
        "result_digest",
        "unattested_result",
        "torn_tail_bytes",
    }
    assert seal(manifest) == manifest


def test_content_slots_are_captured_by_digest_never_inline(recorded: Any) -> None:
    attempt, _ = recorded
    for reference in attempt.manifest.content:
        assert isinstance(reference.slot, CapturedContent)
        data = attempt.store.read_blob(attempt.paths, reference.slot.tier, reference.slot.digest)
        assert data is not None and len(data) == reference.slot.size


# ---- chunk bounds (AC 11) ------------------------------------------------------------------


def test_chunk_bounds_default_and_are_respected(tmp_path: Path) -> None:
    assert (recording().max_events, recording().max_bytes) == (1000, 1024 * 1024)
    spec = recorded_spec(tmp_path / "out", TOOL_SCRIPT, max_events=3, max_bytes=4096)
    run(spec, TOOL_SCRIPT)
    manifest = only_attempt(spec).manifest
    assert len(manifest.chunks) > 1
    assert all(chunk.events <= 3 and chunk.bytes <= 4096 for chunk in manifest.chunks)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_events": 0},
        {"max_events": 10_001},
        {"max_bytes": 4095},
        {"max_bytes": 16 * 1024 * 1024 + 1},
        {"max_events": 2.0},
        {"max_events": True},
        {"max_bytes": "4096"},
    ],
)
def test_chunk_bounds_outside_their_ranges_are_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TrajectoryRecordingSpec(writer=WRITER, **changes)


# ---- purity (AC 12) ------------------------------------------------------------------------


def test_the_trajectory_contracts_are_pure_core() -> None:
    code = (
        "import sys, xaytune.core.trajectory\n"
        "bad = ('xaytune.storage', 'xaytune.trajectory', 'xaytune.harness', 'xaytune.agent',"
        " 'xaytune.runtimes', 'torch', 'transformers')\n"
        "print(sorted(m for m in sys.modules if m.startswith(bad)))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"
