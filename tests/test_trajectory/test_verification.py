"""Evidence and publication verification, for every way a run can end (ADR-021 T5, T7).

Acceptance criteria 5 (integrity and the four operations), 6 (agreement for
every H03 termination reason) and 8 (replay), and failure scenario F16.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.test_harness.fixtures import SCRIPT, RecordingTools
from tests.test_trajectory.fixtures import (
    harness_with,
    only_attempt,
    recorded_spec,
    run,
)
from xaytune.core.trajectory import WriterIdentity, result_disagreements
from xaytune.harness import verify_replay
from xaytune.trajectory import verify_publication, verify_trajectory

VERIFIER = WriterIdentity(name="independent-verifier", version="1")


def _say(text: str, *calls: dict[str, Any], final: bool = False) -> dict[str, Any]:
    return {"text": text, "tool_calls": list(calls), "final": final}


def _search(call_id: str, query: Any = "refund") -> dict[str, Any]:
    return {"call_id": call_id, "tool": "kb.search", "arguments": {"query": query}}


def _searching(turns: int) -> list[dict[str, Any]]:
    return [_say(f"search {i}", _search(f"c{i}", f"q{i}")) for i in range(turns)] + [
        _say("done", final=True)
    ]


def _with(script: Any, section: str, **changes: Any) -> dict[str, Any]:
    record = harness_with(script)
    record[section] = {**record[section], **changes}
    return record


class Raising(RecordingTools):
    def search(self, arguments: Mapping[str, Any]) -> str:
        raise LookupError("the index is offline")


class NotText(RecordingTools):
    def search(self, arguments: Mapping[str, Any]) -> Any:
        return {"passages": []}


def _cancelling() -> tuple[RecordingTools, threading.Event]:
    cancel = threading.Event()

    class CancelsAfterSearch(RecordingTools):
        def search(self, arguments: Mapping[str, Any]) -> str:
            cancel.set()
            return super().search(arguments)

    return CancelsAfterSearch(), cancel


TWO_CALLS = [_say("two", _search("c1"), _search("c2", "fees")), _say("done", final=True)]
REFUSALS = [
    _say(
        "refusals",
        {"call_id": "c1", "tool": "kb.unknown", "arguments": {}},
        _search("c2", 5),
    ),
    _say("done", final=True),
]


def _scenario(name: str) -> tuple[Any, dict[str, Any] | None, dict[str, Any]]:
    """(script, harness record or None, run options) for one way a run ends."""
    if name == "final-answer":
        return SCRIPT, None, {}
    if name == "no-tool-call":
        script = [_say("thinking"), _say("answer", final=True)]
        return script, _with(script, "stopping", terminate_on=["no-tool-call"]), {}
    if name == "max-turns":
        script = _searching(5)
        return script, _with(script, "stopping", max_turns=2), {}
    if name == "token-budget":
        script = _searching(5)
        return script, _with(script, "stopping", max_total_tokens=150), {}
    if name == "tool-call-budget":
        script = _searching(5)
        return script, _with(script, "stopping", max_tool_calls=2), {}
    if name == "context-overflow":
        return SCRIPT, _with(SCRIPT, "context", max_context_tokens=1), {}
    if name == "recent-turns":
        script = _searching(3)
        return script, _with(script, "context", selection={"kind": "recent-turns", "turns": 1}), {}
    if name == "truncate-oldest":
        script = _searching(4)
        record = _with(
            script,
            "context",
            selection={"kind": "full-history"},
            max_context_tokens=70,
            overflow="truncate-oldest",
        )
        return script, record, {}
    if name == "model-error":
        script = [_say("searching", _search("c1"))]
        return script, None, {}
    if name == "cancelled":
        tools, cancel = _cancelling()
        return TWO_CALLS, None, {"tools": tools, "cancel": cancel}
    if name == "tool-raised":
        return SCRIPT, None, {"tools": Raising()}
    if name == "non-text-result":
        return SCRIPT, None, {"tools": NotText()}
    if name == "refusals":
        return REFUSALS, None, {}
    raise AssertionError(name)


SCENARIOS = (
    "final-answer",
    "no-tool-call",
    "max-turns",
    "token-budget",
    "tool-call-budget",
    "context-overflow",
    "recent-turns",
    "truncate-oldest",
    "model-error",
    "cancelled",
    "tool-raised",
    "non-text-result",
    "refusals",
)


def _recorded(tmp_path: Path, name: str) -> tuple[Any, Any]:
    script, record, options = _scenario(name)
    spec = recorded_spec(tmp_path / name, script, record=record)
    result = run(spec, script, **options)
    return only_attempt(spec), result


@pytest.mark.parametrize("name", SCENARIOS)
def test_every_ending_is_recorded_agrees_verifies_and_replays(tmp_path: Path, name: str) -> None:
    attempt, result = _recorded(tmp_path, name)
    manifest = attempt.manifest
    assert manifest.completeness == "complete"
    assert manifest.termination == result.termination
    assert result_disagreements(attempt.events, result) == ()
    assert verify_trajectory(attempt.store, attempt.paths, manifest) == ()
    assert verify_publication(attempt.paths, manifest, attempt.location) == ()
    replay = verify_replay(attempt.store, attempt.paths, manifest, attempt.spec, verifier=VERIFIER)
    assert replay.outcome == "equivalent", replay


def test_the_scenarios_cover_every_termination_and_every_tool_outcome(tmp_path: Path) -> None:
    terminations, outcomes, rules = set(), set(), set()
    for name in SCENARIOS:
        attempt, result = _recorded(tmp_path, name)
        terminations.add(result.termination)
        for event in attempt.events:
            if event.kind == "tool-call-finished":
                outcomes.add((event.outcome, event.reason))
            if event.kind == "context-changed":
                rules.add(event.rule)
    assert terminations == {
        "final-answer",
        "no-tool-call",
        "max-turns",
        "token-budget",
        "tool-call-budget",
        "context-overflow",
        "model-error",
        "cancelled",
    }
    assert outcomes == {
        ("executed", None),
        ("refused", "not-enabled"),
        ("refused", "invalid-arguments"),
        ("failed", "tool-raised"),
        ("failed", "non-text-result"),
    }
    assert rules == {"recent-turns", "truncate-oldest"}


# ---- publication verification (AC 5) --------------------------------------------------------


@pytest.fixture
def complete(tmp_path: Path) -> Any:
    spec = recorded_spec(tmp_path / "out")
    run(spec)
    return only_attempt(spec)


def test_evidence_verification_does_not_depend_on_publication(complete: Any) -> None:
    manifest = complete.manifest
    assert verify_trajectory(complete.store, complete.paths, manifest) == ()
    complete.location.unlink()
    assert verify_trajectory(complete.store, complete.paths, manifest) == ()
    assert verify_publication(complete.paths, manifest, complete.location) == (
        "no result is published at the declared location",
    )


def test_a_copy_is_not_this_attempts_publication(complete: Any) -> None:
    data = complete.location.read_bytes()
    complete.location.unlink()
    complete.location.write_bytes(data)
    assert verify_publication(complete.paths, complete.manifest, complete.location) == (
        "the published result is not this attempt's file",
    )


def test_another_attempts_identical_file_is_not_this_attempts_publication(
    complete: Any, tmp_path: Path
) -> None:
    other = tmp_path / "other.json"
    shutil.copyfile(complete.paths.result, other)
    complete.location.unlink()
    os.link(other, complete.location)
    assert "the published result is not this attempt's file" in verify_publication(
        complete.paths, complete.manifest, complete.location
    )


def test_a_published_result_with_another_digest_is_refused(complete: Any) -> None:
    os.chmod(complete.location, 0o600)
    with open(complete.location, "ab") as handle:
        handle.write(b"\n")
    assert "the published result does not have the attested digest" in verify_publication(
        complete.paths, complete.manifest, complete.location
    )


def test_a_symlink_at_the_published_path_is_refused(complete: Any) -> None:
    complete.location.unlink()
    complete.location.symlink_to(complete.paths.result)
    assert verify_publication(complete.paths, complete.manifest, complete.location) == (
        "the declared result location is a symlink",
    )


# ---- integrity (F16) -----------------------------------------------------------------------


def _first_chunk(attempt: Any) -> Path:
    return sorted(attempt.paths.chunks.iterdir())[0]


def _rewrite(path: Path, data: bytes) -> None:
    os.chmod(path, 0o600)
    path.write_bytes(data)


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        pytest.param(
            lambda a: _rewrite(
                _first_chunk(a),
                _first_chunk(a).read_bytes().replace(b"trajectory_id", b"trajectory_iD", 1),
            ),
            "does not have its digest",
            id="edited-chunk",
        ),
        pytest.param(lambda a: _first_chunk(a).unlink(), "is missing", id="dropped-chunk"),
        pytest.param(
            lambda a: _rewrite(a.paths.result, a.paths.result.read_bytes() + b" "),
            "result.json does not have the result digest",
            id="edited-result",
        ),
        pytest.param(
            lambda a: next((a.paths.directory / "blobs" / "ordinary").iterdir()).unlink(),
            "is missing",
            id="missing-blob",
        ),
    ],
)
def test_tampered_evidence_is_refused_with_its_reason(
    tmp_path: Path, damage: Any, reason: str
) -> None:
    spec = recorded_spec(tmp_path / "out", max_events=3)
    run(spec)
    attempt = only_attempt(spec)
    manifest = attempt.manifest
    damage(attempt)
    problems = verify_trajectory(attempt.store, attempt.paths, manifest)
    assert any(reason in problem for problem in problems), problems


def test_reordered_or_replaced_chunks_break_the_chain(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out", max_events=3)
    run(spec)
    attempt = only_attempt(spec)
    manifest = attempt.manifest
    first, second = sorted(attempt.paths.chunks.iterdir())[:2]
    a, b = first.read_bytes(), second.read_bytes()
    _rewrite(first, b)
    _rewrite(second, a)
    problems = verify_trajectory(attempt.store, attempt.paths, manifest)
    assert any("does not have its digest" in p for p in problems)


def test_an_edited_sealed_manifest_does_not_load(complete: Any) -> None:
    data = json.loads(complete.paths.manifest.read_text())
    data["termination"] = "max-turns"
    _rewrite(complete.paths.manifest, json.dumps(data).encode())
    from xaytune.trajectory.sealing import read_manifest

    assert read_manifest(complete.paths.manifest) is None
