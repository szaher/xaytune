"""Secrets, containment, redaction, tiers and held-out references (ADR-021 T8; criterion 9).

Failure scenarios F11, F12 and F20.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.test_benchmarks.fixtures import TASK_IDS
from tests.test_benchmarks.fixtures import answer as held_out_answer
from tests.test_benchmarks.fixtures import task as benchmark_task
from tests.test_harness.fixtures import SCRIPT, RecordingTools
from tests.test_trajectory.fixtures import (
    attempts,
    only_attempt,
    recorded_spec,
    recording,
    run,
)
from xaytune.core.trajectory import (
    CapturedContent,
    RedactedContent,
    TrajectoryAttribution,
    TrajectoryTask,
)
from xaytune.harness import UnattestedResultError
from xaytune.harness.runner import RECORDING_RUNNER
from xaytune.trajectory import (
    LocalTrajectoryStore,
    SecretDetectedError,
    SecretRegistrationClosedError,
    TrajectoryWriter,
    contain_attempt,
    verify_trajectory,
)

SECRET = "sk-live-7f3a9c2e41"


def _every_byte(root: Path) -> bytes:
    found = b""
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            os.chmod(path, 0o600)
            found += path.name.encode() + path.read_bytes()
    return found


def _secret_traces(root: Path) -> list[str]:
    data = _every_byte(root)
    digest = hashlib.sha256(SECRET.encode()).hexdigest().encode()
    return [
        label
        for label, needle in (("value", SECRET.encode()), ("digest", digest))
        if needle in data
    ]


# ---- the frozen secret set (F11) -----------------------------------------------------------


@pytest.mark.parametrize("where", ["model-output", "tool-output"])
def test_f11_a_frozen_set_secret_is_never_written(tmp_path: Path, where: str) -> None:
    if where == "model-output":
        script = [{"text": f"The key is {SECRET}.", "final": True}]
        tools = RecordingTools()
    else:
        script = list(SCRIPT)
        tools = RecordingTools(search_result=f"internal key {SECRET}")
    spec = recorded_spec(tmp_path / "out", script)
    with pytest.raises(SecretDetectedError) as refused:
        run(spec, script, tools=tools, trajectory_secrets={SECRET})
    assert SECRET not in str(refused.value)
    assert not Path(spec.outputs[0].uri).exists(), "no result is published"
    attempt = only_attempt(spec)
    assert attempt.manifest.abandonment == "secret-detected"
    error = attempt.events[-1]
    assert (error.kind, error.code) == ("error-observed", "secret-detected")
    assert _secret_traces(tmp_path / "out") == []


def test_the_open_record_counts_secrets_and_never_holds_them(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out")
    run(spec, trajectory_secrets={SECRET, "another-secret-value"})
    record = json.loads(only_attempt(spec).paths.open_record.read_text())
    assert record["secrets_registered"] == 2
    assert _secret_traces(tmp_path / "out") == []


def _writer(store: LocalTrajectoryStore, **options: Any) -> TrajectoryWriter:
    return TrajectoryWriter.open(
        store,
        attribution=TrajectoryAttribution(
            request_digest="sha256:" + "1" * 64,
            execution_fingerprint="sha256:" + "2" * 64,
            harness_fingerprint="sha256:" + "3" * 64,
            task=TrajectoryTask(task_id="support/refund-001"),
            seed=7,
            replicate=0,
        ),
        recording=options.pop("recording", recording()),
        runner=RECORDING_RUNNER,
        **options,
    )


def test_registering_a_secret_after_the_attempt_opens_is_refused(tmp_path: Path) -> None:
    writer = _writer(LocalTrajectoryStore(tmp_path / "trajectory"), secrets=[SECRET])
    try:
        with pytest.raises(SecretRegistrationClosedError):
            writer.register_secret("rotated-key")
    finally:
        writer.close()


# ---- secrets JSON would escape (F11) -------------------------------------------------------

ESCAPED = {
    "quote": 'sk-"quoted"-7f3a9c',
    "backslash": "sk\\back\\slash-7f3a9c",
    "newline": "sk-new\nline-7f3a9c",
    "tab": "sk-tab\tbed-7f3a9c",
    "non-ascii": "sk-\u00fcn\u00efcode-7f3a9c",
}


def _traces(root: Path, secret: str) -> list[str]:
    """Every form of *secret* the store could hold: raw, JSON-escaped either way, or a digest."""
    data = _every_byte(root)
    forms = {
        "value": secret.encode(),
        "json": json.dumps(secret)[1:-1].encode(),
        "json-utf8": json.dumps(secret, ensure_ascii=False)[1:-1].encode(),
        "digest": hashlib.sha256(secret.encode()).hexdigest().encode(),
    }
    return [label for label, needle in forms.items() if needle in data]


def _script_with(secret: str, where: str) -> tuple[list[dict[str, Any]], RecordingTools]:
    tools = RecordingTools()
    if where == "model-output":
        return [{"text": f"The key is {secret}.", "final": True}], tools
    if where == "tool-arguments":
        call = {"call_id": "c1", "tool": "kb.search", "arguments": {"query": {"key": [secret]}}}
    elif where == "call-id":
        call = {"call_id": secret, "tool": "kb.search", "arguments": {"query": "refund"}}
    else:
        call = dict(SCRIPT[0]["tool_calls"][0])
        tools = RecordingTools(search_result=f"internal key {secret}")
    return [{"text": "Looking.", "tool_calls": [call]}, dict(SCRIPT[1])], tools


@pytest.mark.parametrize("where", ["model-output", "tool-arguments", "call-id", "tool-output"])
@pytest.mark.parametrize("kind", sorted(ESCAPED))
def test_f11_a_secret_json_would_escape_is_still_never_written(
    tmp_path: Path, kind: str, where: str
) -> None:
    secret = ESCAPED[kind]
    assert json.dumps(secret)[1:-1] != secret, "serialized, it no longer occurs literally"
    script, tools = _script_with(secret, where)
    spec = recorded_spec(tmp_path / "out", script)
    with pytest.raises(SecretDetectedError) as refused:
        run(spec, script, tools=tools, trajectory_secrets={secret})
    assert secret not in str(refused.value)
    assert not Path(spec.outputs[0].uri).exists(), "no result is published"
    attempt = only_attempt(spec)
    assert attempt.manifest.abandonment == "secret-detected"
    assert _traces(tmp_path / "out", secret) == []


@pytest.mark.parametrize("kind", sorted(ESCAPED))
def test_a_secret_in_the_open_records_sources_is_refused_before_it_is_written(
    tmp_path: Path, kind: str
) -> None:
    secret = ESCAPED[kind]
    root = tmp_path / "trajectory"
    with pytest.raises(SecretDetectedError, match="the open record"):
        _writer(LocalTrajectoryStore(root), secrets=[secret], secret_sources=[f"vault:{secret}"])
    assert _traces(root, secret) == []
    assert not list(root.rglob("open.json")), "no open record was written"


def test_a_secret_nested_in_json_content_is_refused(tmp_path: Path) -> None:
    secret = ESCAPED["quote"]
    writer = _writer(LocalTrajectoryStore(tmp_path / "trajectory"), secrets=[secret])
    try:
        text = json.dumps({"outer": [{"inner": {secret: "key as well as value"}}]})
        assert secret not in text, "only the parsed fields hold it literally"
        with pytest.raises(SecretDetectedError):
            writer.content(text, role="tool-call.arguments", media_type="application/json")
    finally:
        writer.close()
    assert _traces(tmp_path / "trajectory", secret) == []


# ---- containment (F12) ---------------------------------------------------------------------


def test_f12_a_late_secret_contains_the_whole_attempt_before_the_commit(tmp_path: Path) -> None:
    store = LocalTrajectoryStore(tmp_path / "trajectory")
    writer = _writer(store)
    writer.content(f"a tool output carrying {SECRET}", role="tool-call.output")
    paths = writer.paths
    contained = writer.contain("a credential was rotated after it leaked")
    assert not paths.directory.exists(), "nothing of the attempt is left outside the quarantine"
    assert contained.directory.parent == store.quarantine_directory
    assert stat.S_IMODE(os.stat(store.quarantine_directory).st_mode) == 0o700
    assert not (contained.directory / "manifest.json").exists(), "nothing is sealed"
    incident = json.loads((contained.directory / "incident.json").read_text())
    assert incident["kind"] == "late-secret"
    assert SECRET not in json.dumps(incident)
    assert hashlib.sha256(SECRET.encode()).hexdigest() not in json.dumps(incident)
    assert list(store.attempts()) == []


def test_f12_a_late_secret_after_the_commit_restricts_the_result_and_replay_refuses(
    tmp_path: Path,
) -> None:
    spec = recorded_spec(tmp_path / "out")
    run(spec)
    attempt = only_attempt(spec)
    location = Path(spec.outputs[0].uri)
    os.chmod(location, 0o644)
    contain_attempt(attempt.store, attempt.paths, "found in the answer", location=location)
    assert stat.S_IMODE(os.stat(location).st_mode) == 0o600
    assert attempts(spec) == []
    with pytest.raises(UnattestedResultError, match="no attempt instance published"):
        run(spec)


# ---- redaction and tiers --------------------------------------------------------------------


class Policy:
    """Removes any tool output that mentions an internal host."""

    policy_digest = "sha256:" + "a" * 64

    def rule_for(self, role: str, text: str) -> str | None:
        return "internal-hosts" if role == "tool-call.output" and "intranet" in text else None


def test_a_redaction_records_its_rule_and_the_trajectory_still_verifies(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out", redaction_policy_digest=Policy.policy_digest)
    run(spec, tools=RecordingTools(search_result="see intranet.example/kb"), redactor=Policy())
    attempt = only_attempt(spec)
    finished = next(e for e in attempt.events if e.kind == "tool-call-finished")
    assert isinstance(finished.output, RedactedContent)
    assert (finished.output.rule, finished.output.secret) == ("internal-hosts", False)
    assert verify_trajectory(attempt.store, attempt.paths, attempt.manifest) == ()


def test_a_recording_that_names_a_policy_needs_its_redactor(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out", redaction_policy_digest=Policy.policy_digest)
    with pytest.raises(ValueError, match="redaction policy"):
        run(spec)


def test_restricted_content_is_referenced_from_ordinary_chunks_by_digest_only(
    tmp_path: Path,
) -> None:
    store = LocalTrajectoryStore(tmp_path / "trajectory")
    writer = _writer(store)
    try:
        slot = writer.content("customer card ends 4242", role="tool-call.output", sensitive=True)
    finally:
        writer.close()
    assert isinstance(slot, CapturedContent) and slot.tier == "restricted"
    assert writer.paths.blob("restricted", slot.digest).exists()
    assert not writer.paths.blob("ordinary", slot.digest).exists()


def test_error_events_carry_no_bodies_or_input_values(tmp_path: Path) -> None:
    class Leaky(RecordingTools):
        def search(self, arguments: Mapping[str, Any]) -> str:
            raise RuntimeError(f"upstream said: {arguments['query']} -> 500 body")

    spec = recorded_spec(tmp_path / "out")
    run(spec, tools=Leaky())
    attempt = only_attempt(spec)
    chunks = b"".join(path.read_bytes() for path in attempt.paths.chunks.iterdir())
    assert b"upstream said" not in chunks and b"500 body" not in chunks
    finished = next(e for e in attempt.events if e.kind == "tool-call-finished")
    assert (finished.reason, finished.error_class) == ("tool-raised", "RuntimeError")


def test_every_file_and_directory_is_owner_only(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out")
    run(spec)
    root = Path(spec.outputs[1].uri)
    for path in [root, *root.rglob("*")]:
        mode = stat.S_IMODE(os.lstat(path).st_mode)
        assert mode & 0o077 == 0, f"{path.relative_to(root)} is {oct(mode)}"


# ---- held-out references (F20) -------------------------------------------------------------


def test_f20_an_answer_equal_to_the_reference_is_recorded_and_the_reference_stays_unreachable(
    tmp_path: Path,
) -> None:
    task = benchmark_task(TASK_IDS[0])
    reference = task.reference
    assert reference is not None
    answer = held_out_answer(TASK_IDS[0])
    script = [{"text": answer, "final": True}]
    spec = recorded_spec(tmp_path / "out", script)
    result = run(spec, script)
    assert result.answer == answer
    attempt = only_attempt(spec)
    terminated = attempt.events[-1]
    # The agent's own correct answer: its digest may equal the reference's.
    assert isinstance(terminated.answer, CapturedContent)
    assert terminated.answer.digest == reference.digest
    # The reference artifact itself is unreachable: no locator, no reference role.
    everything = _every_byte(Path(spec.outputs[1].uri))
    assert reference.uri.encode() not in everything
    assert b'"reference"' not in everything


def test_the_writer_has_no_way_to_accept_a_reference_artifact(tmp_path: Path) -> None:
    reference = benchmark_task(TASK_IDS[0]).reference
    writer = _writer(LocalTrajectoryStore(tmp_path / "trajectory"))
    try:
        with pytest.raises((TypeError, AttributeError)):
            writer.content(reference, role="answer")  # type: ignore[arg-type]
    finally:
        writer.close()
    assert list((writer.paths.directory / "blobs" / "ordinary").iterdir()) == []
