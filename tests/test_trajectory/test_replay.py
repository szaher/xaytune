"""Re-driving against recorded responses (ADR-021 T7; criterion 8).

Failure scenarios F17, F18 and F19.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import xaytune.harness.runner as runner_module
import xaytune.trajectory.writer as writer_module
from tests.test_harness.fixtures import RecordingTools
from tests.test_trajectory.fixtures import (
    SimulatedCrash,
    die,
    only_attempt,
    recorded_spec,
    run,
    store_of,
    track_writers,
)
from xaytune.core.trajectory import ReplayVerification, TierPolicy, WriterIdentity
from xaytune.harness import ScriptedHarnessModel, verify_replay
from xaytune.trajectory import WRITER, expire_blob, reconcile_trajectories
from xaytune.trajectory.sealing import read_manifest
from xaytune.trajectory.writer import TrajectoryWriter

VERIFIER = WriterIdentity(name="independent-verifier", version="1")


def _verify(attempt: Any, spec: Any = None) -> ReplayVerification:
    return verify_replay(
        attempt.store, attempt.paths, attempt.manifest, spec or attempt.spec, verifier=VERIFIER
    )


def test_a_re_drive_calls_no_model_and_no_host_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")
    run(spec)
    attempt = only_attempt(spec)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a re-drive called a real model or tool")

    monkeypatch.setattr(ScriptedHarnessModel, "respond", forbidden)
    monkeypatch.setattr(RecordingTools, "search", forbidden)
    verification = _verify(attempt)
    assert verification.outcome == "equivalent"
    assert verification.trajectory_fingerprint == attempt.manifest.trajectory_fingerprint
    assert verification.verifier == VERIFIER


def test_a_partial_trajectory_is_checked_as_a_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")
    original = ScriptedHarnessModel.respond

    def dies_on_second_turn(self: Any, request: Any, *, turn: int) -> Any:
        if turn == 1:
            raise SimulatedCrash("the process died during the second call")
        return original(self, request, turn=turn)

    monkeypatch.setattr(ScriptedHarnessModel, "respond", dies_on_second_turn)
    track_writers(monkeypatch)
    monkeypatch.setattr(TrajectoryWriter, "abandon", lambda self, *a, **k: self.close())
    with pytest.raises(SimulatedCrash):
        run(spec)
    die()
    monkeypatch.undo()
    reconcile_trajectories(store_of(spec), Path(spec.outputs[0].uri))
    attempt = only_attempt(spec)
    assert attempt.manifest.completeness == "partial"
    assert _verify(attempt).outcome == "prefix-equivalent"


def test_f17_changed_harness_logic_diverges_where_it_first_decides_differently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")
    run(spec)
    attempt = only_attempt(spec)
    counted = runner_module._tokens
    monkeypatch.setattr(runner_module, "_tokens", lambda text: counted(text) + 1)
    verification = _verify(attempt)
    assert verification.outcome == "diverged"
    assert verification.divergence is not None
    assert (verification.divergence.sequence, verification.divergence.field) == (
        2,
        "output_tokens",
    )


def test_a_different_spec_is_ineligible_not_divergent(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out")
    run(spec)
    other = recorded_spec(tmp_path / "other")
    verification = _verify(only_attempt(spec), other)
    assert verification.outcome == "ineligible"
    assert verification.unmet == ("the spec is not the one the trajectory attributes",)


class Redacting:
    policy_digest = "sha256:" + "b" * 64

    def rule_for(self, role: str, text: str) -> str | None:
        return "no-tool-output" if role == "tool-call.output" else None


def test_f18_a_redacted_observation_makes_a_trajectory_ineligible(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out", redaction_policy_digest=Redacting.policy_digest)
    run(spec, redactor=Redacting())
    verification = _verify(only_attempt(spec))
    assert verification.outcome == "ineligible"
    assert verification.unmet == ("sequence 4 output was redacted",)


def test_f18_an_expired_observation_makes_a_trajectory_ineligible(tmp_path: Path) -> None:
    spec = recorded_spec(
        tmp_path / "out", ordinary=TierPolicy(readers="experiment", retention_seconds=60)
    )
    run(spec)
    attempt = only_attempt(spec)
    reply = next(e for e in attempt.events if e.kind == "model-call-finished").reply
    later = datetime.fromisoformat(attempt.manifest.sealed_at) + timedelta(minutes=2)
    expire_blob(attempt.paths, attempt.manifest, reply.digest, "ordinary", actor="gc", now=later)
    verification = _verify(attempt)
    assert verification.outcome == "ineligible"
    assert "sequence 2 reply is expired" in verification.unmet


def test_a_none_level_trajectory_is_never_replayable(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out", replay="none")
    run(spec)
    verification = _verify(only_attempt(spec))
    assert verification.outcome == "ineligible"
    assert "the trajectory was recorded at replay level 'none'" in verification.unmet


def test_f19_the_writer_cannot_verify_its_own_trajectory(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out")
    run(spec)
    attempt = only_attempt(spec)
    with pytest.raises(ValueError, match="writer cannot verify its own"):
        verify_replay(attempt.store, attempt.paths, attempt.manifest, spec, verifier=WRITER)
    assert not hasattr(TrajectoryWriter, "verify_replay")
    assert "verification" not in type(attempt.manifest).model_fields


def test_a_verification_names_a_known_projection_and_a_coherent_outcome() -> None:
    base = {
        "verifier": VERIFIER,
        "trajectory_fingerprint": "sha256:" + "4" * 64,
        "runner": VERIFIER,
    }
    with pytest.raises(ValidationError):
        ReplayVerification(
            **base, outcome="equivalent", projection="xaytune.trajectory-replay-projection/v2"
        )
    with pytest.raises(ValidationError, match="divergence"):
        ReplayVerification(**base, outcome="diverged")
    with pytest.raises(ValidationError, match="unmet"):
        ReplayVerification(**base, outcome="ineligible")


# ---- only a sealed, published trajectory is re-driven (T4, T7) -----------------------------


def _crash_at_link(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """The process dies as ``link_new`` is about to link *name*: nothing after it runs."""
    original = writer_module.link_new

    def crashing(source: Path, target: Path) -> bool:
        if Path(target).name == name:
            raise SimulatedCrash(name)
        return original(source, target)

    monkeypatch.setattr(writer_module, "link_new", crashing)
    track_writers(monkeypatch)
    monkeypatch.setattr(TrajectoryWriter, "abandon", lambda self, *a, **k: self.close())


def _staged(spec: Any) -> Any:
    (paths,) = store_of(spec).attempts()
    staged = read_manifest(paths.staged)
    assert staged is not None and staged.completeness == "complete"
    return paths, staged


def test_a_staged_manifest_before_the_commit_is_never_re_driven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")
    _crash_at_link(monkeypatch, "agent-run.json")
    with pytest.raises(SimulatedCrash):
        run(spec)
    die()
    monkeypatch.undo()
    paths, staged = _staged(spec)
    assert not paths.manifest.exists() and not Path(spec.outputs[0].uri).exists()
    verification = verify_replay(store_of(spec), paths, staged, spec, verifier=VERIFIER)
    assert verification.outcome == "ineligible"
    assert verification.unmet == ("the manifest is not the attempt's sealed manifest",)
    reconcile_trajectories(store_of(spec), Path(spec.outputs[0].uri))
    sealed = read_manifest(paths.manifest)
    assert sealed is not None and sealed.completeness == "partial"
    again = verify_replay(store_of(spec), paths, staged, spec, verifier=VERIFIER)
    assert again.unmet == ("the manifest is not the attempt's sealed manifest",), (
        "the staged complete manifest never stands in for the partial seal"
    )


def test_a_committed_but_unsealed_trajectory_is_re_driven_only_after_its_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")
    _crash_at_link(monkeypatch, "manifest.json")
    with pytest.raises(SimulatedCrash):
        run(spec)
    die()
    monkeypatch.undo()
    paths, staged = _staged(spec)
    assert Path(spec.outputs[0].uri).exists(), "past the commit point"
    verification = verify_replay(store_of(spec), paths, staged, spec, verifier=VERIFIER)
    assert verification.unmet == ("the manifest is not the attempt's sealed manifest",)
    reconcile_trajectories(store_of(spec), Path(spec.outputs[0].uri))
    assert _verify(only_attempt(spec)).outcome == "equivalent"


@pytest.mark.parametrize("publication", ["removed", "a-copy"])
def test_a_complete_trajectory_whose_result_this_attempt_did_not_publish_is_ineligible(
    tmp_path: Path, publication: str
) -> None:
    spec = recorded_spec(tmp_path / "out")
    run(spec)
    attempt = only_attempt(spec)
    location = attempt.location
    os.unlink(location)
    if publication == "a-copy":
        shutil.copyfile(attempt.paths.result, location)
    verification = _verify(attempt)
    assert verification.outcome == "ineligible"
    assert verification.unmet == ("the attested result is not published by this attempt",)


def test_a_sealed_manifest_altered_in_memory_is_not_the_sealed_manifest(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out")
    run(spec)
    attempt = only_attempt(spec)
    later = datetime.fromisoformat(attempt.manifest.sealed_at) + timedelta(days=1)
    other = attempt.manifest.model_copy(update={"sealed_at": later.isoformat()})
    verification = verify_replay(attempt.store, attempt.paths, other, spec, verifier=VERIFIER)
    assert verification.unmet == ("the manifest is not the attempt's sealed manifest",)
