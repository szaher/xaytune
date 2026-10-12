"""The writer's own guarantees: identity, redelivery, bounds and closed attempts (ADR-021 T1, T3).

Acceptance criteria 1 (an oversize event), 2 (identity), 3 (redelivery) and 4
(nothing appends to a sealed or abandoned attempt).
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Any

import pytest

import xaytune.trajectory.writer as writer_module
from tests.test_trajectory.fixtures import (
    SimulatedCrash,
    attempts,
    die,
    recorded_spec,
    recording,
    run,
    store_of,
    track_writers,
)
from xaytune.core.trajectory import (
    ModelCallFinished,
    ModelCallStarted,
    TrajectoryAttribution,
    TrajectoryTask,
)
from xaytune.harness.runner import RECORDING_RUNNER
from xaytune.trajectory import (
    AttemptExistsError,
    LocalTrajectoryStore,
    TrajectoryRecordingError,
    TrajectoryWriter,
    reconcile_trajectories,
)
from xaytune.trajectory.sealing import parse_tail


def _writer(store: LocalTrajectoryStore, **changes: Any) -> TrajectoryWriter:
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
        recording=recording(**changes),
        runner=RECORDING_RUNNER,
    )


# ---- identity (AC 2) -----------------------------------------------------------------------


def test_each_execution_of_one_request_is_its_own_attempt_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")
    location = Path(spec.outputs[0].uri)
    original = writer_module.link_new

    def crash_at_commit(source: Path, target: Path) -> bool:
        if Path(target) == location:
            raise SimulatedCrash("before the commit")
        return original(source, target)

    monkeypatch.setattr(writer_module, "link_new", crash_at_commit)
    track_writers(monkeypatch)
    monkeypatch.setattr(TrajectoryWriter, "abandon", lambda self, *a, **k: self.close())
    with pytest.raises(SimulatedCrash):
        run(spec)
    die()
    monkeypatch.undo()
    run(spec)
    first, second = attempts(spec)
    reconcile_trajectories(store_of(spec), location)
    assert first.manifest.trajectory_id != second.manifest.trajectory_id
    assert first.manifest.attribution == second.manifest.attribution, (
        "the same request and replicate, and two distinct attempt instances"
    )
    assert str(first.manifest.trajectory_id).startswith("ainst_")


def test_an_attempt_instance_id_is_never_reused(tmp_path: Path) -> None:
    store = LocalTrajectoryStore(tmp_path / "trajectory")
    writer = _writer(store)
    writer.close()
    with pytest.raises(AttemptExistsError):
        store.create_attempt(str(writer.trajectory_id))


def test_the_open_record_is_persisted_before_anything_else(tmp_path: Path) -> None:
    store = LocalTrajectoryStore(tmp_path / "trajectory")
    writer = _writer(store)
    try:
        record = store.read_open_record(writer.paths)
        assert record is not None and record.trajectory_id == writer.trajectory_id
        assert store.read_tail(writer.paths) == b""
    finally:
        writer.close()


# ---- bounds (AC 1) -------------------------------------------------------------------------


def test_an_event_larger_than_a_chunk_is_refused_and_never_written(tmp_path: Path) -> None:
    store = LocalTrajectoryStore(tmp_path / "trajectory")
    writer = _writer(store, max_bytes=4096)
    try:
        request = writer.content("{}", role="model-call.request", media_type="application/json")
        start = writer.record(
            ModelCallStarted,
            parent=None,
            turn=0,
            request=request,
            messages_digest="sha256:" + "4" * 64,
            tools_digest="sha256:" + "5" * 64,
            context_rule="full-history",
            messages_shown=1,
        )
        reply = writer.content("{}", role="model-call.reply", media_type="application/json")
        before = store.read_tail(writer.paths)
        with pytest.raises(TrajectoryRecordingError, match="larger than the chunk"):
            writer.record(
                ModelCallFinished,
                parent=start.sequence,
                outcome="replied",
                reply=reply,
                tool_calls_requested=tuple("c" * 190 + str(i) for i in range(30)),
            )
        assert store.read_tail(writer.paths) == before
    finally:
        writer.close()


# ---- redelivery (AC 3) ---------------------------------------------------------------------


def test_a_redelivered_identical_event_is_deduplicated(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out", max_events=2)
    run(spec)
    (attempt,) = attempts(spec)
    rolled = attempt.events
    redelivered = b"".join((event.model_dump_json() + "\n").encode() for event in rolled[:3])
    events, torn, broke = parse_tail(redelivered, rolled)
    assert (events, torn, broke) == ((), 0, False)


def test_a_conflicting_event_with_a_recorded_id_breaks_the_tail(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out", max_events=2)
    run(spec)
    (attempt,) = attempts(spec)
    rolled = attempt.events
    changed = rolled[2].model_copy(update={"latency_ms": 999_999})
    assert changed.event_id == rolled[2].event_id
    _, _, broke = parse_tail((changed.model_dump_json() + "\n").encode(), rolled)
    assert broke


# ---- closed attempts (AC 4) ----------------------------------------------------------------


def test_nothing_is_recorded_after_a_seal_or_an_abandon(tmp_path: Path) -> None:
    store = LocalTrajectoryStore(tmp_path / "trajectory")
    writer = _writer(store)
    manifest = writer.abandon("recording-failed")
    assert manifest.completeness == "partial" and manifest.result_digest is None
    with pytest.raises(TrajectoryRecordingError, match="nothing more is recorded"):
        writer.content("late", role="answer")
    with pytest.raises(TrajectoryRecordingError):
        writer.abandon("writer-lost")


def test_a_partial_seal_that_cannot_be_written_still_releases_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk while sealing partial: the lock is released, so reconciliation recovers it."""
    store = LocalTrajectoryStore(tmp_path / "trajectory")
    writer = _writer(store)
    original = writer_module.write_new

    def full_disk(path: Path, data: bytes) -> bool:
        if Path(path) == writer.paths.manifest:
            raise OSError(28, "No space left on device")
        return original(path, data)

    monkeypatch.setattr(writer_module, "write_new", full_disk)
    with pytest.raises(OSError, match="No space left"):
        writer.abandon("recording-failed", error_code="recording-failed")
    fd = os.open(writer.paths.lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if the lock were still held
    finally:
        os.close(fd)
    with pytest.raises(TrajectoryRecordingError, match="released"):
        writer.content("late", role="answer")
    monkeypatch.undo()
    (outcome,) = reconcile_trajectories(store, tmp_path / "agent-run.json")
    assert outcome.manifest is not None and outcome.manifest.completeness == "partial"
    assert outcome.manifest.abandonment == "writer-lost"
