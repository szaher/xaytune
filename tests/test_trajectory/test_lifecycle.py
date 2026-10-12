"""The lifecycle, the commit point and crash recovery (ADR-021 T4; acceptance criterion 4).

Failure scenarios F1-F10, F21-F23 and F25-F27. A *crash* is simulated by
raising :class:`SimulatedCrash` at the chosen point with ``abandon`` disabled,
so nothing after that point runs, as if the process had died; one scenario
kills a real process instead.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any

import pytest

import xaytune.trajectory._files as files
import xaytune.trajectory.store as store_module
import xaytune.trajectory.writer as writer_module
from tests.test_harness.fixtures import SCRIPT, RecordingTools
from tests.test_trajectory.fixtures import (
    SimulatedCrash,
    attempts,
    die,
    only_attempt,
    recorded_spec,
    recording,
    run,
    store_of,
    track_writers,
)
from xaytune.core.trajectory import TrajectoryAttribution, TrajectoryTask
from xaytune.harness import ScriptedHarnessModel, UnattestedResultError
from xaytune.harness.runner import RECORDING_RUNNER
from xaytune.trajectory import (
    LocalTrajectoryStore,
    TrajectoryRecordingError,
    TrajectoryWriter,
    reconcile_trajectories,
)
from xaytune.trajectory._files import UnsafePathError, digest_of
from xaytune.trajectory.reconcile import reconcile_attempt
from xaytune.trajectory.writer import TrajectoryWriter as Writer

TOOL = "kb.search"


def _crash_without_abandon(monkeypatch: pytest.MonkeyPatch) -> None:
    track_writers(monkeypatch)
    monkeypatch.setattr(Writer, "abandon", lambda self, *a, **k: self.close())


def _crash_when(monkeypatch: pytest.MonkeyPatch, name: str, when: Any) -> None:
    """Crash in ``writer_module.<name>`` when *when(*args)* holds."""
    original = getattr(writer_module, name)

    def crashing(*args: Any, **kwargs: Any) -> Any:
        if when(*args):
            raise SimulatedCrash(name)
        return original(*args, **kwargs)

    monkeypatch.setattr(writer_module, name, crashing)
    _crash_without_abandon(monkeypatch)


def _fail_when(monkeypatch: pytest.MonkeyPatch, module: Any, name: str, when: Any) -> None:
    """Make ``module.<name>`` raise ``OSError`` when *when(*args)* holds: a store failure."""
    original = getattr(module, name)

    def failing(*args: Any, **kwargs: Any) -> Any:
        if when(*args):
            raise OSError("injected store failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, failing)


def _named(name: str) -> Any:
    return lambda *args: Path(args[-2] if len(args) > 1 else args[0]).name == name


def _target(name: str) -> Any:
    return lambda *args: Path(args[1]).name == name


# ---- the normal path -------------------------------------------------------------------------


def test_a_recorded_run_seals_complete_and_publishes_its_own_file(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out")
    result = run(spec)
    attempt = only_attempt(spec)
    manifest = attempt.manifest
    assert manifest.completeness == "complete"
    assert manifest.result_digest == digest_of(attempt.location.read_bytes())
    assert files.same_file(attempt.location, attempt.paths.result)
    assert attempt.store.read_tail(attempt.paths) == b""
    assert result.completed


def test_every_start_is_durable_before_its_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A test model and tool read the tail from disk when they are called (write-ahead rule 3)."""
    spec = recorded_spec(tmp_path / "out")
    seen: list[str] = []
    original = ScriptedHarnessModel.respond

    def tail_kind() -> dict[str, Any]:
        (attempt,) = attempts(spec)
        lines = attempt.store.read_tail(attempt.paths).splitlines()
        return json.loads(lines[-1])

    def respond(self: Any, request: Any, *, turn: int) -> Any:
        last = tail_kind()
        assert (last["kind"], last["turn"]) == ("model-call-started", turn)
        seen.append("model")
        return original(self, request, turn=turn)

    class Tools(RecordingTools):
        def search(self, arguments: Any) -> str:
            last = tail_kind()
            assert (last["kind"], last["tool"]) == ("tool-call-started", TOOL)
            seen.append("tool")
            return super().search(arguments)

    monkeypatch.setattr(ScriptedHarnessModel, "respond", respond)
    run(spec, tools=Tools())
    assert seen == ["model", "tool", "model"]


def test_directories_are_synced_after_the_link_and_after_the_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[Path] = []
    original = files.sync_directory
    monkeypatch.setattr(files, "sync_directory", lambda d: (synced.append(Path(d)), original(d)))
    spec = recorded_spec(tmp_path / "out")
    run(spec)
    attempt = only_attempt(spec)
    result_sync = max(i for i, d in enumerate(synced) if d == attempt.location.parent)
    seal_sync = max(i for i, d in enumerate(synced) if d == attempt.paths.directory)
    assert result_sync < seal_sync, "the result's directory is synced at the commit, then the seal"


def test_the_tail_is_named_durably_before_the_first_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first start event is a write-ahead point: the tail's directory entry is synced first."""
    spec = recorded_spec(tmp_path / "out")
    log: list[tuple[str, Path | None, bool]] = []
    original_sync, original_respond = files.sync_directory, ScriptedHarnessModel.respond

    def sync(directory: Path) -> None:
        log.append(("sync", Path(directory), (Path(directory) / "tail.jsonl").exists()))
        original_sync(directory)

    def respond(self: Any, request: Any, *, turn: int) -> Any:
        log.append(("model", None, False))
        return original_respond(self, request, turn=turn)

    monkeypatch.setattr(files, "sync_directory", sync)
    monkeypatch.setattr(ScriptedHarnessModel, "respond", respond)
    run(spec)
    directory = only_attempt(spec).paths.directory
    first_call = log.index(("model", None, False))
    assert any(
        op == "sync" and where == directory and tail for op, where, tail in log[:first_call]
    ), "the attempt directory is synced after the tail exists, before any call"


def test_creating_a_file_by_appending_syncs_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[Path] = []
    original = files.sync_directory
    monkeypatch.setattr(files, "sync_directory", lambda d: (synced.append(Path(d)), original(d)))
    path = tmp_path / "log.jsonl"
    files.append_durably(path, b"first\n")
    assert synced == [tmp_path], "the new file's name is made durable"
    files.append_durably(path, b"second\n")
    assert synced == [tmp_path], "an append to an existing file needs no directory sync"
    assert path.read_bytes() == b"first\nsecond\n"


def test_creating_the_tail_never_reuses_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "tail.jsonl"
    files.create_durably(path)
    assert path.read_bytes() == b"" and oct(path.stat().st_mode & 0o777) == oct(0o600)
    with pytest.raises(FileExistsError):
        files.create_durably(path)


# ---- crashes (F1-F5) -----------------------------------------------------------------------


def test_f1_a_crash_before_the_open_record_is_durable_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")
    called: list[int] = []
    monkeypatch.setattr(
        ScriptedHarnessModel, "respond", lambda self, request, *, turn: called.append(turn)
    )
    _crash_when(monkeypatch, "write_new", lambda path, data: Path(path).name == "open.json")
    with pytest.raises(SimulatedCrash):
        run(spec)
    die()
    assert called == [] and not Path(spec.outputs[0].uri).exists()
    (outcome,) = reconcile_trajectories(store_of(spec), Path(spec.outputs[0].uri))
    assert outcome.outcome == "unreadable" and outcome.manifest is None
    monkeypatch.undo()
    run(spec)
    assert [a.manifest.completeness for a in attempts(spec)[1:]] == ["complete"]


def test_f2_a_crash_during_a_model_call_leaves_a_start_with_no_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")

    def dies(self: Any, request: Any, *, turn: int) -> Any:
        raise SimulatedCrash("the model call never returned")

    monkeypatch.setattr(ScriptedHarnessModel, "respond", dies)
    _crash_without_abandon(monkeypatch)
    with pytest.raises(SimulatedCrash):
        run(spec)
    die()
    (outcome,) = reconcile_trajectories(store_of(spec), Path(spec.outputs[0].uri))
    manifest = outcome.manifest
    assert manifest is not None and manifest.completeness == "partial"
    assert manifest.abandonment == "writer-lost"
    assert [e.kind for e in only_attempt(spec).events] == ["run-started", "model-call-started"]


def test_f2_a_real_process_killed_mid_call_leaves_durable_evidence(tmp_path: Path) -> None:
    """The process really dies (``os._exit``) inside the model call; its start is on disk."""
    output = tmp_path / "out"
    script = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {str(Path.cwd())!r})
        from pathlib import Path
        from tests.test_trajectory.fixtures import recorded_spec, run
        from xaytune.harness import ScriptedHarnessModel
        ScriptedHarnessModel.respond = lambda self, request, *, turn: os._exit(9)
        run(recorded_spec(Path({str(output)!r})))
        """
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 9, done.stderr
    spec = recorded_spec(output)
    (outcome,) = reconcile_trajectories(store_of(spec), Path(spec.outputs[0].uri))
    assert outcome.outcome == "sealed", "the dead process's lock was released"
    assert [e.kind for e in only_attempt(spec).events] == ["run-started", "model-call-started"]


def test_f3_a_torn_final_line_is_discarded_and_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")
    monkeypatch.setattr(
        ScriptedHarnessModel,
        "respond",
        lambda self, request, *, turn: (_ for _ in ()).throw(SimulatedCrash("x")),
    )
    _crash_without_abandon(monkeypatch)
    with pytest.raises(SimulatedCrash):
        run(spec)
    die()
    (attempt,) = attempts(spec)
    files.append_durably(attempt.paths.tail, b'{"kind":"model-call-fini')
    (outcome,) = reconcile_trajectories(store_of(spec), attempt.location)
    assert outcome.manifest is not None
    assert outcome.manifest.torn_tail_bytes == len(b'{"kind":"model-call-fini')
    assert outcome.manifest.abandonment == "writer-lost"
    assert len(attempt.events) == 2


def test_f4_a_crash_before_the_commit_point_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")
    location = Path(spec.outputs[0].uri)
    _crash_when(monkeypatch, "link_new", lambda source, target: Path(target) == location)
    with pytest.raises(SimulatedCrash):
        run(spec)
    die()
    assert not location.exists()
    (outcome,) = reconcile_trajectories(store_of(spec), location)
    assert outcome.manifest is not None
    assert outcome.manifest.abandonment == "result-unpublished"
    assert outcome.manifest.result_digest is None
    monkeypatch.undo()
    first_events = len(only_attempt(spec).events)
    run(spec)
    old, new = attempts(spec)
    assert len(old.events) == first_events, "nothing appends to the abandoned attempt"
    assert new.manifest.completeness == "complete"


def test_f5_a_crash_after_the_commit_point_is_sealed_complete_by_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")
    _crash_when(
        monkeypatch, "link_new", lambda source, target: Path(target).name == "manifest.json"
    )
    with pytest.raises(SimulatedCrash):
        run(spec)
    die()
    location = Path(spec.outputs[0].uri)
    assert location.exists(), "the result was committed"
    (outcome,) = reconcile_trajectories(store_of(spec), location)
    assert outcome.manifest is not None and outcome.manifest.completeness == "complete"


def _fail_after_the_link(monkeypatch: pytest.MonkeyPatch, location: Path) -> list[bool]:
    """Fail the first directory sync after the result's link exists: ``os.link`` succeeded."""
    failed: list[bool] = []
    original = files.sync_directory

    def sync(directory: Path) -> None:
        if not failed and Path(directory) == location.parent and location.exists():
            failed.append(True)
            raise OSError("injected failure after the link")
        original(directory)

    monkeypatch.setattr(files, "sync_directory", sync)
    return failed


def test_a_failure_right_after_a_successful_link_is_past_the_commit_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The link exists, the commit raised: the writer is committed and nothing is abandoned."""
    spec = recorded_spec(tmp_path / "out")
    location = Path(spec.outputs[0].uri)
    failed = _fail_after_the_link(monkeypatch, location)
    committed: list[bool] = []
    original_commit = Writer.commit

    def commit(self: Any, where: Path) -> bool:
        try:
            return original_commit(self, where)
        finally:
            committed.append(self.committed)

    monkeypatch.setattr(Writer, "commit", commit)
    with pytest.raises(TrajectoryRecordingError, match="committed but its trajectory was not"):
        run(spec)
    assert failed == [True], "the failure came after os.link() succeeded"
    assert committed == [True], "the writer follows the filesystem, not the exception"
    (attempt,) = attempts(spec)
    assert files.same_file(location, attempt.paths.result)
    assert not attempt.paths.manifest.exists(), "never sealed partial as a pre-commit failure"
    monkeypatch.undo()
    (outcome,) = reconcile_trajectories(store_of(spec), location)
    assert outcome.manifest is not None and outcome.manifest.completeness == "complete"
    assert run(spec).completed, "replay returns the attested result"
    assert len(attempts(spec)) == 1, "and opens no new attempt"


def test_a_failure_after_the_link_is_completed_by_replay_without_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")
    location = Path(spec.outputs[0].uri)
    _fail_after_the_link(monkeypatch, location)
    with pytest.raises(TrajectoryRecordingError):
        run(spec)
    monkeypatch.undo()
    assert run(spec).completed
    attempt = only_attempt(spec)
    assert attempt.manifest.completeness == "complete"
    assert attempt.manifest.result_digest == digest_of(location.read_bytes())


# ---- concurrency (F6-F9) -------------------------------------------------------------------


def test_f6_racing_reconcilers_seal_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = recorded_spec(tmp_path / "out")
    monkeypatch.setattr(
        ScriptedHarnessModel,
        "respond",
        lambda self, request, *, turn: (_ for _ in ()).throw(SimulatedCrash("x")),
    )
    _crash_without_abandon(monkeypatch)
    with pytest.raises(SimulatedCrash):
        run(spec)
    die()
    store, location = store_of(spec), Path(spec.outputs[0].uri)
    (paths,) = store.attempts()
    outcomes: list[Any] = []
    barrier = threading.Barrier(4)

    def reconcile() -> None:
        barrier.wait()
        outcomes.append(reconcile_attempt(store, paths, location))

    threads = [threading.Thread(target=reconcile) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    sealed = [o for o in outcomes if o.outcome == "sealed"]
    assert len(sealed) <= 1 and {o.outcome for o in outcomes} <= {
        "sealed",
        "skipped-live",
        "already-sealed",
    }
    final = reconcile_attempt(store, paths, location)
    assert final.outcome in ("already-sealed", "sealed")
    assert paths.manifest.exists()


def test_f7_a_live_writer_is_never_reconciled(tmp_path: Path) -> None:
    store = LocalTrajectoryStore(tmp_path / "out" / "trajectory")
    writer = TrajectoryWriter.open(
        store,
        attribution=TrajectoryAttribution(
            request_digest="sha256:" + "1" * 64,
            execution_fingerprint="sha256:" + "2" * 64,
            harness_fingerprint="sha256:" + "3" * 64,
            task=TrajectoryTask(task_id="support/refund-001"),
            seed=7,
            replicate=0,
        ),
        recording=recording(),
        runner=RECORDING_RUNNER,
    )
    try:
        (outcome,) = reconcile_trajectories(store, tmp_path / "out" / "agent-run.json")
        assert outcome.outcome == "skipped-live"
    finally:
        writer.close()
    (outcome,) = reconcile_trajectories(store, tmp_path / "out" / "agent-run.json")
    assert outcome.outcome == "sealed" and outcome.manifest is not None
    assert outcome.manifest.abandonment == "writer-lost"


def test_f8_only_the_attempt_that_published_seals_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte-identical results: a digest cannot say who published; link identity can."""
    spec = recorded_spec(tmp_path / "out")
    location = Path(spec.outputs[0].uri)
    _crash_when(monkeypatch, "link_new", lambda source, target: Path(target) == location)
    with pytest.raises(SimulatedCrash):
        run(spec)
    die()
    monkeypatch.undo()
    run(spec)  # a second execution publishes the byte-identical result
    loser, winner = attempts(spec)
    assert loser.paths.result.read_bytes() == location.read_bytes()
    reconcile_trajectories(store_of(spec), location)
    assert winner.manifest.completeness == "complete"
    assert loser.manifest.abandonment == "result-published-by-another-attempt"


def test_f8_a_live_commit_that_finds_the_location_taken_seals_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")
    location = Path(spec.outputs[0].uri)
    original = Writer.commit

    def taken(self: Any, target: Path) -> bool:
        target.write_bytes(self.paths.result.read_bytes())  # someone else's identical bytes
        return original(self, target)

    monkeypatch.setattr(Writer, "commit", taken)
    with pytest.raises(UnattestedResultError, match="no attempt instance published"):
        run(spec)
    assert only_attempt(spec).manifest.abandonment == "result-published-by-another-attempt"
    assert location.exists()


def test_f9_replaying_a_published_result_opens_no_attempt(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out")
    first = run(spec)
    model = ScriptedHarnessModel(SCRIPT)
    from xaytune.harness import ScriptedHarnessRunner

    again = ScriptedHarnessRunner().run(
        spec,
        model=model,
        toolbox=RecordingTools().toolbox(),
        authorized_tools={"kb.search", "tickets.close"},
    )
    assert again == first and model.requests == ()
    assert len(attempts(spec)) == 1


# ---- the commit point (F10, F21, F22) ------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "name", "when"),
    [
        pytest.param(writer_module, "append_durably", lambda path, data: True, id="tail-append"),
        pytest.param(
            store_module, "write_new", lambda path, data: "blobs" in Path(path).parts, id="blob"
        ),
        pytest.param(
            store_module,
            "write_new",
            lambda path, data: "chunks" in Path(path).parts,
            id="chunk",
        ),
        pytest.param(store_module, "replace_atomically", lambda path, data: True, id="progress"),
        pytest.param(
            writer_module,
            "write_new",
            lambda path, data: Path(path).name == "result.json",
            id="result-copy",
        ),
        pytest.param(
            writer_module,
            "write_new",
            lambda path, data: Path(path).name == "manifest.staged.json",
            id="staged-manifest",
        ),
    ],
)
def test_f10_a_store_failure_before_the_commit_point_publishes_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: Any, name: str, when: Any
) -> None:
    spec = recorded_spec(tmp_path / "out")
    _fail_when(monkeypatch, module, name, when)
    with pytest.raises((TrajectoryRecordingError, OSError)):
        run(spec)
    monkeypatch.undo()
    assert not Path(spec.outputs[0].uri).exists()
    reconcile_trajectories(store_of(spec), Path(spec.outputs[0].uri))
    manifest = only_attempt(spec).manifest
    assert manifest.completeness == "partial" and manifest.result_digest is None


def test_f10_staged_evidence_that_does_not_verify_publishes_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")
    monkeypatch.setattr(writer_module, "verify_trajectory", lambda *a: ("injected",))
    with pytest.raises(TrajectoryRecordingError, match="does not verify"):
        run(spec)
    assert not Path(spec.outputs[0].uri).exists()
    assert only_attempt(spec).manifest.abandonment == "recording-failed"


def test_f21_a_seal_failure_after_the_commit_is_completed_before_replay_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")
    _fail_when(monkeypatch, writer_module, "link_new", lambda s, t: Path(t).name == "manifest.json")
    with pytest.raises(TrajectoryRecordingError, match="committed"):
        run(spec)
    monkeypatch.undo()
    (attempt,) = attempts(spec)
    assert attempt.location.exists() and not attempt.paths.manifest.exists()
    result = run(spec)  # replay reconciles the winning attempt first
    assert attempt.manifest.completeness == "complete"
    assert result.completed


def test_f22_evidence_lost_after_the_commit_leaves_an_unattested_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = recorded_spec(tmp_path / "out")
    _fail_when(monkeypatch, writer_module, "link_new", lambda s, t: Path(t).name == "manifest.json")
    with pytest.raises(TrajectoryRecordingError):
        run(spec)
    monkeypatch.undo()
    (attempt,) = attempts(spec)
    for chunk in attempt.paths.chunks.iterdir():
        chunk.unlink()
    with pytest.raises(UnattestedResultError, match="partial"):
        run(spec)
    manifest = attempt.manifest
    assert manifest.abandonment == "post-commit-evidence-lost"
    assert manifest.unattested_result == digest_of(attempt.location.read_bytes())
    incident = json.loads(attempt.paths.incident.read_text())
    assert incident["kind"] == "post-commit-evidence-lost"


def test_f23_replay_refuses_a_result_whose_trajectory_was_tampered_with(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out")
    run(spec)
    (chunk,) = only_attempt(spec).paths.chunks.iterdir()
    os.chmod(chunk, 0o600)
    chunk.write_bytes(chunk.read_bytes().replace(b"model-call", b"model-cal_"))
    with pytest.raises(UnattestedResultError, match="digest"):
        run(spec)


def test_f25_a_result_no_attempt_claims_is_refused(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out")
    run(spec)
    location = Path(spec.outputs[0].uri)
    copy = location.read_bytes()
    location.unlink()
    location.write_bytes(copy)  # the same bytes, another inode
    with pytest.raises(UnattestedResultError, match="no attempt instance published"):
        run(spec)


def test_f26_a_symlinked_trajectory_directory_is_refused_before_anything_runs(
    tmp_path: Path,
) -> None:
    spec = recorded_spec(tmp_path / "out")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "trajectory").symlink_to(elsewhere)
    with pytest.raises(UnsafePathError):
        run(spec)
    assert not Path(spec.outputs[0].uri).exists()
    assert list(elsewhere.iterdir()) == []


def test_f26_a_symlinked_result_location_is_never_written_through(tmp_path: Path) -> None:
    spec = recorded_spec(tmp_path / "out")
    target = tmp_path / "target.json"
    location = Path(spec.outputs[0].uri)
    location.parent.mkdir(parents=True)
    location.symlink_to(target)
    with pytest.raises(Exception):
        run(spec)
    assert not target.exists()


def test_f27_a_result_on_another_filesystem_is_refused_before_the_attempt_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xaytune.harness.runner as runner_module

    spec = recorded_spec(tmp_path / "out")
    store_root = Path(spec.outputs[1].uri)
    real = runner_module.device_of
    monkeypatch.setattr(
        runner_module, "device_of", lambda p: real(p) + (1 if Path(p) == store_root else 0)
    )
    with pytest.raises(UnsafePathError, match="different filesystems"):
        run(spec)
    assert attempts(spec) == []
