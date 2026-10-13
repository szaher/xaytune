"""ADR-022 V3: one replicate's evidence, assembled from the store and verified against it.

Every negative case either breaks the store or edits a record and recomputes
its digest: a self-consistent record is still refused when it is not what the
store holds.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

import xaytune.trajectory.writer as writer_module
from tests.test_harness.fixtures import SCRIPT
from tests.test_harness_evaluation import fixtures as fx
from xaytune.core.harness_evaluation import (
    HarnessRunEvidence,
    HarnessRunEvidenceBody,
    run_evidence_digest,
)
from xaytune.core.trajectory import ChunkRef, TrajectoryManifestBody, seal
from xaytune.harness import UnattestedResultError
from xaytune.harness.evidence import (
    EvidenceRefusedError,
    HarnessEvidenceStore,
    assemble_evidence,
    verify_run_evidence,
)
from xaytune.trajectory import LocalTrajectoryStore, SecretDetectedError, TrajectoryWriter
from xaytune.trajectory.sealing import read_manifest
from xaytune.trajectory.store import AttemptPaths

SECRET = "five days"
"""The scripted answer contains it: registering it makes an attempt seal partial."""


class World:
    def __init__(self, root: Path) -> None:
        self.suite = fx.suite()
        self.source = fx.source()
        self.assignment = fx.assignment(the_suite=self.suite)
        self.store = HarnessEvidenceStore(root)
        self.specs = fx.replicate_specs(self.store, self.suite, self.assignment)

    def spec(self, task_id: str) -> Any:
        return next(s for s in self.specs if s.task.task_id == task_id)

    def trajectories(self, task_id: str) -> LocalTrajectoryStore:
        return LocalTrajectoryStore(Path(self.spec(task_id).outputs[1].uri))

    def attempts(self, task_id: str) -> tuple[AttemptPaths, ...]:
        return self.trajectories(task_id).attempts()

    def assemble(self, **options: Any) -> HarnessRunEvidence:
        return assemble_evidence(
            self.specs,
            self.suite,
            fx.AllTasks(),
            self.assignment,
            self.source,
            self.store,
            **options,
        )

    def verify(self, evidence: HarnessRunEvidence) -> HarnessRunEvidence:
        return verify_run_evidence(evidence, self.suite, self.source, self.store)


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path / "replicate")


def _run_all(world: World, **per_task: Any) -> None:
    for spec in world.specs:
        handler = per_task.get(spec.task.task_id)
        if handler is None:
            fx.run(spec)
        else:
            handler(spec)


def _partial(spec: Any) -> None:
    with pytest.raises(SecretDetectedError):
        fx.run(spec, trajectory_secrets={SECRET})


def _retried(spec: Any) -> None:
    _partial(spec)
    fx.run(spec)


def _rehash(evidence: HarnessRunEvidence, **changes: Any) -> HarnessRunEvidence:
    """*evidence* edited, with its digest recomputed: self-consistent, and still a lie."""
    fields = {name: getattr(evidence, name) for name in HarnessRunEvidenceBody.model_fields}
    fields.update(changes)
    body = HarnessRunEvidenceBody(**fields)
    return HarnessRunEvidence(**fields, evidence_digest=run_evidence_digest(body))


def _with_entry(evidence: HarnessRunEvidence, task_id: str, **changes: Any) -> HarnessRunEvidence:
    entries = tuple(
        e.model_copy(update=changes) if e.task_id == task_id else e for e in evidence.entries
    )
    return _rehash(evidence, entries=entries)


def _refused(call: Any, *needles: str) -> tuple[str, ...]:
    with pytest.raises(EvidenceRefusedError) as caught:
        call()
    for needle in needles:
        assert any(needle in reason for reason in caught.value.reasons), (
            needle,
            caught.value.reasons,
        )
    return caught.value.reasons


def _corrupt(path: Path) -> None:
    os.chmod(path, 0o600)
    path.write_bytes(path.read_bytes().replace(b"model-call", b"model-cal_"))


# ---- positive ------------------------------------------------------------------------------


class TestAssembly:
    def test_every_task_attested_in_assignment_order(self, world: World) -> None:
        _run_all(world)
        evidence = world.assemble()
        assert [e.task_id for e in evidence.entries] == list(world.assignment.task_ids)
        for entry in evidence.entries:
            assert entry.state == "attested"
            assert len(entry.attempts) == 1 and entry.attempts[0].completeness == "complete"
            assert entry.outcome is not None
            assert entry.outcome.trajectory_id == entry.attempts[0].trajectory_id
        assert evidence.evidence_digest == run_evidence_digest(evidence)
        assert evidence.seed == world.assignment.seed

    def test_the_digest_is_stable(self, world: World) -> None:
        _run_all(world)
        first, second = world.assemble(), world.assemble()
        assert first == second
        assert HarnessRunEvidence.model_validate_json(first.model_dump_json()) == first

    def test_the_facets_and_configuration_are_recorded(self, world: World) -> None:
        _run_all(world)
        evidence = world.assemble(timing_environment="ci-macos-14")
        assert evidence.facets.timing_environment == "ci-macos-14"
        assert evidence.facets.runner_name == "xaytune.scripted-harness-runner"
        assert evidence.token_budget == world.specs[0].harness.stopping.max_total_tokens

    def test_g3_every_attempt_partial_is_a_valid_no_attested_result(self, world: World) -> None:
        _run_all(world, **{"support/refund-001": _partial})
        evidence = world.assemble()
        entry = evidence.entries[0]
        assert entry.state == "no-attested-result" and entry.outcome is None
        assert [a.abandonment for a in entry.attempts] == ["secret-detected"]

    def test_a_retry_is_recorded_and_the_complete_attempt_is_the_outcome(
        self, world: World
    ) -> None:
        _run_all(world, **{"support/refund-001": _retried})
        entry = world.assemble().entries[0]
        assert entry.state == "attested"
        assert sorted(a.completeness for a in entry.attempts) == ["complete", "partial"]
        complete = next(a for a in entry.attempts if a.completeness == "complete")
        assert entry.outcome is not None and entry.outcome.trajectory_id == complete.trajectory_id

    def test_g3e_a_verified_post_commit_evidence_loss_is_no_attested_result(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _lose_evidence(world, monkeypatch)
        entry = world.assemble().entries[0]
        assert entry.state == "no-attested-result"
        assert [a.abandonment for a in entry.attempts] == ["post-commit-evidence-lost"]

    def test_no_reference_reaches_the_evidence(self, world: World) -> None:
        _run_all(world)
        text = world.assemble().model_dump_json()
        for task in world.suite.tasks:
            assert task.reference is not None
            assert task.reference.uri not in text and task.reference.digest not in text
        assert "Orders ship within two days." not in text

    def test_verification_writes_nothing(self, world: World) -> None:
        _run_all(world)
        evidence = world.assemble()
        before = _tree(world.store.root)
        world.verify(evidence)
        assert _tree(world.store.root) == before


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(root.rglob("*"))
    }


def _lose_evidence(world: World, monkeypatch: pytest.MonkeyPatch) -> AttemptPaths:
    """Commit refund-001's result, fail the seal, then lose its chunks (ADR-021 F22)."""
    spec = world.spec("support/refund-001")
    original = writer_module.link_new

    def failing(*args: Any, **kwargs: Any) -> Any:
        if Path(args[-1]).name == "manifest.json":
            raise OSError("injected")
        return original(*args, **kwargs)

    monkeypatch.setattr(writer_module, "link_new", failing)
    with pytest.raises(Exception):  # noqa: B017 - the seal fails after the commit
        fx.run(spec)
    monkeypatch.undo()
    (paths,) = world.attempts("support/refund-001")
    for chunk in paths.chunks.iterdir():
        chunk.unlink()
    with pytest.raises(UnattestedResultError):
        fx.run(spec)
    for other in world.specs[1:]:
        fx.run(other)
    return paths


# ---- refused at assembly -----------------------------------------------------------------


class TestAssemblyRefuses:
    def test_g3c_an_open_attempt(self, world: World) -> None:
        _run_all(world)
        spec = world.spec("support/refund-002")
        (sealed,) = world.attempts("support/refund-002")
        record = world.trajectories("support/refund-002").read_open_record(sealed)
        assert record is not None
        writer = TrajectoryWriter.open(
            world.trajectories("support/refund-002"),
            attribution=record.attribution,
            recording=record.recording,
            runner=record.runner,
        )
        writer.close()
        assert spec is not None
        _refused(world.assemble, "still open")

    def test_g3d_a_partial_attempt_that_does_not_verify(self, world: World) -> None:
        _run_all(world, **{"support/refund-001": _retried})
        partial = next(
            p
            for p in world.attempts("support/refund-001")
            if (m := read_manifest(p.manifest)) and m.completeness == "partial"
        )
        for chunk in partial.chunks.iterdir():
            _corrupt(chunk)
        _refused(world.assemble, "does not verify")

    def test_g3e_prime_an_unverifiable_evidence_loss_prefix(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths = _lose_evidence(world, monkeypatch)
        manifest = read_manifest(paths.manifest)
        assert manifest is not None
        forged = ChunkRef(
            index=0,
            first_sequence=0,
            last_sequence=0,
            digest="sha256:" + "0" * 64,
            bytes=10,
            events=1,
        )
        body = TrajectoryManifestBody(
            **{
                name: getattr(manifest, name)
                for name in TrajectoryManifestBody.model_fields
                if name != "chunks"
            },
            chunks=(forged,),
        )
        os.chmod(paths.manifest, 0o600)
        paths.manifest.write_text(seal(body).model_dump_json())
        _refused(world.assemble, "does not verify")

    def test_g4_an_edited_chunk(self, world: World) -> None:
        _run_all(world)
        (paths,) = world.attempts("support/refund-002")
        for chunk in paths.chunks.iterdir():
            _corrupt(chunk)
        _refused(world.assemble, "does not verify")

    def test_a_quarantined_attempt(self, world: World) -> None:
        _run_all(world, **{"support/refund-001": _retried})
        store = world.trajectories("support/refund-001")
        partial = next(
            p
            for p in store.attempts()
            if (m := read_manifest(p.manifest)) and m.completeness == "partial"
        )
        store.contain(partial)
        _refused(world.assemble, "quarantined")

    def test_specs_out_of_assignment_order(self, world: World) -> None:
        _run_all(world)
        world.specs.reverse()
        _refused(world.assemble, "assignment order")


# ---- refused at verification: records edited after assembly, digests recomputed -----------


@pytest.fixture
def retried(world: World) -> HarnessRunEvidence:
    _run_all(world, **{"support/refund-001": _retried})
    return world.assemble()


class TestVerificationRefuses:
    def test_g3g_an_entry_with_no_attempts(self, world: World, retried: HarnessRunEvidence) -> None:
        edited = _with_entry(retried, "support/refund-002", attempts=())
        _refused(lambda: world.verify(edited), "lists no attempts")

    def test_g3g_an_empty_no_attested_result_entry(
        self, world: World, retried: HarnessRunEvidence
    ) -> None:
        edited = _with_entry(
            retried, "support/refund-002", attempts=(), state="no-attested-result", outcome=None
        )
        _refused(lambda: world.verify(edited), "lists no attempts")

    def test_g3f_a_retry_left_out(self, world: World, retried: HarnessRunEvidence) -> None:
        entry = retried.entries[0]
        kept = tuple(a for a in entry.attempts if a.completeness == "complete")
        edited = _with_entry(retried, entry.task_id, attempts=kept)
        _refused(lambda: world.verify(edited), "left out")

    def test_an_attempt_added(self, world: World, retried: HarnessRunEvidence) -> None:
        entry = retried.entries[1]
        extra = entry.attempts[0].model_copy(update={"trajectory_id": "att_" + "f" * 26})
        edited = _with_entry(retried, entry.task_id, attempts=(*entry.attempts, extra))
        _refused(lambda: world.verify(edited), "not in the store")

    def test_g3a_an_attested_claim_on_a_partial_attempt(
        self, world: World, retried: HarnessRunEvidence
    ) -> None:
        entry = retried.entries[0]
        partial = next(a for a in entry.attempts if a.completeness == "partial")
        assert entry.outcome is not None
        outcome = entry.outcome.model_copy(
            update={
                "trajectory_id": partial.trajectory_id,
                "trajectory_fingerprint": partial.trajectory_fingerprint,
            }
        )
        edited = _with_entry(retried, entry.task_id, outcome=outcome)
        reasons = _refused(lambda: world.verify(edited), "sealed partial")
        assert not any("no attested result" in r for r in reasons), "never downgraded"

    def test_g3b_a_result_declared_absent(self, world: World, retried: HarnessRunEvidence) -> None:
        edited = _with_entry(
            retried, "support/refund-002", state="no-attested-result", outcome=None
        )
        _refused(lambda: world.verify(edited), "cannot be declared absent")

    def test_another_seed(self, world: World, retried: HarnessRunEvidence) -> None:
        _refused(lambda: world.verify(_rehash(retried, seed=12)), "seed")

    def test_another_cohort(self, world: World, retried: HarnessRunEvidence) -> None:
        assignment = retried.assignment.model_copy(update={"cohort": "sha256:" + "1" * 64})
        _refused(lambda: world.verify(_rehash(retried, assignment=assignment)), "cohort")

    def test_g5_another_task_fingerprint(self, world: World, retried: HarnessRunEvidence) -> None:
        edited = _with_entry(retried, "support/refund-002", task_fingerprint="sha256:" + "2" * 64)
        _refused(lambda: world.verify(edited), "task fingerprint")

    def test_a_missing_task(self, world: World, retried: HarnessRunEvidence) -> None:
        edited = _rehash(retried, entries=retried.entries[:2])
        _refused(lambda: world.verify(edited), "missing")

    def test_a_repeated_task(self, world: World, retried: HarnessRunEvidence) -> None:
        entries = (*retried.entries, retried.entries[1])
        _refused(lambda: world.verify(_rehash(retried, entries=entries)), "more than one entry")

    def test_reordered_tasks(self, world: World, retried: HarnessRunEvidence) -> None:
        entries = tuple(reversed(retried.entries))
        _refused(lambda: world.verify(_rehash(retried, entries=entries)), "assignment order")

    def test_g4_a_chunk_edited_after_assembly(
        self, world: World, retried: HarnessRunEvidence
    ) -> None:
        (paths,) = world.attempts("support/refund-002")
        for chunk in paths.chunks.iterdir():
            _corrupt(chunk)
        reasons = _refused(lambda: world.verify(retried), "does not verify")
        assert any(paths.trajectory_id in r for r in reasons)

    def test_an_edit_without_a_new_digest(self, world: World, retried: HarnessRunEvidence) -> None:
        edited = retried.model_copy(update={"replicate": 3})
        _refused(lambda: world.verify(edited), "digest")

    def test_another_store(self, world: World, retried: HarnessRunEvidence, tmp_path: Path) -> None:
        elsewhere = HarnessEvidenceStore(tmp_path / "empty")
        _refused(
            lambda: verify_run_evidence(retried, world.suite, world.source, elsewhere),
            "not in the store",
        )

    def test_a_suite_whose_bytes_changed(self, world: World, retried: HarnessRunEvidence) -> None:
        held = fx.source()
        broken = fx.MappingBenchmarkSource(manifests=(fx.manifest(),), tasks=fx.tasks(), blobs={})
        assert held is not None
        _refused(lambda: verify_run_evidence(retried, world.suite, broken, world.store), "suite")

    def test_a_deleted_result(self, world: World, retried: HarnessRunEvidence) -> None:
        location = world.store.result_location(1)
        location.unlink()
        _refused(lambda: world.verify(retried), "not published here")


def test_a_whole_attempt_directory_deleted_is_the_documented_limit(world: World) -> None:
    """V3's limit: a deleted attempt directory cannot be listed, so it is not detected."""
    _run_all(world, **{"support/refund-001": _retried})
    store = world.trajectories("support/refund-001")
    partial = next(
        p
        for p in store.attempts()
        if (m := read_manifest(p.manifest)) and m.completeness == "partial"
    )
    shutil.rmtree(partial.directory)
    entry = world.assemble().entries[0]
    assert len(entry.attempts) == 1, "the durable path's attempt ordinals close this gap"


def test_a_script_is_reused_for_every_task() -> None:
    assert SCRIPT[-1]["text"] == fx.ANSWER


def test_a_symlinked_task_directory_is_refused(world: World, tmp_path: Path) -> None:
    _run_all(world)
    evidence = world.assemble()
    task = world.store.task_output(2)
    moved = tmp_path / "elsewhere"
    task.rename(moved)
    task.symlink_to(moved)
    with pytest.raises(ValueError, match="symlink"):
        world.verify(evidence)
