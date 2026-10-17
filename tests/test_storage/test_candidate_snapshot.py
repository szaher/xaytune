"""A node can hold a harness candidate (ADR-018 D10; ADR-019 M6, step 2; migration 021).

A harness node can be recorded, read back and told apart from a training node,
but not run: everything that only a training candidate can do refuses it with
a typed error, before anything is written. Training nodes are unchanged.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from tests.test_core.test_harness import _harness
from tests.test_storage.conftest import make_experiment, make_node, make_run
from xaytune.core.domain.candidate_envelope import (
    CandidateKind,
    ExperimentCandidate,
    NotATrainingCandidateError,
)
from xaytune.core.domain.experiment import CandidateSpecSnapshot, ExperimentNode
from xaytune.core.ids import ExperimentNodeId
from xaytune.core.refs import Actor
from xaytune.storage import write_transaction
from xaytune.storage.control_plane import ControlPlaneRepository
from xaytune.storage.database import connect
from xaytune.storage.errors import StorageError
from xaytune.storage.migrations import available_migrations, migrate

ACTOR = Actor(type="system", id="controller")


def _repository(path: Path) -> ControlPlaneRepository:
    connection = connect(path)
    migrate(connection)
    return ControlPlaneRepository(connection)


def harness_node(experiment, **changes) -> ExperimentNode:
    harness = _harness()
    fields = {
        "id": ExperimentNodeId.generate(),
        "experiment_id": experiment.id,
        "candidate": CandidateSpecSnapshot.of(ExperimentCandidate.agent_harness(harness)),
        "candidate_fingerprint": harness.harness_fingerprint(),
        "created_by": ACTOR,
        **changes,
    }
    return ExperimentNode(**fields)


def _seeded(tmp_path: Path):
    repo = _repository(tmp_path / "state.db")
    experiment = make_experiment()
    repo.create_experiment(experiment, actor=ACTOR)
    return repo, experiment


def _row(repo: ControlPlaneRepository, node: ExperimentNode) -> sqlite3.Row:
    return repo._connection.execute(
        "SELECT candidate_kind, candidate_fingerprint, payload_json FROM experiment_nodes "
        "WHERE id = ?",
        (str(node.id),),
    ).fetchone()


def test_a_harness_node_is_recorded_and_read_back_with_its_kind(tmp_path: Path) -> None:
    repo, experiment = _seeded(tmp_path)
    node = repo.create_node(harness_node(experiment), actor=ACTOR)
    row = _row(repo, node)
    assert row["candidate_kind"] == "agent-harness"
    assert row["candidate_fingerprint"] == _harness().harness_fingerprint()
    assert json.loads(row["payload_json"])["candidate"]["spec_version"] == "2"
    loaded = repo.aggregates.load_node(str(node.id))
    assert loaded == node
    assert loaded.candidate_kind is CandidateKind.AGENT_HARNESS
    assert loaded.candidate.experiment_candidate().payload.harness == _harness()


def test_a_training_node_is_recorded_exactly_as_before(tmp_path: Path) -> None:
    repo, experiment = _seeded(tmp_path)
    node = repo.create_node(make_node(experiment), actor=ACTOR)
    row = _row(repo, node)
    assert row["candidate_kind"] == "training"
    snapshot = json.loads(row["payload_json"])["candidate"]
    assert set(snapshot) == {"candidate", "spec_version", "metadata"}
    assert snapshot["spec_version"] == "1"


def test_a_harness_nodes_fingerprint_must_be_its_harness(tmp_path: Path) -> None:
    repo, experiment = _seeded(tmp_path)
    wrong = harness_node(experiment, candidate_fingerprint="sha256:" + "0" * 64)
    with pytest.raises(StorageError, match="would not describe the proposition"):
        repo.create_node(wrong, actor=ACTOR)


def test_a_harness_node_cannot_be_run(tmp_path: Path) -> None:
    """ADR-019 M1: harness work fails closed, not by accident."""
    repo, experiment = _seeded(tmp_path)
    node = repo.create_node(harness_node(experiment), actor=ACTOR)
    with pytest.raises(NotATrainingCandidateError, match="harness execution is not durable yet"):
        repo.create_run(make_run(node), actor=ACTOR)
    assert repo._connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_training_only_reads_refuse_a_harness_node(tmp_path: Path) -> None:
    repo, experiment = _seeded(tmp_path)
    node = repo.create_node(harness_node(experiment), actor=ACTOR)
    with pytest.raises(NotATrainingCandidateError):
        node.training_candidate()
    with pytest.raises(NotATrainingCandidateError):
        repo.planning_context(str(experiment.id))


def test_the_kind_column_is_the_kind_its_payload_holds_and_never_changes(tmp_path: Path) -> None:
    repo, experiment = _seeded(tmp_path)
    training = repo.create_node(make_node(experiment), actor=ACTOR)
    connection = repo._connection
    with pytest.raises(sqlite3.IntegrityError, match="never changes"):
        connection.execute(
            "UPDATE experiment_nodes SET candidate_kind = 'agent-harness' WHERE id = ?",
            (str(training.id),),
        )
    with pytest.raises(sqlite3.IntegrityError, match="not the kind its candidate holds"):
        connection.execute(
            "INSERT INTO experiment_nodes (id, experiment_id, status, revision, "
            "candidate_fingerprint, candidate_kind, payload_json, created_at, updated_at) "
            "SELECT 'node_forged', experiment_id, status, revision, candidate_fingerprint, "
            "'agent-harness', payload_json, created_at, updated_at FROM experiment_nodes "
            "WHERE id = ?",
            (str(training.id),),
        )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO experiment_nodes (id, experiment_id, status, revision, "
            "candidate_fingerprint, candidate_kind, payload_json, created_at, updated_at) "
            "VALUES ('node_x', ?, 'created', 0, 'f', 'joint', '{}', 'now', 'now')",
            (str(experiment.id),),
        )


def test_existing_nodes_are_training_after_the_migration_and_keep_their_bytes(
    tmp_path: Path,
) -> None:
    """A database from before 021: its nodes become 'training', and no payload is rewritten."""
    older = tmp_path / "older-build"
    older.mkdir()
    for migration in available_migrations():
        if migration.version < 21:
            shutil.copy(migration.path, older / migration.path.name)
    connection = connect(tmp_path / "state.db")
    migrate(connection, older)
    before = ControlPlaneRepository(connection)
    experiment = make_experiment()
    before.create_experiment(experiment, actor=ACTOR)
    node = make_node(experiment)
    with write_transaction(connection):
        # As the build before 021 wrote it: no candidate_kind column yet.
        connection.execute(
            "INSERT INTO experiment_nodes (id, experiment_id, status, revision, "
            "candidate_fingerprint, payload_json, created_at, updated_at) "
            "VALUES (?, ?, 'created', 0, ?, ?, 'now', 'now')",
            (
                str(node.id),
                str(experiment.id),
                node.candidate_fingerprint,
                json.dumps(node.model_dump(mode="json", by_alias=True), sort_keys=True),
            ),
        )
    payload = connection.execute(
        "SELECT payload_json FROM experiment_nodes WHERE id = ?", (str(node.id),)
    ).fetchone()[0]
    assert migrate(connection) == (21,)
    row = connection.execute(
        "SELECT candidate_kind, payload_json FROM experiment_nodes WHERE id = ?", (str(node.id),)
    ).fetchone()
    assert (row["candidate_kind"], row["payload_json"]) == ("training", payload)
    loaded = ControlPlaneRepository(connection).aggregates.load_node(str(node.id))
    assert loaded.candidate_kind is CandidateKind.TRAINING
    assert loaded.training_candidate() == node.training_candidate()


def test_a_build_without_021_still_writes_training_nodes_into_a_migrated_database(
    tmp_path: Path,
) -> None:
    """Compatibility: an older build inserts nodes without the column; they are training."""
    repo, experiment = _seeded(tmp_path)
    node = make_node(experiment)
    connection = repo._connection
    with write_transaction(connection):
        connection.execute(
            "INSERT INTO experiment_nodes (id, experiment_id, status, revision, "
            "candidate_fingerprint, payload_json, created_at, updated_at) "
            "VALUES (?, ?, 'created', 0, ?, ?, 'now', 'now')",
            (
                str(node.id),
                str(experiment.id),
                node.candidate_fingerprint,
                json.dumps(node.model_dump(mode="json", by_alias=True), sort_keys=True),
            ),
        )
    assert _row(repo, node)["candidate_kind"] == "training"
    assert repo.aggregates.load_node(str(node.id)).training_candidate() == node.training_candidate()
