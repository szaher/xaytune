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
from typing import Any

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
from xaytune.core.state.status import ExperimentNodeStatus
from xaytune.storage import write_transaction
from xaytune.storage.control_plane import ControlPlaneRepository
from xaytune.storage.database import connect
from xaytune.storage.errors import NodeRecordMismatchError, StorageError
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


# ---- integrity on update and on read (review of #90) --------------------------------------------


def _payload(repo: ControlPlaneRepository, node: ExperimentNode) -> dict:
    return json.loads(_row(repo, node)["payload_json"])


def _write_payload(repo: ControlPlaneRepository, node: ExperimentNode, payload: dict) -> None:
    repo._connection.execute(
        "UPDATE experiment_nodes SET payload_json = ? WHERE id = ?",
        (json.dumps(payload, sort_keys=True), str(node.id)),
    )


def test_an_ordinary_transition_of_either_kind_still_writes(tmp_path: Path) -> None:
    repo, experiment = _seeded(tmp_path)
    for node in (make_node(experiment), harness_node(experiment)):
        created = repo.create_node(node, actor=ACTOR)
        moved = repo.transition_node(
            created.id,
            expected_revision=created.revision,
            new_status=ExperimentNodeStatus.PLANNED,
            actor=ACTOR,
        )
        assert repo.aggregates.load_node(str(moved.id)) == moved


def test_an_update_cannot_change_a_nodes_candidate_identity(tmp_path: Path) -> None:
    repo, experiment = _seeded(tmp_path)
    node = repo.create_node(make_node(experiment), actor=ACTOR)
    other = make_node(experiment, fingerprint="b")
    with pytest.raises(sqlite3.IntegrityError, match="candidate_fingerprint never changes"):
        repo._connection.execute(
            "UPDATE experiment_nodes SET candidate_fingerprint = ? WHERE id = ?",
            (other.candidate_fingerprint, str(node.id)),
        )
    replaced = {
        **_payload(repo, node),
        "candidate": json.loads(other.model_dump_json(by_alias=True))["candidate"],
        "candidate_fingerprint": other.candidate_fingerprint,
    }
    with pytest.raises(sqlite3.IntegrityError, match="cannot change its candidate"):
        _write_payload(repo, node, replaced)
    assert repo.aggregates.load_node(str(node.id)) == node


def test_an_update_cannot_turn_a_training_payload_into_a_harness(tmp_path: Path) -> None:
    """Even with the old fingerprint copied over, the payload's kind is checked."""
    repo, experiment = _seeded(tmp_path)
    node = repo.create_node(make_node(experiment), actor=ACTOR)
    harness = json.loads(harness_node(experiment).model_dump_json(by_alias=True))["candidate"]
    swapped = {**_payload(repo, node), "candidate": harness}
    assert swapped["candidate_fingerprint"] == node.candidate_fingerprint
    with pytest.raises(sqlite3.IntegrityError, match="cannot change its candidate"):
        _write_payload(repo, node, swapped)
    assert repo.aggregates.load_node(str(node.id)).candidate_kind is CandidateKind.TRAINING


# ---- the snapshot version fails closed (review of #90, second round) -----------------------------


def _unknown_snapshots(training: dict, harness: dict) -> dict[str, Any]:
    """Every way a snapshot can be one this schema does not know, keyed by what is wrong.

    The version cases change the node's own snapshot; the shape cases are built
    from a training and a harness snapshot, whatever the node's kind.
    """

    def version(value: Any) -> Any:
        return lambda own: {**own, "spec_version": value}

    envelope = harness.get("envelope", {})
    unkind = {**envelope, "payload": {"harness": envelope.get("payload", {}).get("harness")}}
    return {
        "missing": lambda own: {k: v for k, v in own.items() if k != "spec_version"},
        "unknown": version("3"),
        "empty": version(""),
        "null": version(None),
        "a number": version(1),
        "an object": version({"version": "1"}),
        "1 also holding an envelope": lambda own: {**training, "envelope": envelope},
        "1 without its candidate": lambda own: {
            k: v for k, v in training.items() if k != "candidate"
        },
        "2 holding a spec": lambda own: {**harness, "candidate": training["candidate"]},
        "2 whose envelope names no kind": lambda own: {**harness, "envelope": unkind},
        "not a snapshot": lambda own: "training",
    }


_UNKNOWN = sorted(_unknown_snapshots({}, {}))


def _stored(repo: ControlPlaneRepository) -> list[tuple]:
    """Every node row exactly as stored."""
    return [
        tuple(row)
        for row in repo._connection.execute(
            "SELECT id, candidate_kind, candidate_fingerprint, payload_json, revision "
            "FROM experiment_nodes ORDER BY id"
        )
    ]


def _with_snapshot(repo: ControlPlaneRepository, node: ExperimentNode, problem: str) -> dict:
    experiment = make_experiment()
    training = json.loads(make_node(experiment).model_dump_json(by_alias=True))["candidate"]
    harness = json.loads(harness_node(experiment).model_dump_json(by_alias=True))["candidate"]
    payload = _payload(repo, node)
    snapshot = _unknown_snapshots(training, harness)[problem](payload["candidate"])
    return {**payload, "candidate": snapshot}


@pytest.mark.parametrize("problem", _UNKNOWN)
def test_a_snapshot_of_an_unknown_version_is_never_inserted(tmp_path: Path, problem: str) -> None:
    """Refused, not read as training -- and the records already there are untouched."""
    repo, experiment = _seeded(tmp_path)
    node = repo.create_node(make_node(experiment), actor=ACTOR)
    before = _stored(repo)
    forged = _with_snapshot(repo, node, problem)
    with pytest.raises(sqlite3.IntegrityError, match="not a version this schema knows"):
        with write_transaction(repo._connection):
            repo._connection.execute(
                "INSERT INTO experiment_nodes (id, experiment_id, status, revision, "
                "candidate_fingerprint, candidate_kind, payload_json, created_at, updated_at) "
                "VALUES ('node_forged', ?, 'created', 0, ?, 'training', ?, 'now', 'now')",
                (
                    str(experiment.id),
                    node.candidate_fingerprint,
                    json.dumps(forged, sort_keys=True),
                ),
            )
    assert _stored(repo) == before
    assert repo.aggregates.nodes_for_experiment(str(experiment.id)) == (node,)


@pytest.mark.parametrize("problem", _UNKNOWN)
@pytest.mark.parametrize("kind", ["training", "agent-harness"])
def test_a_snapshot_cannot_be_updated_to_an_unknown_version(
    tmp_path: Path, problem: str, kind: str
) -> None:
    """On a node of either kind, with its fingerprint kept: refused, and the row unchanged."""
    repo, experiment = _seeded(tmp_path)
    made = make_node(experiment) if kind == "training" else harness_node(experiment)
    node = repo.create_node(made, actor=ACTOR)
    before = _stored(repo)
    with pytest.raises(sqlite3.IntegrityError, match="not a version this schema knows"):
        with write_transaction(repo._connection):
            _write_payload(repo, node, _with_snapshot(repo, node, problem))
    assert _stored(repo) == before
    assert repo.aggregates.load_node(str(node.id)) == node


def test_malformed_payload_json_is_never_written(tmp_path: Path) -> None:
    """Not JSON at all: SQLite's JSON functions refuse it before any trigger can decide."""
    repo, experiment = _seeded(tmp_path)
    node = repo.create_node(make_node(experiment), actor=ACTOR)
    before = _stored(repo)
    with pytest.raises(sqlite3.Error, match="malformed JSON"):
        with write_transaction(repo._connection):
            repo._connection.execute(
                "UPDATE experiment_nodes SET payload_json = '{not json' WHERE id = ?",
                (str(node.id),),
            )
    with pytest.raises(sqlite3.Error, match="malformed JSON"):
        with write_transaction(repo._connection):
            repo._connection.execute(
                "INSERT INTO experiment_nodes (id, experiment_id, status, revision, "
                "candidate_fingerprint, candidate_kind, payload_json, created_at, updated_at) "
                "VALUES ('node_forged', ?, 'created', 0, ?, 'training', '{not json', 'now', 'now')",
                (str(experiment.id), node.candidate_fingerprint),
            )
    assert _stored(repo) == before


def test_both_known_versions_are_still_written(tmp_path: Path) -> None:
    """The positive half: "1" and "2", in their shapes, insert and update as before."""
    repo, experiment = _seeded(tmp_path)
    for made in (make_node(experiment), harness_node(experiment)):
        node = repo.create_node(made, actor=ACTOR)
        with write_transaction(repo._connection):
            _write_payload(repo, node, _payload(repo, node))
        assert repo.aggregates.load_node(str(node.id)) == node


def _around_the_triggers(repo: ControlPlaneRepository) -> None:
    """What a writer that bypasses the schema would leave: no 021 triggers at all."""
    for (name,) in repo._connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'experiment_node_%'"
    ).fetchall():
        repo._connection.execute(f"DROP TRIGGER {name}")


def test_a_read_refuses_a_kind_column_its_payload_contradicts(tmp_path: Path) -> None:
    repo, experiment = _seeded(tmp_path)
    node = repo.create_node(harness_node(experiment), actor=ACTOR)
    _around_the_triggers(repo)
    repo._connection.execute(
        "UPDATE experiment_nodes SET candidate_kind = 'training' WHERE id = ?", (str(node.id),)
    )
    with pytest.raises(NodeRecordMismatchError, match="candidate_kind column says 'training'"):
        repo.aggregates.get_node(str(node.id))
    with pytest.raises(NodeRecordMismatchError):
        repo.aggregates.nodes_for_experiment(str(experiment.id))
    with pytest.raises(NodeRecordMismatchError):
        repo.graph.roots(str(experiment.id))


def test_a_read_refuses_a_replaced_candidate_even_with_its_fingerprints_copied(
    tmp_path: Path,
) -> None:
    """The payload's fingerprint and the column agree; the candidate is someone else's."""
    repo, experiment = _seeded(tmp_path)
    node = repo.create_node(make_node(experiment), actor=ACTOR)
    other = make_node(experiment, fingerprint="b")
    _around_the_triggers(repo)
    forged = {
        **_payload(repo, node),
        "candidate": json.loads(other.model_dump_json(by_alias=True))["candidate"],
    }
    _write_payload(repo, node, forged)
    with pytest.raises(NodeRecordMismatchError, match="not the candidate's identity"):
        repo.aggregates.load_node(str(node.id))


def test_a_read_refuses_a_fingerprint_column_the_payload_does_not_name(tmp_path: Path) -> None:
    repo, experiment = _seeded(tmp_path)
    node = repo.create_node(make_node(experiment), actor=ACTOR)
    _around_the_triggers(repo)
    repo._connection.execute(
        "UPDATE experiment_nodes SET candidate_fingerprint = 'sha256:forged' WHERE id = ?",
        (str(node.id),),
    )
    with pytest.raises(NodeRecordMismatchError) as refused:
        repo.aggregates.get_node(str(node.id))
    assert set(refused.value.problems) == {
        "the candidate_fingerprint column is not the payload's",
        "the candidate_fingerprint column is not the candidate's identity",
    }
    assert refused.value.node_id == str(node.id)


def test_a_historical_node_recorded_under_its_v1_fingerprint_still_reads(
    tmp_path: Path,
) -> None:
    """Training nodes recorded before fingerprint v2 hold v1: still one of their identities."""
    repo, experiment = _seeded(tmp_path)
    node = make_node(experiment)
    (v2, v1) = node.candidate.experiment_candidate().fingerprints_for_lookup()
    historical = ExperimentNode.model_validate(
        {**node.model_dump(mode="json", by_alias=True), "candidate_fingerprint": v1}
    )
    with write_transaction(repo._connection):
        repo.aggregates._insert_node(historical)
    assert repo.aggregates.load_node(str(node.id)).candidate_fingerprint == v1
