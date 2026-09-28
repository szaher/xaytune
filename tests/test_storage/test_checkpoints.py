from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys

import pytest
from pydantic import ValidationError

from tests.test_checkpoints.helpers import make_bundle
from xaytune.checkpoints import (
    CheckpointCorruptionError,
    CheckpointManager,
    LocalCheckpointStore,
    SerializedStateCodec,
)
from xaytune.core.checkpoint import RestoreContext
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.errors import IdempotencyConflictError
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor
from xaytune.runtimes import RuntimeEventEnvelope, TrainingEventPayload
from xaytune.storage import ControlPlaneRepository, connect, write_transaction
from xaytune.storage.control_plane import ProvenanceError

ACTOR = Actor(type="system", id="checkpoint-controller")


def setup_bundle(connection, seeded, tmp_path):
    repo = ControlPlaneRepository(connection)
    attempt = type(seeded["attempt"]).model_validate(
        {**seeded["attempt"].model_dump(mode="json"), "execution_fingerprint": "execution-a"}
    )
    with write_transaction(connection):
        connection.execute(
            "UPDATE run_attempts SET payload_json = ? WHERE id = ?",
            (attempt.model_dump_json(), str(attempt.id)),
        )
    state, context, restore = make_bundle(
        tmp_path / "source",
        attempt_id=attempt.id,
        candidate=seeded["run"].candidate_fingerprint,
    )
    store = LocalCheckpointStore(tmp_path / "bundles")
    manager = CheckpointManager(SerializedStateCodec(), store)
    ref = asyncio.run(manager.save(state, context))
    manifest = asyncio.run(store.get(ref)).manifest
    payload = manifest.committed_payload(ref)
    envelope = RuntimeEventEnvelope(
        event_id="checkpoint-commit",
        sequence=3,
        target=RuntimeOperationTarget(kind="training-attempt", id=str(attempt.id)),
        payload=TrainingEventPayload(data=payload),
    )
    return repo, attempt, envelope


def record(repo, attempt, event, *, destinations=()):
    return repo.record_checkpoint(
        attempt.id,
        event.payload.data,
        evidence=FrozenDict(event.model_dump(mode="json")),
        actor=ACTOR,
        destinations=destinations,
    )


def test_replay_reopen_and_reemission_record_one_checkpoint(connection, seeded, tmp_path, db_path):
    repo, attempt, event = setup_bundle(connection, seeded, tmp_path)
    original = record(repo, attempt, event, destinations=("audit",))
    assert record(repo, attempt, event) == original
    reemitted = event.model_copy(update={"stream_generation": 1, "sequence": 0})
    assert record(repo, attempt, reemitted) == original
    reopened = connect(db_path)
    try:
        restarted = ControlPlaneRepository(reopened)
        assert record(restarted, attempt, event) == original
        assert record(restarted, attempt, reemitted) == original
        assert restarted.checkpoints.for_attempt(str(attempt.id)) == (original,)
        assert restarted.aggregates.telemetry_position(str(attempt.id)) == (1, 0)
    finally:
        reopened.close()
    assert repo.aggregates.load_attempt(str(attempt.id)) == attempt
    assert original.payload == event.payload.data
    assert original.evidence == FrozenDict(event.model_dump(mode="json"))
    assert original.context.run_id == str(attempt.run_id)
    assert original.candidate_fingerprint == seeded["run"].candidate_fingerprint
    for table, count in (
        ("checkpoints", 1),
        ("checkpoint_receipts", 2),
        ("events", 1),
        ("outbox", 1),
        ("actions", 0),
        ("runtime_operations", 0),
        ("run_attempts", 1),
        ("experiment_nodes", 1),
    ):
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == count


def test_same_position_different_evidence_and_new_position_different_payload_conflict(
    connection, seeded, tmp_path
):
    repo, attempt, event = setup_bundle(connection, seeded, tmp_path)
    original = record(repo, attempt, event)
    changed = event.model_copy(update={"event_id": "different-observation"})
    with pytest.raises(IdempotencyConflictError, match="evidence"):
        record(repo, attempt, changed)
    payload = event.payload.data
    changed = event.model_copy(
        update={
            "sequence": 4,
            "payload": TrainingEventPayload(
                data=payload.model_copy(
                    update={
                        "checkpoint_ref": payload.checkpoint_ref.model_copy(
                            update={"uri": "file:///elsewhere"}
                        )
                    }
                )
            ),
        }
    )
    with pytest.raises(IdempotencyConflictError, match="payload"):
        record(repo, attempt, changed)
    assert repo.checkpoints.for_attempt(str(attempt.id)) == (original,)
    assert repo.aggregates.telemetry_position(str(attempt.id)) == (0, 3)


@pytest.mark.parametrize("boundary", ["event", "outbox", "receipt", "cursor"])
def test_commit_report_is_atomic(connection, seeded, tmp_path, monkeypatch, boundary):
    repo, attempt, event = setup_bundle(connection, seeded, tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    with monkeypatch.context() as patch:
        if boundary == "event":
            patch.setattr(repo.events, "_append", fail)
        elif boundary == "outbox":
            patch.setattr(repo.events, "_enqueue", fail)
        elif boundary == "receipt":
            patch.setattr(repo.checkpoints, "_insert_receipt", fail)
        else:
            patch.setattr(repo.aggregates, "_advance_telemetry", fail)
        with pytest.raises(RuntimeError, match="injected"):
            record(repo, attempt, event, destinations=("audit",))
    for table in ("checkpoints", "checkpoint_receipts", "events", "outbox"):
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert repo.aggregates.telemetry_position(str(attempt.id)) == (0, -1)
    assert record(repo, attempt, event).payload == event.payload.data


@pytest.mark.parametrize(
    "field",
    [
        "experiment_id",
        "node_id",
        "run_id",
        "attempt_id",
        "evaluation_run_id",
        "evaluation_attempt_id",
    ],
)
def test_checkpoint_correlation_cannot_claim_another_owner(connection, seeded, tmp_path, field):
    repo, attempt, event = setup_bundle(connection, seeded, tmp_path)
    raw = event.model_dump(mode="json")
    raw["context"] = {field: "another-owner"}
    with pytest.raises(ProvenanceError):
        repo.record_checkpoint(
            attempt.id, event.payload.data, evidence=FrozenDict(raw), actor=ACTOR
        )
    assert repo.checkpoints.for_attempt(str(attempt.id)) == ()


def test_another_attempt_cannot_claim_the_same_checkpoint(connection, seeded, tmp_path):
    from .conftest import make_attempt

    repo, attempt, event = setup_bundle(connection, seeded, tmp_path)
    record(repo, attempt, event)
    other = make_attempt(seeded["run"], attempt_number=2)
    other = type(other).model_validate(
        {**other.model_dump(mode="json"), "execution_fingerprint": "execution-a"}
    )
    with write_transaction(connection):
        repo.aggregates._insert_attempt(other)
    forged = event.model_copy(
        update={"target": RuntimeOperationTarget(kind="training-attempt", id=str(other.id))}
    )
    with pytest.raises(ProvenanceError, match="producer"):
        record(repo, other, forged)
    assert repo.checkpoints.for_attempt(str(other.id)) == ()


@pytest.mark.parametrize("table", ["checkpoints", "checkpoint_receipts"])
@pytest.mark.parametrize("operation", ["UPDATE", "DELETE"])
def test_checkpoint_reports_and_receipts_are_immutable_through_sql(
    connection, seeded, tmp_path, table, operation
):
    repo, attempt, event = setup_bundle(connection, seeded, tmp_path)
    record(repo, attempt, event)
    sql = (
        f"DELETE FROM {table}"
        if operation == "DELETE"
        else (
            f"UPDATE {table} SET "
            + ("optimizer_step = 999" if table == "checkpoints" else "sequence = 999")
        )
    )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"), write_transaction(connection):
        connection.execute(sql)


def test_checkpoint_payload_cannot_be_detached_from_authoritative_evidence(
    connection, seeded, tmp_path
):
    repo, attempt, event = setup_bundle(connection, seeded, tmp_path)
    raw = event.model_dump(mode="json")
    raw["payload"]["data"]["optimizer_step"] = True
    with pytest.raises(ValidationError, match="evidence"):
        repo.record_checkpoint(
            attempt.id, event.payload.data, evidence=FrozenDict(raw), actor=ACTOR
        )


def test_concurrent_processes_record_the_same_checkpoint_once(
    connection, seeded, tmp_path, db_path
):
    repo, attempt, event = setup_bundle(connection, seeded, tmp_path)
    code = """
import sys
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor
from xaytune.core.ids import RunAttemptId
from xaytune.runtimes import RuntimeEventEnvelope
from xaytune.storage import ControlPlaneRepository, connect
connection = connect(sys.argv[1])
event = RuntimeEventEnvelope.model_validate_json(sys.argv[2])
record = ControlPlaneRepository(connection).record_checkpoint(
    RunAttemptId.validate(event.target.id), event.payload.data,
    evidence=FrozenDict(event.model_dump(mode='json')), actor=Actor(type='system', id='child'))
print(record.created_at.isoformat())
connection.close()
"""
    children = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(db_path), event.model_dump_json()],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    outputs = []
    for child in children:
        output, error = child.communicate(timeout=20)
        assert child.returncode == 0, error
        outputs.append(output.strip())
    assert outputs[0] == outputs[1]
    assert len(repo.checkpoints.for_attempt(str(attempt.id))) == 1
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


@pytest.mark.parametrize("field", ["candidate_fingerprint", "execution_fingerprint"])
def test_localized_bundle_must_match_its_durable_provenance(connection, seeded, tmp_path, field):
    repo, attempt, event = setup_bundle(connection, seeded, tmp_path)
    original = record(repo, attempt, event)
    store = LocalCheckpointStore(tmp_path / "bundles")
    localized = asyncio.run(store.get(original.payload.checkpoint_ref))
    manifest = localized.manifest
    restore = RestoreContext(
        candidate_fingerprint=original.candidate_fingerprint,
        compatibility=manifest.context.compatibility,
        dataset_fingerprint=manifest.data_cursor.dataset_fingerprint,
        ordering_fingerprint=manifest.data_cursor.ordering_fingerprint,
    )
    forged = original.model_copy(update={field: "another-producer-snapshot"})
    with pytest.raises(CheckpointCorruptionError, match="durable"):
        asyncio.run(
            CheckpointManager(SerializedStateCodec(), store).restore_recorded(forged, restore)
        )
