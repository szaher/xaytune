"""Observation, diagnosis, event/outbox and replay position are one durable fact."""

from __future__ import annotations

import sqlite3
import subprocess
import sys

import pytest

from tests.test_resilience.test_incidents import classify, envelope
from xaytune.core.domain.incident import DetectorProvenance, IncidentCategory
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor
from xaytune.core.telemetry import EvaluationFailedPayload, IncidentObservedPayload
from xaytune.runtimes import EvaluationEventPayload, RuntimeEventEnvelope
from xaytune.storage import ControlPlaneRepository, connect, write_transaction
from xaytune.storage.control_plane import ProvenanceError
from xaytune.storage.journal import IdempotencyConflictError

ACTOR = Actor(type="system", id="incident-controller")


def _setup(connection, seeded):
    repo = ControlPlaneRepository(connection)
    target = RuntimeOperationTarget(kind="training-attempt", id=str(seeded["attempt"].id))
    owner = repo.incident_context(target)
    event = envelope(owner)
    return repo, owner, event, classify(event, owner)


def test_replay_and_database_reopen_return_the_original_incident(
    connection, seeded, db_path
) -> None:
    repo, owner, event, incident = _setup(connection, seeded)
    recorded = repo.record_incident(incident, actor=ACTOR, destinations=("audit",))
    replay = classify(event, owner).model_copy(
        update={"classifier": DetectorProvenance(name="upgraded-classifier", version="2")}
    )
    assert repo.record_incident(replay, actor=ACTOR, destinations=("audit",)) == recorded
    reopened = connect(db_path)
    try:
        after_restart = ControlPlaneRepository(reopened)
        assert after_restart.record_incident(replay, actor=ACTOR) == recorded
        assert after_restart.incidents.for_attempt(owner.target) == (recorded,)
        assert after_restart.incidents.for_experiment(str(owner.experiment_id)) == (recorded,)
        assert after_restart.incidents.get(str(recorded.id)) == recorded
        assert after_restart.aggregates.telemetry_position(owner.target.id) == (0, 3)
    finally:
        reopened.close()
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 1
    assert repo.aggregates.load_attempt(owner.target.id) == seeded["attempt"]
    assert connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM runtime_operations").fetchone()[0] == 0


def test_same_observation_with_changed_evidence_is_refused(connection, seeded) -> None:
    repo, owner, event, incident = _setup(connection, seeded)
    repo.record_incident(incident, actor=ACTOR)
    changed = envelope(owner, IncidentObservedPayload(reason="cuda-oom", detail="new evidence"))
    with pytest.raises(IdempotencyConflictError, match="evidence"):
        repo.record_incident(classify(changed, owner), actor=ACTOR)
    assert repo.incidents.for_attempt(owner.target) == (incident,)


@pytest.mark.parametrize("failure", ["event", "outbox", "cursor"])
def test_a_partial_write_rolls_back_every_fact(connection, seeded, monkeypatch, failure) -> None:
    repo, owner, event, incident = _setup(connection, seeded)

    def fail(*args, **kwargs):
        raise RuntimeError("injected write failure")

    if failure == "event":
        monkeypatch.setattr(repo.events, "_append", fail)
    elif failure == "outbox":
        monkeypatch.setattr(repo.events, "_enqueue", fail)
    else:
        monkeypatch.setattr(repo.aggregates, "_advance_telemetry", fail)
    with pytest.raises(RuntimeError, match="injected"):
        repo.record_incident(incident, actor=ACTOR, destinations=("audit",))
    for table in ("incidents", "events", "outbox"):
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert repo.aggregates.telemetry_position(owner.target.id) == (0, -1)
    monkeypatch.undo()
    assert repo.record_incident(incident, actor=ACTOR) == incident


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
def test_envelope_correlation_cannot_claim_another_owner(connection, seeded, field) -> None:
    repo, owner, event, incident = _setup(connection, seeded)
    evidence = event.model_dump(mode="json")
    evidence["context"] = {field: "another-owner"}
    forged = incident.model_copy(update={"evidence": FrozenDict(evidence)})
    with pytest.raises(ProvenanceError, match=field):
        repo.record_incident(forged, actor=ACTOR)
    assert repo.incidents.for_attempt(owner.target) == ()
    assert repo.aggregates.telemetry_position(owner.target.id) == (0, -1)


def test_incident_context_is_resolved_again_in_the_write(connection, seeded) -> None:
    repo, owner, event, incident = _setup(connection, seeded)
    forged = incident.model_copy(
        update={"context": owner.model_copy(update={"run_id": "other-run"})}
    )
    with pytest.raises(ProvenanceError, match="context"):
        repo.record_incident(forged, actor=ACTOR)


@pytest.mark.parametrize(
    "statement", ["UPDATE incidents SET category = 'UNKNOWN'", "DELETE FROM incidents"]
)
def test_recorded_incidents_are_immutable_even_through_sql(connection, seeded, statement) -> None:
    repo, owner, event, incident = _setup(connection, seeded)
    repo.record_incident(incident, actor=ACTOR)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"), write_transaction(connection):
        connection.execute(statement)
    assert repo.incidents.for_attempt(owner.target) == (incident,)


def test_sql_cannot_bind_an_incident_to_another_run(connection, seeded) -> None:
    repo, owner, event, incident = _setup(connection, seeded)
    forged = incident.model_copy(
        update={"context": owner.model_copy(update={"run_id": "other-run"})}
    )
    with (
        pytest.raises(sqlite3.IntegrityError, match="does not belong"),
        write_transaction(connection),
    ):
        repo.incidents._insert(forged)


def test_another_generation_is_a_new_observation_and_replay_never_rewinds(
    connection, seeded
) -> None:
    repo, owner, event, incident = _setup(connection, seeded)
    next_generation = classify(
        event.model_copy(update={"stream_generation": 1, "sequence": 0}), owner
    )
    repo.record_incident(incident, actor=ACTOR)
    repo.record_incident(next_generation, actor=ACTOR)
    assert repo.record_incident(classify(event, owner), actor=ACTOR) == incident
    assert repo.incidents.for_attempt(owner.target) == (incident, next_generation)
    assert repo.aggregates.telemetry_position(owner.target.id) == (1, 0)


def test_two_processes_record_the_same_observation_once(connection, seeded, db_path) -> None:
    repo, owner, event, incident = _setup(connection, seeded)
    code = (
        "import sys; from xaytune.core.domain.incident import Incident; "
        "from xaytune.core.refs import Actor; "
        "from xaytune.storage import ControlPlaneRepository, connect; "
        "connection=connect(sys.argv[1]); "
        "result=ControlPlaneRepository(connection).record_incident("
        "Incident.model_validate_json(sys.argv[2]), actor=Actor(type='system',id='child')); "
        "print(result.id); connection.close()"
    )
    candidates = (incident, classify(event, owner))
    children = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(db_path), candidate.model_dump_json()],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for candidate in candidates
    ]
    outputs = []
    for child in children:
        output, error = child.communicate(timeout=20)
        assert child.returncode == 0, error
        outputs.append(output.strip())
    assert outputs[0] == outputs[1]
    (recorded,) = repo.incidents.for_attempt(owner.target)
    assert str(recorded.id) == outputs[0]
    assert recorded in candidates
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_another_attempt_has_its_own_observation_identity(connection, seeded) -> None:
    from .conftest import make_attempt

    repo, owner, event, incident = _setup(connection, seeded)
    other = make_attempt(seeded["run"], attempt_number=2)
    with write_transaction(connection):
        repo.aggregates._insert_attempt(other)
    other_target = RuntimeOperationTarget(kind="training-attempt", id=str(other.id))
    other_owner = repo.incident_context(other_target)
    other_incident = classify(envelope(other_owner), other_owner)
    repo.record_incident(incident, actor=ACTOR)
    repo.record_incident(other_incident, actor=ACTOR)
    assert incident.observation_key != other_incident.observation_key
    assert repo.incidents.for_attempt(other_target) == (other_incident,)
    assert repo.incidents.for_attempt(owner.target) == (incident,)


def test_evaluation_incidents_use_the_same_durable_contract(connection, seeded) -> None:
    from xaytune.core.state.status import ExperimentNodeStatus

    from .test_evaluation_lifecycle import _attempt, _begin, _run

    repo = ControlPlaneRepository(connection)
    node = seeded["node"]
    for status in (
        ExperimentNodeStatus.PLANNED,
        ExperimentNodeStatus.READY,
        ExperimentNodeStatus.ACTIVE,
    ):
        node = repo.transition_node(
            node.id, expected_revision=node.revision, new_status=status, actor=ACTOR
        )
    run = _run(node)
    _begin(repo, node, run)
    attempt, operation = _attempt(repo, run)
    target = RuntimeOperationTarget(kind="evaluation-attempt", id=str(attempt.id))
    owner = repo.incident_context(target)
    event = RuntimeEventEnvelope(
        event_id="evaluation-failure",
        target=target,
        sequence=2,
        payload=EvaluationEventPayload(data=EvaluationFailedPayload(reason="cuda-oom")),
    )
    incident = classify(event, owner)
    assert repo.record_incident(incident, actor=ACTOR) == incident
    assert repo.record_incident(classify(event, owner), actor=ACTOR) == incident
    assert incident.category is IncidentCategory.CUDA_OOM
    assert owner.run_id == str(run.id)
    assert repo.aggregates.telemetry_position(target.id, kind=target.kind) == (0, 2)
    assert connection.execute("SELECT COUNT(*) FROM runtime_operations").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 0
