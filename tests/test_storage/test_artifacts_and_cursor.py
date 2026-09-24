"""Recording what an attempt produced, and reading history from a cursor.

Two additions PR-012's controller needs from the repository:

- ``record_artifact``: an attempt's artifact is durable state, written with
  its event in one commit, revision-guarded like every other write.
- ``events_for_experiment_after``: the database sequence as a cursor, which is
  what makes ``ExperimentHandle.events()`` one query for replay and follow.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from xaytune.core import Actor
from xaytune.core.errors import ConcurrentModificationError
from xaytune.core.ids import ArtifactId
from xaytune.core.refs import ArtifactRef
from xaytune.storage import ControlPlaneRepository

from .conftest import make_attempt, make_experiment, make_node, make_run

ACTOR = Actor(type="system", id="test")


@pytest.fixture
def repo(connection: sqlite3.Connection) -> ControlPlaneRepository:
    return ControlPlaneRepository(connection)


@pytest.fixture
def attempt(repo: ControlPlaneRepository) -> Any:
    experiment = repo.create_experiment(make_experiment(), actor=ACTOR)
    node = repo.create_node(make_node(experiment), actor=ACTOR)
    run = repo.create_run(make_run(node), actor=ACTOR)
    created, _ = repo.create_attempt_with_submit_intent(
        make_attempt(run), request_digest="sha256:x", actor=ACTOR
    )
    return created


def _artifact(**metadata: object) -> ArtifactRef:
    return ArtifactRef(id=ArtifactId.generate(), kind="model", uri="/out/model", metadata=metadata)


def test_an_artifact_is_recorded_with_its_event_in_one_commit(
    repo: ControlPlaneRepository, attempt: Any
) -> None:
    artifact = _artifact()

    recorded = repo.record_artifact(
        attempt.id, artifact, expected_revision=attempt.revision, actor=ACTOR
    )

    assert recorded.artifact_refs == (artifact,)
    assert recorded.revision == attempt.revision + 1
    assert repo.aggregates.load_attempt(str(attempt.id)) == recorded

    (event,) = [
        e
        for e in repo.events.events_for_aggregate(str(attempt.id))
        if e.event_type == "ArtifactRecorded"
    ]
    assert event.aggregate_revision == recorded.revision
    assert event.payload["artifact"]["uri"] == "/out/model"


def test_a_nested_payload_is_stored_as_json(
    repo: ControlPlaneRepository, attempt: Any, connection: sqlite3.Connection
) -> None:
    """The journal encoded payloads shallowly, so the first nested one failed.

    Event payloads are frozen all the way down; an artifact's metadata is a
    mapping inside a mapping inside the payload.
    """
    repo.record_artifact(
        attempt.id,
        _artifact(tokenizer={"vocab": 9}),
        expected_revision=attempt.revision,
        actor=ACTOR,
    )

    (raw,) = connection.execute(
        "SELECT payload_json FROM events WHERE event_type = 'ArtifactRecorded'"
    ).fetchone()
    assert json.loads(raw)["artifact"]["metadata"] == {"tokenizer": {"vocab": 9}}


def test_a_stale_writer_cannot_record(repo: ControlPlaneRepository, attempt: Any) -> None:
    repo.record_artifact(attempt.id, _artifact(), expected_revision=attempt.revision, actor=ACTOR)

    with pytest.raises(ConcurrentModificationError):
        repo.record_artifact(
            attempt.id, _artifact(), expected_revision=attempt.revision, actor=ACTOR
        )


def test_the_same_artifact_is_recorded_once(repo: ControlPlaneRepository, attempt: Any) -> None:
    """A replayed ``ArtifactProduced`` must not appear twice in the record."""
    artifact = _artifact()
    recorded = repo.record_artifact(
        attempt.id, artifact, expected_revision=attempt.revision, actor=ACTOR
    )

    with pytest.raises(ValueError, match="already records"):
        repo.record_artifact(attempt.id, artifact, expected_revision=recorded.revision, actor=ACTOR)


def test_the_cursor_returns_exactly_what_came_after(
    repo: ControlPlaneRepository, attempt: Any
) -> None:
    experiment_id = str(repo.aggregates.load_run(str(attempt.run_id)).experiment_id)
    everything = repo.events.events_for_experiment(experiment_id)
    middle = everything[len(everything) // 2].sequence
    assert middle is not None

    after = repo.events.events_for_experiment_after(experiment_id, middle)

    assert [e.id for e in after] == [e.id for e in everything if (e.sequence or 0) > middle]
    assert repo.events.events_for_experiment_after(experiment_id, 0) == everything


def test_the_cursor_is_bounded_per_call(repo: ControlPlaneRepository, attempt: Any) -> None:
    """A long history is read in pages, not in one unbounded fetch."""
    experiment_id = str(repo.aggregates.load_run(str(attempt.run_id)).experiment_id)
    everything = repo.events.events_for_experiment(experiment_id)

    first = repo.events.events_for_experiment_after(experiment_id, 0, limit=2)

    assert [e.id for e in first] == [e.id for e in everything[:2]]
