"""Migration 019: agent invocations are durable, append-forward and fenced (PR-032)."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from tests.test_core.test_agent_invocation import ANSWER, FAILURE, intent, proposal
from tests.test_storage.conftest import make_experiment
from xaytune.core.domain.agent_invocation import AgentInvocationStatus as S
from xaytune.core.refs import Actor
from xaytune.storage import ControlPlaneRepository
from xaytune.storage.agent_invocations import RepositoryAgentInvocationJournal

_ACTOR = Actor(type="system", id="test")


@pytest.fixture
def repository(connection: sqlite3.Connection) -> ControlPlaneRepository:
    return ControlPlaneRepository(connection)


@pytest.fixture
def experiment_id(repository: ControlPlaneRepository) -> Any:
    experiment = make_experiment()
    repository.create_experiment(experiment, actor=_ACTOR)
    return experiment.id


def _intent(experiment_id: Any, **changes: Any) -> Any:
    return intent(experiment_id=experiment_id, **changes)


def test_an_invocation_is_written_intended_before_anything_else(
    repository: ControlPlaneRepository, experiment_id: Any
) -> None:
    invocation = repository.begin_agent_invocation(_intent(experiment_id))
    loaded = repository.agent_invocations.get(invocation.id)
    assert loaded == invocation
    assert loaded is not None and loaded.status is S.INTENDED and loaded.attempt == 1


def test_the_whole_lifecycle_round_trips_through_the_record(
    repository: ControlPlaneRepository, experiment_id: Any
) -> None:
    journal = repository.agent_invocation_journal()
    invocation = journal.begin(_intent(experiment_id))
    answered = journal.answered(invocation, ANSWER)
    made = proposal(answered)
    done = journal.completed(answered, made)
    assert repository.agent_invocations.get(done.id) == done
    assert repository.agent_invocations.for_experiment(str(experiment_id)) == (done,)
    assert done.proposal_fingerprint == made.proposal_fingerprint()


def test_a_restarted_controller_replays_an_answered_round(
    repository: ControlPlaneRepository, experiment_id: Any, connection: sqlite3.Connection
) -> None:
    journal = repository.agent_invocation_journal()
    answered = journal.answered(journal.begin(_intent(experiment_id)), ANSWER)
    restarted = ControlPlaneRepository(connection)
    assert restarted.begin_agent_invocation(_intent(experiment_id)) == answered
    assert len(restarted.agent_invocations.for_experiment(str(experiment_id))) == 1


def test_an_orphaned_intent_ends_unknown_and_a_new_attempt_begins(
    repository: ControlPlaneRepository, experiment_id: Any
) -> None:
    orphan = repository.begin_agent_invocation(_intent(experiment_id))
    retry = repository.begin_agent_invocation(_intent(experiment_id))
    assert retry.attempt == 2 and retry.status is S.INTENDED
    closed = repository.agent_invocations.get(orphan.id)
    assert closed is not None and closed.status is S.OUTCOME_UNKNOWN
    assert closed.settled_at is not None


def test_a_failed_attempt_is_kept_and_followed_by_the_next(
    repository: ControlPlaneRepository, experiment_id: Any
) -> None:
    journal = repository.agent_invocation_journal()
    failed = journal.failed(journal.begin(_intent(experiment_id)), FAILURE)
    second = journal.begin(_intent(experiment_id))
    assert second.attempt == 2
    assert repository.agent_invocations.get(failed.id) == failed


def test_rounds_are_kept_apart(repository: ControlPlaneRepository, experiment_id: Any) -> None:
    one = repository.begin_agent_invocation(_intent(experiment_id))
    other = repository.begin_agent_invocation(
        _intent(experiment_id, context_fingerprint="sha256:later", request_fingerprint="sha256:r2")
    )
    assert (one.attempt, other.attempt) == (1, 1)
    assert repository.agent_invocations.get(one.id).status is S.INTENDED  # type: ignore[union-attr]


def test_writes_are_fenced(connection: sqlite3.Connection, experiment_id: Any) -> None:
    class Refuses:
        def check(self, connection: sqlite3.Connection) -> None:
            raise PermissionError("another controller holds the lease")

    fenced = ControlPlaneRepository(connection, fence=Refuses())
    with pytest.raises(PermissionError):
        fenced.begin_agent_invocation(_intent(experiment_id))
    assert (
        ControlPlaneRepository(connection).agent_invocations.for_experiment(str(experiment_id))
        == ()
    )


def test_the_repository_journal_is_an_agent_invocation_journal(
    repository: ControlPlaneRepository,
) -> None:
    from xaytune.core.domain.agent_invocation import AgentInvocationJournal

    assert isinstance(repository.agent_invocation_journal(), RepositoryAgentInvocationJournal)
    assert isinstance(repository.agent_invocation_journal(), AgentInvocationJournal)


# ---- the schema holds the contract on its own --------------------------------------------


def _row(repository: ControlPlaneRepository, experiment_id: Any, *, answered: bool = False) -> str:
    journal = repository.agent_invocation_journal()
    invocation = journal.begin(_intent(experiment_id))
    if answered:
        journal.answered(invocation, ANSWER)
    return str(invocation.id)


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        (
            "UPDATE agent_invocations SET intent_json = '{}', revision = revision + 1, "
            "status = 'outcome_unknown', settled_at = 'x' WHERE id = ?",
            "immutable",
        ),
        (
            "UPDATE agent_invocations SET request_fingerprint = 'sha256:x', "
            "revision = revision + 1, "
            "status = 'outcome_unknown', settled_at = 'x' WHERE id = ?",
            "immutable",
        ),
        (
            "UPDATE agent_invocations SET status = 'completed', revision = revision + 1, "
            "settled_at = 'x' WHERE id = ?",
            "forward",
        ),
        (
            "UPDATE agent_invocations SET status = 'outcome_unknown', settled_at = 'x' "
            "WHERE id = ?",
            "forward",
        ),
        ("DELETE FROM agent_invocations WHERE id = ?", "permanent"),
    ],
    ids=["intent", "request", "skip-the-answer", "no-revision", "delete"],
)
def test_the_schema_refuses_rewriting_history(
    repository: ControlPlaneRepository,
    experiment_id: Any,
    connection: sqlite3.Connection,
    statement: str,
    message: str,
) -> None:
    invocation_id = _row(repository, experiment_id)
    with pytest.raises(sqlite3.IntegrityError, match=message):
        with connection:
            connection.execute(statement, (invocation_id,))


def test_the_schema_keeps_a_recorded_answer(
    repository: ControlPlaneRepository, experiment_id: Any, connection: sqlite3.Connection
) -> None:
    invocation_id = _row(repository, experiment_id, answered=True)
    with pytest.raises(sqlite3.IntegrityError, match="never changed"):
        with connection:
            connection.execute(
                "UPDATE agent_invocations SET response_json = '{\"content\": {}}', "
                "status = 'completed', settled_at = 'x', revision = revision + 1 WHERE id = ?",
                (invocation_id,),
            )


def test_the_schema_admits_a_new_attempt_only_after_a_failed_or_unknown_one(
    repository: ControlPlaneRepository, experiment_id: Any, connection: sqlite3.Connection
) -> None:
    invocation_id = _row(repository, experiment_id)
    row = connection.execute(
        "SELECT * FROM agent_invocations WHERE id = ?", (invocation_id,)
    ).fetchone()
    columns = [key for key in row.keys() if key != "id"]
    with pytest.raises(sqlite3.IntegrityError, match="follows only a failed or unknown"):
        with connection:
            connection.execute(
                f"INSERT INTO agent_invocations (id, {', '.join(columns)}) "
                f"SELECT 'agentinv_X', {', '.join(c if c != 'attempt' else '2' for c in columns)} "
                "FROM agent_invocations WHERE id = ?",
                (invocation_id,),
            )
    with pytest.raises(sqlite3.IntegrityError, match="consecutively"):
        with connection:
            connection.execute(
                "UPDATE agent_invocations SET status = 'outcome_unknown', settled_at = 'x', "
                "revision = revision + 1 WHERE id = ?",
                (invocation_id,),
            )
            connection.execute(
                f"INSERT INTO agent_invocations (id, {', '.join(columns)}) "
                f"SELECT 'agentinv_Y', {', '.join(c if c != 'attempt' else '3' for c in columns)} "
                "FROM agent_invocations WHERE id = ?",
                (invocation_id,),
            )


@pytest.mark.parametrize(
    ("answered", "statement", "message"),
    [
        (
            False,
            "UPDATE agent_invocations SET status = 'answered', revision = revision + 1 "
            "WHERE id = ?",
            "CHECK",
        ),
        (
            False,
            "UPDATE agent_invocations SET response_json = '{}', status = 'outcome_unknown', "
            "settled_at = 'x', revision = revision + 1 WHERE id = ?",
            "CHECK",
        ),
        (
            True,
            "UPDATE agent_invocations SET status = 'failed', failure_json = '{}', "
            "settled_at = 'x', revision = revision + 1 WHERE id = ?",
            "along its edges",
        ),
    ],
    ids=["answered-without-answer", "unknown-with-answer", "answered-to-failed"],
)
def test_the_schema_holds_where_an_answer_must_and_must_not_be(
    repository: ControlPlaneRepository,
    experiment_id: Any,
    connection: sqlite3.Connection,
    answered: bool,
    statement: str,
    message: str,
) -> None:
    invocation_id = _row(repository, experiment_id, answered=answered)
    with pytest.raises(sqlite3.IntegrityError, match=message):
        with connection:
            connection.execute(statement, (invocation_id,))


def test_the_schema_refuses_an_intended_row_carrying_an_answer(
    repository: ControlPlaneRepository, experiment_id: Any, connection: sqlite3.Connection
) -> None:
    invocation_id = _row(repository, experiment_id)
    row = connection.execute(
        "SELECT * FROM agent_invocations WHERE id = ?", (invocation_id,)
    ).fetchone()
    columns = [key for key in row.keys() if key != "id"]
    replaced = {"context_fingerprint": "'sha256:another-round'", "response_json": "'{}'"}
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        with connection:
            connection.execute(
                f"INSERT INTO agent_invocations (id, {', '.join(columns)}) "
                f"SELECT 'agentinv_Z', {', '.join(replaced.get(c, c) for c in columns)} "
                "FROM agent_invocations WHERE id = ?",
                (invocation_id,),
            )
