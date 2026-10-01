"""A directed effect is a fact only once confirmed, and only confirmed trajectories succeed.

```text
successor exits cleanly, initial directive unconfirmed
  → Attempt SUCCEEDED (mechanically true) · Action FAILED · Run cannot be SUCCEEDED
InterventionApplied confirmation
  → application + telemetry cursor, one commit (a replay still advances the cursor)
```
"""

from __future__ import annotations

import sqlite3

import pytest

from tests.test_storage.numerical_fixtures import ACTOR, checkpointed_lr_run, executed_successor
from tests.test_storage.test_numerical_successor_model import governed_intervention
from xaytune.core.domain.action import ActionStatus
from xaytune.core.domain.intervention import TrainingPosition
from xaytune.core.state.status import RunAttemptStatus, RunStatus
from xaytune.storage.control_plane import ProvenanceError


@pytest.fixture
def directed(connection, tmp_path):
    """Attempt 2 restored from C1 and directed to apply I, still running."""
    repo, w = checkpointed_lr_run(connection, tmp_path)
    _, intervention = governed_intervention(repo, w["attempt"])
    attempt2, (directive,), _ = executed_successor(
        repo, w["run"], w["attempt"], intervention.action_id, w["c100"]
    )
    for status in (RunAttemptStatus.QUEUED, RunAttemptStatus.STARTING, RunAttemptStatus.RUNNING):
        attempt2 = repo.transition_attempt(
            attempt2.id, expected_revision=attempt2.revision, new_status=status, actor=ACTOR
        )
    return repo, {**w, "attempt2": attempt2, "directive": directive, "I": intervention}


def record(repo, directive, *, position=None, previous=None):
    return repo.record_intervention_application(
        directive.intervention_id,
        application_id=directive.application_id,
        attempt_id=directive.attempt_id,
        position=TrainingPosition(optimizer_step=100),
        observed_previous_value=previous or directive.expected_previous_value,
        applied_value=directive.applied_value,
        actor=ACTOR,
        telemetry_position=position,
    )


def finish(repo, attempt):
    return repo.transition_attempt(
        attempt.id,
        expected_revision=repo.aggregates.load_attempt(str(attempt.id)).revision,
        new_status=RunAttemptStatus.SUCCEEDED,
        actor=ACTOR,
    )


def count(repo, table):
    return int(repo._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_an_unconfirmed_successor_cannot_make_its_run_succeed(directed):
    repo, w = directed
    finish(repo, w["attempt2"])
    assert repo.actions.get(str(w["I"].action_id)).status is ActionStatus.FAILED
    assert repo.unconfirmed_final_directives(str(w["run"].id)) == (w["directive"],)

    run = repo.aggregates.load_run(str(w["run"].id))
    with pytest.raises(ProvenanceError, match="did not confirm"):
        repo.transition_run(
            run.id, expected_revision=run.revision, new_status=RunStatus.SUCCEEDED, actor=ACTOR
        )
    with pytest.raises(sqlite3.DatabaseError, match="unconfirmed intervention directives"):
        repo._connection.execute(
            "UPDATE runs SET status = 'succeeded' WHERE id = ?", (str(run.id),)
        )
    assert repo.aggregates.load_run(str(run.id)).status is RunStatus.ACTIVE
    failed = repo.transition_run(
        run.id, expected_revision=run.revision, new_status=RunStatus.FAILED, actor=ACTOR
    )
    assert failed.status is RunStatus.FAILED


def test_a_confirmed_successor_may_succeed(directed):
    repo, w = directed
    record(repo, w["directive"])
    finish(repo, w["attempt2"])
    assert repo.actions.get(str(w["I"].action_id)).status is ActionStatus.SUCCEEDED
    run = repo.aggregates.load_run(str(w["run"].id))
    assert repo.unconfirmed_final_directives(str(run.id)) == ()
    assert (
        repo.transition_run(
            run.id, expected_revision=run.revision, new_status=RunStatus.SUCCEEDED, actor=ACTOR
        ).status
        is RunStatus.SUCCEEDED
    )


def test_application_and_telemetry_cursor_commit_together(directed):
    repo, w = directed
    attempt_id = str(w["attempt2"].id)
    before = repo.aggregates.telemetry_position(attempt_id)
    with pytest.raises(ProvenanceError):
        record(repo, w["directive"], position=(0, 7), previous=3e-4)
    assert repo.aggregates.telemetry_position(attempt_id) == before, "nothing committed"
    assert count(repo, "intervention_applications") == 0

    record(repo, w["directive"], position=(0, 7))
    assert repo.aggregates.telemetry_position(attempt_id) == (0, 7)
    assert count(repo, "intervention_applications") == 1


def test_a_replayed_confirmation_advances_a_stale_cursor_without_a_duplicate(directed):
    repo, w = directed
    attempt_id = str(w["attempt2"].id)
    original = record(repo, w["directive"])
    stale = repo.aggregates.telemetry_position(attempt_id)
    assert stale < (0, 7)

    replayed = record(repo, w["directive"], position=(0, 7))
    assert replayed == original
    assert count(repo, "intervention_applications") == 1
    assert repo.aggregates.telemetry_position(attempt_id) == (0, 7)

    record(repo, w["directive"], position=(0, 3))
    assert repo.aggregates.telemetry_position(attempt_id) == (0, 7), "never winds back"
