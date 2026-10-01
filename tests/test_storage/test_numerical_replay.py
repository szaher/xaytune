"""REAPPLY_AFTER_ROLLBACK through the real successor transaction (ADR-011, spec 08 §9a).

```text
attempt 1   LR 2e-4 ── NaN ── E1 ── intervention I (1e-4)
attempt 2   restore C1, apply I → A1 ── checkpoint C2 embodies A1 ── NaN ── E2 ── I2 (5e-5)
attempt 3   restore C1 (before A1) → re-apply SAME I → A2, then I2 → A3
attempt 3'  restore C2 (embodies A1) → no re-application, only I2
```
"""

from __future__ import annotations

import sqlite3

import pytest

from tests.test_storage.numerical_fixtures import (
    ACTOR,
    checkpointed_lr_run,
    executed_successor,
    record_checkpoint,
    restored_successor,
)
from tests.test_storage.test_numerical_successor_model import confirm, governed_intervention
from xaytune.core.domain.action import ActionStatus
from xaytune.core.domain.intervention import (
    InterventionDirective,
    InterventionDirectiveKind,
)
from xaytune.core.domain.intervention_replay import InterventionReplayError
from xaytune.core.domain.realization import rebuild_run_realization
from xaytune.core.ids import InterventionApplicationId
from xaytune.core.state.status import RunAttemptStatus
from xaytune.storage import write_transaction
from xaytune.storage.control_plane import ProvenanceError


def count(repo, table):
    return int(repo._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


@pytest.fixture
def scenario(connection, tmp_path):
    """Up to E2's authorized intervention I2, with C2 embodying A1."""
    repo, w = checkpointed_lr_run(connection, tmp_path)
    c1 = w["c100"]
    _, first = governed_intervention(repo, w["attempt"])
    attempt2, (directive,), _ = executed_successor(
        repo, w["run"], w["attempt"], first.action_id, c1
    )
    a1 = confirm(repo, directive, step=100)
    c2 = record_checkpoint(repo, attempt2, tmp_path, "c150", 150, 2, embodied=(a1.id,))
    attempt2 = repo.transition_attempt(
        attempt2.id,
        expected_revision=repo.aggregates.load_attempt(str(attempt2.id)).revision,
        new_status=RunAttemptStatus.FAILED,
        actor=ACTOR,
    )
    _, second = governed_intervention(repo, attempt2, sequence=5)
    assert second.mutation.learning_rate == pytest.approx(5e-5)
    return repo, {
        **w,
        "c1": c1,
        "c2": c2,
        "attempt2": attempt2,
        "I": first,
        "I2": second,
        "A1": a1,
    }


def realization(repo, run_id):
    stored = repo.get_run_realization(run_id)
    attempts, checkpoints = repo.run_ancestry(run_id)
    assert stored == rebuild_run_realization(
        repo.aggregates.load_run(str(run_id)), repo.events_for_run(run_id), attempts, checkpoints
    )
    return stored


def test_restore_before_a1_reapplies_the_same_intervention_as_a_new_application(scenario):
    repo, s = scenario
    actions, interventions = count(repo, "actions"), count(repo, "training_interventions")
    attempt3, directives, _ = executed_successor(
        repo, s["run"], s["attempt2"], s["I2"].action_id, s["c1"]
    )
    reapply, initial = directives
    assert reapply.kind is InterventionDirectiveKind.REAPPLY_AFTER_ROLLBACK
    assert reapply.intervention_id == s["I"].id
    assert (reapply.expected_previous_value, reapply.applied_value) == (2e-4, 1e-4)
    assert initial.kind is InterventionDirectiveKind.INITIAL
    assert initial.intervention_id == s["I2"].id
    assert initial.expected_previous_value == 1e-4

    a2 = confirm(repo, reapply, step=100)
    a3 = confirm(repo, initial, step=100)
    assert a2.intervention_id == s["I"].id and a2.id != s["A1"].id
    assert a2.attempt_id == attempt3.id
    assert a2.checkpoint_ancestor == s["c1"].payload.checkpoint_ref
    assert [a.id for a in repo.intervention_applications.for_intervention(str(s["I"].id))] == [
        s["A1"].id,
        a2.id,
    ]
    # Same decision, same Action: re-application creates neither.
    assert count(repo, "actions") == actions
    assert count(repo, "training_interventions") == interventions
    assert repo.actions.get(str(s["I"].action_id)).status is ActionStatus.SUCCEEDED
    assert repo.actions.get(str(s["I2"].action_id)).status is ActionStatus.SUCCEEDED

    projected = realization(repo, s["run"].id)
    assert [a.application_id for a in projected.applied_interventions] == [
        s["A1"].id,
        a2.id,
        a3.id,
    ]
    assert projected.trajectory is not None
    assert projected.trajectory.application_ids == (a2.id, a3.id), "A1 was rolled back"
    assert projected.trajectory.checkpoint_ancestry == (s["c1"].payload.checkpoint_ref.id,)


def test_restore_of_a_checkpoint_embodying_a1_does_not_reapply(scenario):
    repo, s = scenario
    _, directives, _ = executed_successor(repo, s["run"], s["attempt2"], s["I2"].action_id, s["c2"])
    (initial,) = directives
    assert initial.kind is InterventionDirectiveKind.INITIAL
    assert initial.intervention_id == s["I2"].id
    assert initial.expected_previous_value == 1e-4, "A1's effect survives in C2"
    a3 = confirm(repo, initial, step=150)
    assert len(repo.intervention_applications.for_intervention(str(s["I"].id))) == 1

    projected = realization(repo, s["run"].id)
    assert projected.trajectory is not None
    assert projected.trajectory.application_ids == (s["A1"].id, a3.id)
    assert projected.trajectory.checkpoint_ancestry == (
        s["c1"].payload.checkpoint_ref.id,
        s["c2"].payload.checkpoint_ref.id,
    )


def test_database_refuses_a_reapplication_the_restore_already_embodies(scenario):
    repo, s = scenario
    attempt3 = restored_successor(repo, s["run"], s["attempt2"], s["c2"], number=3)
    forged = InterventionDirective(
        application_id=InterventionApplicationId.generate(),
        intervention_id=s["I"].id,
        attempt_id=attempt3.id,
        ordinal=0,
        kind=InterventionDirectiveKind.REAPPLY_AFTER_ROLLBACK,
        mutation=s["I"].mutation,
        expected_previous_value=1e-4,
    )
    with pytest.raises(sqlite3.IntegrityError, match="not justified"):
        with write_transaction(repo._connection):
            repo.intervention_directives._insert(forged, run_id=str(s["run"].id))
    assert count(repo, "intervention_directives") == 1, "only I's initial directive exists"


def test_ambiguous_checkpoint_ancestry_fails_closed(scenario, tmp_path):
    repo, s = scenario
    unknown = record_checkpoint(
        repo,
        s["attempt2"],
        tmp_path,
        "c-unknown",
        140,
        1,
        embodied=(InterventionApplicationId.generate(),),
    )
    with pytest.raises(InterventionReplayError, match="never recorded"):
        repo.plan_successor_interventions(
            s["run"].id, unknown.payload.checkpoint_ref, initial=s["I2"]
        )
    with pytest.raises(InterventionReplayError):
        executed_successor(repo, s["run"], s["attempt2"], s["I2"].action_id, unknown)
    assert count(repo, "numerical_recovery_executions") == 1


def test_directives_must_be_confirmed_in_order(scenario):
    repo, s = scenario
    _, (reapply, initial), _ = executed_successor(
        repo, s["run"], s["attempt2"], s["I2"].action_id, s["c1"]
    )
    with pytest.raises(ProvenanceError, match="recorded order"):
        confirm(repo, initial)
    confirm(repo, reapply)
    confirm(repo, initial)


def test_successor_that_ends_unconfirmed_fails_the_action_but_keeps_the_decision(scenario):
    repo, s = scenario
    attempt3, _, _ = executed_successor(repo, s["run"], s["attempt2"], s["I2"].action_id, s["c2"])
    repo.transition_attempt(
        attempt3.id,
        expected_revision=attempt3.revision,
        new_status=RunAttemptStatus.FAILED,
        actor=ACTOR,
    )
    assert repo.actions.get(str(s["I2"].action_id)).status is ActionStatus.FAILED
    assert repo.training_interventions.get(str(s["I2"].id)) == s["I2"]
    assert repo.intervention_applications.for_intervention(str(s["I2"].id)) == ()
