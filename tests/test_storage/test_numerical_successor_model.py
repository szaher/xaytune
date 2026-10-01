"""PR-021 v1 execution model: a checkpoint-backed successor attempt carries the LR change.

```text
attempt 1 ── NaN ── episode E1 ── ChangeLearningRate ── TrainingIntervention
    └─ checkpoint C ── attempt 2 (restored from C) ── InterventionApplication
                         └─ E1 closed by the successor; a later NaN opens E2
```

Application provenance is derived from durable state: a caller can manufacture
neither the previous learning rate nor the checkpoint ancestor.
"""

from __future__ import annotations

import sqlite3

import pytest

from tests.test_storage.numerical_fixtures import (
    ACTOR,
    ALLOW,
    HALVE,
    checkpointed_lr_run,
    nonfinite_plan,
    restored_successor,
)
from xaytune.core.domain.event import DomainEvent
from xaytune.core.domain.intervention import InterventionApplication, TrainingPosition
from xaytune.core.domain.numerical_recovery import NumericalLRProposal
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.domain.realization import rebuild_run_realization
from xaytune.core.ids import EventId, InterventionApplicationId
from xaytune.core.immutable import FrozenDict
from xaytune.core.state.status import RunAttemptStatus
from xaytune.resilience.numerical import NumericalRecoveryPlanner
from xaytune.storage import write_transaction
from xaytune.storage.control_plane import ProvenanceError

PLANNER = NumericalRecoveryPlanner()


def count(repo, table):
    return int(repo._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def governed_intervention(repo, attempt):
    _, plan = nonfinite_plan(repo, attempt, sequence=9)
    inputs = repo.numerical_recovery_inputs(str(plan.id), HALVE)
    proposal = PLANNER.plan(inputs)
    assert isinstance(proposal, NumericalLRProposal)
    governed = repo.propose_numerical_recovery_action(
        inputs,
        proposal,
        proposed_by=ACTOR,
        reason="stabilise the continuing trajectory",
        policy=ALLOW,
        capabilities=None,
    )
    intervention = repo.record_numerical_intervention(
        governed.action.id, actor=ACTOR, rationale="loss became nonfinite"
    )
    return plan, intervention


def apply(repo, intervention, attempt, *, observed, step=101, application_id=None):
    return repo.record_intervention_application(
        intervention.id,
        application_id=application_id or InterventionApplicationId.generate(),
        attempt_id=attempt.id,
        position=TrainingPosition(optimizer_step=step),
        observed_previous_value=observed,
        applied_value=intervention.mutation.learning_rate,
        actor=ACTOR,
    )


@pytest.fixture
def world(connection, tmp_path):
    repo, world = checkpointed_lr_run(connection, tmp_path)
    plan, intervention = governed_intervention(repo, world["attempt"])
    return repo, {**world, "plan": plan, "intervention": intervention}


def test_successor_closes_e1_and_later_nan_opens_e2_which_may_reduce_again(world):
    repo, w = world
    e1 = w["plan"].episode_id
    assert repo.recovery_plans.is_effective_and_fresh(str(w["plan"].id))

    successor = restored_successor(repo, w["run"], w["attempt"], w["c100"])
    assert not repo.recovery_plans.is_effective_and_fresh(str(w["plan"].id)), "E1 closed"
    applied = apply(repo, w["intervention"], successor, observed=2e-4)
    assert applied.previous_value == 2e-4 and applied.applied_value == 1e-4

    successor = repo.transition_attempt(
        successor.id,
        expected_revision=successor.revision,
        new_status=RunAttemptStatus.FAILED,
        actor=ACTOR,
    )
    _, plan2 = nonfinite_plan(repo, successor, sequence=5)
    assert plan2.episode_id != e1
    e2 = repo.recovery_episodes.for_attempt(
        RuntimeOperationTarget(kind="training-attempt", id=str(successor.id))
    )
    assert e2 is not None and e2.id == plan2.episode_id
    assert repo.recovery_episodes.get(str(e1)).context.target.id == str(w["attempt"].id)

    inputs = repo.numerical_recovery_inputs(str(plan2.id), HALVE)
    assert inputs.current_learning_rate.application_id == applied.id
    assert [p.episode_id for p in inputs.prior_interventions] == [e1]
    proposal = PLANNER.plan(inputs)
    assert isinstance(proposal, NumericalLRProposal)
    assert proposal.action_spec.learning_rate == pytest.approx(5e-5)


def test_true_restore_ancestor_is_derived_and_projected(world):
    repo, w = world
    successor = restored_successor(repo, w["run"], w["attempt"], w["c100"])
    applied = apply(repo, w["intervention"], successor, observed=2e-4)
    assert applied.checkpoint_ancestor == w["c100"].payload.checkpoint_ref

    realization = repo.get_run_realization(w["run"].id)
    attempts, checkpoints = repo.run_ancestry(w["run"].id)
    assert realization == rebuild_run_realization(
        repo.aggregates.load_run(str(w["run"].id)),
        repo.events_for_run(w["run"].id),
        attempts,
        checkpoints,
    )
    assert realization.trajectory is not None
    assert realization.trajectory.checkpoint_ancestry == (w["c100"].payload.checkpoint_ref.id,)
    assert realization.trajectory.application_ids == (applied.id,)


def test_fresh_start_has_no_checkpoint_ancestor(world):
    repo, w = world
    successor = restored_successor(repo, w["run"], w["attempt"], None)
    assert apply(repo, w["intervention"], successor, observed=2e-4).checkpoint_ancestor is None


def test_false_previous_value_is_refused(world):
    repo, w = world
    successor = restored_successor(repo, w["run"], w["attempt"], w["c100"])
    events = count(repo, "events")
    with pytest.raises(ProvenanceError, match="retained-trajectory rate"):
        apply(repo, w["intervention"], successor, observed=9e-9)
    assert count(repo, "intervention_applications") == 0
    assert count(repo, "events") == events


def test_rollback_discards_the_earlier_application_from_the_previous_value(world):
    """Applied on attempt 1 after C100; attempt 2 restores C100, so it is back at 2e-4."""
    repo, w = world
    first = apply(repo, w["intervention"], w["attempt"], observed=2e-4, step=150)
    successor = restored_successor(repo, w["run"], w["attempt"], w["c100"])
    with pytest.raises(ProvenanceError, match="retained-trajectory rate"):
        apply(repo, w["intervention"], successor, observed=1e-4)
    again = apply(repo, w["intervention"], successor, observed=2e-4, step=100)
    assert again.event_sequence > first.event_sequence
    assert again.position.optimizer_step < first.position.optimizer_step
    realization = repo.get_run_realization(w["run"].id)
    assert [a.application_id for a in realization.applied_interventions] == [first.id, again.id]
    assert realization.trajectory is not None
    assert realization.trajectory.application_ids == (again.id,)


def test_unrelated_same_run_checkpoint_cannot_be_named_as_ancestor(world):
    """Even past the repository, the database ties the ancestor to the attempt's restore."""
    repo, w = world
    successor = restored_successor(repo, w["run"], w["attempt"], w["c100"])
    application_id = InterventionApplicationId.generate()
    forged = InterventionApplication(
        id=application_id,
        intervention_id=w["intervention"].id,
        attempt_id=successor.id,
        event_sequence=1,
        position=TrainingPosition(optimizer_step=101),
        previous_value=2e-4,
        applied_value=1e-4,
        checkpoint_ancestor=w["c50"].payload.checkpoint_ref,
    )
    with pytest.raises(sqlite3.IntegrityError, match="ancestor"):
        with write_transaction(repo._connection):
            sequence = repo._write_sequenced_event(
                DomainEvent(
                    id=EventId.generate(),
                    experiment_id=str(w["run"].experiment_id),
                    aggregate_type="RunAttempt",
                    aggregate_id=str(successor.id),
                    aggregate_revision=successor.revision,
                    event_type="InterventionApplied",
                    actor=ACTOR,
                    payload=FrozenDict({"application_id": str(application_id)}),
                ),
                (),
            )
            repo.intervention_applications._insert(
                forged.model_copy(update={"event_sequence": sequence}), run_id=str(w["run"].id)
            )
    assert count(repo, "intervention_applications") == 0


def test_replay_returns_original_even_after_the_trajectory_moved(world):
    repo, w = world
    successor = restored_successor(repo, w["run"], w["attempt"], w["c100"])
    application_id = InterventionApplicationId.generate()
    first = apply(repo, w["intervention"], successor, observed=2e-4, application_id=application_id)
    # The trajectory's rate is now 1e-4, but a replay of the same confirmation is the same fact.
    replayed = apply(
        repo, w["intervention"], successor, observed=2e-4, application_id=application_id
    )
    assert replayed == first
