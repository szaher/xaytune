"""Governed numerical recovery records a TrainingIntervention, never a node or override."""

from __future__ import annotations

import sqlite3

import pytest

from tests.test_storage.conftest import make_attempt
from tests.test_storage.numerical_fixtures import (
    ACTOR,
    ALLOW,
    APPROVAL,
    DENY,
    HALVE,
    REVIEWER,
    checkpointed_lr_run,
    human_intervention,
    nonfinite_plan,
    restored_successor,
    seeded_lr_run,
)
from tests.test_storage.test_recovery_episodes import another_run, incident, signal
from xaytune.core.domain.action import ActionStatus, ActionTarget
from xaytune.core.domain.actions import ChangeCheckpointInterval, ChangeLearningRate
from xaytune.core.domain.event import DomainEvent
from xaytune.core.domain.intervention import (
    IncidentTrigger,
    InterventionOrigin,
    InterventionReplayPolicy,
    LearningRateMutation,
    ManualTrigger,
    TrainingIntervention,
    TrainingPosition,
)
from xaytune.core.domain.numerical_recovery import (
    NumericalEscalation,
    NumericalEscalationCode,
    NumericalLRProposal,
    NumericalRecoveryActionBinding,
)
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.domain.realization import rebuild_run_realization
from xaytune.core.ids import EventId, InterventionApplicationId, InterventionId
from xaytune.core.immutable import FrozenDict
from xaytune.core.state.status import RunAttemptStatus
from xaytune.resilience.numerical import NumericalRecoveryPlanner
from xaytune.storage import ControlPlaneRepository, connect, migrate, write_transaction
from xaytune.storage.control_plane import (
    InterventionNotAuthorizedError,
    ProvenanceError,
    StaleRecoveryContextError,
)
from xaytune.storage.errors import AggregateNotFoundError
from xaytune.storage.journal import IdempotencyConflictError


def count(repo, table):
    return int(repo._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def prepared(connection, reason="numerical-nan", tmp_path=None):
    """A nonfinite episode on attempt 1; with *tmp_path*, attempt 1 also has checkpoints."""
    if tmp_path is None:
        repo, world = seeded_lr_run(connection)
    else:
        repo, world = checkpointed_lr_run(connection, tmp_path)
    observed, plan = nonfinite_plan(repo, world["attempt"], reason=reason)
    inputs = repo.numerical_recovery_inputs(str(plan.id), HALVE)
    proposal = NumericalRecoveryPlanner().plan(inputs)
    assert isinstance(proposal, NumericalLRProposal)
    return repo, world, observed, inputs, proposal


def propose(repo, inputs, proposal, *, policy=ALLOW, **kwargs):
    return repo.propose_numerical_recovery_action(
        inputs,
        proposal,
        proposed_by=ACTOR,
        reason="stabilise the continuing trajectory",
        policy=policy,
        capabilities=None,
        destinations=("audit",),
        **kwargs,
    )


def record(repo, action_id, **kwargs):
    return repo.record_numerical_intervention(
        action_id, actor=ACTOR, rationale="loss became nonfinite; lower LR and continue", **kwargs
    )


def apply(repo, intervention, attempt, *, step=14_250, previous=2e-4, application_id=None):
    return repo.record_intervention_application(
        intervention.id,
        application_id=application_id or InterventionApplicationId.generate(),
        attempt_id=attempt.id,
        position=TrainingPosition(optimizer_step=step),
        observed_previous_value=previous,
        applied_value=intervention.mutation.learning_rate,
        actor=ACTOR,
        destinations=("audit",),
    )


def realization_matches_rebuild(repo, run_id):
    stored = repo.get_run_realization(run_id)
    attempts, checkpoints = repo.run_ancestry(run_id)
    rebuilt = rebuild_run_realization(
        repo.aggregates.load_run(str(run_id)), repo.events_for_run(run_id), attempts, checkpoints
    )
    assert stored == rebuilt
    return stored


# ---- governance -------------------------------------------------------------------------


def test_adr011_scenario_lr_drop_is_an_intervention_on_the_same_run(connection, db_path, tmp_path):
    """Candidate LR 2e-4, loss nonfinite, proposal 1e-4, authorized, intervention recorded.

    Applied (v1, spec 08 §9a) on the checkpoint-backed successor attempt.
    """
    repo, world, observed, inputs, proposal = prepared(connection, tmp_path=tmp_path)
    run, node, attempt = world["run"], world["node"], world["attempt"]
    nodes_before = count(repo, "experiment_nodes")
    before = realization_matches_rebuild(repo, run.id)

    governed = propose(repo, inputs, proposal)
    assert governed.action.type == "change-learning-rate"
    assert governed.action.status is ActionStatus.VALIDATED
    binding = repo.numerical_recovery_bindings.for_action(str(governed.action.id))
    assert binding == NumericalRecoveryActionBinding.for_proposal(
        governed.action.id, proposal
    ).model_copy(update={"created_at": binding.created_at})
    assert count(repo, "training_interventions") == 0, "authorization alone is no intervention"

    intervention = record(repo, governed.action.id)
    assert intervention.run_id == run.id
    assert intervention.action_id == governed.action.id
    assert intervention.origin is InterventionOrigin.REACTIVE_POLICY
    assert intervention.trigger.incident_id == observed.id
    assert intervention.trigger.incident_category.value == "NUMERICAL_NAN"
    assert intervention.replay_policy is InterventionReplayPolicy.REAPPLY_AFTER_ROLLBACK
    assert intervention.mutation == LearningRateMutation(learning_rate=1e-4)
    assert count(repo, "intervention_applications") == 0, "no effect was confirmed"
    assert count(repo, "experiment_nodes") == nodes_before
    assert repo.aggregates.load_run(str(run.id)).candidate_fingerprint == run.candidate_fingerprint
    assert repo.aggregates.load_node(str(node.id)).candidate_fingerprint == (
        node.candidate_fingerprint
    )
    assert repo.aggregates.load_attempt(str(attempt.id)).execution_overrides == ()

    decided = realization_matches_rebuild(repo, run.id)
    assert decided.interventions == (intervention.id,)
    assert decided.history_fingerprint == before.history_fingerprint

    successor = restored_successor(repo, run, attempt, world["c100"])
    applied = apply(repo, intervention, successor)
    assert applied.checkpoint_ancestor == world["c100"].payload.checkpoint_ref
    after = realization_matches_rebuild(repo, run.id)
    assert after.candidate_fingerprint == before.candidate_fingerprint == run.candidate_fingerprint
    assert after.history_fingerprint != before.history_fingerprint
    assert after.artifact_lineage_fingerprint != before.artifact_lineage_fingerprint
    assert after.trajectory is not None and after.trajectory.application_ids == (applied.id,)

    with connect(db_path) as reopened:
        migrate(reopened)
        restored = ControlPlaneRepository(reopened)
        assert restored.training_interventions.for_action(str(governed.action.id)) == intervention
        assert restored.get_run_realization(run.id) == after


def test_proposal_is_bound_atomically_and_replays_without_policy(connection):
    repo, _, _, inputs, proposal = prepared(connection)
    first = propose(repo, inputs, proposal)
    before = tuple(count(repo, t) for t in ("actions", "policy_decisions", "outbox"))
    assert propose(repo, inputs, proposal, policy=None) == first
    assert tuple(count(repo, t) for t in ("actions", "policy_decisions", "outbox")) == before
    assert repo.numerical_recovery_bindings.for_plan(str(inputs.plan.id)).action_id == (
        first.action.id
    )


def test_denied_action_creates_no_intervention(connection):
    repo, _, _, inputs, proposal = prepared(connection)
    governed = propose(repo, inputs, proposal, policy=DENY)
    assert governed.action.status is ActionStatus.REJECTED
    with pytest.raises(InterventionNotAuthorizedError):
        record(repo, governed.action.id)
    assert count(repo, "training_interventions") == 0


def test_approval_is_required_before_an_intervention_exists(connection):
    repo, _, _, inputs, proposal = prepared(connection)
    governed = propose(repo, inputs, proposal, policy=APPROVAL)
    assert governed.action.status is ActionStatus.APPROVAL_PENDING
    with pytest.raises(InterventionNotAuthorizedError):
        record(repo, governed.action.id)
    assert count(repo, "training_interventions") == 0
    repo.approve_action(governed.action.id, approver=REVIEWER, reason="stabilise")
    intervention = record(repo, governed.action.id)
    assert intervention.action_id == governed.action.id


def test_stale_plan_cannot_manufacture_a_current_intervention(connection):
    repo, world, _, inputs, proposal = prepared(connection)
    governed = propose(repo, inputs, proposal)
    incident(repo, world["attempt"], sequence=10, signal=signal("numerical-inf"))
    events_before = count(repo, "events")
    with pytest.raises(StaleRecoveryContextError):
        record(repo, governed.action.id)
    assert count(repo, "training_interventions") == 0
    assert count(repo, "events") == events_before


def test_new_evidence_before_proposal_is_rejected_under_the_write_lock(connection):
    repo, world, _, inputs, proposal = prepared(connection)
    incident(repo, world["attempt"], sequence=10, signal=signal("process-failure"))
    with pytest.raises(StaleRecoveryContextError):
        propose(repo, inputs, proposal)
    assert count(repo, "actions") == count(repo, "numerical_recovery_action_bindings") == 0


def test_operational_action_cannot_manufacture_a_scientific_intervention(connection):
    repo, world, _, _, _ = prepared(connection)
    run = world["run"]
    governed = repo.propose_action(
        ChangeCheckpointInterval(target=ActionTarget(kind="run", id=str(run.id)), every_steps=10),
        experiment_id=run.experiment_id,
        proposed_by=REVIEWER,
        reason="operational",
        policy=ALLOW,
        capabilities=None,
    )
    assert governed.action.status is ActionStatus.VALIDATED
    decided = TrainingIntervention(
        run_id=run.id,
        action_id=governed.action.id,
        origin=InterventionOrigin.REACTIVE_HUMAN,
        trigger=ManualTrigger(actor=REVIEWER),
        replay_policy=InterventionReplayPolicy.APPLY_ONCE,
        mutation=LearningRateMutation(learning_rate=1e-4),
        rationale="not an LR action",
    )
    with pytest.raises(ProvenanceError, match="does not decide"):
        repo.record_training_intervention(decided, actor=ACTOR)

    # And the database refuses it even if the repository's checks were bypassed.
    with pytest.raises(sqlite3.IntegrityError, match="authorized scientific Action"):
        with write_transaction(connection):
            sequence = repo._write_sequenced_event(
                DomainEvent(
                    id=EventId.generate(),
                    experiment_id=str(run.experiment_id),
                    aggregate_type="Run",
                    aggregate_id=str(run.id),
                    aggregate_revision=run.revision,
                    event_type="TrainingInterventionRecorded",
                    actor=ACTOR,
                    payload=FrozenDict({"intervention_id": str(decided.id)}),
                ),
                (),
            )
            repo.training_interventions._insert(
                decided, experiment_id=str(run.experiment_id), event_sequence=sequence
            )
    assert count(repo, "training_interventions") == 0


def _human_lr_action(repo, run, proposer=REVIEWER):
    return repo.propose_action(
        ChangeLearningRate(target=ActionTarget(kind="run", id=str(run.id)), learning_rate=5e-5),
        experiment_id=run.experiment_id,
        proposed_by=proposer,
        reason="researcher lowers LR",
        policy=ALLOW,
        capabilities=None,
    ).action


def _human_intervention(action, run, **overrides):
    fields = {
        "run_id": run.id,
        "action_id": action.id,
        "origin": InterventionOrigin.REACTIVE_HUMAN,
        "trigger": ManualTrigger(actor=REVIEWER),
        "replay_policy": InterventionReplayPolicy.APPLY_ONCE,
        "mutation": LearningRateMutation(learning_rate=5e-5),
        "rationale": "researcher judgement",
    }
    fields.update(overrides)
    return TrainingIntervention(**fields)


def test_origin_must_match_governed_provenance(connection):
    repo, world, _, _, _ = prepared(connection)
    run = world["run"]
    action = _human_lr_action(repo, run)
    with pytest.raises(ProvenanceError, match="bound recovery decision"):
        repo.record_training_intervention(
            _human_intervention(
                action,
                run,
                origin=InterventionOrigin.REACTIVE_POLICY,
                trigger=world_trigger(repo, world),
                replay_policy=InterventionReplayPolicy.REAPPLY_AFTER_ROLLBACK,
            ),
            actor=ACTOR,
        )
    with pytest.raises(ProvenanceError, match="llm_agent"):
        repo.record_training_intervention(
            _human_intervention(
                action,
                run,
                origin=InterventionOrigin.REACTIVE_AGENT,
                trigger=world_trigger(repo, world),
            ),
            actor=ACTOR,
        )
    with pytest.raises(ProvenanceError, match="does not decide"):
        repo.record_training_intervention(
            _human_intervention(action, run, mutation=LearningRateMutation(learning_rate=1e-5)),
            actor=ACTOR,
        )
    recorded = repo.record_training_intervention(_human_intervention(action, run), actor=ACTOR)
    assert recorded.origin is InterventionOrigin.REACTIVE_HUMAN


def world_trigger(repo, world):
    target = RuntimeOperationTarget(kind="training-attempt", id=str(world["attempt"].id))
    observed = repo.incidents.for_attempt(target)[0]
    return IncidentTrigger(incident_category=observed.category, incident_id=observed.id)


def test_wrong_run_is_rejected(connection):
    repo, world, _, _, _ = prepared(connection)
    action = _human_lr_action(repo, world["run"])
    other_run, _ = another_run(repo, {"node": world["node"]})
    with pytest.raises(ProvenanceError):
        repo.record_training_intervention(_human_intervention(action, other_run), actor=ACTOR)
    assert count(repo, "training_interventions") == 0


def test_intervention_replay_is_idempotent_and_changes_conflict(connection):
    repo, _, _, inputs, proposal = prepared(connection)
    governed = propose(repo, inputs, proposal)
    first = record(repo, governed.action.id, intervention_id=InterventionId.generate())
    events = count(repo, "events")
    assert repo.record_training_intervention(first, actor=ACTOR) == first
    assert count(repo, "events") == events
    with pytest.raises(IdempotencyConflictError):
        repo.record_training_intervention(
            first.model_copy(update={"rationale": "a different story"}), actor=ACTOR
        )
    with pytest.raises(IdempotencyConflictError):
        record(repo, governed.action.id)  # another id for the same Action


# ---- applications -----------------------------------------------------------------------


def _recorded(connection):
    """A generic (human) intervention: application mechanics, not the numerical v1 path."""
    repo, world = seeded_lr_run(connection)
    return repo, world, human_intervention(repo, world["run"], learning_rate=1e-4)


def test_application_is_ordered_by_event_sequence_and_replays(connection):
    repo, world, intervention = _recorded(connection)
    application_id = InterventionApplicationId.generate()
    first = apply(repo, intervention, world["attempt"], application_id=application_id)
    assert first.event_sequence > 0
    assert repo.intervention_applications.for_run(str(world["run"].id)) == (first,)
    events = count(repo, "events")
    replayed = apply(repo, intervention, world["attempt"], application_id=application_id)
    assert replayed == first and count(repo, "events") == events
    with pytest.raises(IdempotencyConflictError):
        apply(repo, intervention, world["attempt"], application_id=application_id, step=1)
    # A rollback's re-application is a *new* record at an earlier position, later in sequence.
    again = apply(repo, intervention, world["attempt"], step=12_000, previous=1e-4)
    ordered = repo.intervention_applications.for_intervention(str(intervention.id))
    assert ordered == (first, again)
    assert again.event_sequence > first.event_sequence
    assert again.position.optimizer_step < first.position.optimizer_step


def test_application_rejects_wrong_attempt_value_or_intervention(connection):
    repo, world, intervention = _recorded(connection)
    _, foreign = another_run(repo, {"node": world["node"]})
    with pytest.raises(ProvenanceError, match="does not belong"):
        apply(repo, intervention, foreign)
    with pytest.raises(ProvenanceError, match="decided value"):
        repo.record_intervention_application(
            intervention.id,
            application_id=InterventionApplicationId.generate(),
            attempt_id=world["attempt"].id,
            position=TrainingPosition(optimizer_step=1),
            observed_previous_value=2e-4,
            applied_value=3e-5,
            actor=ACTOR,
        )
    with pytest.raises(AggregateNotFoundError):
        apply(
            repo,
            intervention.model_copy(update={"id": InterventionId.generate()}),
            world["attempt"],
        )
    assert count(repo, "intervention_applications") == 0


def test_failed_transaction_leaves_no_half_recorded_event_or_state(connection, monkeypatch):
    repo, world, intervention = _recorded(connection)
    events, outbox = count(repo, "events"), count(repo, "outbox")

    def boom(*args, **kwargs):
        raise RuntimeError("crash between event and row")

    monkeypatch.setattr(repo.intervention_applications, "_insert", boom)
    with pytest.raises(RuntimeError):
        apply(repo, intervention, world["attempt"])
    assert (count(repo, "events"), count(repo, "outbox")) == (events, outbox)
    assert count(repo, "intervention_applications") == 0


def test_failed_intervention_transaction_rolls_back(connection, monkeypatch):
    repo, _, _, inputs, proposal = prepared(connection)
    governed = propose(repo, inputs, proposal)
    events = count(repo, "events")
    monkeypatch.setattr(
        repo.training_interventions,
        "_insert",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")),
    )
    with pytest.raises(RuntimeError):
        record(repo, governed.action.id)
    assert count(repo, "events") == events
    assert count(repo, "training_interventions") == 0


@pytest.mark.parametrize(
    "table",
    ["training_interventions", "intervention_applications", "numerical_recovery_action_bindings"],
)
@pytest.mark.parametrize("operation", ["UPDATE {t} SET created_at = 'x'", "DELETE FROM {t}"])
def test_intervention_tables_are_append_only(connection, tmp_path, table, operation):
    repo, world, _, inputs, proposal = prepared(connection, tmp_path=tmp_path)
    intervention = record(repo, propose(repo, inputs, proposal).action.id)
    successor = restored_successor(repo, world["run"], world["attempt"], world["c100"])
    apply(repo, intervention, successor)
    assert count(repo, table) >= 1
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with write_transaction(connection):
            connection.execute(operation.format(t=table))


# ---- loop protection from durable state -------------------------------------------------


def _successor_nan(repo, world, number=2):
    attempt = make_attempt(world["run"], attempt_number=number)
    with write_transaction(repo._connection):
        repo.aggregates._insert_attempt(attempt)
    attempt = repo.transition_attempt(
        attempt.id,
        expected_revision=attempt.revision,
        new_status=RunAttemptStatus.FAILED,
        actor=ACTOR,
    )
    _, plan = nonfinite_plan(repo, attempt, sequence=3)
    return attempt, plan


def test_promised_reduction_absent_from_the_trajectory_escalates(connection, tmp_path):
    """E1 promised 1e-4, but the successor never confirmed applying it; E2 must not stack."""
    repo, world, _, inputs, proposal = prepared(connection, tmp_path=tmp_path)
    intervention = record(repo, propose(repo, inputs, proposal).action.id)
    successor = restored_successor(repo, world["run"], world["attempt"], world["c100"])
    successor = repo.transition_attempt(
        successor.id,
        expected_revision=successor.revision,
        new_status=RunAttemptStatus.FAILED,
        actor=ACTOR,
    )
    _, plan = nonfinite_plan(repo, successor, sequence=3)
    inputs = repo.numerical_recovery_inputs(str(plan.id), HALVE)
    assert inputs.current_learning_rate.value == 2e-4
    assert [p.intervention_id for p in inputs.prior_interventions] == [intervention.id]
    result = NumericalRecoveryPlanner().plan(inputs)
    assert isinstance(result, NumericalEscalation)
    assert result.code is NumericalEscalationCode.PRIOR_INTERVENTION_NOT_REFLECTED


def test_closed_episode_cannot_record_its_intervention(connection):
    repo, world, _, inputs, proposal = prepared(connection)
    governed = propose(repo, inputs, proposal)
    _successor_nan(repo, world)
    with pytest.raises(StaleRecoveryContextError):
        record(repo, governed.action.id)
    assert count(repo, "training_interventions") == 0
