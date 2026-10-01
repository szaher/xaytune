"""TrainingIntervention / InterventionApplication semantics (ADR-011), and the projection."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from xaytune.core.domain.action import ActionTarget
from xaytune.core.domain.actions import ChangeLearningRate
from xaytune.core.domain.incident import IncidentCategory
from xaytune.core.domain.intervention import (
    IncidentTrigger,
    InterventionApplication,
    InterventionOrigin,
    InterventionReplayPolicy,
    LearningRateMutation,
    ManualTrigger,
    MetricAggregation,
    MetricSource,
    MetricTrigger,
    OptimizerStepTrigger,
    PolicyTrigger,
    StepTrigger,
    TokenCountTrigger,
    TrainingIntervention,
    TrainingPosition,
    mutation_for_action,
)
from xaytune.core.domain.realization import (
    AttemptAncestry,
    CheckpointAncestry,
    project_run_realization,
    retained_trajectory,
)
from xaytune.core.domain.run import Run
from xaytune.core.ids import (
    ActionId,
    CheckpointId,
    ExperimentId,
    ExperimentNodeId,
    IncidentId,
    InterventionApplicationId,
    InterventionId,
    RunAttemptId,
    RunId,
)
from xaytune.core.refs import Actor

RUN = RunId.generate()
HUMAN = Actor(type="human", id="researcher")
METRIC = MetricTrigger(
    metric_ref="train/loss",
    metric_schema_version="1",
    source=MetricSource.TRAINING_STREAM,
    aggregation=MetricAggregation.MEAN,
    operator=">",
    threshold=4.0,
    window=50,
)
INCIDENT = IncidentTrigger(
    incident_category=IncidentCategory.NUMERICAL_NAN, incident_id=IncidentId.generate()
)


def intervention(**overrides):
    fields = {
        "run_id": RUN,
        "action_id": ActionId.generate(),
        "origin": InterventionOrigin.REACTIVE_POLICY,
        "trigger": INCIDENT,
        "replay_policy": InterventionReplayPolicy.REAPPLY_AFTER_ROLLBACK,
        "mutation": LearningRateMutation(learning_rate=1e-4),
        "rationale": "loss became nonfinite",
    }
    fields.update(overrides)
    return TrainingIntervention(**fields)


def test_valid_numerical_intervention_is_immutable_and_has_no_status():
    decided = intervention(evidence_refs=(str(INCIDENT.incident_id),))
    assert decided.trigger == INCIDENT
    assert decided.replay_policy is InterventionReplayPolicy.REAPPLY_AFTER_ROLLBACK
    assert "status" not in TrainingIntervention.model_fields
    with pytest.raises(ValidationError):
        decided.rationale = "changed"  # type: ignore[misc]
    assert TrainingIntervention.model_validate_json(decided.model_dump_json()) == decided


def test_replay_policy_has_no_default_and_is_never_inferred_from_origin():
    fields = intervention().model_dump()
    del fields["replay_policy"]
    with pytest.raises(ValidationError, match="replay_policy"):
        TrainingIntervention.model_validate(fields)


@pytest.mark.parametrize(
    "trigger",
    [
        INCIDENT,
        ManualTrigger(actor=HUMAN),
        StepTrigger(global_step=20_000),
        OptimizerStepTrigger(optimizer_step=20_000),
        TokenCountTrigger(tokens_seen=10**9),
        PolicyTrigger(policy_rule_id="r", policy_version="1", policy_digest="sha256:x"),
    ],
)
def test_rearm_is_refused_at_creation_for_non_rearmable_triggers(trigger):
    origin = (
        InterventionOrigin.REACTIVE_HUMAN
        if isinstance(trigger, ManualTrigger)
        else InterventionOrigin.REACTIVE_POLICY
    )
    with pytest.raises(ValidationError, match="cannot be re-armed"):
        intervention(
            trigger=trigger, origin=origin, replay_policy=InterventionReplayPolicy.REARM_TRIGGER
        )


def test_reconstructible_metric_trigger_can_be_rearmed():
    decided = intervention(
        trigger=METRIC,
        origin=InterventionOrigin.REACTIVE_AGENT,
        replay_policy=InterventionReplayPolicy.REARM_TRIGGER,
    )
    assert decided.trigger == METRIC
    with pytest.raises(ValidationError, match="evaluator"):
        MetricTrigger.model_validate({**METRIC.model_dump(), "source": "evaluator"})


def test_scheduled_origin_and_manual_trigger_consistency():
    with pytest.raises(ValidationError, match="schedule"):
        intervention(origin=InterventionOrigin.SCHEDULED)
    with pytest.raises(ValidationError, match="schedule"):
        intervention(schedule_ref="lr-stage-2")
    with pytest.raises(ValidationError, match="human"):
        intervention(trigger=ManualTrigger(actor=HUMAN))
    with pytest.raises(ValidationError, match="human"):
        ManualTrigger(actor=Actor(type="llm_agent", id="planner"))


def test_learning_rate_mutation_is_typed_versioned_and_finite():
    spec = ChangeLearningRate(target=ActionTarget(kind="run", id=str(RUN)), learning_rate=1e-4)
    assert mutation_for_action(spec) == LearningRateMutation(learning_rate=1e-4)
    assert LearningRateMutation(learning_rate=1e-4).schema_version.endswith("/v1alpha1")
    for bad in (0.0, -1e-4, float("inf"), float("nan"), "1e-4"):
        with pytest.raises(ValidationError):
            LearningRateMutation(learning_rate=bad)
    with pytest.raises(ValidationError):
        intervention(mutation={"learning_rate": 1e-4, "optimizer": "sgd"})


def test_training_position_needs_a_coordinate_and_is_not_an_ordering_key():
    with pytest.raises(ValidationError, match="coordinate"):
        TrainingPosition()
    assert TrainingPosition(optimizer_step=22_000).optimizer_step == 22_000


def _application(intervention_id, attempt, sequence, step, applied=1e-4):
    return InterventionApplication(
        id=InterventionApplicationId.generate(),
        intervention_id=intervention_id,
        attempt_id=attempt,
        event_sequence=sequence,
        position=TrainingPosition(optimizer_step=step),
        previous_value=2e-4,
        applied_value=applied,
    )


def _run():
    return Run(
        id=RUN,
        node_id=ExperimentNodeId.generate(),
        experiment_id=ExperimentId.generate(),
        seed=7,
        candidate_fingerprint="sha256:candidate",
    )


def test_rollback_retains_only_applications_on_the_restored_trajectory():
    """ADR-011: apply at 20k, roll back to a checkpoint taken at 15k, apply again."""
    decided = intervention(id=InterventionId.generate())
    first, second = RunAttemptId.generate(), RunAttemptId.generate()
    checkpoint = CheckpointId.generate()
    early = _application(decided.id, first, sequence=10, step=20_000)
    again = _application(decided.id, second, sequence=20, step=15_000)
    attempts = (
        AttemptAncestry(attempt_id=first, attempt_number=1),
        AttemptAncestry(attempt_id=second, attempt_number=2, restored_from=checkpoint),
    )
    checkpoints = (
        CheckpointAncestry(
            checkpoint_id=checkpoint, producer_attempt_id=first, embodied_application_ids=()
        ),
    )
    trajectory = retained_trajectory(second, attempts, checkpoints, (early, again))
    assert trajectory is not None
    assert trajectory.application_ids == (again.id,)
    assert trajectory.checkpoint_ancestry == (checkpoint,)

    realization = project_run_realization(_run(), (decided,), (again, early), attempts, checkpoints)
    assert [r.application_id for r in realization.applied_interventions] == [early.id, again.id]
    once = project_run_realization(
        _run(),
        (decided,),
        (again.model_copy(update={"event_sequence": 10}),),
        attempts,
        checkpoints,
    )
    assert realization.history_fingerprint != once.history_fingerprint
    assert realization.artifact_lineage_fingerprint == once.artifact_lineage_fingerprint
    assert realization.candidate_fingerprint == once.candidate_fingerprint == "sha256:candidate"


def test_unknown_embodied_applications_make_lineage_unknown_not_empty():
    first, second = RunAttemptId.generate(), RunAttemptId.generate()
    checkpoint = CheckpointId.generate()
    attempts = (
        AttemptAncestry(attempt_id=first, attempt_number=1),
        AttemptAncestry(attempt_id=second, attempt_number=2, restored_from=checkpoint),
    )
    unknown = (
        CheckpointAncestry(
            checkpoint_id=checkpoint, producer_attempt_id=first, embodied_application_ids=None
        ),
    )
    realization = project_run_realization(_run(), (), (), attempts, unknown)
    assert realization.trajectory is None
    assert realization.artifact_lineage_fingerprint is None
    assert retained_trajectory(second, attempts[1:], unknown, ()) is None


def test_projection_is_deterministic():
    decided = intervention(id=InterventionId.generate())
    attempt = RunAttemptId.generate()
    application = _application(decided.id, attempt, sequence=5, step=100)
    attempts = (AttemptAncestry(attempt_id=attempt, attempt_number=1),)
    first = project_run_realization(_run(), (decided,), (application,), attempts, ())
    assert first == project_run_realization(_run(), (decided,), (application,), attempts, ())
    assert first.trajectory is not None and first.trajectory.application_ids == (application.id,)
    empty = project_run_realization(_run(), (decided,), (), attempts, ())
    assert first.history_fingerprint != empty.history_fingerprint
