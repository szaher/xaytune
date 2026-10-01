"""The pure replay planner: which interventions a restored successor must apply."""

from __future__ import annotations

import pytest

from xaytune.core.domain.incident import IncidentCategory
from xaytune.core.domain.intervention import (
    IncidentTrigger,
    InterventionApplication,
    InterventionDirectiveKind,
    InterventionOrigin,
    InterventionReplayPolicy,
    LearningRateMutation,
    ManualTrigger,
    MetricAggregation,
    MetricSource,
    MetricTrigger,
    TrainingIntervention,
    TrainingPosition,
)
from xaytune.core.domain.intervention_replay import (
    InterventionReplayError,
    plan_intervention_directives,
)
from xaytune.core.ids import (
    ActionId,
    IncidentId,
    InterventionApplicationId,
    RunAttemptId,
    RunId,
)
from xaytune.core.refs import Actor

RUN = RunId.generate()


def decided(lr, policy=InterventionReplayPolicy.REAPPLY_AFTER_ROLLBACK):
    if policy is InterventionReplayPolicy.APPLY_ONCE:
        trigger, origin = (
            ManualTrigger(actor=Actor(type="human", id="r")),
            (InterventionOrigin.REACTIVE_HUMAN),
        )
    elif policy is InterventionReplayPolicy.REARM_TRIGGER:
        trigger = MetricTrigger(
            metric_ref="train/loss",
            metric_schema_version="1",
            source=MetricSource.TRAINING_STREAM,
            aggregation=MetricAggregation.LAST,
            operator=">",
            threshold=9.0,
        )
        origin = InterventionOrigin.REACTIVE_AGENT
    else:
        trigger = IncidentTrigger(
            incident_category=IncidentCategory.NUMERICAL_NAN, incident_id=IncidentId.generate()
        )
        origin = InterventionOrigin.REACTIVE_POLICY
    return TrainingIntervention(
        run_id=RUN,
        action_id=ActionId.generate(),
        origin=origin,
        trigger=trigger,
        replay_policy=policy,
        mutation=LearningRateMutation(learning_rate=lr),
        rationale="r",
    )


def applied(intervention, sequence, previous):
    return InterventionApplication(
        id=InterventionApplicationId.generate(),
        intervention_id=intervention.id,
        attempt_id=RunAttemptId.generate(),
        event_sequence=sequence,
        position=TrainingPosition(optimizer_step=100),
        previous_value=previous,
        applied_value=intervention.mutation.learning_rate,
    )


def plan(interventions, applications, embodied, initial=None, restored=True):
    return plan_intervention_directives(
        declared_learning_rate=2e-4,
        interventions=interventions,
        applications=applications,
        restored=restored,
        embodied_application_ids=embodied,
        initial=initial,
    )


def test_initial_only_on_a_clean_run():
    first = decided(1e-4)
    result = plan([first], [], (), initial=first)
    (directive,) = result.directives
    assert directive.kind is InterventionDirectiveKind.INITIAL
    assert (directive.expected_previous_value, result.final_learning_rate) == (2e-4, 1e-4)


def test_rolled_back_application_is_reapplied_before_the_new_intervention():
    first, second = decided(1e-4), decided(5e-5)
    a1 = applied(first, 10, 2e-4)
    result = plan([first, second], [a1], (), initial=second)
    kinds = [(d.intervention_id, d.kind, d.expected_previous_value) for d in result.directives]
    assert kinds == [
        (first.id, InterventionDirectiveKind.REAPPLY_AFTER_ROLLBACK, 2e-4),
        (second.id, InterventionDirectiveKind.INITIAL, 1e-4),
    ]
    assert result.restored_learning_rate == 2e-4


def test_embodied_effect_survives_and_is_not_reapplied():
    first, second = decided(1e-4), decided(5e-5)
    a1 = applied(first, 10, 2e-4)
    result = plan([first, second], [a1], (str(a1.id),), initial=second)
    assert [d.kind for d in result.directives] == [InterventionDirectiveKind.INITIAL]
    assert result.restored_learning_rate == 1e-4


def test_any_embodied_application_of_an_intervention_counts_as_surviving():
    first = decided(1e-4)
    a1, a2 = applied(first, 10, 2e-4), applied(first, 20, 2e-4)
    assert plan([first], [a1, a2], (str(a2.id),)).directives == ()


def test_apply_once_is_deliberately_not_reapplied():
    human = decided(5e-5, InterventionReplayPolicy.APPLY_ONCE)
    a1 = applied(human, 10, 2e-4)
    result = plan([human], [a1], ())
    assert result.directives == () and result.restored_learning_rate == 2e-4


def test_rearm_fails_closed():
    rearm = decided(5e-5, InterventionReplayPolicy.REARM_TRIGGER)
    with pytest.raises(InterventionReplayError, match="re-armed"):
        plan([rearm], [applied(rearm, 10, 2e-4)], ())


@pytest.mark.parametrize("embodied", [None, ("intapp_unknown",)])
def test_unknown_or_ambiguous_lineage_fails_closed(embodied):
    first = decided(1e-4)
    with pytest.raises(InterventionReplayError):
        plan([first], [applied(first, 10, 2e-4)], embodied)


def test_unknown_lineage_also_refuses_a_first_effect():
    first = decided(1e-4)
    with pytest.raises(InterventionReplayError, match="does not record"):
        plan([first], [], None, initial=first)


def test_no_restore_cannot_carry_or_reapply():
    first = decided(1e-4)
    with pytest.raises(InterventionReplayError, match="restores no checkpoint"):
        plan([first], [applied(first, 10, 2e-4)], None, restored=False)
    assert plan([], [], None, restored=False).directives == ()


def test_an_intervention_that_already_took_effect_is_not_initial_again():
    first = decided(1e-4)
    with pytest.raises(InterventionReplayError, match="already taken effect"):
        plan([first], [applied(first, 10, 2e-4)], (), initial=first)


def test_reapplication_order_follows_first_effect_not_recorded_order():
    early, late = decided(1e-4), decided(5e-5)
    a_late = applied(late, 5, 2e-4)
    a_early = applied(early, 10, 5e-5)
    result = plan([early, late], [a_early, a_late], ())
    assert [d.intervention_id for d in result.directives] == [late.id, early.id]
    assert [d.expected_previous_value for d in result.directives] == [2e-4, 5e-5]
