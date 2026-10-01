"""The numerical planner is pure, evidence-structured and never guesses a reduction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.test_storage.numerical_fixtures import HALVE, nonfinite_plan, seeded_lr_run
from tests.test_storage.test_recovery_episodes import decide, incident, signal
from xaytune.core.domain.incident import IncidentCategory
from xaytune.core.domain.intervention import InterventionReplayPolicy
from xaytune.core.domain.numerical_recovery import (
    EffectiveLearningRate,
    NumericalEscalation,
    NumericalEscalationCode,
    NumericalLRProposal,
    NumericalRecoveryInputsV1,
    NumericalRecoveryPolicyV1,
    PriorNumericalIntervention,
)
from xaytune.core.ids import (
    ActionId,
    InterventionApplicationId,
    InterventionId,
    RecoveryEpisodeId,
)
from xaytune.resilience.numerical import NumericalRecoveryPlanner

PLANNER = NumericalRecoveryPlanner()


def inputs_for(connection, reason="numerical-nan", policy=HALVE, extra=None):
    repo, world = seeded_lr_run(connection)
    observed, plan = nonfinite_plan(repo, world["attempt"], reason=reason)
    if extra is not None:
        later = incident(repo, world["attempt"], sequence=10, signal=signal(extra))
        plan = decide(repo, later)
    return repo, world, observed, repo.numerical_recovery_inputs(str(plan.id), policy)


@pytest.mark.parametrize(
    "reason,category",
    [
        ("numerical-nan", IncidentCategory.NUMERICAL_NAN),
        ("numerical-inf", IncidentCategory.NUMERICAL_INF),
    ],
)
def test_nonfinite_incident_yields_one_explicit_lr_proposal(connection, reason, category):
    _, world, observed, inputs = inputs_for(connection, reason=reason)
    proposal = PLANNER.plan(inputs)
    assert isinstance(proposal, NumericalLRProposal)
    assert proposal.action_spec.learning_rate == pytest.approx(1e-4)
    assert proposal.action_spec.target.id == str(world["run"].id)
    assert proposal.previous_learning_rate == EffectiveLearningRate(value=2e-4)
    assert proposal.trigger.incident_category is category
    assert proposal.trigger.incident_id == observed.id
    assert proposal.replay_policy is InterventionReplayPolicy.REAPPLY_AFTER_ROLLBACK
    assert proposal.policy_fingerprint == HALVE.policy_fingerprint
    assert (proposal.plan_id, proposal.plan_sequence) == (inputs.plan.id, inputs.plan.sequence)


def test_same_inputs_same_output_and_fingerprints(connection):
    _, _, _, inputs = inputs_for(connection)
    assert PLANNER.plan(inputs) == PLANNER.plan(inputs)
    assert (
        inputs.input_fingerprint
        == NumericalRecoveryInputsV1.model_validate_json(inputs.model_dump_json()).input_fingerprint
    )
    other = inputs.model_copy(
        update={
            "policy": NumericalRecoveryPolicyV1(
                learning_rate_multiplier=0.25, minimum_learning_rate=None
            )
        }
    )
    assert other.input_fingerprint != inputs.input_fingerprint


def test_without_an_explicit_policy_the_run_stays_human_governed(connection):
    _, _, _, inputs = inputs_for(connection, policy=None)
    result = PLANNER.plan(inputs)
    assert isinstance(result, NumericalEscalation)
    assert result.code is NumericalEscalationCode.NO_POLICY


def test_policy_has_no_defaults_and_bounds_the_multiplier():
    with pytest.raises(ValidationError):
        NumericalRecoveryPolicyV1(learning_rate_multiplier=0.5)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        NumericalRecoveryPolicyV1(minimum_learning_rate=None)  # type: ignore[call-arg]
    for bad in (0.0, 1.0, 1.5, -0.5, float("nan")):
        with pytest.raises(ValidationError):
            NumericalRecoveryPolicyV1(learning_rate_multiplier=bad, minimum_learning_rate=None)


@pytest.mark.parametrize(
    "extra,code",
    [
        ("cuda-oom", NumericalEscalationCode.CONFLICTING_SPECIALISED_EVIDENCE),
        ("checkpoint-write-failure", NumericalEscalationCode.UNSUPPORTED_EVIDENCE),
        ("config-error", NumericalEscalationCode.PLAN_NOT_AWAITING_REVIEW),
    ],
)
def test_mixed_evidence_fails_closed(connection, extra, code):
    _, _, _, inputs = inputs_for(connection, extra=extra)
    result = PLANNER.plan(inputs)
    assert isinstance(result, NumericalEscalation)
    assert result.code is code


def test_the_failed_workers_transient_exit_does_not_block_recovery(connection):
    """As for OOM: the nonzero exit that follows the failure is transient evidence."""
    _, _, observed, inputs = inputs_for(connection, extra="process-failure")
    proposal = PLANNER.plan(inputs)
    assert isinstance(proposal, NumericalLRProposal)
    assert proposal.trigger.incident_id == observed.id


def test_numerical_family_without_a_nonfinite_diagnosis_is_unsupported(connection):
    _, _, _, inputs = inputs_for(connection, reason="gradient-explosion")
    result = PLANNER.plan(inputs)
    assert isinstance(result, NumericalEscalation)
    assert result.code is NumericalEscalationCode.UNSUPPORTED_EVIDENCE


def test_incident_diagnosed_both_nan_and_inf_is_ambiguous(connection):
    _, _, _, inputs = inputs_for(connection)
    evidence = inputs.plan.inputs.accepted_evidence[0].model_copy(
        update={
            "categories": (IncidentCategory.NUMERICAL_INF, IncidentCategory.NUMERICAL_NAN),
        }
    )
    both = inputs.plan.inputs.model_copy(update={"accepted_evidence": (evidence,)})
    plan = inputs.plan.model_copy(
        update={"inputs": both, "accepted_evidence_fingerprint": both.accepted_evidence_fingerprint}
    )
    result = PLANNER.plan(inputs.model_copy(update={"plan": plan}))
    assert isinstance(result, NumericalEscalation)
    assert result.code is NumericalEscalationCode.AMBIGUOUS_EVIDENCE


def test_plan_reason_text_is_not_an_eligibility_api(connection):
    _, _, _, inputs = inputs_for(connection)
    reworded = inputs.model_copy(
        update={"plan": inputs.plan.model_copy(update={"reason": "adaptive-execution please"})}
    )
    original, changed = PLANNER.plan(inputs), PLANNER.plan(reworded)
    assert isinstance(original, NumericalLRProposal) and isinstance(changed, NumericalLRProposal)
    assert changed.action_spec == original.action_spec
    assert changed.trigger == original.trigger


def test_configured_floor_escalates_rather_than_clamps(connection):
    floor = NumericalRecoveryPolicyV1(learning_rate_multiplier=0.5, minimum_learning_rate=1.5e-4)
    _, _, _, inputs = inputs_for(connection, policy=floor)
    result = PLANNER.plan(inputs)
    assert isinstance(result, NumericalEscalation)
    assert result.code is NumericalEscalationCode.MINIMUM_LEARNING_RATE_REACHED
    exact = NumericalRecoveryPolicyV1(learning_rate_multiplier=0.5, minimum_learning_rate=1e-4)
    assert isinstance(
        PLANNER.plan(inputs.model_copy(update={"policy": exact})), NumericalLRProposal
    )


def _prior(episode=None, promised=1e-4):
    return PriorNumericalIntervention(
        intervention_id=InterventionId.generate(),
        action_id=ActionId.generate(),
        episode_id=episode or RecoveryEpisodeId.generate(),
        previous_learning_rate=2e-4,
        promised_learning_rate=promised,
    )


def test_promised_reduction_not_reflected_escalates(connection):
    _, _, _, inputs = inputs_for(connection)
    prior = _prior()
    unreflected = inputs.model_copy(update={"prior_interventions": (prior,)})
    result = PLANNER.plan(unreflected)
    assert isinstance(result, NumericalEscalation)
    assert result.code is NumericalEscalationCode.PRIOR_INTERVENTION_NOT_REFLECTED

    # Right value, but not attested by that intervention's application: still refused.
    unattested = unreflected.model_copy(
        update={"current_learning_rate": EffectiveLearningRate(value=1e-4)}
    )
    assert PLANNER.plan(unattested).code is (  # type: ignore[union-attr]
        NumericalEscalationCode.PRIOR_INTERVENTION_NOT_REFLECTED
    )

    reflected = unreflected.model_copy(
        update={
            "current_learning_rate": EffectiveLearningRate(
                value=1e-4,
                application_id=InterventionApplicationId.generate(),
                intervention_id=prior.intervention_id,
            )
        }
    )
    proposal = PLANNER.plan(reflected)
    assert isinstance(proposal, NumericalLRProposal)
    assert proposal.action_spec.learning_rate == pytest.approx(5e-5)


def test_one_episode_yields_at_most_one_intervention(connection):
    _, _, _, inputs = inputs_for(connection)
    same = _prior(episode=inputs.plan.episode_id)
    result = PLANNER.plan(
        inputs.model_copy(
            update={
                "prior_interventions": (same,),
                "current_learning_rate": EffectiveLearningRate(
                    value=1e-4,
                    application_id=InterventionApplicationId.generate(),
                    intervention_id=same.intervention_id,
                ),
            }
        )
    )
    assert isinstance(result, NumericalEscalation)
    assert result.code is NumericalEscalationCode.EPISODE_ALREADY_INTERVENED


def test_inputs_must_bind_the_plans_run_and_attempt(connection):
    _, _, _, inputs = inputs_for(connection)
    fields = inputs.model_dump()
    fields["candidate_fingerprint"] = "sha256:other"
    with pytest.raises(ValidationError, match="bind"):
        NumericalRecoveryInputsV1.model_validate(fields)
    fields = inputs.model_dump()
    del fields["policy"]
    with pytest.raises(ValidationError, match="policy"):
        NumericalRecoveryInputsV1.model_validate(fields)
