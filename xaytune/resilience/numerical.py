"""Pure deterministic numerical recovery planning; no action, storage or runtime effects.

The planner answers one question: may a governed ``ChangeLearningRate`` be
*proposed* for this nonfinite episode? Eligibility comes only from structured
accepted evidence and PR-019 arbitration, never from ``RecoveryPlan.reason``.
Everything else -- policy, approval, applicability -- is the Action path's.
"""

from __future__ import annotations

from xaytune.core.domain.action import ActionTarget
from xaytune.core.domain.actions.builtin import ChangeLearningRate
from xaytune.core.domain.intervention import IncidentTrigger, InterventionReplayPolicy
from xaytune.core.domain.numerical_recovery import (
    NONFINITE_CATEGORIES,
    NUMERICAL_FAMILY,
    NumericalEscalation,
    NumericalEscalationCode,
    NumericalLRProposal,
    NumericalRecoveryInputsV1,
)
from xaytune.core.domain.recovery import Recoverability, RecoveryStrategy, diagnosis_requirement

__all__ = ["NumericalRecoveryPlanner"]


class NumericalRecoveryPlanner:
    """Propose one learning-rate reduction, or a deterministic reason to escalate.

    Version 1 semantics, fixed by ``version`` rather than configuration:

    - the trigger is the earliest accepted nonfinite incident, by membership;
    - the replay policy is ``REAPPLY_AFTER_ROLLBACK``. An incident is a past
      occurrence and cannot be re-armed, and a stabilising reduction that a
      restore rewinds past must still hold on the retained trajectory.
    """

    name = "deterministic-numerical"
    version = "1"
    replay_policy = InterventionReplayPolicy.REAPPLY_AFTER_ROLLBACK

    def plan(self, inputs: NumericalRecoveryInputsV1) -> NumericalLRProposal | NumericalEscalation:
        plan = inputs.plan

        def escalate(code: NumericalEscalationCode, reason: str) -> NumericalEscalation:
            return NumericalEscalation(
                episode_id=plan.episode_id,
                plan_id=plan.id,
                plan_sequence=plan.sequence,
                input_fingerprint=inputs.input_fingerprint,
                code=code,
                reason=reason,
            )

        if (
            plan.strategy is not RecoveryStrategy.PAUSE_FOR_APPROVAL
            or plan.recoverability is not Recoverability.REQUIRES_HUMAN
        ):
            return escalate(
                NumericalEscalationCode.PLAN_NOT_AWAITING_REVIEW,
                "the effective recovery plan does not await specialised review",
            )

        pairs = [
            (evidence, category, diagnosis_requirement(category))
            for evidence in plan.inputs.accepted_evidence
            for category in evidence.categories
        ]
        families = {r.specialised_family for _, _, r in pairs if r.specialised_family is not None}
        if families - {NUMERICAL_FAMILY}:
            return escalate(
                NumericalEscalationCode.CONFLICTING_SPECIALISED_EVIDENCE,
                "accepted evidence requires another specialised recovery family",
            )
        if any(r.specialised_family != NUMERICAL_FAMILY for _, _, r in pairs):
            return escalate(
                NumericalEscalationCode.UNSUPPORTED_EVIDENCE,
                "accepted evidence includes diagnoses outside numerical recovery",
            )
        nonfinite = sorted(
            (
                evidence
                for evidence in plan.inputs.accepted_evidence
                if NONFINITE_CATEGORIES & set(evidence.categories)
            ),
            key=lambda evidence: evidence.membership_sequence,
        )
        if not nonfinite:
            return escalate(
                NumericalEscalationCode.UNSUPPORTED_EVIDENCE,
                "no accepted nonfinite diagnosis; v1 answers only NaN or Inf",
            )
        trigger_evidence = nonfinite[0]
        categories = NONFINITE_CATEGORIES & set(trigger_evidence.categories)
        if len(categories) != 1:
            return escalate(
                NumericalEscalationCode.AMBIGUOUS_EVIDENCE,
                "the triggering incident is diagnosed as both NaN and Inf",
            )

        policy = inputs.policy
        if policy is None:
            return escalate(
                NumericalEscalationCode.NO_POLICY,
                "no explicit numerical recovery policy; the run stays human-governed",
            )

        priors = inputs.prior_interventions
        if any(prior.episode_id == plan.episode_id for prior in priors):
            return escalate(
                NumericalEscalationCode.EPISODE_ALREADY_INTERVENED,
                "this episode already produced a numerical intervention",
            )
        current = inputs.current_learning_rate
        if priors:
            latest = priors[-1]
            if (
                current.intervention_id != latest.intervention_id
                or current.value != latest.promised_learning_rate
            ):
                return escalate(
                    NumericalEscalationCode.PRIOR_INTERVENTION_NOT_REFLECTED,
                    "the previous numerical intervention is not the trajectory's "
                    "applied learning rate",
                )

        proposed = current.value * policy.learning_rate_multiplier
        floor = policy.minimum_learning_rate
        if not 0 < proposed < current.value or (floor is not None and proposed < floor):
            return escalate(
                NumericalEscalationCode.MINIMUM_LEARNING_RATE_REACHED,
                "the reduced learning rate would fall below the configured minimum",
            )
        return NumericalLRProposal(
            episode_id=plan.episode_id,
            plan_id=plan.id,
            plan_sequence=plan.sequence,
            input_fingerprint=inputs.input_fingerprint,
            run_id=inputs.run_id,
            candidate_fingerprint=inputs.candidate_fingerprint,
            source_attempt_id=inputs.source_attempt_id,
            source_execution_state_fingerprint=inputs.execution_state_fingerprint,
            policy_fingerprint=policy.policy_fingerprint,
            previous_learning_rate=current,
            trigger=IncidentTrigger(
                incident_category=next(iter(categories)),
                incident_id=trigger_evidence.incident_id,
            ),
            replay_policy=self.replay_policy,
            action_spec=ChangeLearningRate(
                target=ActionTarget(kind="run", id=str(inputs.run_id)),
                learning_rate=proposed,
            ),
        )
