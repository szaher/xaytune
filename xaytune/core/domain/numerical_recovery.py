"""Immutable inputs, outputs and Action binding for numerical recovery (PR-021).

Numerical instability is a *scientific* matter. Lowering the learning rate to
stabilise a continuing trajectory changes training semantics, so it is a
``TrainingIntervention`` decided through a governed ``ChangeLearningRate``
Action -- never an ``ExecutionOverride`` and never a new ``ExperimentNode``.

A proposal is intent for governance, never execution authority. Its algorithm
parameters come only from an explicit, versioned ``NumericalRecoveryPolicyV1``:
there is no built-in reduction factor.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from xaytune.core.checkpoint import Digest
from xaytune.core.clock import utc_now
from xaytune.core.domain.actions.builtin import ChangeLearningRate
from xaytune.core.domain.incident import IncidentCategory
from xaytune.core.domain.intervention import (
    IncidentTrigger,
    InterventionReplayPolicy,
    replay_policy_problem,
)
from xaytune.core.domain.recovery import RecoveryPlan
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import (
    ActionId,
    InterventionApplicationId,
    InterventionId,
    RecoveryEpisodeId,
    RecoveryPlanId,
    RunAttemptId,
    RunId,
)
from xaytune.core.immutable import FrozenDomainModel

__all__ = [
    "NONFINITE_CATEGORIES",
    "NUMERICAL_FAMILY",
    "EffectiveLearningRate",
    "NumericalEscalation",
    "NumericalEscalationCode",
    "NumericalLRProposal",
    "NumericalRecoveryActionBinding",
    "NumericalRecoveryInputsV1",
    "NumericalRecoveryPlannerResult",
    "NumericalRecoveryPolicyV1",
    "PriorNumericalIntervention",
    "UnsupportedNumericalRecoveryError",
]

NUMERICAL_FAMILY = "numerical-intervention"
"""The specialised family PR-019 arbitration assigns to numerical diagnoses."""

NONFINITE_CATEGORIES: frozenset[IncidentCategory] = frozenset(
    {IncidentCategory.NUMERICAL_NAN, IncidentCategory.NUMERICAL_INF}
)
"""The diagnoses the v1 planner can answer with a learning-rate reduction."""

_LearningRate = Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)]


class UnsupportedNumericalRecoveryError(ValueError):
    """Numerical recovery was armed where it cannot be honoured. Refused at submission."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("numerical recovery cannot be armed: " + "; ".join(reasons))


class NumericalRecoveryPolicyV1(FrozenDomainModel):
    """Explicit algorithm parameters. Every field is required; none has a default.

    ``minimum_learning_rate`` is ``None`` only when the caller states that no
    floor applies. A reduction that would fall below a stated floor escalates;
    it is never clamped.
    """

    schema_version: Literal["xaytune.numerical-recovery-policy/v1alpha1"] = (
        "xaytune.numerical-recovery-policy/v1alpha1"
    )
    learning_rate_multiplier: float = Field(gt=0, lt=1, strict=True, allow_inf_nan=False)
    minimum_learning_rate: _LearningRate | None

    @property
    def policy_fingerprint(self) -> str:
        return fingerprint(self)


class EffectiveLearningRate(FrozenDomainModel):
    """The base learning rate the source trajectory actually runs at, and why.

    Either the candidate's declared value (no application on the retained
    trajectory), or the value of one recorded ``InterventionApplication``.
    """

    value: _LearningRate
    application_id: InterventionApplicationId | None = None
    intervention_id: InterventionId | None = None

    @model_validator(mode="after")
    def _attested(self) -> EffectiveLearningRate:
        if (self.application_id is None) != (self.intervention_id is None):
            raise ValueError("an applied learning rate names its application and intervention")
        return self


class PriorNumericalIntervention(FrozenDomainModel):
    """An earlier numerical-recovery intervention on this run, in recorded order."""

    intervention_id: InterventionId
    action_id: ActionId
    episode_id: RecoveryEpisodeId
    previous_learning_rate: _LearningRate
    promised_learning_rate: _LearningRate

    @model_validator(mode="after")
    def _reduces(self) -> PriorNumericalIntervention:
        if self.promised_learning_rate >= self.previous_learning_rate:
            raise ValueError("a numerical intervention promised a reduction")
        return self


class NumericalRecoveryInputsV1(FrozenDomainModel):
    """Decision-bound facts, derived from durable records by the repository.

    ``policy`` is required and has no default: ``None`` states that no
    numerical recovery policy was configured, and the planner escalates.
    """

    schema_version: Literal["xaytune.numerical-recovery-inputs/v1alpha1"] = (
        "xaytune.numerical-recovery-inputs/v1alpha1"
    )
    plan: RecoveryPlan
    run_id: RunId
    candidate_fingerprint: str = Field(min_length=1)
    source_attempt_id: RunAttemptId
    execution_state_fingerprint: Digest
    current_learning_rate: EffectiveLearningRate
    prior_interventions: tuple[PriorNumericalIntervention, ...]
    policy: NumericalRecoveryPolicyV1 | None

    @model_validator(mode="after")
    def _bound_to_episode(self) -> NumericalRecoveryInputsV1:
        context = self.plan.inputs.context
        if (
            context.target.kind != "training-attempt"
            or self.run_id != context.run_id
            or str(self.source_attempt_id) != context.target.id
            or self.candidate_fingerprint != self.plan.inputs.candidate_fingerprint
            or self.execution_state_fingerprint != self.plan.execution_state_fingerprint
        ):
            raise ValueError("numerical inputs must bind the plan's training attempt and run")
        ids = [prior.intervention_id for prior in self.prior_interventions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate prior intervention")
        return self

    @property
    def input_fingerprint(self) -> str:
        return fingerprint(self)


class NumericalRecoveryPlannerResult(FrozenDomainModel):
    """Common freshness binding for a governed proposal or an escalation."""

    episode_id: RecoveryEpisodeId
    plan_id: RecoveryPlanId
    plan_sequence: StrictInt = Field(ge=1)
    input_fingerprint: Digest


class NumericalLRProposal(NumericalRecoveryPlannerResult):
    """One ``ChangeLearningRate`` intent and the intervention provenance it would carry.

    ``trigger`` and ``replay_policy`` are decided here, explicitly, and bound to
    the Action; the intervention later copies them rather than inferring them.
    """

    run_id: RunId
    candidate_fingerprint: str = Field(min_length=1)
    source_attempt_id: RunAttemptId
    source_execution_state_fingerprint: Digest
    policy_fingerprint: Digest
    previous_learning_rate: EffectiveLearningRate
    trigger: IncidentTrigger
    replay_policy: InterventionReplayPolicy
    action_spec: ChangeLearningRate

    @model_validator(mode="after")
    def _reduces_for_an_incident(self) -> NumericalLRProposal:
        spec = self.action_spec
        if (
            spec.target.kind != "run"
            or spec.target.id != str(self.run_id)
            or not spec.learning_rate < self.previous_learning_rate.value
        ):
            raise ValueError("numerical recovery must reduce the run's learning rate")
        if self.trigger.incident_id is None or (
            self.trigger.incident_category not in NONFINITE_CATEGORIES
        ):
            raise ValueError("numerical recovery is triggered by one exact nonfinite incident")
        problem = replay_policy_problem(self.trigger, self.replay_policy)
        if problem is not None:
            raise ValueError(problem)
        return self


class NumericalEscalationCode(str, Enum):
    PLAN_NOT_AWAITING_REVIEW = "plan-not-awaiting-review"
    CONFLICTING_SPECIALISED_EVIDENCE = "conflicting-specialised-evidence"
    UNSUPPORTED_EVIDENCE = "unsupported-evidence"
    AMBIGUOUS_EVIDENCE = "ambiguous-evidence"
    NO_POLICY = "no-policy"
    EPISODE_ALREADY_INTERVENED = "episode-already-intervened"
    PRIOR_INTERVENTION_NOT_REFLECTED = "prior-intervention-not-reflected"
    MINIMUM_LEARNING_RATE_REACHED = "minimum-learning-rate-reached"


class NumericalEscalation(NumericalRecoveryPlannerResult):
    """No autonomous learning-rate proposal; the run stays human-governed."""

    code: NumericalEscalationCode
    reason: str = Field(min_length=1)


class NumericalRecoveryActionBinding(FrozenDomainModel):
    """Append-only proof that a ChangeLearningRate Action answers one plan revision.

    Separate from the OOM ``RecoveryActionBinding`` v1alpha1, whose schema
    embeds an OOM proposal and is left unchanged. Action reason text is never
    authority; this binding is.
    """

    schema_version: Literal["xaytune.numerical-recovery-action-binding/v1alpha1"] = (
        "xaytune.numerical-recovery-action-binding/v1alpha1"
    )
    action_id: ActionId
    episode_id: RecoveryEpisodeId
    plan_id: RecoveryPlanId
    plan_sequence: int = Field(ge=1, strict=True)
    proposal: NumericalLRProposal
    proposal_fingerprint: Digest
    input_fingerprint: Digest
    source_execution_state_fingerprint: Digest
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _bound(self) -> NumericalRecoveryActionBinding:
        proposal = self.proposal
        if (
            self.episode_id != proposal.episode_id
            or self.plan_id != proposal.plan_id
            or self.plan_sequence != proposal.plan_sequence
            or self.proposal_fingerprint != fingerprint(proposal)
            or self.input_fingerprint != proposal.input_fingerprint
            or self.source_execution_state_fingerprint
            != proposal.source_execution_state_fingerprint
        ):
            raise ValueError("numerical Action binding disagrees with its proposal")
        return self

    @classmethod
    def for_proposal(
        cls, action_id: ActionId, proposal: NumericalLRProposal
    ) -> NumericalRecoveryActionBinding:
        return cls(
            action_id=action_id,
            episode_id=proposal.episode_id,
            plan_id=proposal.plan_id,
            plan_sequence=proposal.plan_sequence,
            proposal=proposal,
            proposal_fingerprint=fingerprint(proposal),
            input_fingerprint=proposal.input_fingerprint,
            source_execution_state_fingerprint=proposal.source_execution_state_fingerprint,
        )
