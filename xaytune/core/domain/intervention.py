"""Scientific changes to a continuing run: decisions and their applications (ADR-011).

```text
TrainingIntervention     the durable scientific decision an authorized Action produced
                         "lower LR to 1e-4 because incident inc_... was nonfinite"

InterventionApplication  one confirmed occurrence of that decision taking effect
                         "intervention_... applied at optimizer step 20,000"
```

An intervention is never an execution mechanism and never an
``ExecutionOverride``: it changes training semantics while leaving the node and
its ``CandidateFingerprint`` intact. It carries no status. Whether it took
effect is answered only by its append-only applications, which a rollback never
deletes.

Replay behaviour is explicit, required and immutable. Nothing here infers it
from origin, and a trigger that cannot be re-armed is refused when the
intervention is built, not during a recovery.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import Field, StrictBool, StrictInt, model_validator

from xaytune.core.clock import utc_now
from xaytune.core.domain.actions.builtin import ChangeLearningRate
from xaytune.core.domain.incident import IncidentCategory
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import (
    ActionId,
    IncidentId,
    InterventionApplicationId,
    InterventionId,
    RunAttemptId,
    RunId,
)
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.refs import Actor, CheckpointRef

__all__ = [
    "IncidentTrigger",
    "InterventionApplication",
    "InterventionOrigin",
    "InterventionReplayPolicy",
    "InterventionTrigger",
    "LearningRateMutation",
    "ManualTrigger",
    "MetricAggregation",
    "MetricSource",
    "MetricTrigger",
    "OptimizerStepTrigger",
    "PolicyTrigger",
    "StepTrigger",
    "TokenCountTrigger",
    "TrainingIntervention",
    "TrainingMutation",
    "TrainingPosition",
    "TriggerEvaluation",
    "mutation_for_action",
    "replay_policy_problem",
]

_Count = Annotated[int, Field(ge=0, strict=True)]
_LearningRate = Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)]


class InterventionOrigin(str, Enum):
    """Who or what decided the change. Recorded, never inferred (ADR-011 §4)."""

    SCHEDULED = "scheduled"
    REACTIVE_AGENT = "reactive-agent"
    REACTIVE_HUMAN = "reactive-human"
    REACTIVE_POLICY = "reactive-policy"


class InterventionReplayPolicy(str, Enum):
    """What happens to an intervention when a restore rewinds past its application.

    Required and without a default. The ADR-011 table of per-origin
    recommendations is advice to the proposer, never a fallback.
    """

    REAPPLY_AFTER_ROLLBACK = "reapply-after-rollback"
    APPLY_ONCE = "apply-once"
    REARM_TRIGGER = "rearm-trigger"


# ---- triggers: why an intervention exists ------------------------------------------------


class StepTrigger(FrozenDomainModel):
    type: Literal["step"] = "step"
    global_step: _Count


class OptimizerStepTrigger(FrozenDomainModel):
    type: Literal["optimizer-step"] = "optimizer-step"
    optimizer_step: _Count


class TokenCountTrigger(FrozenDomainModel):
    type: Literal["tokens"] = "tokens"
    tokens_seen: _Count


class MetricSource(str, Enum):
    TRAINING_STREAM = "training-stream"
    EVALUATOR = "evaluator"
    DERIVED = "derived"


class MetricAggregation(str, Enum):
    LAST = "last"
    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"


class MetricTrigger(FrozenDomainModel):
    """A standing condition on one identified metric series.

    Every field that re-evaluation needs is required, so a persisted metric
    trigger is reconstructible against restored state by construction: the
    series identity and schema, its source and aggregation, and the comparator.
    """

    type: Literal["metric"] = "metric"
    metric_ref: str = Field(min_length=1)
    metric_schema_version: str = Field(min_length=1)
    source: MetricSource
    aggregation: MetricAggregation
    operator: Literal[">", ">=", "<", "<=", "=="]
    threshold: float = Field(strict=True, allow_inf_nan=False)
    window: int | None = Field(default=None, ge=1, strict=True)
    consecutive: int | None = Field(default=None, ge=1, strict=True)
    evaluator_ref: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _evaluator_named(self) -> MetricTrigger:
        if (self.source is MetricSource.EVALUATOR) != (self.evaluator_ref is not None):
            raise ValueError("an evaluator-sourced metric names its evaluator, and only then")
        return self


class IncidentTrigger(FrozenDomainModel):
    """A past occurrence. It is not a standing condition and cannot be re-armed."""

    type: Literal["incident"] = "incident"
    incident_category: IncidentCategory
    incident_id: IncidentId | None = None


class ManualTrigger(FrozenDomainModel):
    """A human judgement. It cannot be re-derived from restored state."""

    type: Literal["manual"] = "manual"
    actor: Actor

    @model_validator(mode="after")
    def _human(self) -> ManualTrigger:
        if self.actor.type != "human":
            raise ValueError("a manual trigger records the human who decided")
        return self


class PolicyTrigger(FrozenDomainModel):
    """The policy rule, at the exact version and digest that was evaluated."""

    type: Literal["policy"] = "policy"
    policy_rule_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    policy_digest: str = Field(min_length=1)


InterventionTrigger = Annotated[
    Union[  # noqa: UP007 -- a runtime union, which the discriminator needs
        StepTrigger,
        OptimizerStepTrigger,
        TokenCountTrigger,
        MetricTrigger,
        IncidentTrigger,
        ManualTrigger,
        PolicyTrigger,
    ],
    Field(discriminator="type"),
]

_REARMABLE: frozenset[str] = frozenset({"metric"})
"""Triggers whose persisted form can be re-evaluated against restored state.

``policy`` is deliberately absent in v1. ADR-011 permits it only for a
state-dependent rule, and no reconstruction contract for policy-rule state
exists yet, so claiming re-armability would be claiming what cannot be checked.
"""


def replay_policy_problem(
    trigger: StepTrigger
    | OptimizerStepTrigger
    | TokenCountTrigger
    | MetricTrigger
    | IncidentTrigger
    | ManualTrigger
    | PolicyTrigger,
    replay_policy: InterventionReplayPolicy,
) -> str | None:
    """Why *trigger* cannot use *replay_policy*, or ``None``. Checked at creation."""
    if replay_policy is InterventionReplayPolicy.REARM_TRIGGER and trigger.type not in _REARMABLE:
        return (
            f"a {trigger.type} trigger cannot be re-armed: it is monotone, a past "
            f"occurrence, a human decision, or not reconstructible from restored state"
        )
    return None


# ---- mutations: what the intervention changes --------------------------------------------


class LearningRateMutation(FrozenDomainModel):
    """Replace the run's base learning rate, which any declared schedule then scales.

    The only mutation in v1. ``TrainingMutation`` becomes a discriminated
    union on ``type`` when a second one is accepted.
    """

    type: Literal["learning-rate"] = "learning-rate"
    schema_version: Literal["xaytune.training-mutation.learning-rate/v1alpha1"] = (
        "xaytune.training-mutation.learning-rate/v1alpha1"
    )
    learning_rate: _LearningRate


TrainingMutation = LearningRateMutation


def mutation_for_action(spec: ChangeLearningRate) -> TrainingMutation:
    """The mutation an authorized ``ChangeLearningRate`` Action decides."""
    return LearningRateMutation(learning_rate=spec.learning_rate)


# ---- the decision ------------------------------------------------------------------------


class TrainingIntervention(FrozenDomainModel):
    """The immutable scientific decision recorded from one authorized Action.

    It has no status: governance lives on the Action, effect on applications.
    """

    schema_version: Literal["xaytune.training-intervention/v1alpha1"] = (
        "xaytune.training-intervention/v1alpha1"
    )
    id: InterventionId = Field(default_factory=InterventionId.generate)
    run_id: RunId
    action_id: ActionId

    origin: InterventionOrigin
    trigger: InterventionTrigger
    replay_policy: InterventionReplayPolicy

    schedule_ref: str | None = Field(default=None, min_length=1)
    derived_from: InterventionId | None = None

    mutation: TrainingMutation
    rationale: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = ()

    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _semantics(self) -> TrainingIntervention:
        problem = replay_policy_problem(self.trigger, self.replay_policy)
        if problem is not None:
            raise ValueError(problem)
        if (self.origin is InterventionOrigin.SCHEDULED) != (self.schedule_ref is not None):
            raise ValueError("a scheduled intervention names its schedule entry, and only it does")
        if isinstance(self.trigger, ManualTrigger) and (
            self.origin is not InterventionOrigin.REACTIVE_HUMAN
        ):
            raise ValueError("a manual trigger is a human's reactive decision")
        if self.derived_from == self.id:
            raise ValueError("an intervention cannot derive from itself")
        if any(not ref for ref in self.evidence_refs) or len(set(self.evidence_refs)) != len(
            self.evidence_refs
        ):
            raise ValueError("evidence references are nonempty and unique")
        return self

    def semantic_fingerprint(self) -> str:
        """Identity for idempotent replay: everything but the creation time."""
        return fingerprint(self.model_dump(mode="json", exclude={"created_at"}))


# ---- the effect --------------------------------------------------------------------------


class TrainingPosition(FrozenDomainModel):
    """Where in training something happened. Never an ordering key.

    A restore moves it backwards; the durable event sequence orders history.
    """

    global_step: _Count | None = None
    optimizer_step: _Count | None = None
    tokens_seen: _Count | None = None
    examples_seen: _Count | None = None

    @model_validator(mode="after")
    def _known(self) -> TrainingPosition:
        if all(
            value is None
            for value in (
                self.global_step,
                self.optimizer_step,
                self.tokens_seen,
                self.examples_seen,
            )
        ):
            raise ValueError("a training position records at least one coordinate")
        return self


class TriggerEvaluation(FrozenDomainModel):
    """The observed values an application decided on, kept for later audit."""

    matched: StrictBool
    observed_values: FrozenDict = Field(default_factory=FrozenDict)
    evaluated_at: datetime


class InterventionApplication(FrozenDomainModel):
    """One confirmed occurrence of an intervention taking effect. Append-only.

    ``event_sequence`` is assigned by the repository at commit, so only the
    repository constructs a complete application. It is the canonical order;
    ``position`` is where in training the effect landed.
    """

    schema_version: Literal["xaytune.intervention-application/v1alpha1"] = (
        "xaytune.intervention-application/v1alpha1"
    )
    id: InterventionApplicationId
    intervention_id: InterventionId
    attempt_id: RunAttemptId

    event_sequence: StrictInt = Field(ge=1)
    position: TrainingPosition

    trigger_evaluation: TriggerEvaluation | None = None

    previous_value: _LearningRate
    applied_value: _LearningRate
    checkpoint_ancestor: CheckpointRef | None = None

    created_at: datetime = Field(default_factory=utc_now)

    def semantic_fingerprint(self) -> str:
        """Identity for idempotent replay, without the repository's sequence or time."""
        return fingerprint(self.model_dump(mode="json", exclude={"event_sequence", "created_at"}))
