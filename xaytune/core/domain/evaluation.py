"""Evaluation: what is measured, how it executes, and what it found (ADR-007, ADR-015).

```text
EvaluationSpec        what to measure -- the scientific contract, no seed
      ↓ bound per subject, seed and replicate
EvaluationRun         one logical evaluation of one artifact
      ↓
EvaluationAttempt     one infrastructure attempt at it
      ↓
EvaluationResult      the metrics that run produced, attributed to it
```

Evaluation is a workload, not a function call: it queues, fails, is preempted
and can be in flight when the controller restarts. So it has the same run and
attempt split as training -- following the same lifecycle principles, over its
own states. It writes no checkpoints, so there is no ``CHECKPOINTING`` and
nothing to recover into; a failed evaluation is retried as a new attempt.

It is **not** training, and is not folded into a generic ``Execution``
aggregate either (ADR-015 §2): an evaluation consumes a subject and a spec and
mutates no training state. What the two share is the transport -- the operation
journal, the telemetry envelope, the runtime -- not the domain object.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from xaytune.core.clock import utc_now
from xaytune.core.fingerprint import fingerprint
from xaytune.core.ids import (
    EvaluationAttemptId,
    EvaluationId,
    EvaluationRunId,
    ExperimentId,
    ExperimentNodeId,
)
from xaytune.core.immutable import AggregateModel, FrozenDict, FrozenDomainModel
from xaytune.core.observability import Finite, Name, NonNegative
from xaytune.core.refs import ArtifactRef, DatasetRef, RuntimeRef
from xaytune.core.state.machines import EVALUATION_ATTEMPT_MACHINE, EVALUATION_RUN_MACHINE
from xaytune.core.state.status import EvaluationAttemptStatus, EvaluationRunStatus

__all__ = [
    "EvaluationAttempt",
    "EvaluationResult",
    "EvaluationRun",
    "EvaluationSpec",
    "EvaluatorDeterminism",
    "EvaluatorSpec",
    "MetricResult",
    "evaluation_identity_v1",
]


class EvaluatorDeterminism(str, Enum):
    """Whether a result can stand in for running the evaluation again (ADR-015 §3).

    Declared by the evaluator, not assumed: an LLM judge, an agent environment
    and a sampled benchmark are not deterministic, and treating one of their
    results as *the* answer would silently turn a request for another sample
    into the previous sample.
    """

    DETERMINISTIC = "deterministic"
    """Same inputs, same output, always."""

    SEEDED = "seeded"
    """Reproducible given the same seed, and only where execution is local."""

    STOCHASTIC = "stochastic"
    """Each run is a sample; a finished one is history, not the answer."""


class EvaluatorSpec(FrozenDomainModel):
    """Which evaluator measures, configured how.

    ``version`` and ``determinism`` are ``None`` as a request and set once
    bound: the host resolves the name through its registry and records what
    that evaluator declares, so the record says which implementation measured
    rather than which name was asked for -- the same rule as ``CompilerSpec``.
    """

    name: str = Field(min_length=1)
    version: str | None = None
    determinism: EvaluatorDeterminism | None = None
    config: FrozenDict = Field(default_factory=FrozenDict)


class EvaluationSpec(FrozenDomainModel):
    """What to measure: the scientific contract of an evaluation.

    **No seed.** A seed here would make "this evaluation with seed 1" and "with
    seed 2" two different evaluations rather than two samples of one, and put
    replicate identity into :func:`evaluation_identity_v1`. It belongs to
    :class:`EvaluationRun`, exactly as a training seed belongs to ``Run`` and
    not to the candidate (ADR-015 §3). ``extra="forbid"`` refuses one.

    Not part of any candidate either: the same grader used as a reward is
    training; scoring a finished artifact is evaluation, and contributes to
    the evaluation's identity only (ADR-006).

    **One evaluator**, which may measure many metrics. So one run is one
    subject, one spec, one evaluator -- one determinism class -- and one seed
    and replicate, and nothing is ambiguous about whether its result can be
    reused. Several evaluators are several ``EvaluationRun`` s in a cycle,
    each with its own reproducibility, never one run mixing them.
    """

    api_version: Literal["xaytune.eval/v1alpha1"] = "xaytune.eval/v1alpha1"
    evaluator: EvaluatorSpec
    dataset: DatasetRef | None = None
    slices: tuple[str, ...] = Field(default_factory=tuple)
    metadata: FrozenDict = Field(default_factory=FrozenDict)

    def evaluation_fingerprint(self) -> str:
        """This evaluation's identity: :func:`evaluation_identity_v1`, hashed."""
        return fingerprint(evaluation_identity_v1(self))


def evaluation_identity_v1(spec: EvaluationSpec) -> Mapping[str, Any]:
    """What makes two evaluations the same evaluation, version 1.

    An explicit projection rather than the model's dump, so a field added to
    :class:`EvaluationSpec` later does not change the identity of every
    evaluation already recorded (the rule ``candidate_identity_v1`` follows).

    Covers the evaluator's name, bound version and configuration, the dataset
    and the slices. Not ``determinism``, which is the evaluator's
    statement about reuse rather than a property of what is measured, and not
    ``metadata``, which describes rather than defines.
    """
    return {
        "kind": "evaluation",
        "identity_version": 1,
        "api_version": spec.api_version,
        "evaluator": {
            "name": spec.evaluator.name,
            "version": spec.evaluator.version,
            "config": spec.evaluator.config,
        },
        "dataset": spec.dataset,
        "slices": list(spec.slices),
    }


class MetricResult(FrozenDomainModel):
    """One measured value, with what it was measured on and how sure it is.

    Never a bare ``dict[str, float]``: a number without its sample count,
    slice, evaluator and uncertainty cannot be compared honestly with another
    one, and a decision drawn from two such numbers is a guess.
    """

    name: Name
    value: Finite
    sample_count: int | None = Field(default=None, ge=0)
    dataset_ref: DatasetRef | None = None
    slice: str | None = None
    evaluator_name: Name
    evaluator_version: str | None = None
    seed: int | None = None
    """The seed of the :class:`EvaluationRun` that produced it."""
    confidence_interval: tuple[Finite, Finite] | None = None
    standard_error: NonNegative | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)

    @field_validator("confidence_interval")
    @classmethod
    def _ordered(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and value[0] > value[1]:
            raise ValueError(f"confidence interval {value} has its lower bound above its upper")
        return value


class EvaluationResult(FrozenDomainModel):
    """What one evaluation run measured, attributed to that run.

    ``evaluation_run_id`` is required, not convenience. A node can hold several
    runs over the same subject and fingerprint -- replicates of a stochastic
    evaluation -- and without it nothing says which execution drew which
    sample, which is the provenance a variance estimate depends on (ADR-015
    AC-4c). ``node_id``, ``subject`` and ``evaluation_fingerprint`` repeat the
    run's; the repository refuses a result that disagrees with its run.
    """

    id: EvaluationId
    evaluation_run_id: EvaluationRunId
    node_id: ExperimentNodeId
    subject: ArtifactRef
    evaluation_fingerprint: str
    metrics: tuple[MetricResult, ...] = Field(min_length=1)
    artifacts: tuple[ArtifactRef, ...] = Field(default_factory=tuple)
    created_at: datetime = Field(default_factory=utc_now)


class EvaluationRun(AggregateModel):
    """One logical evaluation of one subject (ADR-015 §1).

    ``evaluation_cycle`` names the node's evaluation round this run belongs to.
    A node can go ``ACTIVE -> EVALUATING -> DECIDING -> ACTIVE -> EVALUATING``,
    and the runs of the first round must not satisfy the second: only runs of
    the node's current cycle count when its evaluation is reconciled.
    """

    id: EvaluationRunId
    experiment_id: ExperimentId
    node_id: ExperimentNodeId
    evaluation_cycle: int = Field(ge=1)

    spec: EvaluationSpec
    subject: ArtifactRef
    evaluation_fingerprint: str

    seed: int | None = None
    replicate: int | None = None

    status: EvaluationRunStatus = EvaluationRunStatus.CREATED

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    revision: int = 0

    @model_validator(mode="after")
    def _fingerprint_describes_spec(self) -> EvaluationRun:
        # Derived, so a supplied value that disagrees with the spec would index
        # one evaluation under another's identity.
        expected = self.spec.evaluation_fingerprint()
        if self.evaluation_fingerprint != expected:
            raise ValueError(
                f"evaluation_fingerprint {self.evaluation_fingerprint!r} does not describe "
                f"the spec, which fingerprints as {expected!r}"
            )
        return self

    def with_status(self, new_status: EvaluationRunStatus) -> EvaluationRun:
        """Return a copy in *new_status*, with the revision bumped.

        Raises:
            InvalidTransitionError: If the transition is not permitted.
        """
        EVALUATION_RUN_MACHINE.validate(self.status, new_status)
        return self._validated_copy(
            {"status": new_status, "revision": self.revision + 1, "updated_at": utc_now()}
        )

    @property
    def is_terminal(self) -> bool:
        """Whether this run has reached a final state."""
        return EVALUATION_RUN_MACHINE.is_terminal(self.status)


class EvaluationAttempt(AggregateModel):
    """One infrastructure attempt at an evaluation run (ADR-015 §1).

    Its telemetry cursor -- generation and sequence -- is a pair of columns the
    controller assigns, as for a training attempt (ADR-014), not a field here.
    """

    id: EvaluationAttemptId
    evaluation_run_id: EvaluationRunId
    attempt_number: int = Field(ge=1)

    status: EvaluationAttemptStatus = EvaluationAttemptStatus.CREATED
    runtime_ref: RuntimeRef | None = None

    started_at: datetime | None = None
    ended_at: datetime | None = None
    revision: int = 0

    def with_status(self, new_status: EvaluationAttemptStatus) -> EvaluationAttempt:
        """Return a copy in *new_status*, with the revision bumped.

        ``started_at`` is stamped on entering ``RUNNING`` and ``ended_at`` on
        reaching a terminal state.

        Raises:
            InvalidTransitionError: If the transition is not permitted.
        """
        EVALUATION_ATTEMPT_MACHINE.validate(self.status, new_status)
        update: dict[str, Any] = {"status": new_status, "revision": self.revision + 1}
        if new_status is EvaluationAttemptStatus.RUNNING and self.started_at is None:
            update["started_at"] = utc_now()
        if EVALUATION_ATTEMPT_MACHINE.is_terminal(new_status) and self.ended_at is None:
            update["ended_at"] = utc_now()
        return self._validated_copy(update)

    @property
    def is_terminal(self) -> bool:
        """Whether this attempt has reached a final state."""
        return EVALUATION_ATTEMPT_MACHINE.is_terminal(self.status)
