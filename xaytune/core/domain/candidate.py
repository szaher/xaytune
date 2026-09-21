"""The candidate: a whole scientific proposition (ADR-011).

A candidate is *what we are testing*, not *how we train it*. That distinction is
the substance of ADR-011, and it decides what belongs here:

```text
CandidateSpec                     Run
├── model:       ModelSpec        ├── seed
├── data:        DataSpec         └── replicate
├── training:    TrainingSpec
├── reward:      RewardSpec?
├── environment: EnvironmentSpec?
└── schedule:    TrainingSchedule?
```

Three things are deliberately **outside** it:

* **Seed and replicate** belong to :class:`~xaytune.core.domain.run.Run`. Two
  replicates differing only by seed are the same scientific candidate run
  twice; folding the seed into candidate identity would make the replicate
  concept meaningless.
* **`EvaluationSpec`** is not part of a candidate. Making a grader part of
  candidate identity would mean changing a grader implies a retrain, which is
  the failure ADR-006 exists to prevent.
* **`ExecutionFingerprint`** — compiler, runtime, GPU type, topology — never
  enters candidate identity. Running the same candidate on different hardware
  does not make it a different hypothesis.

The same grader can appear in both roles: used inside the training loop it is a
reward and belongs to `RewardSpec`; used to score the artifact it is an
evaluator and contributes only to `EvaluationFingerprint`. The role decides,
not the object.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from xaytune.core.fingerprint import fingerprint
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.refs import DatasetRef, ModelRef

__all__ = [
    "AdapterSpec",
    "CandidateSpec",
    "DataSpec",
    "EnvironmentSpec",
    "ModelSpec",
    "OptimizationSpec",
    "RewardSpec",
    "ScheduledIntervention",
    "TrainingKind",
    "TrainingSchedule",
    "TrainingSpec",
]


class TrainingKind(str, Enum):
    """The training program a candidate runs."""

    SFT = "sft"

    CONTINUED_PRETRAIN = "continued_pretrain"
    """Adapting an existing base model on new corpora.

    Not ``PRETRAIN``. Xaytune is not a frontier-scale pretraining runtime, and
    an enum member reading `PRETRAIN` in the control-plane API invites exactly
    the opposite conclusion from an agent choosing a training kind. The legacy
    ``xaytune.pretrain()`` entry point keeps its own name.
    """

    DPO = "dpo"
    GRPO = "grpo"


class ModelSpec(FrozenDomainModel):
    """The base model a candidate starts from."""

    model: ModelRef
    revision: str | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class DataSpec(FrozenDomainModel):
    """The data a candidate trains on."""

    dataset: DatasetRef
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class AdapterSpec(FrozenDomainModel):
    """A parameter-efficient adapter, when one is used."""

    type: str
    rank: int | None = None
    alpha: float | None = None
    target_modules: tuple[str, ...] = Field(default_factory=tuple)
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class OptimizationSpec(FrozenDomainModel):
    """Declared optimization intent.

    Micro-batch size and gradient accumulation are here because the *effective*
    batch is scientific. A recovery that halves one and doubles the other
    preserves intent and is an `ExecutionOverride`, not a change to this spec.
    """

    learning_rate: float | None = None
    micro_batch_size: int | None = None
    gradient_accumulation: int | None = None
    epochs: int | None = None
    max_steps: int | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class RewardSpec(FrozenDomainModel):
    """A grader used *inside* the training loop.

    The same grader scoring a finished artifact is an evaluator instead, and
    contributes only to `EvaluationFingerprint`. The role decides.
    """

    graders: tuple[str, ...] = Field(default_factory=tuple)
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class EnvironmentSpec(FrozenDomainModel):
    """The environment an agent candidate acts in."""

    name: str
    revision: str | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class ScheduledIntervention(FrozenDomainModel):
    """A change declared *before* training starts.

    Pre-registered, so it is part of what the candidate proposes to test and
    contributes to its identity. A reactive intervention — decided mid-run in
    response to something observed — does not: it changes the run's history,
    not the hypothesis (ADR-011).
    """

    id: str
    trigger: FrozenDict
    mutation: FrozenDict
    rationale: str | None = None


class TrainingSchedule(FrozenDomainModel):
    """Pre-registered interventions, in declared order."""

    interventions: tuple[ScheduledIntervention, ...] = Field(default_factory=tuple)


class TrainingSpec(FrozenDomainModel):
    """The training **program**, and nothing else.

    Model, data, reward and environment are siblings of this on
    :class:`CandidateSpec`, not fields of it, and the seed belongs to the
    `Run`. A GRPO candidate has a reward and an environment at the same level
    as its training program, which is only expressible once the program stops
    being the whole proposition.
    """

    api_version: str = "xaytune.ai/v1alpha1"
    kind: TrainingKind

    adapter: AdapterSpec | None = None
    optimization: OptimizationSpec = Field(default_factory=OptimizationSpec)

    metadata: FrozenDict = Field(default_factory=FrozenDict)


class CandidateSpec(FrozenDomainModel):
    """One whole scientific proposition.

    Frozen: a scientific change creates a new node rather than editing a
    candidate, because the comparability rule turns on being able to compare
    before and after as alternatives (ADR-011).
    """

    model: ModelSpec
    data: DataSpec
    training: TrainingSpec

    reward: RewardSpec | None = None
    environment: EnvironmentSpec | None = None
    schedule: TrainingSchedule | None = None

    metadata: FrozenDict = Field(default_factory=FrozenDict)

    def candidate_fingerprint(self) -> str:
        """Return this candidate's identity: *has this hypothesis been explored?*

        Covers everything declared — including any pre-registered schedule,
        which is part of the proposal. It does **not** cover the seed, the
        evaluator, or anything about execution: two replicates of one candidate
        share this fingerprint, a changed grader must never imply a retrain,
        and the same hypothesis run on different hardware is the same
        hypothesis.
        """
        return fingerprint(self)
