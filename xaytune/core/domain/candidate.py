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

from collections.abc import Mapping
from enum import Enum
from typing import Any

from pydantic import Field

from xaytune.core.fingerprint import fingerprint
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.refs import DatasetRef, ModelRef

__all__ = [
    "AdapterSpec",
    "AlgorithmSpec",
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
    "CheckpointIntent",
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
    """The base model a candidate starts from.

    The revision lives on :class:`ModelRef` and only there. Carrying a second
    one here would create two authorities for the same fact, and nothing would
    say which wins when they disagree.
    """

    model: ModelRef
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class DataSpec(FrozenDomainModel):
    """The data a candidate trains on."""

    dataset: DatasetRef
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class OptimizerSpec(FrozenDomainModel):
    """Which optimizer, and its declared hyperparameters.

    Typed rather than left to metadata, because AdamW and SGD are different
    scientific propositions and a fingerprint must say so.
    """

    name: str
    learning_rate: float | None = None
    weight_decay: float | None = None
    betas: tuple[float, ...] = Field(default_factory=tuple)
    params: FrozenDict = Field(default_factory=FrozenDict)


class LRScheduleSpec(FrozenDomainModel):
    """The declared learning-rate schedule.

    Part of candidate identity: a cosine decay and a constant rate over the
    same steps are different experiments.
    """

    name: str
    warmup_steps: int | None = None
    warmup_ratio: float | None = None
    params: FrozenDict = Field(default_factory=FrozenDict)


class PrecisionSpec(FrozenDomainModel):
    """Declared numerical precision.

    Identity-bearing: bf16 and fp32 can reach different results from the same
    candidate, so they are not the same candidate.
    """

    dtype: str | None = None
    grad_accum_dtype: str | None = None
    params: FrozenDict = Field(default_factory=FrozenDict)


class CheckpointIntent(FrozenDomainModel):
    """How often the candidate intends to checkpoint, and what it keeps.

    **Intent, not mechanism.** Declaring "every 500 optimizer steps" is a
    control-plane statement about the experiment. Owning a trainer's
    checkpoint save/restore lifecycle is a runtime concern and stays there
    (TASK-029); keeping the two apart is the point of the compile/execute
    boundary.
    """

    every_optimizer_steps: int | None = None
    keep_last: int | None = None
    params: FrozenDict = Field(default_factory=FrozenDict)


class AlgorithmSpec(FrozenDomainModel):
    """Algorithm-specific parameters beyond the training kind.

    DPO's beta or GRPO's group size live here rather than in generic metadata,
    so they are visibly part of the scientific proposition.
    """

    name: str | None = None
    params: FrozenDict = Field(default_factory=FrozenDict)


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

    optimizer: OptimizerSpec | None = None
    lr_schedule: LRScheduleSpec | None = None

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

    algorithm: AlgorithmSpec = Field(default_factory=AlgorithmSpec)
    adapter: AdapterSpec | None = None
    optimization: OptimizationSpec = Field(default_factory=OptimizationSpec)
    precision: PrecisionSpec = Field(default_factory=PrecisionSpec)
    checkpoint: CheckpointIntent = Field(default_factory=CheckpointIntent)

    metadata: FrozenDict = Field(default_factory=FrozenDict)


def _optional(value: FrozenDomainModel | None) -> Any:
    """Project an optional component, distinguishing absent from empty."""
    return None if value is None else value.model_dump(mode="json", by_alias=True)


def candidate_identity_v1(candidate: CandidateSpec) -> Mapping[str, Any]:
    """The explicit, versioned projection that defines candidate identity.

    **Why this is not just the model.** Hashing a Pydantic model makes the
    *current schema* define identity. Add a field with a default six months
    from now and every candidate already on disk fingerprints differently,
    though no researcher changed anything -- and reuse, comparison and lineage
    all silently stop matching. Listing the fields here means adding one later
    is a deliberate act: either it joins ``v2``, or it does not bear identity.

    **What is excluded, and why.** Generic ``metadata`` does not enter identity.
    A ticket number or an owner's name is not a scientific proposition, and
    ``metadata={"ticket": "RHOAI-1234"}`` must not mint a new candidate.
    Anything that genuinely should bear identity gets a typed field -- which is
    what ``AlgorithmSpec.params`` and the other ``params`` fields are for.

    The projection is domain-separated by ``type``, so a candidate and a run
    history can never collide even if their contents coincided.
    """
    training = candidate.training
    return {
        "type": "candidate",
        "version": 1,
        "model": candidate.model.model.model_dump(mode="json", by_alias=True),
        "data": candidate.data.dataset.model_dump(mode="json", by_alias=True),
        "training": {
            "kind": training.kind.value,
            "algorithm": training.algorithm.model_dump(mode="json", by_alias=True),
            "adapter": _optional(training.adapter),
            "optimization": training.optimization.model_dump(mode="json", by_alias=True),
            "precision": training.precision.model_dump(mode="json", by_alias=True),
            "checkpoint": training.checkpoint.model_dump(mode="json", by_alias=True),
        },
        "reward": _optional(candidate.reward),
        "environment": _optional(candidate.environment),
        "schedule": _optional(candidate.schedule),
    }


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

        Hashes the **v1 identity projection**, not the model. See
        :func:`candidate_identity_v1` for what that covers and why it is not
        simply the model's fields.

        It does not cover the seed, the evaluator, or anything about execution:
        two replicates of one candidate share this fingerprint, a changed
        grader must never imply a retrain, and the same hypothesis on different
        hardware is the same hypothesis.
        """
        return fingerprint(candidate_identity_v1(self))
