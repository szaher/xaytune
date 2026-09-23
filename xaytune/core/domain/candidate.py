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
  enters candidate identity. Running the same candidate on different hardware,
  or through a different compiler, does not make it a different hypothesis --
  though runs through different compilers are not interchangeable replicates
  (ADR-011 §5).

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
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw
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
    """The data a candidate trains on, and what it becomes before training.

    ``format``, ``max_seq_length`` and ``packing`` are here because they
    decide the tokens the model actually sees. The same file read as ``text``
    or as ``alpaca``, truncated at 512 or at 2048, packed or not, is a
    different training set -- so they bear identity, and none of them can be
    left for a compiler to fill in.

    Optional at the type level and required by the compilers that need them.
    A GRPO or agent candidate may have no meaningful notion of a data format;
    making it mandatory here would force every workload to invent one.
    """

    dataset: DatasetRef
    format: str | None = None
    max_seq_length: int | None = Field(default=None, gt=0)
    packing: bool | None = None
    metadata: FrozenDict = Field(default_factory=FrozenDict)


class OptimizerSpec(FrozenDomainModel):
    """Which optimizer, and its declared hyperparameters.

    Typed rather than left to metadata, because AdamW and SGD are different
    scientific propositions and a fingerprint must say so.

    The learning rate is **not** here. It lives on
    :class:`OptimizationSpec.learning_rate` and only there: two places to put
    it means nothing says which one a compiler should read when they disagree,
    and that question would then have to be answered separately in every
    compiler.
    """

    name: str
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

    max_grad_norm: float | None = Field(default=None, ge=0)
    """Gradient clipping threshold; ``0`` disables clipping.

    Scientific rather than operational: clipping changes the optimization
    trajectory, and a candidate that left it out would train under whatever
    the trainer happened to default to. ``None`` means *undeclared*, as it does
    throughout this module -- which is why "no clipping" is ``0`` rather than
    ``None``.
    """

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


def _model_identity_v1(spec: ModelSpec) -> Mapping[str, Any]:
    """The base model, by identity rather than by whatever fields it has."""
    ref = spec.model
    return {
        "uri": ref.uri,
        "revision": ref.revision,
        "digest": getattr(ref, "digest", None),
    }


def _data_identity_v1(spec: DataSpec) -> Mapping[str, Any]:
    """The data, including the transformations that decide what it becomes.

    A tokenizer or template change produces different training data from the
    same source, so those fingerprints are identity even though the URI is
    unchanged.
    """
    ref = spec.dataset
    return {
        "uri": ref.uri,
        "revision": ref.revision,
        "split": ref.split,
        "content_digest": ref.content_digest,
        "transform_fingerprint": ref.transform_fingerprint,
        "tokenizer_fingerprint": ref.tokenizer_fingerprint,
        "template_fingerprint": ref.template_fingerprint,
    }


def _data_identity_v2(spec: DataSpec) -> Mapping[str, Any]:
    """v1, plus the preprocessing that decides what the data becomes."""
    return {
        **_data_identity_v1(spec),
        "format": spec.format,
        "max_seq_length": spec.max_seq_length,
        "packing": spec.packing,
    }


def _adapter_identity_v1(spec: AdapterSpec | None) -> Mapping[str, Any] | None:
    if spec is None:
        return None
    return {
        "type": spec.type,
        "rank": spec.rank,
        "alpha": spec.alpha,
        "target_modules": list(spec.target_modules),
    }


def _optimizer_identity_v1(spec: OptimizerSpec | None) -> Mapping[str, Any] | None:
    if spec is None:
        return None
    return {
        "name": spec.name,
        "weight_decay": spec.weight_decay,
        "betas": list(spec.betas),
        "params": thaw(spec.params),
    }


def _lr_schedule_identity_v1(spec: LRScheduleSpec | None) -> Mapping[str, Any] | None:
    if spec is None:
        return None
    return {
        "name": spec.name,
        "warmup_steps": spec.warmup_steps,
        "warmup_ratio": spec.warmup_ratio,
        "params": thaw(spec.params),
    }


def _optimization_identity_v1(spec: OptimizationSpec) -> Mapping[str, Any]:
    return {
        "optimizer": _optimizer_identity_v1(spec.optimizer),
        "lr_schedule": _lr_schedule_identity_v1(spec.lr_schedule),
        "learning_rate": spec.learning_rate,
        "micro_batch_size": spec.micro_batch_size,
        "gradient_accumulation": spec.gradient_accumulation,
        "epochs": spec.epochs,
        "max_steps": spec.max_steps,
    }


def _optimization_identity_v2(spec: OptimizationSpec) -> Mapping[str, Any]:
    """v1, plus gradient clipping."""
    return {**_optimization_identity_v1(spec), "max_grad_norm": spec.max_grad_norm}


def _training_identity_v2(spec: TrainingSpec) -> Mapping[str, Any]:
    """v1 with the v2 optimization projection substituted, and nothing else."""
    return {
        **_training_identity_v1(spec),
        "optimization": _optimization_identity_v2(spec.optimization),
    }


def _training_identity_v1(spec: TrainingSpec) -> Mapping[str, Any]:
    return {
        "kind": spec.kind.value,
        "algorithm": {"name": spec.algorithm.name, "params": thaw(spec.algorithm.params)},
        "adapter": _adapter_identity_v1(spec.adapter),
        "optimization": _optimization_identity_v1(spec.optimization),
        "precision": {
            "dtype": spec.precision.dtype,
            "grad_accum_dtype": spec.precision.grad_accum_dtype,
            "params": thaw(spec.precision.params),
        },
        "checkpoint": {
            "every_optimizer_steps": spec.checkpoint.every_optimizer_steps,
            "keep_last": spec.checkpoint.keep_last,
            "params": thaw(spec.checkpoint.params),
        },
    }


def _reward_identity_v1(spec: RewardSpec | None) -> Mapping[str, Any] | None:
    if spec is None:
        return None
    return {"graders": list(spec.graders)}


def _environment_identity_v1(spec: EnvironmentSpec | None) -> Mapping[str, Any] | None:
    if spec is None:
        return None
    return {"name": spec.name, "revision": spec.revision}


def _schedule_identity_v1(spec: TrainingSchedule | None) -> Mapping[str, Any] | None:
    if spec is None:
        return None
    return {
        "interventions": [
            {
                "id": item.id,
                "trigger": thaw(item.trigger),
                "mutation": thaw(item.mutation),
            }
            for item in spec.interventions
        ]
    }


def candidate_identity_v1(candidate: CandidateSpec) -> Mapping[str, Any]:
    """The explicit, versioned projection that defines candidate identity.

    **Why this is not just the model.** Hashing a Pydantic model makes the
    *current schema* define identity. Add a field with a default six months
    from now and every candidate already on disk fingerprints differently,
    though no researcher changed anything -- and reuse, comparison and lineage
    all silently stop matching. Enumerating the fields means adding one later
    is a deliberate act: either it joins ``v2``, or it does not bear identity.

    **Enumerated all the way down.** Every nested component has its own
    projection rather than being dumped. A projection that named the top-level
    fields but dumped ``ModelRef`` underneath would only move the problem one
    level down -- and, worse, would read as though it had solved it.

    **What is excluded, and why.** Generic ``metadata`` does not enter
    identity, at *any* level. A ticket number or an owner's name is not a
    scientific proposition. Anything that genuinely should bear identity gets
    a typed field, which is what the ``params`` fields are for -- those are
    carried, because algorithm and optimizer parameters are the experiment.

    A ``rationale`` on a scheduled intervention is excluded for the same
    reason: rewording why a change was planned does not change what is planned.

    The projection is domain-separated by ``type``, so a candidate and a run
    history can never collide even if their contents coincided.
    """
    return {
        "type": "candidate",
        "version": 1,
        "model": _model_identity_v1(candidate.model),
        "data": _data_identity_v1(candidate.data),
        "training": _training_identity_v1(candidate.training),
        "reward": _reward_identity_v1(candidate.reward),
        "environment": _environment_identity_v1(candidate.environment),
        "schedule": _schedule_identity_v1(candidate.schedule),
    }


def candidate_identity_v2(candidate: CandidateSpec) -> Mapping[str, Any]:
    """v1, plus the fields v1 could not see: data preprocessing and clipping.

    **Why a new version rather than an edit to v1.** Every candidate fingerprint
    already recorded -- in ``experiment_nodes.candidate_fingerprint``, indexed,
    with no column saying which projection produced it -- was computed by v1.
    Adding a key to v1 would change all of them while leaving v1's name
    unchanged, which is precisely the silent drift the version number exists
    to prevent. So v1 is frozen and still reproduces every historical value,
    and the new fields enter identity here.

    ``"version": 2`` separates the two domains, so a v1 and a v2 fingerprint
    cannot collide even for a candidate where the new fields are all unset.

    **The v1 helpers are now shared, and therefore frozen.** The unchanged
    components -- model, reward, environment, schedule -- reuse their v1
    projections. A later change to one of them would move *both* versions, so
    any such change must fork the helper instead. The pinned v1 literal in the
    test suite is the tripwire for that.
    """
    return {
        "type": "candidate",
        "version": 2,
        "model": _model_identity_v1(candidate.model),
        "data": _data_identity_v2(candidate.data),
        "training": _training_identity_v2(candidate.training),
        "reward": _reward_identity_v1(candidate.reward),
        "environment": _environment_identity_v1(candidate.environment),
        "schedule": _schedule_identity_v1(candidate.schedule),
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

        Hashes the **v2 identity projection**, not the model. See
        :func:`candidate_identity_v2` for what changed from v1 and why v1 was
        left alone, and :func:`candidate_identity_v1` for why identity is a
        projection rather than the model's fields.

        It does not cover the seed, the evaluator, or anything about execution:
        two replicates of one candidate share this fingerprint, a changed
        grader must never imply a retrain, and the same hypothesis on different
        hardware is the same hypothesis.
        """
        return fingerprint(candidate_identity_v2(self))

    def candidate_fingerprint_v1(self) -> str:
        """This candidate's identity under the frozen v1 projection.

        For finding what was recorded before v2. A stored fingerprint is only a
        digest -- ``experiment_nodes.candidate_fingerprint`` carries no
        projection version -- so a v1 value and a v2 value for the same
        candidate simply do not compare equal, and nothing in the stored value
        says why. This is how a caller asks for the historical one on purpose.
        """
        return fingerprint(candidate_identity_v1(self))

    def candidate_fingerprint_v2(self) -> str:
        """This candidate's identity under v2, the current projection."""
        return fingerprint(candidate_identity_v2(self))

    def candidate_fingerprints_for_lookup(self) -> tuple[str, ...]:
        """Every identity this candidate may have been recorded under, newest first.

        For a lookup that has to find a candidate whichever projection stored
        it. The two are deliberately **not** treated as equivalent: v1 cannot
        see data preprocessing or clipping, so a v1 match is weaker evidence
        than a v2 one, and a caller that needs to tell them apart can, by
        position. Deciding what a historical match permits -- reuse, or only
        comparison -- is the reuse policy's decision (ADR-017), not this one's.
        """
        return (self.candidate_fingerprint_v2(), self.candidate_fingerprint_v1())
