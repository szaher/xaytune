"""What ``TRLCompiler`` emits and ``TRLWorker`` accepts.

The wire contract between the two, and deliberately **not** ``SFTConfig``.
``SFTConfig`` is TRL's API: it inherits ``transformers.TrainingArguments``,
carries over a hundred fields, and changes between releases. A plan carrying it
would make every recorded plan depend on the TRL version that wrote it, and
every TRL upgrade a change to the execution contract.

Torch-free and TRL-free, because the compiler runs in the controller: TRL is an
optional dependency of the *worker*, and a controller host needs it no more
than it needs a GPU.

Every field is required, as in the native schema, and for the same reason: a
default here would be a third place a value could come from.

**Narrower than the native schema on purpose.** There is no ``format`` choice
and no ``packing`` field: v1alpha1 carries plain-text SFT only, where the whole
sequence is the training target. Prompt/completion and chat data make TRL
decide whether loss covers only the completion -- a choice the candidate cannot
express yet -- and TRL's packing is best-fit-decreasing, not what the native
trainer does. Each becomes a field when the candidate can say what it means.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from xaytune.core.immutable import FrozenDomainModel

__all__ = [
    "TRL_SFT_API_VERSION",
    "TRLData",
    "TRLModel",
    "TRLOptimization",
    "TRLRealization",
    "TRLSftConfig",
]

TRLSftApiVersion = Literal["xaytune.trl-sft/v1alpha1"]
TRL_SFT_API_VERSION: TRLSftApiVersion = "xaytune.trl-sft/v1alpha1"

Scheduler = Literal["cosine", "linear", "constant", "constant_with_warmup"]
Precision = Literal["fp16", "bf16", "fp32"]


class TRLModel(FrozenDomainModel):
    uri: str


class TRLData(FrozenDomainModel):
    """A local JSONL file of ``{"text": ...}`` records."""

    path: str
    max_length: int = Field(ge=1)


class TRLOptimization(FrozenDomainModel):
    """AdamW, a schedule, and the step arithmetic -- all of it explicit.

    ``warmup_steps`` is a count and only a count. ``TrainingArguments`` reads a
    value below 1 as a *ratio*, rounded up; the native trainer rounds a ratio
    down. A ratio would therefore mean a different step count depending on the
    trainer, so the compiler refuses one and this field cannot carry one.

    ``adam_epsilon`` has no candidate field. It is carried at ``1e-8`` --
    ``torch.optim.AdamW``'s own default, and so what the native trainer uses --
    so that "undeclared" means the same number on both trainers rather than
    whatever ``TrainingArguments`` defaults it to in a given release.
    """

    learning_rate: float = Field(gt=0)
    micro_batch_size: int = Field(ge=1)
    gradient_accumulation: int = Field(ge=1)
    epochs: int = Field(ge=1)
    max_steps: int | None = Field(ge=1)
    max_grad_norm: float = Field(ge=0)
    weight_decay: float = Field(ge=0)
    adam_beta1: float = Field(ge=0, lt=1)
    adam_beta2: float = Field(ge=0, lt=1)
    adam_epsilon: float = Field(gt=0)
    scheduler: Scheduler
    warmup_steps: int = Field(ge=0)
    mixed_precision: Precision


class TRLRealization(FrozenDomainModel):
    seed: int
    output_dir: str


class TRLSftConfig(FrozenDomainModel):
    api_version: TRLSftApiVersion = TRL_SFT_API_VERSION
    model: TRLModel
    data: TRLData
    optimization: TRLOptimization
    realization: TRLRealization
