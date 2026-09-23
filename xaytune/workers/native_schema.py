"""What ``NativeCompiler`` emits and ``NativeWorker`` accepts.

The wire contract between the two, and deliberately **not** ``TrainConfig``.
``TrainConfig`` is the native trainer's own configuration and carries every
field and default it has accumulated. Putting it at the boundary would make the
control plane inherit all of them permanently, and every future change to the
trainer's config would become a change to the execution contract.

Torch-free, because the compiler runs in the controller. A controller host that
needed the training stack installed to *compile* a plan -- rather than to run
one -- would have lost the separation ADR-010 exists to keep.

Every field is required. There are no defaults here on purpose: anything the
worker needs arrives explicitly from the compiler, which got it explicitly from
the candidate or the realization, or refused. A default in this schema would be
a third place a value could come from, and the one nobody would think to check.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from xaytune.core.immutable import FrozenDomainModel

__all__ = [
    "NATIVE_SFT_API_VERSION",
    "NativeData",
    "NativeModel",
    "NativeOptimization",
    "NativeRealization",
    "NativeSftConfig",
]

NativeSftApiVersion = Literal["xaytune.native-sft/v1alpha1"]
NATIVE_SFT_API_VERSION: NativeSftApiVersion = "xaytune.native-sft/v1alpha1"

Scheduler = Literal["cosine", "linear", "constant", "constant_with_warmup"]
Precision = Literal["fp16", "bf16", "fp32"]


class NativeModel(FrozenDomainModel):
    """The model artifact, as a location the worker resolves normally."""

    uri: str


class NativeData(FrozenDomainModel):
    """The data, and what it becomes before the model sees it."""

    path: str
    format: str
    max_seq_length: int = Field(gt=0)
    packing: bool


class NativeOptimization(FrozenDomainModel):
    """How the model is optimized. All of it scientific, all of it declared."""

    learning_rate: float = Field(gt=0)
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation: int = Field(gt=0)
    epochs: int = Field(gt=0)
    max_steps: int | None = Field(gt=0)
    """``None`` means no step cap; training length is then bounded by epochs."""

    max_grad_norm: float = Field(ge=0)
    weight_decay: float = Field(ge=0)
    scheduler: Scheduler
    warmup_steps: int | None = Field(ge=0)
    warmup_ratio: float | None = Field(ge=0, le=1)
    mixed_precision: Precision


class NativeRealization(FrozenDomainModel):
    """What this *run* is, as opposed to what the candidate proposes.

    Separate because replicates of one candidate share everything above and
    differ here, which is what lets them share a fingerprint.
    """

    seed: int
    output_dir: str
    checkpoint_every_optimizer_steps: int = Field(ge=0)
    """``0`` means no periodic checkpoints."""


class NativeSftConfig(FrozenDomainModel):
    """One native SFT run, fully specified."""

    api_version: NativeSftApiVersion = NATIVE_SFT_API_VERSION
    model: NativeModel
    data: NativeData
    optimization: NativeOptimization
    realization: NativeRealization
