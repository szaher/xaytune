"""What ``NativeEvaluator`` emits and the native evaluation worker accepts.

The wire contract between the two, torch-free for the reason
:mod:`xaytune.workers.native_schema` is: the evaluator prepares in the
controller, which must not need the training stack to do it.

Every field is required. Anything that changes a measured number arrives
here explicitly -- from the evaluation spec, the subject, or the run -- and a
default in this schema would be a place a value could come from that no
fingerprint covers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from xaytune.core.immutable import FrozenDomainModel

__all__ = [
    "NATIVE_EVALUATION_API_VERSION",
    "NATIVE_METRICS",
    "NativeEvaluationData",
    "NativeEvaluationMeasure",
    "NativeEvaluationRealization",
    "NativeEvaluationWorkerConfig",
    "NativeMetric",
    "content_digest",
]

NativeEvaluationApiVersion = Literal["xaytune.native-eval/v1alpha1"]
NATIVE_EVALUATION_API_VERSION: NativeEvaluationApiVersion = "xaytune.native-eval/v1alpha1"

NativeMetric = Literal["loss", "perplexity", "token_accuracy"]
NATIVE_METRICS: tuple[NativeMetric, ...] = ("loss", "perplexity", "token_accuracy")


def content_digest(path: Path) -> str:
    """``sha256:`` and the hex digest of *path*'s bytes: how a data file is named and checked."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


class NativeEvaluationData(FrozenDomainModel):
    """The held-out data, pinned by content, and what it becomes before the model sees it."""

    path: str
    content_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    format: Literal["text"]
    max_seq_length: int = Field(gt=0)


class NativeEvaluationMeasure(FrozenDomainModel):
    """What is computed, and how the batches that compute it are formed."""

    metrics: tuple[NativeMetric, ...] = Field(min_length=1)
    batch_size: int = Field(gt=0)
    precision: Literal["fp32"]

    @field_validator("metrics")
    @classmethod
    def _once_each(cls, value: tuple[NativeMetric, ...]) -> tuple[NativeMetric, ...]:
        if len(set(value)) != len(value):
            raise ValueError(f"metrics {value} names one more than once")
        return value


class NativeEvaluationRealization(FrozenDomainModel):
    """What belongs to this run rather than to the evaluation: who measures, the seed, where to."""

    evaluator_name: str
    evaluator_version: str
    seed: int
    output_dir: str


class NativeEvaluationWorkerConfig(FrozenDomainModel):
    """One evaluation of one model, as the worker runs it."""

    api_version: NativeEvaluationApiVersion
    model_uri: str
    data: NativeEvaluationData
    measure: NativeEvaluationMeasure
    realization: NativeEvaluationRealization
