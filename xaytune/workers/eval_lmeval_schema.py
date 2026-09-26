"""What ``LMEvalEvaluator`` records and emits, and the lm-eval worker accepts.

Torch-free and lm-eval-free, for the reason :mod:`xaytune.workers.native_schema`
is: the controller validates and prepares evaluations, and must not need the
evaluation stack to do it.

Two contracts live here:

- :class:`LMEvalTaskBinding`, what an lm-eval task **resolved to** at
  submission. It is recorded in ``EvaluatorSpec.config["binding"]``, so the
  ``EvaluationFingerprint`` covers it, and it pins everything a task name
  leaves mutable: the task definition (by digest, under an exact lm-eval
  release) and the hub dataset (by commit).
- :class:`LMEvalWorkerConfig`, one evaluation of one model, as the worker
  runs it.

Every field is required. Anything that changes a measured number arrives
explicitly, and a default here would be a place a value could come from that
no fingerprint covers.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import Field, field_validator

from xaytune.core.immutable import FrozenDomainModel

__all__ = [
    "LMEVAL_EVALUATION_API_VERSION",
    "LMEVAL_METRICS",
    "LMEVAL_OUTPUT_TYPES",
    "LM_EVAL_VERSION",
    "LMEvalMeasure",
    "LMEvalMetric",
    "LMEvalRealization",
    "LMEvalTaskBinding",
    "LMEvalWorkerConfig",
    "task_config_digest",
]

LMEvalApiVersion = Literal["xaytune.lm-eval/v1alpha1"]
LMEVAL_EVALUATION_API_VERSION: LMEvalApiVersion = "xaytune.lm-eval/v1alpha1"

LM_EVAL_VERSION = "0.4.13"
"""The one lm-eval release the evaluator binds tasks under and the worker runs.

Equal to the ``eval`` extra's pin. A task's definition -- its YAML and the
Python it references -- ships inside lm-eval, so the release is part of what a
task *is*: the same name under another release can be another benchmark.
"""

LMEVAL_OUTPUT_TYPES: tuple[str, ...] = ("multiple_choice", "loglikelihood")
"""Task kinds scored by log-likelihood alone, with nothing generated.

``generate_until`` is refused: generation brings its own settings -- decoding
kwargs, stop sequences, filters, answer extraction -- that decide the score and
that no binding here pins yet. ``loglikelihood_rolling`` reports perplexities
over documents, which no supported metric describes.
"""

LMEvalMetric = Literal["acc", "acc_norm"]
LMEVAL_METRICS: tuple[str, ...] = ("acc", "acc_norm")
"""The metrics a bound task may produce. Any other refuses the task."""

_SHA256 = r"^sha256:[0-9a-f]{64}$"
_COMMIT = r"^[0-9a-f]{40}$"


def task_config_digest(config: Any) -> str:
    """``sha256:`` over *config* as canonical JSON: how a task definition is named.

    *config* is the task's YAML as lm-eval loads it, includes merged,
    function references unresolved and made independent of where lm-eval is
    installed (see :mod:`xaytune.evaluation.lmeval`).
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LMEvalTaskBinding(FrozenDomainModel):
    """What an lm-eval task resolved to at submission: immutable, so recordable.

    It records what the task **is**, including what makes it unsupported --
    an ``output_type`` or metric outside the supported set -- so that the
    evaluator can refuse the resolved task with the reason, rather than
    resolution failing to describe it.

    Attributes:
        task: The registered lm-eval task name.
        task_version: The task's own ``metadata.version``.
        task_config_digest: :func:`task_config_digest` of its definition.
        lm_eval_version: The lm-eval release it was resolved under.
        dataset_path: The hub dataset repository the task reads.
        dataset_name: Its configuration, if the task names one.
        dataset_revision: The dataset repository's commit, resolved from
            whatever revision the task named -- or its default branch.
        output_type: How lm-eval scores the task.
        num_fewshot: Examples placed before each question.
        metrics: What the task reports, in its ``metric_list`` order.
    """

    task: str = Field(min_length=1)
    task_version: str = Field(min_length=1)
    task_config_digest: str = Field(pattern=_SHA256)
    lm_eval_version: str = Field(min_length=1)
    dataset_path: str = Field(min_length=1)
    dataset_name: str | None
    dataset_revision: str = Field(pattern=_COMMIT)
    output_type: str = Field(min_length=1)
    num_fewshot: int = Field(ge=0)
    metrics: tuple[str, ...] = Field(min_length=1)


class LMEvalMeasure(FrozenDomainModel):
    """How the work is batched, in what precision, and on how many documents."""

    batch_size: int = Field(gt=0)
    precision: Literal["fp32"]
    limit: int | None = Field(gt=0)


class LMEvalRealization(FrozenDomainModel):
    """What belongs to this run rather than to the evaluation: who measures, the seed, where to."""

    evaluator_name: str
    evaluator_version: str
    seed: int
    output_dir: str


class LMEvalWorkerConfig(FrozenDomainModel):
    """One lm-eval task, on one model, as the worker runs it."""

    api_version: LMEvalApiVersion
    model_uri: str
    binding: LMEvalTaskBinding
    measure: LMEvalMeasure
    realization: LMEvalRealization

    @field_validator("binding")
    @classmethod
    def _supported(cls, binding: LMEvalTaskBinding) -> LMEvalTaskBinding:
        # The evaluator refuses these before preparing; a worker handed one
        # anyway must not run it.
        if binding.output_type not in LMEVAL_OUTPUT_TYPES:
            raise ValueError(f"output_type {binding.output_type!r} is not supported")
        unsupported = [metric for metric in binding.metrics if metric not in LMEVAL_METRICS]
        if unsupported:
            raise ValueError(f"metrics {unsupported} are not supported")
        return binding
