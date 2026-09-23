"""What every SFT compiler refuses, whichever trainer it targets.

Some refusals are about a trainer: the native loop builds ``AdamW`` with fixed
betas, TRL decays a different set of parameters. Those stay with the compiler
that has the limitation. The rest are about the **candidate and the boundary**,
and would be wrong to decide twice:

- a candidate that is not plain SFT;
- a value that changes training and was left undeclared;
- an identity constraint -- a digest, a fingerprint, a revision -- that no
  worker can yet verify, which would put a claim in the fingerprint that the
  run does not keep;
- a location that is not an absolute local path, which the worker would
  resolve against whatever it happened to be given;
- checkpoint intent, which no worker yet reports (TASK-029).

Kept here so the two compilers cannot drift on them: a rule one compiler
enforced and the other forgot would make "supported" mean something different
depending on which trainer answered.

Every function reads the declared candidate and nothing else.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from xaytune.core.domain.candidate import CandidateSpec, TrainingKind

SCHEDULERS = frozenset({"cosine", "linear", "constant", "constant_with_warmup"})
PRECISIONS = frozenset({"fp16", "bf16", "fp32"})

_DATASET_IDENTITY_FIELDS = (
    "content_digest",
    "transform_fingerprint",
    "tokenizer_fingerprint",
    "template_fingerprint",
)

_REQUIRED_OPTIMIZATION_FIELDS = (
    "learning_rate",
    "micro_batch_size",
    "gradient_accumulation",
    "epochs",
    "max_grad_norm",
)


def local_path(uri: str) -> str | None:
    """*uri* as an absolute local path, or ``None`` if it is not one.

    Reading the declared string, not the filesystem. ``file://`` is accepted and
    removed, because the worker hands the result to ``Path``; any other scheme
    is not absolute and so is not local.
    """
    location = uri.removeprefix("file://")
    return location if os.path.isabs(location) else None


def sft_refusals(candidate: CandidateSpec, *, trainer: str) -> Iterator[str]:
    """Every boundary-level reason *candidate* cannot be run as declared.

    *trainer* names the implementation in the reasons ("the native trainer"),
    so a planner reading them knows which compiler said no.
    """
    training = candidate.training

    if training.kind is not TrainingKind.SFT:
        yield (
            f"training.kind is {training.kind.value!r}; {trainer} supports only "
            f"'sft' in this release"
        )
    if training.adapter is not None:
        yield f"training.adapter is declared; {trainer} runs full-parameter SFT only"
    if training.algorithm.name is not None or training.algorithm.params:
        yield "training.algorithm is declared; plain SFT has no algorithm variant to select"
    if candidate.reward is not None:
        yield "reward is declared; supervised fine-tuning has no reward"
    if candidate.environment is not None:
        yield "environment is declared; supervised fine-tuning has no environment"
    if candidate.schedule is not None:
        yield f"schedule is declared; {trainer} cannot apply scheduled interventions yet"

    yield from _data_refusals(candidate)
    yield from _model_refusals(candidate)
    yield from _optimization_refusals(candidate, trainer=trainer)
    yield from _precision_refusals(candidate, trainer=trainer)
    yield from _checkpoint_refusals(candidate, trainer=trainer)


def _data_refusals(candidate: CandidateSpec) -> Iterator[str]:
    data = candidate.data
    dataset = data.dataset

    if data.format is None:
        yield "data.format is undeclared; it decides what the file's records become"
    if data.max_seq_length is None:
        yield "data.max_seq_length is undeclared; it decides where examples are truncated"
    if data.packing is None:
        yield "data.packing is undeclared; it decides whether examples share sequences"

    if local_path(dataset.uri) is None:
        yield (
            f"data.dataset.uri {dataset.uri!r} is not an absolute local path; the "
            f"worker reads local JSONL, and a relative path would depend on the "
            f"working directory of whichever process resolved it"
        )
    if dataset.revision is not None:
        yield "data.dataset.revision is declared; a local file has no revisions to select"
    if dataset.split is not None:
        yield "data.dataset.split is declared; the worker reads a whole file"
    for field in _DATASET_IDENTITY_FIELDS:
        if getattr(dataset, field) is not None:
            yield (
                f"data.dataset.{field} is declared; it is part of the candidate's "
                f"identity, but the worker cannot verify it and would train on "
                f"whatever the file and its own preprocessing currently produce"
            )


def _model_refusals(candidate: CandidateSpec) -> Iterator[str]:
    model = candidate.model.model
    if local_path(model.uri) is None:
        # "Qwen/Qwen3-8B" is a hub name, and without a revision the loader
        # fetches whatever that name points at on the day the worker starts.
        # The candidate would name one model and the run could train another;
        # a digest or revision would pin it, and neither can be verified yet.
        yield (
            f"model.model.uri {model.uri!r} is not an absolute local path; the "
            f"worker cannot pin a hub name to the model the candidate means, so "
            f"it would train whatever the name resolves to when it starts"
        )
    if model.revision is not None:
        yield (
            "model.model.revision is declared; the worker's loader takes no revision, "
            "so it would load whatever the URI currently points at"
        )
    if model.digest is not None:
        yield (
            "model.model.digest is declared; it is part of the candidate's identity, "
            "but the worker cannot verify it and would train on whatever the URI "
            "currently holds"
        )


def _optimization_refusals(candidate: CandidateSpec, *, trainer: str) -> Iterator[str]:
    optimization = candidate.training.optimization
    prefix = "training.optimization"

    for field in _REQUIRED_OPTIMIZATION_FIELDS:
        if getattr(optimization, field) is None:
            yield f"{prefix}.{field} is undeclared"

    schedule = optimization.lr_schedule
    if schedule is None:
        yield f"{prefix}.lr_schedule is undeclared"
    else:
        if schedule.name not in SCHEDULERS:
            yield (
                f"{prefix}.lr_schedule.name {schedule.name!r} is not one {trainer} "
                f"implements ({', '.join(sorted(SCHEDULERS))})"
            )
        if schedule.params:
            yield f"{prefix}.lr_schedule.params are declared; {trainer} takes none"

    optimizer = optimization.optimizer
    if optimizer is None:
        yield f"{prefix}.optimizer is undeclared"
        return
    if optimizer.name != "adamw":
        yield (
            f"{prefix}.optimizer.name is {optimizer.name!r}; {trainer} is compiled to "
            f"AdamW and nothing else, so any other optimizer would be silently replaced"
        )
    if optimizer.weight_decay is None:
        yield f"{prefix}.optimizer.weight_decay is undeclared"
    if optimizer.params:
        yield f"{prefix}.optimizer.params are declared; {trainer} passes none"


def _precision_refusals(candidate: CandidateSpec, *, trainer: str) -> Iterator[str]:
    precision = candidate.training.precision
    if precision.dtype is None:
        yield "training.precision.dtype is undeclared"
    elif precision.dtype not in PRECISIONS:
        yield (
            f"training.precision.dtype {precision.dtype!r} is not one {trainer} "
            f"implements ({', '.join(sorted(PRECISIONS))})"
        )
    if precision.grad_accum_dtype is not None:
        yield f"training.precision.grad_accum_dtype is declared; {trainer} cannot set it"
    if precision.params:
        yield f"training.precision.params are declared; {trainer} takes none"


def _checkpoint_refusals(candidate: CandidateSpec, *, trainer: str) -> Iterator[str]:
    """No checkpoint intent is supported yet, so declaring one is refused.

    A trainer may be able to write periodic checkpoints, but no worker emits a
    ``CheckpointCommitted`` for them -- and the controller learns that a
    resumable position exists only when a checkpoint is reported (ADR-014).
    A checkpoint written silently is unusable for recovery and would claim a
    capability this path does not have. Checkpointing with the telemetry that
    makes it real is TASK-029, which flips this refusal.
    """
    checkpoint = candidate.training.checkpoint
    if checkpoint.every_optimizer_steps is not None:
        yield (
            "training.checkpoint.every_optimizer_steps is declared; the worker does "
            "not yet report checkpoints, so a resume point it wrote could not be "
            "found (TASK-029)"
        )
    if checkpoint.keep_last is not None:
        yield (
            f"training.checkpoint.keep_last is declared; {trainer} cannot honour a "
            f"retention count while it writes no checkpoints"
        )
    if checkpoint.params:
        yield f"training.checkpoint.params are declared; {trainer} takes none"
