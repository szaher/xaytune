"""NativeCompiler: an SFT candidate into a plan the native trainer can run.

The first real :class:`~xaytune.compilation.TrainerCompiler`.

**A value that changes what the model learns comes from the candidate, or the
candidate is refused.** The native trainer has a default for nearly every field
it reads, and a compiler that left one unset would not raise -- it would produce
a run the candidate never described, under a fingerprint claiming to describe it
completely. So :meth:`NativeCompiler.supports` refuses anything undeclared, and
refuses anything the native loop would silently replace: it builds
``AdamW(lr, weight_decay)`` and nothing else, so a candidate naming SGD, or AdamW
with other betas, would train on that AdamW regardless.

**Compilation inspects nothing.** No filesystem, no model, no dataset, no
environment, no clock. Paths are opaque strings: a compiler whose output
depended on the machine it ran on would produce plans that could not be trusted
on the machine that executes them.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from xaytune.compilation import CompilationContext, SupportResult, UnsupportedCandidateError
from xaytune.core.capabilities import (
    PLUGIN_API_VERSIONS,
    AlgorithmCapabilities,
    CapabilityDocument,
    DistributedCapabilities,
    PluginDescriptor,
)
from xaytune.core.domain.candidate import CandidateSpec, TrainingKind
from xaytune.core.execution import (
    ArtifactOutput,
    CheckpointExecutionContract,
    CompilerIdentity,
    PythonModuleEntrypoint,
    ResourceRequirements,
    TelemetryContract,
    TrainingExecutionSpec,
)
from xaytune.core.immutable import FrozenDict
from xaytune.workers.native_schema import (
    NativeData,
    NativeModel,
    NativeOptimization,
    NativeRealization,
    NativeSftConfig,
)

__all__ = ["NativeCompiler"]

_WORKER_MODULE = "xaytune.workers.native"

_SCHEDULERS = frozenset({"cosine", "linear", "constant", "constant_with_warmup"})
_PRECISIONS = frozenset({"fp16", "bf16", "fp32"})

_ADAMW_DEFAULT_BETAS = (0.9, 0.999)
"""The betas ``torch.optim.AdamW`` uses when none are passed, which is always.

A candidate may declare them explicitly -- that describes exactly what will
run -- but any other value would be silently replaced by these.
"""


class NativeCompiler:
    """Compiles plain supervised fine-tuning for the native trainer."""

    descriptor = PluginDescriptor(
        api_version=PLUGIN_API_VERSIONS[0],
        name="native",
        plugin_version="0.1.0",
        provider="xaytune",
        xaytune_version="0.6.0",
    )

    def capabilities(self) -> CapabilityDocument:
        """Plain SFT, one worker, full-parameter."""
        return CapabilityDocument(
            algorithms=AlgorithmCapabilities(supported=("sft",)),
            distributed=DistributedCapabilities(strategies=(), min_workers=1, max_workers=1),
        )

    def supports(self, candidate: CandidateSpec) -> SupportResult:
        """Whether this compiler can run *candidate* exactly as declared.

        Every reason, not the first -- a planner that learned one defect per
        round-trip would pay for defects the compiler already knew about.
        """
        reasons = tuple(_refusals(candidate))
        return SupportResult(supported=not reasons, reasons=reasons)

    def compile(
        self, candidate: CandidateSpec, context: CompilationContext
    ) -> TrainingExecutionSpec:
        """Return how to run *candidate*. Mechanical, deterministic, inert.

        Raises:
            UnsupportedCandidateError: With every reason, if ``supports()``
                would refuse the candidate.
            ValueError: If the context lacks the realization a run needs --
                a seed and an output location. Neither is defaulted: a run with
                an invented seed is not reproducible, whatever the seed is.
        """
        supported = self.supports(candidate)
        if not supported:
            raise UnsupportedCandidateError(self.descriptor.name, supported.reasons)

        if context.seed is None:
            raise ValueError(
                "the compilation context has no seed; a run is only reproducible "
                "if its seed was chosen deliberately, so it is not defaulted"
            )
        if context.output_uri is None:
            raise ValueError("the compilation context has no output_uri to write the model to")
        output_dir = _local_path(context.output_uri)
        if output_dir is None:
            raise ValueError(
                f"output_uri {context.output_uri!r} is not an absolute local path; the "
                f"native worker writes the model with save_pretrained() to a local "
                f"directory, so any other location would be written somewhere else "
                f"than the plan and the ArtifactProduced event claim"
            )

        training = candidate.training
        optimization = training.optimization
        # supports() has proved all of these are declared.
        assert optimization.optimizer is not None and optimization.lr_schedule is not None
        assert optimization.learning_rate is not None
        assert optimization.micro_batch_size is not None
        assert optimization.gradient_accumulation is not None
        assert optimization.epochs is not None
        assert optimization.max_grad_norm is not None
        assert optimization.optimizer.weight_decay is not None
        assert training.precision.dtype is not None
        assert candidate.data.format is not None
        assert candidate.data.max_seq_length is not None
        assert candidate.data.packing is not None
        dataset_path = _local_path(candidate.data.dataset.uri)
        assert dataset_path is not None

        # Always 0 while supports() refuses checkpoint intent; kept in the wire
        # schema so TASK-029 changes the refusal, not the contract.
        every_steps = 0

        config = NativeSftConfig(
            model=NativeModel(uri=candidate.model.model.uri),
            data=NativeData(
                path=dataset_path,
                format=candidate.data.format,
                max_seq_length=candidate.data.max_seq_length,
                packing=candidate.data.packing,
            ),
            optimization=NativeOptimization(
                learning_rate=optimization.learning_rate,
                micro_batch_size=optimization.micro_batch_size,
                gradient_accumulation=optimization.gradient_accumulation,
                epochs=optimization.epochs,
                max_steps=optimization.max_steps,
                max_grad_norm=optimization.max_grad_norm,
                weight_decay=optimization.optimizer.weight_decay,
                scheduler=optimization.lr_schedule.name,  # type: ignore[arg-type]
                warmup_steps=optimization.lr_schedule.warmup_steps,
                warmup_ratio=optimization.lr_schedule.warmup_ratio,
                mixed_precision=training.precision.dtype,  # type: ignore[arg-type]
            ),
            realization=NativeRealization(
                seed=context.seed,
                output_dir=output_dir,
                checkpoint_every_optimizer_steps=every_steps,
            ),
        )

        return TrainingExecutionSpec(
            compiler=CompilerIdentity(
                name=self.descriptor.name,
                version=self.descriptor.plugin_version,
                descriptor=self.descriptor,
            ),
            candidate_fingerprint=candidate.candidate_fingerprint(),
            entrypoint=PythonModuleEntrypoint(module=_WORKER_MODULE, function="main"),
            config=FrozenDict(config.model_dump(mode="json")),
            outputs=(ArtifactOutput(name="model", uri=output_dir, kind="model"),),
            resources=ResourceRequirements(workers=1),
            checkpoint=CheckpointExecutionContract(
                store_uri=context.checkpoint_store_uri,
                every_optimizer_steps=every_steps or None,
                boundary="optimizer-step",
                # The native trainer does not commit checkpoints atomically, so
                # requiring it would be a claim the worker cannot keep. Atomic
                # checkpoint commit is TASK-029.
                require_atomic_commit=False,
            ),
            telemetry=TelemetryContract(protocol_version="xaytune.telemetry/v1alpha2"),
        )


def _refusals(candidate: CandidateSpec) -> Iterator[str]:
    """Every reason this compiler cannot run *candidate* as declared."""
    training = candidate.training

    if training.kind is not TrainingKind.SFT:
        yield (
            f"training.kind is {training.kind.value!r}; the native compiler supports "
            f"only 'sft' in this release"
        )
    if training.adapter is not None:
        yield "training.adapter is declared; the native compiler runs full-parameter SFT only"
    if training.algorithm.name is not None or training.algorithm.params:
        yield "training.algorithm is declared; plain SFT has no algorithm variant to select"
    if candidate.reward is not None:
        yield "reward is declared; supervised fine-tuning has no reward"
    if candidate.environment is not None:
        yield "environment is declared; supervised fine-tuning has no environment"
    if candidate.schedule is not None:
        yield "schedule is declared; the native worker cannot apply scheduled interventions yet"

    yield from _data_refusals(candidate)
    yield from _model_refusals(candidate)
    yield from _optimization_refusals(candidate)
    yield from _precision_refusals(candidate)
    yield from _checkpoint_refusals(candidate)


def _data_refusals(candidate: CandidateSpec) -> Iterator[str]:
    data = candidate.data
    dataset = data.dataset

    if data.format is None:
        yield "data.format is undeclared; it decides what the file's records become"
    if data.max_seq_length is None:
        yield "data.max_seq_length is undeclared; it decides where examples are truncated"
    if data.packing is None:
        yield "data.packing is undeclared; it decides whether examples share sequences"

    if _local_path(dataset.uri) is None:
        yield (
            f"data.dataset.uri {dataset.uri!r} is not an absolute local path; the "
            f"native worker reads local JSONL, and a relative path would depend on "
            f"the working directory of whichever process resolved it"
        )
    if dataset.revision is not None:
        yield "data.dataset.revision is declared; a local file has no revisions to select"
    if dataset.split is not None:
        yield "data.dataset.split is declared; the native loader reads a whole file"
    for field in (
        "content_digest",
        "transform_fingerprint",
        "tokenizer_fingerprint",
        "template_fingerprint",
    ):
        if getattr(dataset, field) is not None:
            yield (
                f"data.dataset.{field} is declared; it is part of the candidate's "
                f"identity, but the native worker cannot verify it and would train on "
                f"whatever the file and its own preprocessing currently produce"
            )


def _local_path(uri: str) -> str | None:
    """*uri* as an absolute local path, or ``None`` if it is not one.

    Reading the declared string, not the filesystem. ``file://`` is accepted and
    removed, because the worker hands the result to ``Path``; any other scheme
    is not absolute and so is not local.
    """
    location = uri.removeprefix("file://")
    return location if os.path.isabs(location) else None


def _model_refusals(candidate: CandidateSpec) -> Iterator[str]:
    if candidate.model.model.revision is not None:
        yield (
            "model.model.revision is declared; the native loader takes no revision, "
            "so it would load whatever the URI currently points at"
        )
    if candidate.model.model.digest is not None:
        yield (
            "model.model.digest is declared; it is part of the candidate's identity, "
            "but the native worker cannot verify it and would train on whatever the "
            "URI currently holds"
        )


def _optimization_refusals(candidate: CandidateSpec) -> Iterator[str]:
    optimization = candidate.training.optimization
    prefix = "training.optimization"

    for field in (
        "learning_rate",
        "micro_batch_size",
        "gradient_accumulation",
        "epochs",
        "max_grad_norm",
    ):
        if getattr(optimization, field) is None:
            yield f"{prefix}.{field} is undeclared"

    schedule = optimization.lr_schedule
    if schedule is None:
        yield f"{prefix}.lr_schedule is undeclared"
    else:
        if schedule.name not in _SCHEDULERS:
            yield (
                f"{prefix}.lr_schedule.name {schedule.name!r} is not one the native "
                f"trainer implements ({', '.join(sorted(_SCHEDULERS))})"
            )
        if schedule.params:
            yield f"{prefix}.lr_schedule.params are declared; the native trainer takes none"
        if (schedule.warmup_steps or 0) > 0 and (schedule.warmup_ratio or 0) > 0:
            yield (
                f"{prefix}.lr_schedule declares both warmup_steps and warmup_ratio; the "
                f"native trainer uses warmup_steps and silently ignores the ratio"
            )

    optimizer = optimization.optimizer
    if optimizer is None:
        yield f"{prefix}.optimizer is undeclared"
        return
    if optimizer.name != "adamw":
        yield (
            f"{prefix}.optimizer.name is {optimizer.name!r}; the native trainer builds "
            f"AdamW and nothing else, so any other optimizer would be silently replaced"
        )
    if optimizer.weight_decay is None:
        yield f"{prefix}.optimizer.weight_decay is undeclared"
    if optimizer.betas and tuple(optimizer.betas) != _ADAMW_DEFAULT_BETAS:
        yield (
            f"{prefix}.optimizer.betas {tuple(optimizer.betas)} cannot be set; the native "
            f"trainer always uses AdamW's defaults {_ADAMW_DEFAULT_BETAS}"
        )
    if optimizer.params:
        yield f"{prefix}.optimizer.params are declared; the native trainer passes none"


def _precision_refusals(candidate: CandidateSpec) -> Iterator[str]:
    precision = candidate.training.precision
    if precision.dtype is None:
        yield "training.precision.dtype is undeclared"
    elif precision.dtype not in _PRECISIONS:
        yield (
            f"training.precision.dtype {precision.dtype!r} is not one the native "
            f"trainer implements ({', '.join(sorted(_PRECISIONS))})"
        )
    if precision.grad_accum_dtype is not None:
        yield "training.precision.grad_accum_dtype is declared; the native trainer cannot set it"
    if precision.params:
        yield "training.precision.params are declared; the native trainer takes none"


def _checkpoint_refusals(candidate: CandidateSpec) -> Iterator[str]:
    """No checkpoint intent is supported yet, so declaring one is refused.

    The native trainer can write periodic checkpoints, but it emits no
    ``CheckpointCommitted`` for them -- and the controller learns that a
    resumable position exists only when a checkpoint is reported (ADR-014).
    A checkpoint written silently is unusable for recovery and would claim a
    capability this path does not have. Checkpointing with the telemetry that
    makes it real is TASK-029, which flips this refusal.
    """
    checkpoint = candidate.training.checkpoint
    if checkpoint.every_optimizer_steps is not None:
        yield (
            "training.checkpoint.every_optimizer_steps is declared; the native worker "
            "does not yet report checkpoints, so a resume point it wrote could not be "
            "found (TASK-029)"
        )
    if checkpoint.keep_last is not None:
        yield (
            "training.checkpoint.keep_last is declared; the native trainer keeps only "
            "the last checkpoint and cannot honour a retention count"
        )
    if checkpoint.params:
        yield "training.checkpoint.params are declared; the native trainer takes none"
