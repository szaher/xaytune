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

from collections.abc import Iterator

from xaytune.compilation import CompilationContext, SupportResult, UnsupportedCandidateError
from xaytune.compilation._sft import local_path, sft_refusals
from xaytune.core.capabilities import (
    PLUGIN_API_VERSIONS,
    AlgorithmCapabilities,
    CapabilityDocument,
    DistributedCapabilities,
    PluginDescriptor,
)
from xaytune.core.domain.candidate import CandidateSpec
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
        output_dir = local_path(context.output_uri)
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
        dataset_path = local_path(candidate.data.dataset.uri)
        assert dataset_path is not None
        model_path = local_path(candidate.model.model.uri)
        assert model_path is not None

        # Always 0 while supports() refuses checkpoint intent; kept in the wire
        # schema so TASK-029 changes the refusal, not the contract.
        every_steps = 0

        config = NativeSftConfig(
            model=NativeModel(uri=model_path),
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
    yield from sft_refusals(candidate, trainer="the native trainer")

    optimization = candidate.training.optimization
    prefix = "training.optimization"

    schedule = optimization.lr_schedule
    if (
        schedule is not None
        and (schedule.warmup_steps or 0) > 0
        and (schedule.warmup_ratio or 0) > 0
    ):
        yield (
            f"{prefix}.lr_schedule declares both warmup_steps and warmup_ratio; the "
            f"native trainer uses warmup_steps and silently ignores the ratio"
        )

    optimizer = optimization.optimizer
    if optimizer is not None and optimizer.betas and tuple(optimizer.betas) != _ADAMW_DEFAULT_BETAS:
        yield (
            f"{prefix}.optimizer.betas {tuple(optimizer.betas)} cannot be set; the native "
            f"trainer always uses AdamW's defaults {_ADAMW_DEFAULT_BETAS}"
        )
