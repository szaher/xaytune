"""TRLCompiler: an SFT candidate into a plan TRL's ``SFTTrainer`` can run.

The second :class:`~xaytune.compilation.TrainerCompiler`, and the first for a
trainer Xaytune does not own. That is the point of it: one compiler behind a
boundary shows the boundary can be implemented; two, on unrelated trainers,
are evidence it is not a description of the first one.

Same rule as :mod:`xaytune.compilation.native`: **a value that changes what
the model learns comes from the candidate, or the candidate is refused.** TRL
makes the rule harder to keep, because ``SFTConfig`` has a default for well
over a hundred fields and several of them change training (see
:mod:`xaytune.workers.trl`). The compiler's half of the job is to refuse what
TRL would do *differently from what the candidate says*:

``weight_decay > 0``   ``Trainer`` exempts biases and normalization weights
                       from decay; the native trainer decays every parameter.
                       The candidate does not say which is meant, so a
                       non-zero value would mean different things on the two
                       trainers under one fingerprint.
``warmup_ratio``       ``TrainingArguments`` rounds a ratio up; the native
                       trainer rounds down. Only a step count means one thing.
``format != "text"``   Prompt/completion and chat data make TRL choose
                       completion-only loss, which the candidate cannot yet
                       express.
``packing``            TRL packs best-fit-decreasing; the native trainer does
                       not. Same word, different datasets.

And it can honour what the native compiler must refuse: TRL passes AdamW's
betas through, so a candidate declaring them is supported here. Two compilers
accepting different candidates is the capability resolution ADR-008 describes,
not an inconsistency.

**Compilation inspects nothing**, and imports nothing from TRL: a controller
host compiles plans without the worker's training stack installed.
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
from xaytune.workers.trl_schema import (
    TRLData,
    TRLModel,
    TRLOptimization,
    TRLRealization,
    TRLSftConfig,
)

__all__ = ["TRLCompiler"]

_WORKER_MODULE = "xaytune.workers.trl"
_TRAINER = "the TRL trainer"

_ADAMW_DEFAULT_BETAS = (0.9, 0.999)
_ADAMW_DEFAULT_EPSILON = 1e-8
"""``torch.optim.AdamW``'s defaults, which the native trainer always uses.

An undeclared value must mean the same number on both trainers, so it is
pinned to the optimizer's own default rather than left to
``TrainingArguments``, whose defaults are TRL's and transformers' to change.
"""


class TRLCompiler:
    """Compiles plain-text supervised fine-tuning for TRL's ``SFTTrainer``."""

    descriptor = PluginDescriptor(
        api_version=PLUGIN_API_VERSIONS[0],
        name="trl",
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
        """Whether TRL can run *candidate* exactly as declared, with every reason."""
        reasons = tuple(_refusals(candidate))
        return SupportResult(supported=not reasons, reasons=reasons)

    def compile(
        self, candidate: CandidateSpec, context: CompilationContext
    ) -> TrainingExecutionSpec:
        """Return how to run *candidate*. Mechanical, deterministic, inert.

        Raises:
            UnsupportedCandidateError: With every reason, if ``supports()``
                would refuse the candidate.
            ValueError: If the context lacks a seed or an output location, or
                the output location is not an absolute local path.
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
                f"TRL worker writes the model with save_pretrained() to a local "
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
        assert candidate.data.max_seq_length is not None
        dataset_path = local_path(candidate.data.dataset.uri)
        assert dataset_path is not None
        model_path = local_path(candidate.model.model.uri)
        assert model_path is not None

        schedule = optimization.lr_schedule
        warmup_steps = schedule.warmup_steps or 0
        scheduler = schedule.name
        if scheduler == "constant" and warmup_steps > 0:
            # transformers' "constant" ignores warmup. The candidate declared a
            # warmup, and the native trainer honours one under this name, so
            # the schedule that does what was declared is the one sent.
            scheduler = "constant_with_warmup"

        beta1, beta2 = tuple(optimization.optimizer.betas) or _ADAMW_DEFAULT_BETAS

        config = TRLSftConfig(
            model=TRLModel(uri=model_path),
            data=TRLData(path=dataset_path, max_length=candidate.data.max_seq_length),
            optimization=TRLOptimization(
                learning_rate=optimization.learning_rate,
                micro_batch_size=optimization.micro_batch_size,
                gradient_accumulation=optimization.gradient_accumulation,
                epochs=optimization.epochs,
                max_steps=optimization.max_steps,
                max_grad_norm=optimization.max_grad_norm,
                weight_decay=optimization.optimizer.weight_decay,
                adam_beta1=beta1,
                adam_beta2=beta2,
                adam_epsilon=_ADAMW_DEFAULT_EPSILON,
                scheduler=scheduler,  # type: ignore[arg-type]
                warmup_steps=warmup_steps,
                mixed_precision=training.precision.dtype,  # type: ignore[arg-type]
            ),
            realization=TRLRealization(seed=context.seed, output_dir=output_dir),
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
                every_optimizer_steps=None,
                boundary="optimizer-step",
                require_atomic_commit=False,
            ),
            telemetry=TelemetryContract(protocol_version="xaytune.telemetry/v1alpha2"),
        )


def _refusals(candidate: CandidateSpec) -> Iterator[str]:
    """Every reason TRL cannot run *candidate* as declared."""
    yield from sft_refusals(candidate, trainer=_TRAINER)

    data = candidate.data
    if data.format is not None and data.format != "text":
        yield (
            f"data.format is {data.format!r}; {_TRAINER} supports only 'text' in this "
            f"release, because other formats make TRL choose completion-only loss, "
            f"which the candidate cannot express"
        )
    if data.packing:
        yield (
            f"data.packing is declared; {_TRAINER} packs best-fit-decreasing, which "
            f"is not what packing means on the native trainer, and the candidate "
            f"cannot say which it means"
        )

    optimization = candidate.training.optimization
    prefix = "training.optimization"

    schedule = optimization.lr_schedule
    if schedule is not None and schedule.warmup_ratio:
        yield (
            f"{prefix}.lr_schedule.warmup_ratio is declared; transformers rounds a "
            f"ratio to steps differently from the native trainer, so only "
            f"warmup_steps means the same thing on both"
        )

    optimizer = optimization.optimizer
    if optimizer is not None and optimizer.betas and len(optimizer.betas) != 2:
        yield (
            f"{prefix}.optimizer.betas {tuple(optimizer.betas)} is not a pair; AdamW "
            f"takes exactly two"
        )
    if optimizer is not None and optimizer.weight_decay:
        yield (
            f"{prefix}.optimizer.weight_decay is {optimizer.weight_decay}; {_TRAINER} "
            f"exempts biases and normalization weights from decay while the native "
            f"trainer decays every parameter, and the candidate does not say which "
            f"is meant"
        )
