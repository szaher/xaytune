"""NativeWorker: runs a compiled native SFT config on the existing trainer.

```text
NativeSftConfig      the wire contract, from NativeCompiler
      ↓ train_config_from
TrainConfig          the native trainer's own configuration
      ↓
recipes.finetune     the existing, unmodified training loop
```

The translation is the only place ``TrainConfig`` appears, which is what keeps
it an implementation detail of this worker rather than part of the execution
contract.

**Every field that changes training is set explicitly**, from the config. The
ones left at ``TrainConfig`` defaults are inert for this workload and are named
below with the reason, because "left at default" is only safe when someone has
checked that the default cannot matter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xaytune.config.schema import (
    DataConfig,
    EvalConfig,
    LoggingConfig,
    ModelConfig,
    OnlineRLConfig,
    OutputConfig,
    TrainConfig,
    TrainerConfig,
)
from xaytune.core.immutable import FrozenDict
from xaytune.core.telemetry import (
    NumericalInstabilityObserved,
    TrainingCompletedPayload,
    TrainingFailedPayload,
    TrainingMetricObserved,
    TrainingStartedPayload,
)
from xaytune.runtimes.worker import (
    OBSERVATIONS_PATH_ENV,
    WORKER_CONFIG_PATH_ENV,
    ObservationWriter,
)
from xaytune.workers.common import (
    failure_reason,
    nonfinite,
    publish_model,
    report,
    require_environment,
)
from xaytune.workers.native_schema import NativeSftConfig

if TYPE_CHECKING:
    from xaytune.trainer.callbacks import CallbackManager

__all__ = [
    "UnsupportedPrecisionError",
    "main",
    "register_native_observation_callbacks",
    "require_honourable_precision",
    "train_config_from",
]


def train_config_from(config: Mapping[str, Any]) -> TrainConfig:
    """Build the native trainer's configuration from a compiled config.

    Validates against :class:`NativeSftConfig` first, so a config from a
    different compiler, or an older schema, is refused here rather than
    half-applied.

    **Every section that can change this run is set here, even where the value
    equals today's default.** A default is only neutral until someone changes
    it; an explicit value is neutral regardless. ``eval`` is set to never run
    rather than trusted to stay idle because no evaluation data happens to be
    supplied, and ``online_rl`` and ``data_prep`` are set off rather than
    assumed off. Evaluation is its own lifecycle (ADR-007), not something a
    training run does to itself.

    Left at ``TrainConfig`` defaults, because each is *gated by a field set
    here* and cannot be read on this path:

    ``lora``        read only when ``method == "lora"``; this is ``"full"``.
    ``ppo``         read only by the PPO recipe; this is ``"finetune"``.
    ``fsdp``,       read only under those strategies. ``strategy="auto"``
    ``deepspeed``   resolves to single-device, because LocalRuntime strips
                    ``WORLD_SIZE`` from the worker's environment.
    """
    spec = NativeSftConfig.model_validate(dict(config))
    optimization = spec.optimization
    realization = spec.realization

    return TrainConfig(
        recipe="finetune",
        method="full",
        model=ModelConfig(
            name=spec.model.uri,
            quantization=None,
            # As the artifact declares, so the checkpoint's own dtype decides
            # rather than one invented here.
            dtype="auto",
            trust_remote_code=False,
        ),
        data=DataConfig(
            path=spec.data.path,
            format=spec.data.format,
            source="local",
            eval_split=0.0,
            eval_path=None,
            packing=spec.data.packing,
            max_seq_length=spec.data.max_seq_length,
            streaming=False,
        ),
        trainer=TrainerConfig(
            strategy="auto",
            mixed_precision=optimization.mixed_precision,
            batch_size=optimization.micro_batch_size,
            gradient_accumulation=optimization.gradient_accumulation,
            learning_rate=optimization.learning_rate,
            num_epochs=optimization.epochs,
            # The trainer's word for "no cap" is -1; the config's is None.
            max_steps=-1 if optimization.max_steps is None else optimization.max_steps,
            warmup_steps=optimization.warmup_steps or 0,
            warmup_ratio=optimization.warmup_ratio or 0.0,
            scheduler=optimization.scheduler,
            weight_decay=optimization.weight_decay,
            max_grad_norm=optimization.max_grad_norm,
            seed=realization.seed,
            checkpoint_every_n_steps=realization.checkpoint_every_optimizer_steps,
            # The run's output is the model published after training, written
            # deliberately. A raw state_dict checkpoint at the end is a
            # different artifact with different meaning, and it should not
            # appear because a config happened to ask for it.
            save_last=False,
            activation_checkpointing=False,
            async_checkpoint=False,
        ),
        output=OutputConfig(dir=realization.output_dir, merge_on_complete=False),
        eval=EvalConfig(
            every_n_steps=0,
            metrics=[],
            benchmarks=[],
            early_stopping_patience=0,
        ),
        logging=LoggingConfig(backends=["console"], log_every_n_steps=10),
        online_rl=OnlineRLConfig(enabled=False),
        data_prep=[],
        method_params={},
        base=None,
    )


class _NativeObservations:
    """Turns trainer events into observations, and does nothing else.

    **Observational only.** Nothing here mutates the ``TrainState``, calls
    ``stop_training()``, changes a learning rate or touches the trainer. A
    callback able to do those things would be a way to change a run from
    inside the telemetry path, bypassing the ``Action → TrainingIntervention``
    boundary that makes such changes visible and attributable (ADR-011).

    **Reports the trainer's numbers as the trainer means them**, which is not
    always what their names suggest:

    ``learning_rate``  The loop records it after ``scheduler.step()``, so at
                       ``step_end`` it is the rate for the *next* step. It is
                       read at ``step_start`` instead, where it is still the
                       rate this step used. ``None`` for the first step, which
                       nothing has recorded yet -- unknown, not zero.
    ``loss``           Overwritten every micro-batch, so under gradient
                       accumulation the value at ``step_end`` is only the
                       *last* micro-batch's. Reported as ``loss`` when there is
                       no accumulation, where it is the step's loss; otherwise
                       as ``final_micro_batch_loss`` in metadata, because
                       labelling it ``loss`` would claim a number the trainer
                       never computed, and averaging would invent one.

    Nothing the trainer does not expose is reported -- no throughput, gradient
    norm or resource figures. Those arrive with a real collector.
    """

    def __init__(self, writer: ObservationWriter, *, gradient_accumulation: int) -> None:
        self._writer = writer
        self._accumulating = gradient_accumulation > 1
        self._learning_rate: float | None = None

    def train_start(self, state: Any) -> None:
        self._report(lambda: TrainingStartedPayload(optimizer_step=state.global_step))

    def step_start(self, state: Any) -> None:
        self._learning_rate = state.metrics.get("learning_rate")

    def step_end(self, state: Any) -> None:
        loss = state.metrics.get("loss")
        step = state.global_step

        if loss is not None and not math.isfinite(loss):
            # The vocabulary's own term for this, and the one signal that the
            # run diverged. A NaN cannot be a JSON number, so it is reported
            # symbolically rather than dropped as an invalid metric.
            self._report(
                lambda: NumericalInstabilityObserved(
                    optimizer_step=step,
                    quantity="loss",
                    observation=nonfinite(loss),
                )
            )
            return

        if self._accumulating:
            self._report(
                lambda: TrainingMetricObserved(
                    optimizer_step=step,
                    learning_rate=self._learning_rate,
                    metadata=FrozenDict({} if loss is None else {"final_micro_batch_loss": loss}),
                )
            )
        else:
            self._report(
                lambda: TrainingMetricObserved(
                    optimizer_step=step, loss=loss, learning_rate=self._learning_rate
                )
            )

    def train_end(self, state: Any) -> None:
        """The loop ended normally -- which includes a legitimate early stop.

        That is a fact about the training loop, not about the experiment: it
        does not say the objective was met, which is evaluation's question.
        """
        self._report(lambda: TrainingCompletedPayload(optimizer_step=state.global_step))

    def _report(self, build: Any) -> None:
        report(self._writer, build)


class UnsupportedPrecisionError(RuntimeError):
    """The declared precision cannot be honoured on the device the model is on."""


def require_honourable_precision(model: Any, precision: str) -> None:
    """Refuse a declared precision the native trainer would silently drop.

    The trainer loop switches autocast off wherever it does not support it --
    on CPU, for any half precision -- and trains in fp32 without a word, so a
    candidate declaring ``bf16`` would run as a different proposition under
    its fingerprint. Execution may fail because hardware cannot do what was
    declared; it may not quietly do something else. So this mirrors the
    loop's own decision, and then checks the hardware:

    - the device must be one the loop runs autocast on at all;
    - on CUDA, ``bf16`` needs ``torch.cuda.is_bf16_supported()`` -- a GPU
      that runs autocast does not necessarily have bf16;
    - and a one-matmul probe on the model's device must actually produce the
      declared dtype, which catches a backend that accepts the request and
      computes in something else.

    The legacy trainer keeps its fallback; this is the control-plane path's
    rule, enforced at the worker boundary.

    Raises:
        UnsupportedPrecisionError: Naming the precision and the device.
    """
    if precision == "fp32":
        return

    import torch

    from xaytune.trainer.device import detect_device_type_from_model, supports_amp

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    device_type = detect_device_type_from_model(model)

    if not supports_amp(device_type):
        raise UnsupportedPrecisionError(
            f"{precision} was declared, but the native trainer runs no mixed precision "
            f"on {device_type}; it would silently train in fp32"
        )
    if device_type == "cuda" and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise UnsupportedPrecisionError(
            "bf16 was declared, but this CUDA device does not support bf16"
        )

    probe = torch.ones(2, 2, device=device_type)
    try:
        with torch.amp.autocast(device_type, dtype=dtype):
            produced = (probe @ probe).dtype
    except (RuntimeError, TypeError) as exc:
        raise UnsupportedPrecisionError(
            f"{precision} was declared, but autocast on {device_type} refused it: {exc}"
        ) from exc
    if produced != dtype:
        raise UnsupportedPrecisionError(
            f"{precision} was declared, but autocast on {device_type} computes in {produced}"
        )


def register_native_observation_callbacks(
    callbacks: CallbackManager,
    writer: ObservationWriter,
    *,
    gradient_accumulation: int,
) -> None:
    """Attach the observational callbacks. Named for what they are allowed to do."""
    observations = _NativeObservations(writer, gradient_accumulation=gradient_accumulation)
    callbacks.on("train_start")(observations.train_start)
    callbacks.on("step_start")(observations.step_start)
    callbacks.on("step_end")(observations.step_end)
    callbacks.on("train_end")(observations.train_end)


def main(*_arguments: str) -> int:
    """Run one compiled native SFT config, reporting as it goes.

    Built on ``setup_training`` rather than ``recipes.finetune``: the recipe
    creates its own callback manager and gives no way to attach one, and this
    worker exists to attach one. The recipe stays the legacy convenience API;
    this is the implementation behind the execution boundary.
    """
    config_path = Path(require_environment(WORKER_CONFIG_PATH_ENV))
    writer = ObservationWriter(Path(require_environment(OBSERVATIONS_PATH_ENV)))
    writer.verify()

    worker_config = NativeSftConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    train_config = train_config_from(worker_config.model_dump(mode="json"))

    # Heavy imports here, so the translator above stays importable -- and
    # testable -- without the training stack.
    from xaytune.recipes.base import setup_training
    from xaytune.trainer.callbacks import CallbackManager

    callbacks = CallbackManager()
    register_native_observation_callbacks(
        callbacks,
        writer,
        gradient_accumulation=worker_config.optimization.gradient_accumulation,
    )

    try:
        components = setup_training(train_config, callback_manager=callbacks)
        # After the model is placed, because only then is the device known,
        # and before train(), which would otherwise degrade silently.
        require_honourable_precision(components.model, worker_config.optimization.mixed_precision)
        components.trainer.train(
            model=components.model,
            train_dataloader=components.train_dataloader,
            resume_state=components.resume_state,
        )
    except Exception as exc:
        # The trainer fires no event on an exception, so this boundary is the
        # only place that can report one. It is training-level evidence; the
        # launcher separately records the non-zero exit as process-level
        # evidence, and the two are not duplicates.
        try:
            writer.write(TrainingFailedPayload(reason=failure_reason(exc), detail=str(exc)))
        except OSError:
            # The channel failing too must not replace the error that matters.
            pass
        raise

    # Its own step, after training and outside the block above: training has
    # completed by now, so a failure here is a failure to publish, and
    # reporting it as TrainingFailed would say something false about the run.
    _publish_model(components.model, components.tokenizer, worker_config, writer)
    return 0


def _publish_model(
    model: Any, tokenizer: Any, worker_config: NativeSftConfig, writer: ObservationWriter
) -> None:
    """Produce the model the plan declared, and say so (see ``publish_model``)."""
    publish_model(model, tokenizer, Path(worker_config.realization.output_dir), writer)
