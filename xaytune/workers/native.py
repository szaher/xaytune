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
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from xaytune.config.schema import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainerConfig,
)
from xaytune.core.ids import ArtifactId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import ArtifactRef
from xaytune.core.telemetry import (
    ArtifactProducedPayload,
    IncidentObservedPayload,
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
from xaytune.workers.native_schema import NativeSftConfig

if TYPE_CHECKING:
    from xaytune.trainer.callbacks import CallbackManager

__all__ = ["main", "register_native_observation_callbacks", "train_config_from"]


def train_config_from(config: Mapping[str, Any]) -> TrainConfig:
    """Build the native trainer's configuration from a compiled config.

    Validates against :class:`NativeSftConfig` first, so a config from a
    different compiler, or an older schema, is refused here rather than
    half-applied.

    Left at ``TrainConfig`` defaults, deliberately, because each is inert for
    full-parameter SFT:

    ``lora``        read only when ``method == "lora"``; this is ``"full"``.
    ``eval``        no evaluation data is ever supplied (``eval_split=0``, no
                    ``eval_path``), so in-training evaluation cannot run, and
                    early stopping is disabled by default. Evaluation is its
                    own lifecycle (ADR-007), not something a training run does
                    to itself.
    ``ppo``,        off; this is not an RL recipe.
    ``online_rl``
    ``fsdp``,       read only under those strategies. ``strategy="auto"``
    ``deepspeed``   resolves to single-device when ``WORLD_SIZE`` is unset,
                    which LocalRuntime does not set.
    ``logging``     console only; nothing leaves the process.
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
            # As the artifact declares. The model artifact is identity already
            # (its URI and revision), so this is determined by the candidate
            # rather than invented here.
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
            save_last=True,
            activation_checkpointing=False,
            async_checkpoint=False,
        ),
        output=OutputConfig(dir=realization.output_dir, merge_on_complete=False),
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
                    observation=_nonfinite(loss),
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
        """Build one observation and write it, keeping the two failures apart.

        A value the vocabulary refuses is one bad observation: it is reported
        on stderr and skipped, and training continues, because the observation
        channel is not authoritative and one malformed metric is no reason to
        lose a run. A failure to *write* is not caught: that is the runtime's
        channel broken, which is an execution-contract failure.
        """
        try:
            observation = build()
        except ValidationError as exc:
            print(f"xaytune: dropped an observation the vocabulary refused: {exc}", file=sys.stderr)
            return
        self._writer.write(observation)


def _nonfinite(value: float) -> Literal["nan", "positive-infinity", "negative-infinity"]:
    if math.isnan(value):
        return "nan"
    return "positive-infinity" if value > 0 else "negative-infinity"


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


def _require_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set; NativeWorker runs under a runtime that provides it, "
            f"and without it there is no config to run or channel to report on"
        )
    return value


def main(*_arguments: str) -> int:
    """Run one compiled native SFT config, reporting as it goes.

    Built on ``setup_training`` rather than ``recipes.finetune``: the recipe
    creates its own callback manager and gives no way to attach one, and this
    worker exists to attach one. The recipe stays the legacy convenience API;
    this is the implementation behind the execution boundary.
    """
    config_path = Path(_require_environment(WORKER_CONFIG_PATH_ENV))
    writer = ObservationWriter(Path(_require_environment(OBSERVATIONS_PATH_ENV)))
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
            writer.write(TrainingFailedPayload(reason=_reason(exc), detail=str(exc)))
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
    """Produce the model the plan declared, and say so.

    The trainer leaves only a checkpoint -- a raw ``state_dict`` -- which
    nothing outside this trainer can load. The plan's declared ``model`` output
    is a loadable artifact, so it is written with ``save_pretrained``, the same
    form a candidate names when it refers to a model: what this run produces
    is something the next candidate can train from.

    Reported after it exists, not before. An ``ArtifactProduced`` sent ahead of
    a write that then failed would be a claim with nothing behind it.

    No content digest yet. Hashing multi-gigabyte weights on the training host
    is a real cost, and whether it happens here, asynchronously, or in an
    artifact store is a decision worth making deliberately rather than by
    default.

    Raises:
        Exception: Whatever the save raised, after reporting it -- the run did
            train, but did not produce what the plan declared, so the process
            must not exit as though it had.
    """
    target = Path(worker_config.realization.output_dir)
    try:
        # Unwrapped if distributed wrapping ever applies; a plain model here.
        getattr(model, "module", model).save_pretrained(target)
        if tokenizer is not None:
            tokenizer.save_pretrained(target)
    except Exception as exc:
        writer.write(
            IncidentObservedPayload(
                reason="artifact-publication-failed",
                detail=f"training completed but the model could not be written: {exc}",
            )
        )
        raise

    writer.write(
        ArtifactProducedPayload(
            # Attribution travels on the envelope's target, which names the
            # attempt, so it is not repeated as producer_attempt_id here.
            artifact_ref=ArtifactRef(id=ArtifactId.generate(), kind="model", uri=str(target))
        )
    )


def _reason(exc: BaseException) -> str:
    """A ``Name``-shaped reason: the exception's type, lower-kebab-cased."""
    name = type(exc).__name__
    kebab = "".join(f"-{c.lower()}" if c.isupper() else c for c in name).lstrip("-")
    return kebab or "error"
