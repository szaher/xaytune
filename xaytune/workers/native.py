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

from collections.abc import Mapping
from typing import Any

from xaytune.config.schema import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainerConfig,
)
from xaytune.workers.native_schema import NativeSftConfig

__all__ = ["train_config_from"]


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
