from __future__ import annotations

from typing import Any

from trainlib.config.schema import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    TrainerConfig,
)
from trainlib.recipes import base as _base
from trainlib.trainer.callbacks import TrainState


def align(
    *,
    config: TrainConfig | None = None,
    model: str | None = None,
    dataset: str | None = None,
    method: str = "dpo",
    format: str = "preference",
    num_epochs: int = 1,
    learning_rate: float = 5e-6,
    batch_size: int = 4,
    **kwargs: Any,
) -> TrainState:
    if config is None:
        if model is None or dataset is None:
            raise ValueError("Either 'config' or both 'model' and 'dataset' are required.")

        trainer_fields = {}
        trainer_param_names = {f for f in TrainerConfig.model_fields}
        for k in list(kwargs.keys()):
            if k in trainer_param_names:
                trainer_fields[k] = kwargs.pop(k)

        config = TrainConfig(
            recipe="align",
            method=method,
            model=ModelConfig(name=model),
            data=DataConfig(path=dataset, format=format),
            trainer=TrainerConfig(
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                **trainer_fields,
            ),
        )

    components = _base.setup_training(config)

    state = components.trainer.train(
        model=components.model,
        train_dataloader=components.train_dataloader,
    )

    return state
