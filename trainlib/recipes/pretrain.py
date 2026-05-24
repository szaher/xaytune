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


def pretrain(
    *,
    config: TrainConfig | None = None,
    model: str | None = None,
    dataset: str | None = None,
    format: str = "text",
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    batch_size: int = 4,
    resume_from: str | None = None,
    **kwargs: Any,
) -> TrainState:
    """Pre-train a language model on raw text with a causal LM objective.

    Accepts either a fully specified ``TrainConfig`` or individual arguments
    for quick one-liner usage.  Extra ``**kwargs`` that match
    ``TrainerConfig`` fields are forwarded automatically.

    Args:
        config: Complete training configuration. When provided, all other
            arguments except ``resume_from`` are ignored.
        model: HuggingFace model name or local path.
        dataset: Path to a JSONL corpus file (each line: ``{"text": "..."}``)
            or a HuggingFace dataset name.
        format: Data format — typically ``"text"`` for pre-training.
        num_epochs: Number of training epochs.
        learning_rate: Peak learning rate.
        batch_size: Per-device batch size.
        resume_from: Path to a checkpoint directory to resume from.
        **kwargs: Additional ``TrainerConfig`` fields (``max_steps``,
            ``mixed_precision``, ``scheduler``, etc.).

    Returns:
        Final training state with loss, global step count, and other metrics.

    Raises:
        ValueError: If neither ``config`` nor both ``model`` and ``dataset``
            are provided.

    Example::

        state = trainlib.pretrain(
            model="gpt2",
            dataset="data/corpus.jsonl",
            num_epochs=1,
            max_steps=1000,
        )
    """
    if config is None:
        if model is None or dataset is None:
            raise ValueError("Either 'config' or both 'model' and 'dataset' are required.")

        trainer_fields = {}
        trainer_param_names = {f for f in TrainerConfig.model_fields}
        for k, v in list(kwargs.items()):
            if k in trainer_param_names:
                trainer_fields[k] = kwargs.pop(k)

        config = TrainConfig(
            recipe="pretrain",
            method="full",
            model=ModelConfig(name=model),
            data=DataConfig(path=dataset, format=format),
            trainer=TrainerConfig(
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                **trainer_fields,
            ),
        )

    components = _base.setup_training(config, resume_from=resume_from)

    state = components.trainer.train(
        model=components.model,
        train_dataloader=components.train_dataloader,
        resume_state=components.resume_state,
        resume_checkpoint_dir=resume_from,
    )

    return state
