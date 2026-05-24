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


def finetune(
    *,
    config: TrainConfig | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    dataset: str | None = None,
    method: str = "full",
    format: str = "alpaca",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    resume_from: str | None = None,
    **kwargs: Any,
) -> TrainState:
    """Fine-tune a pretrained language model on a supervised dataset.

    Accepts either a fully specified ``TrainConfig`` or individual arguments
    for quick one-liner usage.  Extra ``**kwargs`` that match
    ``TrainerConfig`` fields (e.g. ``max_steps``, ``mixed_precision``) are
    forwarded automatically.

    Args:
        config: Complete training configuration. When provided, all other
            arguments except ``resume_from`` are ignored.
        model: HuggingFace model name, local path, or a pre-built
            ``nn.Module`` / ``ModelResult``.  When passing a raw module,
            ``tokenizer`` must also be provided.
        tokenizer: Tokenizer instance — required when ``model`` is a raw
            ``nn.Module``, ignored when ``model`` is a string or ``None``.
        dataset: Path to a JSONL training file or HuggingFace dataset name.
        method: Fine-tuning method — ``"full"``, ``"lora"``, or ``"qlora"``.
        format: Data format — ``"alpaca"``, ``"sharegpt"``, ``"chat"``, or ``"text"``.
        num_epochs: Number of training epochs.
        learning_rate: Peak learning rate.
        batch_size: Per-device batch size.
        resume_from: Path to a checkpoint directory to resume from.
        **kwargs: Additional ``TrainerConfig`` fields (``max_steps``,
            ``mixed_precision``, ``scheduler``, ``warmup_steps``, etc.).

    Returns:
        Final training state with loss, global step count, and other metrics.

    Raises:
        ValueError: If neither ``config`` nor both ``model`` and ``dataset``
            are provided.

    Example::

        state = trainlib.finetune(
            model="meta-llama/Llama-3-8B",
            dataset="data/train.jsonl",
            method="lora",
            num_epochs=3,
            max_steps=100,
        )
        print(f"Final loss: {state.metrics['loss']:.4f}")
    """
    injected_model = None
    if config is None:
        if dataset is None:
            raise ValueError("Either 'config' or both 'model' and 'dataset' are required.")

        model_name = model if isinstance(model, str) else "custom"
        if not isinstance(model, str) and model is not None:
            injected_model = model
        elif model is None:
            raise ValueError("Either 'config' or both 'model' and 'dataset' are required.")

        trainer_fields = {}
        trainer_param_names = {f for f in TrainerConfig.model_fields}
        for k, v in list(kwargs.items()):
            if k in trainer_param_names:
                trainer_fields[k] = kwargs.pop(k)

        config = TrainConfig(
            recipe="finetune",
            method=method,
            model=ModelConfig(name=model_name),
            data=DataConfig(path=dataset, format=format),
            trainer=TrainerConfig(
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                **trainer_fields,
            ),
        )

    components = _base.setup_training(
        config, resume_from=resume_from,
        model=injected_model, tokenizer=tokenizer,
    )

    state = components.trainer.train(
        model=components.model,
        train_dataloader=components.train_dataloader,
        resume_state=components.resume_state,
        resume_checkpoint_dir=resume_from,
    )

    return state
