from __future__ import annotations

from typing import Any, NamedTuple

from torch.utils.data import DataLoader

from trainlib.config.schema import TrainConfig
from trainlib.data import load_dataset
from trainlib.models import apply_lora, load_model
from trainlib.trainer import CallbackManager, Trainer


class TrainingComponents(NamedTuple):
    model: Any
    tokenizer: Any
    train_dataloader: DataLoader
    eval_dataloader: DataLoader | None
    trainer: Trainer


def setup_training(
    config: TrainConfig,
    callback_manager: CallbackManager | None = None,
) -> TrainingComponents:
    quantization = None
    if config.method == "qlora":
        quantization = "4bit"
    elif config.model.quantization:
        quantization = config.model.quantization

    model_result = load_model(
        config.model.name,
        quantization=quantization,
        dtype=config.model.dtype,
        trust_remote_code=config.model.trust_remote_code,
    )

    if config.method in ("lora", "qlora"):
        model_result = apply_lora(
            model_result,
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
        )

    dataset = load_dataset(
        config.data.path,
        format=config.data.format,
        eval_split=config.data.eval_split,
    )

    if config.data.eval_split > 0:
        train_data, eval_data = dataset  # type: ignore[misc]
    else:
        train_data = dataset  # type: ignore[assignment]
        eval_data = None

    train_dataloader: Any = DataLoader(
        train_data,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )

    eval_dataloader: Any = None
    if eval_data is not None:
        eval_dataloader = DataLoader(
            eval_data,
            batch_size=config.trainer.batch_size,
            shuffle=False,
        )

    trainer = Trainer(
        config=config.trainer,
        callback_manager=callback_manager or CallbackManager(),
    )

    return TrainingComponents(
        model=model_result.model,
        tokenizer=model_result.tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        trainer=trainer,
    )
