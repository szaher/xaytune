from __future__ import annotations

from typing import Any, NamedTuple

from torch.utils.data import DataLoader

from trainlib.config.schema import TrainConfig
from trainlib.data import load_dataset
from trainlib.models import apply_lora, load_model
from trainlib.trainer import CallbackManager, Trainer
from trainlib.trainer.distributed import (
    cleanup_distributed,
    get_strategy,
    init_distributed,
    wrap_model_distributed,
)


class TrainingComponents(NamedTuple):
    model: Any
    tokenizer: Any
    train_dataloader: DataLoader
    eval_dataloader: DataLoader | None
    trainer: Trainer
    distributed_ctx: Any = None


def setup_training(
    config: TrainConfig,
    callback_manager: CallbackManager | None = None,
) -> TrainingComponents:
    # Initialize distributed context
    ctx = init_distributed()
    strategy = get_strategy(config.trainer.strategy, ctx.world_size)

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

    # Move model to correct device
    model = model_result.model
    model.to(ctx.device)

    # Wrap model for distributed training
    if strategy != "none":
        model = wrap_model_distributed(
            model,
            strategy=strategy,
            ctx=ctx,
            fsdp_config=config.fsdp,
            deepspeed_config=config.deepspeed_config,
            mixed_precision=config.trainer.mixed_precision,
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

    # Create DataLoaders with DistributedSampler when needed
    sampler = None
    shuffle = True
    if ctx.is_distributed:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            train_data,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=True,
        )
        shuffle = False

    train_dataloader: Any = DataLoader(
        train_data,  # type: ignore[arg-type]
        batch_size=config.trainer.batch_size,
        shuffle=shuffle,
        sampler=sampler,
    )

    eval_sampler = None
    if ctx.is_distributed and eval_data is not None:
        from torch.utils.data.distributed import DistributedSampler

        eval_sampler = DistributedSampler(
            eval_data,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=False,
        )

    eval_dataloader: Any = None
    if eval_data is not None:
        eval_dataloader = DataLoader(
            eval_data,  # type: ignore[arg-type]
            batch_size=config.trainer.batch_size,
            shuffle=False,
            sampler=eval_sampler,
        )

    cb_manager = callback_manager or CallbackManager()

    # Register distributed cleanup callback
    if ctx.is_distributed:

        @cb_manager.on("train_end")
        def _cleanup_distributed(state: Any) -> None:
            cleanup_distributed(ctx)

    trainer = Trainer(
        config=config.trainer,
        callback_manager=cb_manager,
    )

    return TrainingComponents(
        model=model,
        tokenizer=model_result.tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        trainer=trainer,
        distributed_ctx=ctx,
    )
