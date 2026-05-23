from __future__ import annotations

from typing import Any, NamedTuple

from torch.utils.data import DataLoader

from trainlib.config.schema import TrainConfig
from trainlib.data import load_dataset
from trainlib.data.packing import pack_sequences
from trainlib.data.validation import validate_dataset_sample
from trainlib.models import apply_lora, load_model
from trainlib.trainer import CallbackManager, Trainer
from trainlib.trainer.checkpoint_callback import register_checkpoint_callbacks
from trainlib.trainer.checkpointing import load_checkpoint
from trainlib.trainer.device import seed_all
from trainlib.trainer.distributed import (
    cleanup_distributed,
    get_strategy,
    init_distributed,
    wrap_model_distributed,
)
from trainlib.trainer.early_stopping import register_early_stopping_callbacks
from trainlib.trainer.eval_callback import register_eval_callbacks
from trainlib.trainer.progress import register_progress_callbacks


class TrainingComponents(NamedTuple):
    model: Any
    tokenizer: Any
    train_dataloader: DataLoader
    eval_dataloader: DataLoader | None
    trainer: Trainer
    distributed_ctx: Any = None
    resume_state: Any = None


def setup_training(
    config: TrainConfig,
    callback_manager: CallbackManager | None = None,
    resume_from: str | None = None,
) -> TrainingComponents:
    # Set random seeds for reproducibility
    seed_all(config.trainer.seed)

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

    if config.trainer.activation_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

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
        source=config.data.source,
        streaming=config.data.streaming,
        eval_split=config.data.eval_split,
    )

    if config.data.eval_split > 0:
        train_data, eval_data = dataset  # type: ignore[misc]
    else:
        train_data = dataset  # type: ignore[assignment]
        eval_data = None

    if (
        config.data.packing
        and config.data.max_seq_length > 0
        and isinstance(train_data, list)
        and train_data
        and isinstance(train_data[0], dict)
        and "input_ids" in train_data[0]
        and isinstance(train_data[0]["input_ids"], list)
    ):
        pad_id = getattr(model_result.tokenizer, "pad_token_id", 0) or 0
        train_data = pack_sequences(
            train_data,
            max_seq_length=config.data.max_seq_length,
            pad_token_id=pad_id,
        )
        if eval_data is not None:
            eval_data = pack_sequences(
                eval_data,
                max_seq_length=config.data.max_seq_length,
                pad_token_id=pad_id,
            )

    # Create DataLoaders with DistributedSampler when needed
    sampler: Any = None
    shuffle = True
    if ctx.is_distributed:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            train_data,  # type: ignore[arg-type]
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

    eval_sampler: Any = None
    if ctx.is_distributed and eval_data is not None:
        from torch.utils.data.distributed import DistributedSampler

        eval_sampler = DistributedSampler(
            eval_data,  # type: ignore[arg-type]
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

    # Validate a sample batch before training
    validate_dataset_sample(
        train_dataloader,
        max_seq_length=config.data.max_seq_length,
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

    # Set up logging
    from trainlib.logging import setup_logging

    logging_manager = setup_logging(
        config.logging,
        cb_manager,
        output_dir=config.output.dir,
        rank=ctx.rank,
    )

    @cb_manager.on("train_start")
    def _log_config(state: Any) -> None:
        logging_manager.log_config(config.model_dump())

    # Set up async checkpoint saver if requested
    async_saver = None
    if config.trainer.async_checkpoint:
        from trainlib.trainer.async_checkpoint import AsyncCheckpointSaver

        async_saver = AsyncCheckpointSaver()

        @cb_manager.on("train_end")
        def _wait_async_saver(state: Any) -> None:
            async_saver.wait()

    # Register checkpoint callbacks
    register_checkpoint_callbacks(
        callback_manager=cb_manager,
        trainer=trainer,
        model=model,
        output_dir=config.output.dir,
        checkpoint_every_n_steps=config.trainer.checkpoint_every_n_steps,
        save_last=config.trainer.save_last,
        is_main_process=ctx.is_main_process,
        async_saver=async_saver,
    )

    # Register eval callbacks if eval data is available
    if eval_dataloader is not None and config.eval.every_n_steps > 0:
        register_eval_callbacks(
            callback_manager=cb_manager,
            model=model,
            eval_dataloader=eval_dataloader,
            every_n_steps=config.eval.every_n_steps,
            metrics=config.eval.metrics,
            is_main_process=ctx.is_main_process,
        )

    # Register early stopping if configured
    if config.eval.early_stopping_patience > 0 and eval_dataloader is not None:
        register_early_stopping_callbacks(
            callback_manager=cb_manager,
            patience=config.eval.early_stopping_patience,
            metric=config.eval.early_stopping_metric,
            min_delta=config.eval.early_stopping_min_delta,
        )

    # Auto-merge LoRA adapters on training completion
    if config.output.merge_on_complete and config.method in ("lora", "qlora"):

        @cb_manager.on("train_end")
        def _merge_on_complete(state: Any) -> None:
            if not ctx.is_main_process:
                return
            if hasattr(model, "merge_and_unload"):
                merged = model.merge_and_unload()
                save_dir = f"{config.output.dir}/merged"
                from pathlib import Path

                Path(save_dir).mkdir(parents=True, exist_ok=True)
                merged.save_pretrained(save_dir)
                model_result.tokenizer.save_pretrained(save_dir)

    # Register progress bar
    total_steps = len(train_dataloader)
    if config.trainer.gradient_accumulation > 1:
        total_steps = total_steps // config.trainer.gradient_accumulation
    total_steps *= config.trainer.num_epochs
    if config.trainer.max_steps > 0:
        total_steps = min(total_steps, config.trainer.max_steps)

    register_progress_callbacks(
        callback_manager=cb_manager,
        total_steps=total_steps,
        is_main_process=ctx.is_main_process,
    )

    # Resume from checkpoint if requested
    resume_state = None
    if resume_from is not None:

        class _NoopOptimizer:
            def load_state_dict(self, state: Any) -> None:
                pass

        resume_state = load_checkpoint(
            checkpoint_dir=resume_from,
            model=model,
            optimizer=_NoopOptimizer(),
        )

    return TrainingComponents(
        model=model,
        tokenizer=model_result.tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        trainer=trainer,
        distributed_ctx=ctx,
        resume_state=resume_state,
    )
