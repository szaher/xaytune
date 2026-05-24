from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, field_validator


class ModelConfig(BaseModel):
    """Model loading configuration.

    Attributes:
        name: HuggingFace model name or local path.
        quantization: Optional quantization level (``"4bit"`` or ``"8bit"``).
        dtype: Model dtype — ``"auto"``, ``"float16"``, ``"bfloat16"``, etc.
        trust_remote_code: Allow execution of custom model code from the Hub.
    """

    name: str
    quantization: Literal["4bit", "8bit"] | None = None
    dtype: str = "auto"
    trust_remote_code: bool = False

    @field_validator("quantization")
    @classmethod
    def validate_quantization(cls, v: str | None) -> str | None:
        if v is not None and v not in ("4bit", "8bit"):
            raise ValueError(f"quantization must be '4bit' or '8bit', got '{v}'")
        return v


class LoraConfig(BaseModel):
    """LoRA adapter configuration.

    Attributes:
        rank: Rank of the low-rank matrices.
        alpha: LoRA scaling factor (effective scale = ``alpha / rank``).
        dropout: Dropout probability applied to LoRA layers.
        target_modules: Modules to apply LoRA to — ``"auto"`` for
            framework defaults, or a list of module name patterns.
    """

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "auto"


class DataConfig(BaseModel):
    """Dataset configuration.

    Attributes:
        path: Path to a local JSONL file or HuggingFace dataset name.
        format: Data format — ``"alpaca"``, ``"sharegpt"``, ``"chat"``,
            ``"text"``, or ``"preference"``.
        source: ``"local"`` for files on disk, ``"huggingface"`` for Hub datasets.
        eval_split: Fraction of training data to hold out for evaluation.
        eval_path: Optional separate evaluation dataset path.
        packing: Pack short sequences together to reduce padding waste.
        max_seq_length: Maximum sequence length after tokenization.
        streaming: Stream data instead of loading into memory.
    """

    path: str
    format: str
    source: Literal["local", "huggingface"] = "local"
    eval_split: float = 0.0
    eval_path: str | None = None
    packing: bool = True
    max_seq_length: int = 2048
    streaming: bool = False


class TrainerConfig(BaseModel):
    """Training loop configuration.

    Attributes:
        strategy: Distributed strategy — ``"auto"``, ``"ddp"``, ``"fsdp"``,
            or ``"deepspeed"``.
        mixed_precision: AMP dtype — ``"fp16"``, ``"bf16"``, or ``"fp32"``.
        batch_size: Per-device batch size.
        gradient_accumulation: Accumulate gradients over N micro-batches.
        learning_rate: Peak learning rate.
        num_epochs: Number of training epochs.
        max_steps: Stop after this many optimizer steps (``-1`` = unlimited).
        warmup_steps: Linear warmup steps (mutually exclusive with ``warmup_ratio``).
        warmup_ratio: Warmup as a fraction of total steps.
        scheduler: LR schedule — ``"cosine"``, ``"linear"``, ``"constant"``,
            or ``"constant_with_warmup"``.
        weight_decay: AdamW weight decay coefficient.
        max_grad_norm: Gradient clipping norm (``0`` = disabled).
        seed: Random seed for reproducibility.
        checkpoint_every_n_steps: Save a checkpoint every N steps.
        save_last: Save a final checkpoint at training end.
        activation_checkpointing: Trade compute for memory by
            recomputing activations during backward.
        async_checkpoint: Write checkpoints in a background thread.
    """

    strategy: Literal["auto", "ddp", "fsdp", "deepspeed"] = "auto"
    mixed_precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    batch_size: int = 4
    gradient_accumulation: int = 1
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    scheduler: Literal["cosine", "linear", "constant", "constant_with_warmup"] = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    checkpoint_every_n_steps: int = 500
    save_last: bool = True
    activation_checkpointing: bool = False
    async_checkpoint: bool = False

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid = {"auto", "ddp", "fsdp", "deepspeed"}
        if v not in valid:
            raise ValueError(f"strategy must be one of {valid}, got '{v}'")
        return v


class EvalConfig(BaseModel):
    """Evaluation and early stopping configuration.

    Attributes:
        every_n_steps: Run evaluation every N training steps.
        metrics: Metrics to compute — ``"loss"``, ``"perplexity"``,
            ``"token_accuracy"``.
        benchmarks: Optional benchmark names for lm-eval-harness.
        early_stopping_patience: Stop if no improvement for this many
            evaluations (``0`` = disabled).
        early_stopping_metric: Metric to monitor for early stopping.
        early_stopping_min_delta: Minimum improvement to count as progress.
    """

    every_n_steps: int = 500
    metrics: list[str] = ["loss", "perplexity"]
    benchmarks: list[str] = []
    early_stopping_patience: int = 0
    early_stopping_metric: str = "eval_loss"
    early_stopping_min_delta: float = 0.0


class LoggingConfig(BaseModel):
    """Logging backend configuration.

    Attributes:
        backends: Active backends — ``"console"``, ``"tensorboard"``,
            ``"wandb"``.
        project: W&B / TensorBoard project name.
        run_name: Optional run name for experiment tracking.
        log_every_n_steps: Log metrics every N steps.
    """

    backends: list[str] = ["console"]
    project: str | None = None
    run_name: str | None = None
    log_every_n_steps: int = 10


class OutputConfig(BaseModel):
    """Output and artifact configuration.

    Attributes:
        dir: Directory for checkpoints, logs, and exported models.
        merge_on_complete: Auto-merge LoRA adapters at training end.
    """

    dir: str = "output"
    merge_on_complete: bool = False


class FSDPConfig(BaseModel):
    """Fully Sharded Data Parallel (FSDP) configuration.

    Attributes:
        sharding_strategy: How to shard parameters across ranks.
        cpu_offload: Offload parameters and gradients to CPU.
        backward_prefetch: Prefetch strategy for backward pass.
        mixed_precision: Use mixed precision within FSDP.
    """

    sharding_strategy: Literal["full_shard", "shard_grad_op", "no_shard"] = "full_shard"
    cpu_offload: bool = False
    backward_prefetch: Literal["backward_pre", "backward_post"] | None = None
    mixed_precision: bool = True


class DeepSpeedConfig(BaseModel):
    """DeepSpeed integration configuration.

    Attributes:
        config_file: Path to a DeepSpeed JSON config file.
        zero_stage: ZeRO optimization stage (0, 1, 2, or 3).
    """

    config_file: str | None = None
    zero_stage: int = 2


class TrainConfig(BaseModel):
    """Top-level training configuration combining all sub-configs.

    This is the single object that drives ``setup_training()`` and the
    recipe one-liners (``finetune``, ``pretrain``, ``align``).

    Attributes:
        recipe: Training recipe — ``"finetune"``, ``"pretrain"``, or ``"align"``.
        method: Training method (e.g. ``"full"``, ``"lora"``, ``"dpo"``).
        base: Optional path to a base YAML config for inheritance.
        model: Model loading settings.
        data: Dataset settings.
        lora: LoRA adapter settings (used when method is ``"lora"``/``"qlora"``).
        trainer: Training loop settings.
        eval: Evaluation and early stopping settings.
        logging: Logging backend settings.
        output: Output directory and artifact settings.
        method_params: Extra hyperparameters passed to the alignment loss
            function (e.g. ``{"beta": 0.1}`` for DPO).
        fsdp: FSDP settings.
        deepspeed_config: DeepSpeed settings.
    """

    recipe: Literal["finetune", "pretrain", "align"]
    method: str = "full"
    base: str | None = None
    model: ModelConfig
    data: DataConfig
    lora: LoraConfig = LoraConfig()
    trainer: TrainerConfig = TrainerConfig()
    eval: EvalConfig = EvalConfig()
    logging: LoggingConfig = LoggingConfig()
    output: OutputConfig = OutputConfig()
    method_params: dict[str, Any] = {}
    fsdp: FSDPConfig = FSDPConfig()
    deepspeed_config: DeepSpeedConfig = DeepSpeedConfig()

    @field_validator("recipe")
    @classmethod
    def validate_recipe(cls, v: str) -> str:
        valid = {"finetune", "pretrain", "align"}
        if v not in valid:
            raise ValueError(f"recipe must be one of {valid}, got '{v}'")
        return v

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        valid = {"full", "lora", "qlora", "dpo", "grpo", "ppo", "orpo", "simpo", "reinforce"}
        if v not in valid:
            raise ValueError(f"method must be one of {valid}, got '{v}'")
        return v
