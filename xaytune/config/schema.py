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
            ``"full_shard"`` shards params, grads, and optimizer states.
            ``"shard_grad_op"`` only shards grads and optimizer states.
            ``"no_shard"`` disables sharding (equivalent to DDP).
        cpu_offload: Offload parameters and gradients to CPU RAM.
            Reduces GPU memory at the cost of slower training.
        backward_prefetch: Prefetch next layer's params during backward.
            ``"backward_pre"`` is faster, ``"backward_post"`` uses less memory.
        mixed_precision: Use FSDP-native mixed precision (dtype from
            ``TrainerConfig.mixed_precision``).
        auto_wrap_min_params: Minimum parameter count for automatic FSDP
            wrapping. Layers with fewer parameters than this are grouped
            together. Set to 0 to disable auto-wrapping.
        forward_prefetch: Prefetch next layer's params during forward pass.
        sync_module_states: Broadcast module states from rank 0 on init.
            Useful when only rank 0 loads the checkpoint.
        limit_all_gathers: Rate-limit all-gathers to reduce memory spikes.
        activation_checkpointing: Apply activation checkpointing to
            auto-wrapped layers (trades compute for memory).
    """

    sharding_strategy: Literal["full_shard", "shard_grad_op", "no_shard"] = "full_shard"
    cpu_offload: bool = False
    backward_prefetch: Literal["backward_pre", "backward_post"] | None = None
    mixed_precision: bool = True
    auto_wrap_min_params: int = 100_000
    forward_prefetch: bool = False
    sync_module_states: bool = True
    limit_all_gathers: bool = True
    activation_checkpointing: bool = False


class DeepSpeedConfig(BaseModel):
    """DeepSpeed integration configuration.

    Attributes:
        config_file: Path to a DeepSpeed JSON config file. When provided,
            all other fields are ignored and the JSON file is used directly.
        zero_stage: ZeRO optimization stage.
            ``0`` = disabled, ``1`` = optimizer state partitioning,
            ``2`` = gradient + optimizer partitioning,
            ``3`` = full parameter partitioning.
        offload_optimizer: Offload optimizer states to CPU (ZeRO stage 2/3).
        offload_param: Offload parameters to CPU (ZeRO stage 3 only).
        overlap_comm: Overlap gradient communication with backward pass.
        contiguous_gradients: Use contiguous memory for gradients.
        reduce_bucket_size: Size of gradient reduction buckets in bytes.
        stage3_prefetch_bucket_size: Prefetch buffer size for ZeRO-3.
        stage3_param_persistence_threshold: Params smaller than this stay
            on GPU even in ZeRO-3 (reduces communication overhead).
    """

    config_file: str | None = None
    zero_stage: Literal[0, 1, 2, 3] = 2
    offload_optimizer: bool = False
    offload_param: bool = False
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    reduce_bucket_size: int = 500_000_000
    stage3_prefetch_bucket_size: int = 50_000_000
    stage3_param_persistence_threshold: int = 100_000


class GenerationConfig(BaseModel):
    """Generation parameters for online RL alignment.

    Attributes:
        max_new_tokens: Maximum tokens to generate per completion.
        temperature: Sampling temperature (higher = more random).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling (0 = disabled).
        do_sample: Use sampling vs greedy decoding.
        group_size: Completions per prompt (>1 for GRPO group sampling).
    """

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True
    group_size: int = 4


class OnlineRLConfig(BaseModel):
    """Online RL configuration for generating completions during training.

    Attributes:
        enabled: Enable online generation (vs pre-computed advantages).
        generation: Generation parameters.
        reward_name: Registered reward function name.
        reward_kwargs: Extra keyword arguments for the reward function.
        eval_prompts: Prompts to evaluate periodically during training.
        eval_every_n_steps: Run online eval every N training steps.
    """

    enabled: bool = False
    generation: GenerationConfig = GenerationConfig()
    reward_name: str = "default"
    reward_kwargs: dict[str, Any] = {}
    eval_prompts: list[str] = []
    eval_every_n_steps: int = 100


class DataPrepStepConfig(BaseModel):
    """A single data preparation step.

    Exactly one of the fields should be set per step.
    """

    filter: dict[str, Any] | None = None
    deduplicate: dict[str, Any] | None = None
    convert: dict[str, Any] | None = None


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

    recipe: str
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
    online_rl: OnlineRLConfig = OnlineRLConfig()
    data_prep: list[dict[str, Any]] = []
    fsdp: FSDPConfig = FSDPConfig()
    deepspeed_config: DeepSpeedConfig = DeepSpeedConfig()

    @field_validator("recipe")
    @classmethod
    def validate_recipe(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("recipe must be a non-empty string")
        return v

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("method must be a non-empty string")
        return v
