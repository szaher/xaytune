from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, field_validator


class ModelConfig(BaseModel):
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
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Union[str, list[str]] = "auto"


class DataConfig(BaseModel):
    path: str
    format: str
    source: Literal["local", "huggingface"] = "local"
    eval_split: float = 0.0
    eval_path: str | None = None
    packing: bool = True
    max_seq_length: int = 2048
    streaming: bool = False


class TrainerConfig(BaseModel):
    strategy: Literal["auto", "ddp", "fsdp", "deepspeed"] = "auto"
    mixed_precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    batch_size: int = 4
    gradient_accumulation: int = 1
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    checkpoint_every_n_steps: int = 500
    save_last: bool = True

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid = {"auto", "ddp", "fsdp", "deepspeed"}
        if v not in valid:
            raise ValueError(f"strategy must be one of {valid}, got '{v}'")
        return v


class EvalConfig(BaseModel):
    every_n_steps: int = 500
    metrics: list[str] = ["loss", "perplexity"]
    benchmarks: list[str] = []


class LoggingConfig(BaseModel):
    backends: list[str] = ["console"]
    project: str | None = None
    run_name: str | None = None
    log_every_n_steps: int = 10


class OutputConfig(BaseModel):
    dir: str = "output"
    merge_on_complete: bool = False


class TrainConfig(BaseModel):
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
        valid = {"full", "lora", "qlora", "dpo", "grpo", "ppo", "orpo", "simpo"}
        if v not in valid:
            raise ValueError(f"method must be one of {valid}, got '{v}'")
        return v
