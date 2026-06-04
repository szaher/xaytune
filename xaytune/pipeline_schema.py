"""Pydantic models for multi-stage training pipelines."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from xaytune.config.schema import DataConfig, LoraConfig, TrainerConfig


class EvalStageConfig(BaseModel):
    """Configuration for an evaluation stage."""

    metrics: list[str] = ["loss", "perplexity"]
    benchmarks: list[str] = []
    dataset: str | None = None
    num_fewshot: int | None = None


class StageConfig(BaseModel):
    """A single stage in a training pipeline.

    Exactly one of ``recipe``, ``export``, or ``eval`` must be set.
    """

    name: str
    recipe: str | None = None
    export: str | None = None
    eval: EvalStageConfig | None = None
    model_name: str | None = None
    method: str | None = None
    data: DataConfig | None = None
    trainer: TrainerConfig | None = None
    lora: LoraConfig | None = None
    method_params: dict[str, Any] = {}
    repo: str | None = None
    save_to: str | None = None
    quantization: str | None = None


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    name: str = "pipeline"
    output_dir: str = "output"
    stages: list[StageConfig]


class StageResult(BaseModel):
    """Result of a single pipeline stage."""

    type: str
    output: str | None = None
    metrics: dict[str, Any] = {}
    status: str = "completed"
    error: str | None = None


class PipelineResult(BaseModel):
    """Result of a complete pipeline run."""

    name: str
    stages: dict[str, StageResult] = {}
    completed: bool = True
