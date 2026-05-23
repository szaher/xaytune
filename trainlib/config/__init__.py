from pathlib import Path

from trainlib.config.parser import apply_overrides, load_config, merge_dicts
from trainlib.config.schema import (
    DataConfig,
    DeepSpeedConfig,
    EvalConfig,
    FSDPConfig,
    LoggingConfig,
    LoraConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainerConfig,
)
from trainlib.config.validation import ConfigValidationError, preflight_check, validate_config


def get_defaults_dir() -> Path:
    return Path(__file__).parent / "defaults"


__all__ = [
    "apply_overrides",
    "ConfigValidationError",
    "DataConfig",
    "DeepSpeedConfig",
    "EvalConfig",
    "FSDPConfig",
    "get_defaults_dir",
    "load_config",
    "LoggingConfig",
    "LoraConfig",
    "merge_dicts",
    "ModelConfig",
    "OutputConfig",
    "preflight_check",
    "TrainConfig",
    "TrainerConfig",
    "validate_config",
]
