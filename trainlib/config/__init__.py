from pathlib import Path

from trainlib.config.parser import load_config, merge_dicts, apply_overrides
from trainlib.config.schema import (
    DataConfig,
    EvalConfig,
    LoggingConfig,
    LoraConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainerConfig,
)
from trainlib.config.validation import ConfigValidationError, validate_config


def get_defaults_dir() -> Path:
    return Path(__file__).parent / "defaults"


__all__ = [
    "apply_overrides",
    "ConfigValidationError",
    "DataConfig",
    "EvalConfig",
    "get_defaults_dir",
    "load_config",
    "LoggingConfig",
    "LoraConfig",
    "merge_dicts",
    "ModelConfig",
    "OutputConfig",
    "TrainConfig",
    "TrainerConfig",
    "validate_config",
]
