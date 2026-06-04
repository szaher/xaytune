from __future__ import annotations

import os
import warnings
from pathlib import Path

from xaytune.config.schema import TrainConfig


class ConfigValidationError(Exception):
    """Raised when a :class:`~xaytune.config.schema.TrainConfig` has invalid field combinations."""


_FINETUNE_METHODS = {"full", "lora", "qlora"}
_ALIGN_METHODS = {"dpo", "grpo", "ppo", "orpo", "simpo", "reinforce"}

_KNOWN_METHOD_PARAMS: dict[str, set[str]] = {
    "dpo": {"beta"},
    "grpo": {"kl_coeff"},
    "ppo": {"clip_eps"},
    "orpo": {"lambda_weight"},
    "simpo": {"beta", "gamma"},
    "reinforce": set(),
}


def validate_config(config: TrainConfig) -> None:
    """Validate cross-field constraints on a training configuration.

    Checks recipe/method compatibility, mutual exclusivity of warmup
    settings, quantization requirements, and method_params validity.

    Raises:
        ConfigValidationError: With a list of all detected issues.
    """
    errors: list[str] = []

    if config.method == "qlora" and config.model.quantization != "4bit":
        errors.append(
            "QLoRA requires 4bit quantization, but model.quantization="
            f"'{config.model.quantization}'. Suggestion: set model.quantization='4bit'."
        )

    if not 0.0 <= config.data.eval_split <= 1.0:
        errors.append(
            f"data.eval_split must be between 0.0 and 1.0, got {config.data.eval_split}. "
            "Suggestion: set eval_split to a value like 0.05 for a 5% eval split."
        )

    if config.trainer.batch_size < 1:
        errors.append(
            f"trainer.batch_size must be >= 1, got {config.trainer.batch_size}. "
            "Suggestion: set batch_size to at least 1."
        )

    if config.trainer.learning_rate <= 0:
        errors.append(
            f"trainer.learning_rate must be positive, got {config.trainer.learning_rate}. "
            "Suggestion: typical values are 1e-5 to 5e-4."
        )

    if config.trainer.warmup_steps > 0 and config.trainer.warmup_ratio > 0.0:
        errors.append(
            "trainer.warmup_steps and trainer.warmup_ratio are mutually exclusive — "
            "set one to 0. Suggestion: use warmup_steps for an exact count, "
            "or warmup_ratio for a fraction of total steps."
        )

    if config.recipe == "align" and config.method not in _ALIGN_METHODS:
        errors.append(
            f"Recipe 'align' requires an alignment method "
            f"({', '.join(sorted(_ALIGN_METHODS))}), got '{config.method}'. "
            "Suggestion: set method='dpo' or method='grpo'."
        )

    if config.recipe == "finetune" and config.method not in _FINETUNE_METHODS:
        errors.append(
            f"Recipe 'finetune' requires a fine-tuning method "
            f"({', '.join(sorted(_FINETUNE_METHODS))}), got '{config.method}'. "
            "Suggestion: set method='lora' or method='full'."
        )

    if config.method_params:
        known = _KNOWN_METHOD_PARAMS.get(config.method, set())
        if not known and config.method not in _ALIGN_METHODS:
            errors.append(
                f"method_params is only supported for alignment methods "
                f"({', '.join(sorted(_ALIGN_METHODS))}), "
                f"but recipe/method is '{config.recipe}/{config.method}'."
            )
        else:
            unknown = set(config.method_params) - known
            if unknown:
                errors.append(
                    f"Unknown method_params for '{config.method}': "
                    f"{', '.join(sorted(unknown))}. "
                    f"Valid params: {', '.join(sorted(known)) if known else 'none'}."
                )

    if errors:
        raise ConfigValidationError(
            f"Config validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def preflight_check(config: TrainConfig) -> list[str]:
    """Run environment-aware checks before training starts.

    Verifies GPU availability for quantization and mixed precision,
    checks that data paths exist, and validates output directory
    write permissions.

    Returns:
        List of warning/issue strings (empty if everything looks good).
    """
    issues: list[str] = []

    try:
        import torch

        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        has_cuda = False
        has_mps = False

    if config.model.quantization and not has_cuda:
        issues.append(
            f"Quantization ({config.model.quantization}) requires CUDA, "
            "but no CUDA GPU was detected."
        )

    if config.trainer.mixed_precision != "fp32" and not has_cuda and not has_mps:
        warnings.warn(
            f"mixed_precision='{config.trainer.mixed_precision}' selected "
            "but no GPU detected. Training will fall back to CPU (fp32).",
            stacklevel=2,
        )

    if config.data.source == "local":
        data_path = Path(config.data.path)
        if not data_path.exists():
            issues.append(f"Data path not found: {config.data.path}")

    output_parent = Path(config.output.dir).parent
    if output_parent.exists() and not os.access(str(output_parent), os.W_OK):
        issues.append(f"Output directory parent is not writable: {output_parent}")

    return issues
