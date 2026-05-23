from __future__ import annotations

from trainlib.config.schema import TrainConfig


class ConfigValidationError(Exception):
    pass


_FINETUNE_METHODS = {"full", "lora", "qlora"}
_ALIGN_METHODS = {"dpo", "grpo", "ppo", "orpo", "simpo"}


def validate_config(config: TrainConfig) -> None:
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

    if errors:
        raise ConfigValidationError(
            f"Config validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
