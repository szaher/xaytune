from __future__ import annotations

from pathlib import Path
from typing import Any


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _find_examples_dir() -> Path | None:
    """Locate the configs/examples directory relative to the package."""
    pkg_dir = Path(__file__).resolve().parent.parent.parent
    candidates = [
        pkg_dir / "configs" / "examples",
        Path.cwd() / "configs" / "examples",
    ]
    for d in candidates:
        if d.is_dir():
            return d
    return None


def _load_examples() -> dict[str, dict[str, Any]]:
    """Load all example YAML configs into a dict keyed by display name."""
    examples_dir = _find_examples_dir()
    if not examples_dir:
        return {}

    try:
        import yaml
    except ImportError:
        return {}

    result: dict[str, dict[str, Any]] = {}
    for path in sorted(examples_dir.glob("*.yaml")):
        try:
            with path.open() as f:
                raw = yaml.safe_load(f)
            if not isinstance(raw, dict):
                continue

            comment = ""
            with path.open() as f:
                first_line = f.readline().strip()
                if first_line.startswith("#"):
                    comment = first_line.lstrip("# ").strip()

            name = path.stem.replace("_", " ").title()
            if comment:
                name = comment

            result[name] = raw
        except Exception:
            continue

    return result


EXAMPLES = _load_examples()

_RECIPE_FROM_BASE: dict[str, tuple[str, str]] = {
    "full_finetune": ("finetune", "full"),
    "lora": ("finetune", "lora"),
    "qlora": ("finetune", "qlora"),
    "pretrain": ("pretrain", "full"),
}


def load_example_values(name: str) -> dict[str, Any]:
    """Convert a parsed example config into form field values.

    Returns a flat dict with keys matching the form field names used in
    ``create_app``.  Unknown or missing fields use sensible defaults so
    the form is fully populated.
    """
    cfg = EXAMPLES.get(name, {})
    if not cfg:
        return {}

    recipe = cfg.get("recipe", "")
    method = cfg.get("method", "")
    if not recipe and "base" in cfg:
        recipe, method = _RECIPE_FROM_BASE.get(cfg["base"], ("finetune", "full"))

    model = cfg.get("model", {})
    data = cfg.get("data", {})
    trainer = cfg.get("trainer", {})
    lora = cfg.get("lora", {})
    mp = cfg.get("method_params", {})
    output = cfg.get("output", {})
    eval_cfg = cfg.get("eval", {})
    logging_cfg = cfg.get("logging", {})

    quant = model.get("quantization")
    quant_str = quant if quant else "None"

    return {
        "recipe": recipe,
        "method": method,
        "model_name": model.get("name", ""),
        "data_path": data.get("path", ""),
        "data_format": data.get("format", "alpaca"),
        "quantization": quant_str,
        "dtype": model.get("dtype", "auto"),
        "trust_remote_code": model.get("trust_remote_code", False),
        "source": data.get("source", "local"),
        "max_seq_length": _as_int(data.get("max_seq_length"), 2048),
        "packing": data.get("packing", True),
        "streaming": data.get("streaming", False),
        "eval_split": _as_float(data.get("eval_split"), 0.0),
        "eval_path": data.get("eval_path", ""),
        "lora_rank": _as_int(lora.get("rank"), 16),
        "lora_alpha": _as_int(lora.get("alpha"), 32),
        "lora_dropout": _as_float(lora.get("dropout"), 0.05),
        "batch_size": _as_int(trainer.get("batch_size"), 4),
        "gradient_accumulation": _as_int(trainer.get("gradient_accumulation"), 1),
        "learning_rate": _as_float(trainer.get("learning_rate"), 2e-4),
        "num_epochs": _as_int(trainer.get("num_epochs"), 3),
        "max_steps": _as_int(trainer.get("max_steps"), -1),
        "seed": _as_int(trainer.get("seed"), 42),
        "mixed_precision": trainer.get("mixed_precision", "bf16"),
        "scheduler": trainer.get("scheduler", "cosine"),
        "warmup_steps": _as_int(trainer.get("warmup_steps"), 0),
        "warmup_ratio": _as_float(trainer.get("warmup_ratio"), 0.0),
        "weight_decay": _as_float(trainer.get("weight_decay"), 0.01),
        "max_grad_norm": _as_float(trainer.get("max_grad_norm"), 1.0),
        "eval_every_n_steps": _as_int(eval_cfg.get("every_n_steps"), 500),
        "early_stopping_patience": _as_int(eval_cfg.get("early_stopping_patience"), 0),
        "early_stopping_metric": eval_cfg.get("early_stopping_metric", "eval_loss"),
        "early_stopping_min_delta": _as_float(eval_cfg.get("early_stopping_min_delta"), 0.0),
        "log_every_n_steps": _as_int(logging_cfg.get("log_every_n_steps"), 10),
        "output_dir": output.get("dir", "output"),
        "merge_on_complete": output.get("merge_on_complete", False),
        "checkpoint_every_n_steps": _as_int(trainer.get("checkpoint_every_n_steps"), 500),
        "save_last": trainer.get("save_last", True),
        "activation_checkpointing": trainer.get("activation_checkpointing", False),
        "async_checkpoint": trainer.get("async_checkpoint", False),
        "beta": _as_float(mp.get("beta"), 0.1),
        "kl_coeff": _as_float(mp.get("kl_coeff"), 0.04),
        "clip_eps": _as_float(mp.get("clip_eps"), 0.2),
        "lambda_weight": _as_float(mp.get("lambda_weight"), 1.0),
        "gamma": _as_float(mp.get("gamma"), 0.5),
    }
