from __future__ import annotations

from typing import Any

_RECIPE_FUNC: dict[str, str] = {
    "finetune": "finetune",
    "pretrain": "pretrain",
    "align": "align",
}

_RECIPE_DEFAULTS: dict[str, dict[str, Any]] = {
    "finetune": {
        "method": "full",
        "format": "alpaca",
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "batch_size": 4,
    },
    "pretrain": {
        "format": "text",
        "num_epochs": 1,
        "learning_rate": 3e-4,
        "batch_size": 4,
    },
    "align": {
        "method": "dpo",
        "format": "preference",
        "num_epochs": 1,
        "learning_rate": 5e-6,
        "batch_size": 4,
    },
}

_TRAINER_DEFAULTS: dict[str, Any] = {
    "gradient_accumulation": 1,
    "max_steps": -1,
    "seed": 42,
    "mixed_precision": "bf16",
    "scheduler": "cosine",
    "warmup_steps": 0,
    "warmup_ratio": 0.0,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "checkpoint_every_n_steps": 500,
    "save_last": True,
    "activation_checkpointing": False,
    "async_checkpoint": False,
}

_DATA_DEFAULTS: dict[str, Any] = {
    "max_seq_length": 2048,
    "packing": True,
    "streaming": False,
    "eval_split": 0.0,
}

_MODEL_DEFAULTS: dict[str, Any] = {
    "dtype": "auto",
    "trust_remote_code": False,
}

_EVAL_DEFAULTS: dict[str, Any] = {
    "eval_every_n_steps": 500,
    "early_stopping_patience": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_min_delta": 0.0,
}

_LOGGING_DEFAULTS: dict[str, Any] = {
    "log_every_n_steps": 10,
}

_OUTPUT_DEFAULTS: dict[str, Any] = {
    "output_dir": "output",
    "merge_on_complete": False,
}

_LORA_DEFAULTS: dict[str, Any] = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}


def _format_value(v: Any) -> str:
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, float):
        if v != 0 and (abs(v) < 0.01 or abs(v) >= 10000):
            return f"{v:.0e}" if v == int(v) else f"{v:g}"
        return repr(v)
    return repr(v)


def generate_code(
    recipe: str,
    method: str,
    model_name: str,
    data_path: str,
    data_format: str,
    quantization: str | None = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    max_seq_length: int = 2048,
    packing: bool = True,
    streaming: bool = False,
    eval_split: float = 0.0,
    eval_path: str = "",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    batch_size: int = 4,
    gradient_accumulation: int = 1,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    max_steps: int = -1,
    seed: int = 42,
    mixed_precision: str = "bf16",
    scheduler: str = "cosine",
    warmup_steps: int = 0,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    eval_every_n_steps: int = 500,
    early_stopping_patience: int = 0,
    early_stopping_metric: str = "eval_loss",
    early_stopping_min_delta: float = 0.0,
    log_every_n_steps: int = 10,
    output_dir: str = "output",
    merge_on_complete: bool = False,
    checkpoint_every_n_steps: int = 500,
    save_last: bool = True,
    activation_checkpointing: bool = False,
    async_checkpoint: bool = False,
    beta: float = 0.1,
    kl_coeff: float = 0.04,
    clip_eps: float = 0.2,
    lambda_weight: float = 1.0,
    gamma: float = 0.5,
    **_extra: Any,
) -> str:
    """Generate a Python code snippet from form values.

    Returns a clean ``xaytune.finetune()`` / ``align()`` / ``pretrain()``
    call with only non-default parameters included.
    """
    func_name = _RECIPE_FUNC.get(recipe, "finetune")
    defaults = _RECIPE_DEFAULTS.get(recipe, _RECIPE_DEFAULTS["finetune"])

    params: list[tuple[str, str]] = []

    params.append(("model", repr(model_name)))
    params.append(("dataset", repr(data_path)))

    if method != defaults.get("method", method):
        params.append(("method", repr(method)))

    if data_format != defaults.get("format", data_format):
        params.append(("format", repr(data_format)))

    if quantization and quantization != "None":
        params.append(("quantization", repr(quantization)))

    if dtype != _MODEL_DEFAULTS["dtype"]:
        params.append(("dtype", repr(dtype)))
    if trust_remote_code != _MODEL_DEFAULTS["trust_remote_code"]:
        params.append(("trust_remote_code", _format_value(trust_remote_code)))

    if num_epochs != defaults.get("num_epochs", num_epochs):
        params.append(("num_epochs", _format_value(num_epochs)))
    if learning_rate != defaults.get("learning_rate", learning_rate):
        params.append(("learning_rate", _format_value(learning_rate)))
    if batch_size != defaults.get("batch_size", batch_size):
        params.append(("batch_size", _format_value(batch_size)))

    _extra_params: dict[str, Any] = {
        "max_seq_length": max_seq_length,
        "packing": packing,
        "streaming": streaming,
        "eval_split": eval_split,
    }
    for k, v in _extra_params.items():
        if v != _DATA_DEFAULTS.get(k, v):
            params.append((k, _format_value(v)))

    if eval_path and eval_path.strip():
        params.append(("eval_path", repr(eval_path.strip())))

    if method in ("lora", "qlora"):
        lora_vals: dict[str, Any] = {
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        }
        for k, v in lora_vals.items():
            if v != _LORA_DEFAULTS.get(k, v):
                params.append((k, _format_value(v)))

    trainer_vals: dict[str, Any] = {
        "gradient_accumulation": gradient_accumulation,
        "max_steps": max_steps,
        "seed": seed,
        "mixed_precision": mixed_precision,
        "scheduler": scheduler,
        "warmup_steps": warmup_steps,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "checkpoint_every_n_steps": checkpoint_every_n_steps,
        "save_last": save_last,
        "activation_checkpointing": activation_checkpointing,
        "async_checkpoint": async_checkpoint,
    }
    for k, v in trainer_vals.items():
        if v != _TRAINER_DEFAULTS.get(k, v):
            params.append((k, _format_value(v)))

    eval_vals: dict[str, Any] = {
        "eval_every_n_steps": eval_every_n_steps,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_metric": early_stopping_metric,
        "early_stopping_min_delta": early_stopping_min_delta,
    }
    for k, v in eval_vals.items():
        if v != _EVAL_DEFAULTS.get(k, v):
            params.append((k, _format_value(v)))

    log_vals: dict[str, Any] = {
        "log_every_n_steps": log_every_n_steps,
    }
    for k, v in log_vals.items():
        if v != _LOGGING_DEFAULTS.get(k, v):
            params.append((k, _format_value(v)))

    out_vals: dict[str, Any] = {
        "output_dir": output_dir,
        "merge_on_complete": merge_on_complete,
    }
    for k, v in out_vals.items():
        default = _OUTPUT_DEFAULTS.get(k, v)
        if v != default:
            params.append((k, _format_value(v)))

    if recipe == "align":
        from xaytune.studio.app import METHOD_PARAMS_SPEC

        spec = METHOD_PARAMS_SPEC.get(method, [])
        mp_values = {
            "beta": beta,
            "kl_coeff": kl_coeff,
            "clip_eps": clip_eps,
            "lambda_weight": lambda_weight,
            "gamma": gamma,
        }
        for p in spec:
            name = p["name"]
            val = mp_values.get(name)
            if val is not None and val != p["default"]:
                params.append((name, _format_value(val)))

    param_lines = ",\n    ".join(f"{k}={v}" for k, v in params)
    return (
        f"import xaytune\n"
        f"\n"
        f"state = xaytune.{func_name}(\n"
        f"    {param_lines},\n"
        f")\n"
        f"\n"
        f'print(f"Training complete. Final loss: '
        f"{{state.metrics.get('loss', 'N/A')}}\")\n"
    )
