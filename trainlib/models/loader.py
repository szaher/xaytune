from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class ModelResult:
    model: Any
    tokenizer: Any
    name: str
    quantization: str | None = None
    peft_applied: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


_DTYPE_MAP = {
    "auto": "auto",
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _get_quantization_config(quantization: str) -> BitsAndBytesConfig:
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization: {quantization}. Use '4bit' or '8bit'.")


def load_model(
    name_or_path: str,
    *,
    quantization: str | None = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    device_map: str = "auto",
) -> ModelResult:
    from trainlib.models.registry import model_registry

    if model_registry.has(name_or_path):
        loader_fn = model_registry.get(name_or_path)
        result: ModelResult = loader_fn(
            name_or_path,
            quantization=quantization,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )
        return result

    torch_dtype = _DTYPE_MAP.get(dtype, "auto")

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }

    if quantization:
        model_kwargs["quantization_config"] = _get_quantization_config(quantization)

    model = AutoModelForCausalLM.from_pretrained(name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return ModelResult(
        model=model,
        tokenizer=tokenizer,
        name=name_or_path,
        quantization=quantization,
    )
