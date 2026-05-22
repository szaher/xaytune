from __future__ import annotations

from typing import Union

from peft import LoraConfig, get_peft_model, TaskType

from trainlib.models.loader import ModelResult

_AUTO_TARGET_MODULES: dict[str, list[str]] = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
}

_DEFAULT_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def get_target_modules(target_modules: Union[str, list[str]], model: object) -> list[str]:
    if isinstance(target_modules, list):
        return target_modules
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type and model_type in _AUTO_TARGET_MODULES:
        return _AUTO_TARGET_MODULES[model_type]
    return _DEFAULT_MODULES


def apply_lora(
    model_result: ModelResult,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Union[str, list[str]] = "auto",
) -> ModelResult:
    resolved_modules = get_target_modules(target_modules, model_result.model)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=resolved_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    peft_model = get_peft_model(model_result.model, lora_config)
    return ModelResult(
        model=peft_model,
        tokenizer=model_result.tokenizer,
        name=model_result.name,
        quantization=model_result.quantization,
        peft_applied=True,
        metadata={**model_result.metadata, "lora_rank": rank, "lora_alpha": alpha},
    )
