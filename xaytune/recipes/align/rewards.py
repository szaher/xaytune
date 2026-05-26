from __future__ import annotations

import re

from xaytune.utils.registry import Registry

reward_registry = Registry("reward")

register_reward = reward_registry.register


# ---------------------------------------------------------------------------
# Reward Model (neural)
# ---------------------------------------------------------------------------


class RewardModelWrapper:
    """Wraps a HuggingFace reward model for use with the reward registry."""

    def __init__(self, model_name: str, device: str = "auto", dtype: str = "auto") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = torch.bfloat16 if dtype == "auto" else getattr(torch, dtype)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch_dtype)
            .to(device)
            .eval()
        )
        self._device = device

    def score(self, prompt: str, response: str) -> float:
        import torch

        text = prompt + "\n" + response
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        return float(logits.squeeze(-1).item())


_reward_model_cache: dict[str, RewardModelWrapper] = {}


# ---------------------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------------------


class LLMJudgeWrapper:
    """Uses an LLM to judge response quality on a 1-5 scale."""

    DEFAULT_TEMPLATE = (
        "Rate the following response to the prompt on a scale of 1-5.\n\n"
        "Prompt: {prompt}\n\nResponse: {response}\n\n"
        "Criteria: {criteria}\n\n"
        "Output only a single integer (1-5):"
    )

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        template: str | None = None,
    ) -> None:
        from transformers import pipeline

        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map=device if device != "auto" else "auto",
            max_new_tokens=8,
        )
        self._template = template or self.DEFAULT_TEMPLATE

    def judge(
        self,
        prompt: str,
        response: str,
        criteria: str = "helpfulness, correctness, coherence",
    ) -> float:
        text = self._template.format(prompt=prompt, response=response, criteria=criteria)
        out = self._pipe(text, return_full_text=False)
        generated = out[0]["generated_text"].strip()
        match = re.search(r"[1-5]", generated)
        if match:
            return (int(match.group()) - 1) / 4.0
        return 0.0


_llm_judge_cache: dict[str, LLMJudgeWrapper] = {}


@register_reward("default")
def default_reward(prompt: str, response: str) -> float:
    """Baseline reward that always returns 0."""
    return 0.0


@register_reward("length_penalty")
def length_penalty_reward(
    prompt: str,
    response: str,
    *,
    target_length: int = 200,
    penalty_scale: float = 0.001,
) -> float:
    """Penalize responses that deviate from *target_length* characters."""
    diff = abs(len(response) - target_length)
    return -penalty_scale * diff


@register_reward("format_check")
def format_check_reward(
    prompt: str,
    response: str,
    *,
    required_markers: list[str] | None = None,
) -> float:
    """Reward based on the fraction of *required_markers* present in the response."""
    if required_markers is None:
        required_markers = []
    if not required_markers:
        return 0.0
    matched = sum(1 for m in required_markers if m in response)
    return matched / len(required_markers)


@register_reward("composite")
def composite_reward(
    prompt: str,
    response: str,
    *,
    reward_names: list[str] | None = None,
    weights: list[float] | None = None,
) -> float:
    """Weighted combination of multiple registered reward functions."""
    if not reward_names:
        return 0.0
    if weights is None:
        weights = [1.0] * len(reward_names)
    total = 0.0
    for name, weight in zip(reward_names, weights):
        fn = reward_registry.get(name)
        total += weight * fn(prompt, response)
    return total


@register_reward("reward_model")
def reward_model_reward(
    prompt: str,
    response: str,
    *,
    model_name: str = "",
    device: str = "auto",
    dtype: str = "auto",
) -> float:
    """Score using a HuggingFace reward model (e.g. sequence classifier)."""
    if not model_name:
        raise ValueError("reward_model requires 'model_name' in reward_kwargs")
    if model_name not in _reward_model_cache:
        _reward_model_cache[model_name] = RewardModelWrapper(model_name, device, dtype)
    return _reward_model_cache[model_name].score(prompt, response)


@register_reward("llm_judge")
def llm_judge_reward(
    prompt: str,
    response: str,
    *,
    model_name: str = "",
    criteria: str = "helpfulness, correctness, coherence",
    template: str | None = None,
    device: str = "auto",
) -> float:
    """Score using an LLM as a judge on a 1-5 scale, normalized to 0-1."""
    if not model_name:
        raise ValueError("llm_judge requires 'model_name' in reward_kwargs")
    cache_key = model_name
    if cache_key not in _llm_judge_cache:
        _llm_judge_cache[cache_key] = LLMJudgeWrapper(model_name, device, template)
    return _llm_judge_cache[cache_key].judge(prompt, response, criteria)
