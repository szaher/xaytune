from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from xaytune.data.prep.report import PrepReport, PrepResult, StepReport


def _load_input(source: str | list[dict]) -> list[dict[str, Any]]:
    if isinstance(source, list):
        return source
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Seed dataset not found: {source}")
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "API key required for synthetic data generation. "
        "Pass api_key= or set OPENAI_API_KEY environment variable."
    )


def _get_client(api_key: str, api_base: str | None):
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "Synthetic data generation requires the openai package. "
            "Install with: pip install xaytune[synth]"
        )
    kwargs: dict[str, Any] = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAI(**kwargs)


def _call_llm_sync(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return str(response.choices[0].message.content)


_AUGMENT_SYSTEM = (
    "You are a dataset augmentation assistant. Given example training data, "
    "generate a new example in the SAME format and style but with different content. "
    "Respond with ONLY valid JSON matching the example format."
)

_DISTILL_SYSTEM = (
    "You are a dataset generation assistant. Given a topic, generate a training "
    "example as an instruction-response pair. Respond with ONLY valid JSON."
)

_EVOLVE_SYSTEM = (
    "You are a dataset evolution assistant. Given a training example, create a more "
    "complex and challenging version that tests deeper understanding. Keep the same "
    "format but increase difficulty. Respond with ONLY valid JSON."
)

_FORMAT_TEMPLATES: dict[str, str] = {
    "alpaca": '{"instruction": "...", "output": "..."}',
    "sharegpt": (
        '{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}'
    ),
    "chat": (
        '{"messages": [{"role": "user", "content": "..."}, '
        '{"role": "assistant", "content": "..."}]}'
    ),
}


def _augment_prompt(seed: dict, format: str) -> str:
    template = _FORMAT_TEMPLATES.get(format, _FORMAT_TEMPLATES["alpaca"])
    return (
        f"Here is an example from the training dataset:\n\n"
        f"{json.dumps(seed, ensure_ascii=False)}\n\n"
        f"Generate ONE new example in the same format: {template}\n"
        f"Make it different in content but similar in style and quality."
    )


def _distill_prompt(topic: str, format: str) -> str:
    template = _FORMAT_TEMPLATES.get(format, _FORMAT_TEMPLATES["alpaca"])
    return (
        f"Generate ONE training example about: {topic}\n\n"
        f"Format: {template}\n"
        f"Make it educational, accurate, and well-written."
    )


def _evolve_prompt(sample: dict, format: str) -> str:
    template = _FORMAT_TEMPLATES.get(format, _FORMAT_TEMPLATES["alpaca"])
    return (
        f"Here is a training example:\n\n"
        f"{json.dumps(sample, ensure_ascii=False)}\n\n"
        f"Create a MORE COMPLEX version that tests deeper understanding. "
        f"Keep the same format: {template}"
    )


def _parse_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    result: dict[str, Any] = json.loads(text)
    return result


def _apply_post_filter(
    samples: list[dict], post_filter: list[dict[str, Any]], format: str
) -> list[dict]:
    from xaytune.data.prep.filters import filter_dataset

    field_map = {"alpaca": "output", "sharegpt": "conversations", "chat": "messages"}
    field = field_map.get(format, "text")

    for sample in samples:
        if field not in sample:
            for key in ("text", "output", "instruction"):
                if key in sample:
                    field = key
                    break

    result = filter_dataset(samples, filters=post_filter, field=field)
    return result.dataset


def generate(
    *,
    mode: Literal["augment", "distill", "evolve"],
    seed: str | list[dict] | None = None,
    topic: str | None = None,
    n: int = 10,
    rounds: int = 1,
    format: str = "alpaca",
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float = 0.8,
    concurrency: int = 5,
    post_filter: list[dict[str, Any]] | None = None,
) -> PrepResult:
    if mode == "distill" and not topic:
        raise ValueError("mode='distill' requires a topic= argument.")

    if mode in ("augment", "evolve") and seed is None:
        raise ValueError(f"mode='{mode}' requires a seed= argument.")

    resolved_key = _resolve_api_key(api_key)
    client = _get_client(resolved_key, api_base)

    seeds: list[dict] = []
    if seed is not None:
        seeds = _load_input(seed)

    generated: list[dict] = []

    if mode == "augment":
        for i in range(n):
            example = seeds[i % len(seeds)]
            prompt = _augment_prompt(example, format)
            response = _call_llm_sync(client, model, _AUGMENT_SYSTEM, prompt, temperature)
            try:
                generated.append(_parse_response(response))
            except json.JSONDecodeError:
                continue

    elif mode == "distill":
        assert topic is not None
        for _ in range(n):
            prompt = _distill_prompt(topic, format)
            response = _call_llm_sync(client, model, _DISTILL_SYSTEM, prompt, temperature)
            try:
                generated.append(_parse_response(response))
            except json.JSONDecodeError:
                continue

    elif mode == "evolve":
        current = list(seeds)
        for _ in range(rounds):
            evolved = []
            for sample in current:
                prompt = _evolve_prompt(sample, format)
                response = _call_llm_sync(client, model, _EVOLVE_SYSTEM, prompt, temperature)
                try:
                    evolved.append(_parse_response(response))
                except json.JSONDecodeError:
                    evolved.append(sample)
            current = evolved
        generated = current

    if post_filter:
        generated = _apply_post_filter(generated, post_filter, format)

    input_rows = len(seeds) if mode != "distill" else 0

    return PrepResult(
        dataset=generated,
        report=PrepReport(
            input_rows=input_rows,
            output_rows=len(generated),
            steps=[
                StepReport(
                    name="generate",
                    input_rows=input_rows,
                    output_rows=len(generated),
                    details={"mode": mode, "model": model, "n": n},
                )
            ],
        ),
    )
