from __future__ import annotations

import warnings
from typing import Any

from xaytune.data.registry import format_registry

_warned_text_keys: set[tuple[str, ...]] = set()


@format_registry.register("alpaca")
def format_alpaca(sample: dict[str, Any]) -> dict[str, str]:
    """Format an Alpaca-style sample with prompt/response boundary for label masking."""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")
    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}"
            f"\n\n### Response:\n"
        )
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    text = prompt + output
    return {"text": text, "prompt_text": prompt}


@format_registry.register("sharegpt")
def format_sharegpt(sample: dict[str, Any]) -> dict[str, Any]:
    """Format a ShareGPT-style multi-turn conversation with per-turn masking."""
    conversations = sample.get("conversations", [])
    role_map = {"human": "user", "gpt": "assistant"}
    turns = []
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))
        turns.append({"role": role_map.get(role, role), "content": content})
    return {"turns": turns}


@format_registry.register("chat")
def format_chat(sample: dict[str, Any]) -> dict[str, Any]:
    """Format OpenAI-style chat messages with per-turn masking."""
    messages = sample.get("messages", [])
    turns = []
    for msg in messages:
        turns.append({"role": msg.get("role", ""), "content": msg.get("content", "")})
    return {"turns": turns}


@format_registry.register("text")
def format_text(sample: dict[str, Any]) -> dict[str, str]:
    """Pass through a raw text sample as ``{"text": ...}``."""
    text = sample.get("text", sample.get("content", ""))
    if not text:
        keys = tuple(sorted(sample.keys()))
        if keys not in _warned_text_keys:
            _warned_text_keys.add(keys)
            warnings.warn(
                f"Sample has no 'text' or 'content' key. Found keys: {list(keys)}. "
                "Returning empty text — this sample will be skipped during tokenization.",
                stacklevel=2,
            )
    return {"text": text}


def apply_chat_template(
    sample: dict[str, Any],
    tokenizer: Any,
    *,
    format: str = "chat",
) -> dict[str, Any]:
    """Apply the tokenizer's chat template to a conversation sample.

    Returns ``{"turns": [...]}`` for per-turn masking in the tokenizer.
    """
    if format == "sharegpt":
        role_map = {"human": "user", "gpt": "assistant"}
        conversations = sample.get("conversations", [])
        turns = []
        for turn in conversations:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            turns.append({"role": role_map.get(role, role), "content": content})
    else:
        turns = [
            {"role": m.get("role", ""), "content": m.get("content", "")}
            for m in sample.get("messages", [])
        ]

    return {"turns": turns, "_use_chat_template": True}
