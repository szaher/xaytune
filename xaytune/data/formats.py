from __future__ import annotations

from typing import Any

from xaytune.data.registry import format_registry


@format_registry.register("alpaca")
def format_alpaca(sample: dict[str, Any]) -> dict[str, str]:
    """Format an Alpaca-style sample (instruction/input/output) into ``{"text": ...}``."""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")
    if input_text:
        text = (
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}"
            f"\n\n### Response:\n{output}"
        )
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return {"text": text}


@format_registry.register("sharegpt")
def format_sharegpt(sample: dict[str, Any]) -> dict[str, str]:
    """Format a ShareGPT-style multi-turn conversation into ``{"text": ...}``."""
    conversations = sample.get("conversations", [])
    parts = []
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        value = turn.get("value", turn.get("content", ""))
        if role in ("human", "user"):
            parts.append(f"### User:\n{value}")
        elif role in ("gpt", "assistant"):
            parts.append(f"### Assistant:\n{value}")
        elif role == "system":
            parts.append(f"### System:\n{value}")
    return {"text": "\n\n".join(parts)}


@format_registry.register("chat")
def format_chat(sample: dict[str, Any]) -> dict[str, str]:
    """Format an OpenAI-style chat messages list into ``{"text": ...}``."""
    messages = sample.get("messages", [])
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        parts.append(f"### {role.capitalize()}:\n{content}")
    return {"text": "\n\n".join(parts)}


@format_registry.register("text")
def format_text(sample: dict[str, Any]) -> dict[str, str]:
    """Pass through a raw text sample as ``{"text": ...}``."""
    text = sample.get("text", sample.get("content", ""))
    return {"text": text}


def apply_chat_template(
    sample: dict[str, Any],
    tokenizer: Any,
    *,
    format: str = "chat",
) -> dict[str, str]:
    """Apply the tokenizer's chat template to a conversation sample."""
    if format == "sharegpt":
        role_map = {"human": "user", "gpt": "assistant"}
        conversations = sample.get("conversations", [])
        messages = []
        for turn in conversations:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            messages.append({"role": role_map.get(role, role), "content": content})
    else:
        messages = sample.get("messages", [])

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}
