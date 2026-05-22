from __future__ import annotations
from typing import Any
from trainlib.data.registry import format_registry


@format_registry.register("alpaca")
def format_alpaca(sample: dict[str, Any]) -> dict[str, str]:
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")
    if input_text:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return {"text": text}
