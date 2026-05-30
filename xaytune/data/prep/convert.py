from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from xaytune.data.prep.report import PrepReport, PrepResult, StepReport


def _load_input(source: str | list[dict]) -> list[dict[str, Any]]:
    if isinstance(source, list):
        return source
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".parquet":
        return _load_parquet(path)
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_csv(path: Path) -> list[dict[str, Any]]:
    items = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(dict(row))
    return items


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("Parquet support requires pyarrow. Install with: pip install pyarrow")
    table = pq.read_table(path)
    return table.to_pylist()


def _apply_field_map(samples: list[dict], field_map: dict[str, str]) -> list[dict]:
    result = []
    for sample in samples:
        mapped = {}
        for src_key, dst_key in field_map.items():
            if src_key in sample:
                mapped[dst_key] = sample[src_key]
        for k, v in sample.items():
            if k not in field_map:
                mapped[k] = v
        result.append(mapped)
    return result


def _alpaca_to_sharegpt(sample: dict) -> dict:
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")
    user_msg = f"{instruction}\n{input_text}".strip() if input_text else instruction
    return {
        "conversations": [
            {"from": "human", "value": user_msg},
            {"from": "gpt", "value": output},
        ]
    }


def _alpaca_to_chat(sample: dict) -> dict:
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")
    user_msg = f"{instruction}\n{input_text}".strip() if input_text else instruction
    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": output},
        ]
    }


def _sharegpt_to_alpaca(sample: dict) -> dict:
    convs = sample.get("conversations", [])
    instruction = ""
    output = ""
    for turn in convs:
        role = turn.get("from", turn.get("role", ""))
        value = turn.get("value", turn.get("content", ""))
        if role in ("human", "user") and not instruction:
            instruction = value
        elif role in ("gpt", "assistant") and not output:
            output = value
    return {"instruction": instruction, "input": "", "output": output}


def _sharegpt_to_chat(sample: dict) -> dict:
    convs = sample.get("conversations", [])
    role_map = {"human": "user", "gpt": "assistant"}
    messages = []
    for turn in convs:
        role = turn.get("from", turn.get("role", ""))
        value = turn.get("value", turn.get("content", ""))
        messages.append({"role": role_map.get(role, role), "content": value})
    return {"messages": messages}


def _chat_to_alpaca(sample: dict) -> dict:
    messages = sample.get("messages", [])
    instruction = ""
    output = ""
    for msg in messages:
        if msg["role"] == "user" and not instruction:
            instruction = msg["content"]
        elif msg["role"] == "assistant" and not output:
            output = msg["content"]
    return {"instruction": instruction, "input": "", "output": output}


def _chat_to_sharegpt(sample: dict) -> dict:
    messages = sample.get("messages", [])
    role_map = {"user": "human", "assistant": "gpt"}
    convs = []
    for msg in messages:
        convs.append({
            "from": role_map.get(msg["role"], msg["role"]),
            "value": msg["content"],
        })
    return {"conversations": convs}


_CONVERTERS: dict[tuple[str, str], Any] = {
    ("alpaca", "sharegpt"): _alpaca_to_sharegpt,
    ("alpaca", "chat"): _alpaca_to_chat,
    ("sharegpt", "alpaca"): _sharegpt_to_alpaca,
    ("sharegpt", "chat"): _sharegpt_to_chat,
    ("chat", "alpaca"): _chat_to_alpaca,
    ("chat", "sharegpt"): _chat_to_sharegpt,
}


def convert(
    source: str | list[dict],
    *,
    source_format: str,
    target_format: str,
    output: str | None = None,
    field_map: dict[str, str] | None = None,
) -> PrepResult:
    samples = _load_input(source)
    input_rows = len(samples)

    if field_map:
        samples = _apply_field_map(samples, field_map)

    src_fmt = source_format if source_format not in ("csv", "parquet") else _detect_loaded_format(samples)
    if src_fmt in ("csv", "parquet"):
        src_fmt = "alpaca"

    if src_fmt == target_format:
        converted = samples
    else:
        key = (src_fmt, target_format)
        if key not in _CONVERTERS:
            raise ValueError(
                f"Unsupported conversion: {src_fmt} → {target_format}. "
                f"Supported: {', '.join(f'{a}→{b}' for a, b in _CONVERTERS)}"
            )
        converter = _CONVERTERS[key]
        converted = [converter(s) for s in samples]

    result = PrepResult(
        dataset=converted,
        report=PrepReport(
            input_rows=input_rows,
            output_rows=len(converted),
            steps=[StepReport(
                name="convert",
                input_rows=input_rows,
                output_rows=len(converted),
                details={"source_format": source_format, "target_format": target_format},
            )],
        ),
    )

    if output:
        result.save(output)

    return result


def _detect_loaded_format(samples: list[dict]) -> str:
    if not samples:
        return "alpaca"
    sample = samples[0]
    if "conversations" in sample:
        return "sharegpt"
    if "messages" in sample:
        return "chat"
    return "alpaca"
