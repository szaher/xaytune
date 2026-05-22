from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from trainlib.data.registry import format_registry


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _split_dataset(data: list[dict], eval_split: float) -> tuple[list[dict], list[dict]]:
    split_idx = len(data) - int(len(data) * eval_split)
    return data[:split_idx], data[split_idx:]


def load_dataset(
    path: str,
    *,
    format: str,
    eval_split: float = 0.0,
    **kwargs: Any,
) -> Union[list[dict], tuple[list[dict], list[dict]]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    format_fn = format_registry.get(format)
    raw_data = _load_jsonl(path)
    processed = [format_fn(sample) for sample in raw_data]
    if eval_split > 0:
        return _split_dataset(processed, eval_split)
    return processed
