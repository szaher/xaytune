from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trainlib.data.formats import apply_chat_template
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


def _load_huggingface(
    path: str,
    *,
    format: str,
    streaming: bool = False,
    eval_split: float = 0.0,
) -> Any:
    import datasets

    format_fn = format_registry.get(format)

    if eval_split > 0 and not streaming:
        ds = datasets.load_dataset(path, split="train")
        split = ds.train_test_split(test_size=eval_split)
        train = [format_fn(sample) for sample in split["train"]]
        val = [format_fn(sample) for sample in split["test"]]
        return train, val

    if streaming:
        ds = datasets.load_dataset(path, split="train", streaming=True)
        return ds.map(format_fn)

    ds = datasets.load_dataset(path, split="train")
    return [format_fn(sample) for sample in ds]


def _make_format_fn(format: str, tokenizer: Any | None) -> Any:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        if format in ("chat", "sharegpt"):
            return lambda sample: apply_chat_template(sample, tokenizer, format=format)
    return format_registry.get(format)


def load_dataset(
    path: str,
    *,
    format: str,
    source: str = "local",
    streaming: bool = False,
    eval_split: float = 0.0,
    tokenizer: Any | None = None,
    **kwargs: Any,
) -> list[dict] | tuple[list[dict], list[dict]]:
    if source == "huggingface":
        return _load_huggingface(
            path, format=format, streaming=streaming, eval_split=eval_split,
        )

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    format_fn = _make_format_fn(format, tokenizer)
    raw_data = _load_jsonl(path)
    processed = [format_fn(sample) for sample in raw_data]
    if eval_split > 0:
        return _split_dataset(processed, eval_split)
    return processed
