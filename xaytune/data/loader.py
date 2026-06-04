from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Any

from xaytune.data.formats import apply_chat_template
from xaytune.data.registry import format_registry


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _split_dataset(data: list[dict], eval_split: float) -> tuple[list[dict], list[dict]]:
    shuffled = list(data)
    random.Random(42).shuffle(shuffled)
    split_idx = len(shuffled) - int(len(shuffled) * eval_split)
    return shuffled[:split_idx], shuffled[split_idx:]


def _load_huggingface(
    path: str,
    *,
    format: str,
    streaming: bool = False,
    eval_split: float = 0.0,
    tokenizer: Any | None = None,
) -> Any:
    import datasets

    format_fn = _make_format_fn(format, tokenizer)

    if eval_split > 0 and not streaming:
        ds = datasets.load_dataset(path, split="train")
        split = ds.train_test_split(test_size=eval_split)
        train = [format_fn(sample) for sample in split["train"]]
        val = [format_fn(sample) for sample in split["test"]]
        return train, val

    if streaming:
        ds = datasets.load_dataset(path, split="train", streaming=True)
        if eval_split > 0:
            warnings.warn(
                "eval_split is not supported with streaming=True; "
                "evaluation data will not be available. Use streaming=False "
                "or provide a separate eval_path.",
                stacklevel=3,
            )
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
    """Load and format a dataset from a local JSONL file or HuggingFace Hub.

    Each sample is run through the registered format function (``"alpaca"``,
    ``"sharegpt"``, ``"chat"``, ``"text"``, ``"preference"``), converting
    raw fields into a ``{"text": "..."}`` dict ready for tokenization.

    Args:
        path: Local file path or HuggingFace dataset name.
        format: Format name registered in the format registry.
        source: ``"local"`` or ``"huggingface"``.
        streaming: Stream from HuggingFace instead of downloading.
        eval_split: Fraction to hold out for evaluation (0 = no split).
        tokenizer: Optional tokenizer for chat template application.

    Returns:
        A list of formatted samples, or a ``(train, eval)`` tuple when
        ``eval_split > 0``.

    Raises:
        FileNotFoundError: If *source* is ``"local"`` and *path* doesn't exist.
    """
    if source == "huggingface":
        return _load_huggingface(  # type: ignore[no-any-return]
            path,
            format=format,
            streaming=streaming,
            eval_split=eval_split,
            tokenizer=tokenizer,
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
