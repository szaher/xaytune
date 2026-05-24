from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def preview_dataset(
    path: str,
    format: str = "alpaca",
    source: str = "local",
    num_samples: int = 5,
) -> list[dict[str, Any]]:
    """Load and return the first N samples from a dataset file.

    Supports JSONL files (one JSON object per line).  Returns an empty list
    if the file does not exist or cannot be parsed.
    """
    try:
        p = Path(path)
        if not p.exists():
            return []

        samples: list[dict[str, Any]] = []
        with p.open() as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return samples
    except Exception:
        return []


def compute_tokenization_stats(
    path: str,
    format: str = "alpaca",
    tokenizer_name: str = "",
    max_samples: int = 1000,
) -> dict[str, Any]:
    """Compute token length statistics for a dataset.

    Returns a dict with ``count``, ``avg``, ``min``, ``max``, ``p50``,
    ``p90``, ``p99``, and ``histogram`` (list of ``(bin_start, count)``
    tuples).  Returns an empty dict on errors or if the tokenizer cannot
    be loaded.
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        return {}

    p = Path(path)
    if not p.exists():
        return {}

    lengths: list[int] = []
    try:
        with p.open() as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = _extract_text(sample, format)
                if text:
                    tokens = tokenizer.encode(text)
                    lengths.append(len(tokens))
    except Exception:
        return {}

    if not lengths:
        return {}

    lengths.sort()
    n = len(lengths)
    avg = sum(lengths) / n

    num_bins = min(20, max(5, n // 10))
    min_len, max_len = lengths[0], lengths[-1]
    bin_width = max(1, (max_len - min_len + 1) // num_bins)
    histogram: list[tuple[int, int]] = []
    for b in range(num_bins):
        start = min_len + b * bin_width
        end = start + bin_width
        count = sum(1 for ln in lengths if start <= ln < end)
        histogram.append((start, count))

    return {
        "count": n,
        "avg": round(avg, 1),
        "min": lengths[0],
        "max": lengths[-1],
        "p50": lengths[n // 2],
        "p90": lengths[int(n * 0.9)],
        "p99": lengths[int(n * 0.99)],
        "histogram": histogram,
    }


def _extract_text(sample: dict[str, Any], format: str) -> str:
    if "text" in sample:
        return str(sample["text"])
    if "prompt" in sample:
        parts: list[str] = [str(sample.get("prompt", ""))]
        if "chosen" in sample:
            parts.append(str(sample["chosen"]))
        elif "output" in sample:
            parts.append(str(sample["output"]))
        return " ".join(p for p in parts if p)
    if "instruction" in sample:
        parts = [str(sample.get("instruction", ""))]
        if "input" in sample and sample["input"]:
            parts.append(str(sample["input"]))
        if "output" in sample:
            parts.append(str(sample["output"]))
        return " ".join(p for p in parts if p)
    return ""
