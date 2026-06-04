from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from xaytune.data.registry import format_registry

_REQUIRED_FIELDS = {"prompt", "chosen", "rejected"}


@format_registry.register("preference")
def format_preference(sample: dict[str, Any]) -> dict[str, str]:
    """Extract prompt/chosen/rejected fields from a preference sample."""
    return {
        "prompt": sample["prompt"],
        "chosen": sample["chosen"],
        "rejected": sample["rejected"],
    }


def load_preference_dataset(
    path: str,
    *,
    eval_split: float = 0.0,
) -> list[dict] | tuple[list[dict], list[dict]]:
    """Load a preference JSONL file with prompt/chosen/rejected fields.

    Args:
        path: Path to a JSONL file where each line has ``prompt``,
            ``chosen``, and ``rejected`` fields.
        eval_split: Fraction to hold out for evaluation.

    Returns:
        Formatted samples, or a ``(train, eval)`` tuple.

    Raises:
        FileNotFoundError: If *path* doesn't exist.
        ValueError: If any row is missing required fields.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Preference dataset not found: {path}")

    items = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            missing = _REQUIRED_FIELDS - set(sample.keys())
            if missing:
                raise ValueError(
                    f"Row {i}: missing required fields: {', '.join(sorted(missing))}. "
                    f"Preference data must have: prompt, chosen, rejected."
                )
            items.append(format_preference(sample))

    if eval_split > 0:
        shuffled = list(items)
        random.Random(42).shuffle(shuffled)
        split_idx = len(shuffled) - int(len(shuffled) * eval_split)
        return shuffled[:split_idx], shuffled[split_idx:]

    return items
