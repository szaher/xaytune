from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from trainlib.data.registry import format_registry

_REQUIRED_FIELDS = {"prompt", "chosen", "rejected"}


@format_registry.register("preference")
def format_preference(sample: dict[str, Any]) -> dict[str, str]:
    return {
        "prompt": sample["prompt"],
        "chosen": sample["chosen"],
        "rejected": sample["rejected"],
    }


def load_preference_dataset(
    path: str,
    *,
    eval_split: float = 0.0,
) -> Union[list[dict], tuple[list[dict], list[dict]]]:
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
        split_idx = len(items) - int(len(items) * eval_split)
        return items[:split_idx], items[split_idx:]

    return items
