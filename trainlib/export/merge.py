from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def merge(checkpoint_path: str, *, save_to: str) -> None:
    from trainlib.models import load_model

    model_result = load_model(checkpoint_path)

    if not model_result.peft_applied:
        raise ValueError(
            f"Model at '{checkpoint_path}' is not a PEFT model. "
            f"merge() only works with LoRA/QLoRA checkpoints."
        )

    merged_model = model_result.model.merge_and_unload()

    Path(save_to).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(save_to)
    model_result.tokenizer.save_pretrained(save_to)


def save(
    model: Any,
    tokenizer: Any,
    *,
    output_dir: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if metadata:
        meta_path = path / "trainlib_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
