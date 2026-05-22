from __future__ import annotations

from typing import Any


def push_to_hub(
    model_or_path: Any,
    *,
    repo: str | None = None,
    tokenizer: Any | None = None,
) -> None:
    if repo is None:
        raise ValueError("'repo' is required (e.g., 'username/model-name').")

    if isinstance(model_or_path, str):
        from trainlib.models import load_model
        model_result = load_model(model_or_path)
        model = model_result.model
        tokenizer = model_result.tokenizer
    else:
        model = model_or_path

    model.push_to_hub(repo)
    if tokenizer is not None:
        tokenizer.push_to_hub(repo)
