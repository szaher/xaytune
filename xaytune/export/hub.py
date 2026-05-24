from __future__ import annotations

from typing import Any


def push_to_hub(
    model_or_path: Any,
    *,
    repo: str | None = None,
    tokenizer: Any | None = None,
) -> None:
    """Push a model and tokenizer to the HuggingFace Hub.

    Args:
        model_or_path: A model instance or path to a saved checkpoint.
            If a path string, the model and tokenizer are loaded automatically.
        repo: Hub repository id (e.g. ``"username/model-name"``).
        tokenizer: Tokenizer to push alongside the model. Ignored when
            *model_or_path* is a string (tokenizer is loaded from checkpoint).

    Raises:
        ValueError: If *repo* is not provided.
    """
    if repo is None:
        raise ValueError("'repo' is required (e.g., 'username/model-name').")

    if isinstance(model_or_path, str):
        from xaytune.models import load_model

        model_result = load_model(model_or_path)
        model = model_result.model
        tokenizer = model_result.tokenizer
    else:
        model = model_or_path

    model.push_to_hub(repo)
    if tokenizer is not None:
        tokenizer.push_to_hub(repo)
