from __future__ import annotations

from typing import Any

import torch

from trainlib.eval.metrics import metric_registry


def evaluate(
    *,
    model: Any,
    dataset: list[dict[str, Any]],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate a model on a list of batches and compute metrics.

    Args:
        model: A model instance or HuggingFace model name string.
        dataset: List of batch dicts (each passable to ``model(**batch)``).
        metrics: Metric names to compute (default: ``["loss", "perplexity"]``).

    Returns:
        Dict mapping metric names to computed values.
    """
    if metrics is None:
        metrics = ["loss", "perplexity"]

    if isinstance(model, str):
        from trainlib.models import load_model

        model_result = load_model(model)
        model = model_result.model

    losses: list[float] = []

    model.eval() if hasattr(model, "eval") else None

    with torch.no_grad():
        for batch in dataset:
            if isinstance(batch, dict):
                outputs = model(**batch)
            else:
                outputs = model(batch)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                losses.append(outputs.loss.item())

    results: dict[str, float] = {}
    for metric_name in metrics:
        compute_fn = metric_registry.get(metric_name)
        if metric_name in ("loss", "perplexity"):
            results[metric_name] = compute_fn(losses)
        else:
            results[metric_name] = compute_fn([], [])

    return results
