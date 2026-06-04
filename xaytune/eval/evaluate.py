from __future__ import annotations

from typing import Any

import torch

from xaytune.eval.metrics import metric_registry


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
        from xaytune.models import load_model

        model_result = load_model(model)
        model = model_result.model

    device = next(model.parameters()).device

    losses: list[float] = []
    all_predictions: list[int] = []
    all_references: list[int] = []

    if hasattr(model, "eval"):
        model.eval()

    with torch.no_grad():
        for batch in dataset:
            if isinstance(batch, dict):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs = model(**batch)
            else:
                outputs = model(batch)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                losses.append(outputs.loss.item())

            if (
                hasattr(outputs, "logits")
                and isinstance(batch, dict)
                and "labels" in batch
            ):
                preds = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                mask = labels != -100
                all_predictions.extend(preds[mask].cpu().tolist())
                all_references.extend(labels[mask].cpu().tolist())

    results: dict[str, float] = {}
    for metric_name in metrics:
        compute_fn = metric_registry.get(metric_name)
        if metric_name in ("loss", "perplexity"):
            results[metric_name] = compute_fn(losses)
        else:
            results[metric_name] = compute_fn(all_predictions, all_references)

    return results
