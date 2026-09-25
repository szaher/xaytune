from __future__ import annotations

from typing import Any

import torch

from xaytune.eval.causal import next_token_pairs, next_token_targets
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

    For a causal LM, ``token_accuracy`` compares the logits at each position
    with the *next* label, and ``loss`` and ``perplexity`` are means over every
    next-token target in the dataset -- each batch's loss weighted by its
    target count -- so batching the same examples differently gives the same
    answer. Labels of ``-100`` are not targets.
    """
    if metrics is None:
        metrics = ["loss", "perplexity"]

    if isinstance(model, str):
        from xaytune.models import load_model

        model_result = load_model(model)
        model = model_result.model

    device = next(model.parameters()).device

    # Only metrics other than loss and perplexity need the predictions.
    scores_tokens = any(name not in ("loss", "perplexity") for name in metrics)
    losses: list[float] = []
    # Each loss's target count, for a dataset mean that does not depend on how
    # the examples were batched. None once any loss arrives without labels to
    # count, and then the unweighted mean is all that can be computed.
    weights: list[int] | None = []
    all_predictions: list[int] = []
    all_references: list[int] = []

    if hasattr(model, "eval"):
        model.eval()

    with torch.no_grad():
        for batch in dataset:
            if isinstance(batch, dict):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }
                outputs = model(**batch)
            else:
                outputs = model(batch)

            labels = batch.get("labels") if isinstance(batch, dict) else None
            if labels is not None and not isinstance(labels, torch.Tensor):
                # Batch values are not required to be tensors on the way in
                # (the device move above passes non-tensors through).
                labels = torch.as_tensor(labels)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                targets = next_token_targets(labels) if labels is not None else None
                if targets == 0:
                    continue  # a mean over no targets is not a loss
                losses.append(outputs.loss.item())
                if weights is not None and targets is not None:
                    weights.append(targets)
                else:
                    weights = None

            if scores_tokens and hasattr(outputs, "logits") and labels is not None:
                predictions, references = next_token_pairs(outputs.logits, labels)
                all_predictions.extend(predictions)
                all_references.extend(references)

    results: dict[str, float] = {}
    for metric_name in metrics:
        compute_fn = metric_registry.get(metric_name)
        if metric_name in ("loss", "perplexity"):
            results[metric_name] = compute_fn(losses, weights=weights)
        else:
            results[metric_name] = compute_fn(all_predictions, all_references)

    return results
