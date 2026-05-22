from __future__ import annotations

import math
from typing import Any

from trainlib.utils.registry import Registry

metric_registry = Registry("metric")

register_metric = metric_registry.register


@register_metric("loss")
def compute_loss(losses: list[float], *args: Any, **kwargs: Any) -> float:
    if not losses:
        return 0.0
    return sum(losses) / len(losses)


@register_metric("perplexity")
def compute_perplexity(losses: list[float], *args: Any, **kwargs: Any) -> float:
    if not losses:
        return 0.0
    mean_loss = sum(losses) / len(losses)
    return math.exp(mean_loss)


@register_metric("token_accuracy")
def compute_token_accuracy(
    predictions: list[int],
    references: list[int],
    *args: Any,
    **kwargs: Any,
) -> float:
    if not predictions:
        return 0.0
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(predictions)
