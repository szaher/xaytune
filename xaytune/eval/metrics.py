from __future__ import annotations

import math
from typing import Any

from xaytune.utils.registry import Registry

metric_registry = Registry("metric")

register_metric = metric_registry.register


def _mean_loss(losses: list[float], weights: list[int] | None) -> float:
    """The dataset's mean loss: each batch's mean, weighted by the targets it averaged.

    A causal LM's batch loss is a mean over that batch's valid next-token
    targets, so averaging batch means gives a batch of 3 targets the weight of
    a batch of 300, and the answer moves with how the same examples were
    batched (issue #36). Weighted by *weights* -- each batch's target count --
    it is the mean over every target in the dataset. Without weights, the
    unweighted mean, for losses that come with no target count.
    """
    if weights is None:
        return sum(losses) / len(losses)
    if len(weights) != len(losses):
        raise ValueError(f"{len(losses)} losses but {len(weights)} weights")
    total = sum(weights)
    if total <= 0:
        raise ValueError("no batch had a target to average over")
    return sum(loss * weight for loss, weight in zip(losses, weights)) / total


@register_metric("loss")
def compute_loss(
    losses: list[float], *args: Any, weights: list[int] | None = None, **kwargs: Any
) -> float:
    """Mean loss over the dataset: weighted by each batch's target count when given."""
    if not losses:
        return 0.0
    return _mean_loss(losses, weights)


@register_metric("perplexity")
def compute_perplexity(
    losses: list[float], *args: Any, weights: list[int] | None = None, **kwargs: Any
) -> float:
    """Perplexity: ``exp`` of the mean loss, weighted as :func:`compute_loss` weights it."""
    if not losses:
        return 0.0
    return math.exp(_mean_loss(losses, weights))


@register_metric("token_accuracy")
def compute_token_accuracy(
    predictions: list[int],
    references: list[int],
    *args: Any,
    **kwargs: Any,
) -> float:
    """Fraction of targets whose prediction matches.

    Pairs are compared as given. For a causal LM they must already be aligned
    next-token pairs (:func:`xaytune.eval.causal.next_token_pairs`): the
    logits at *i* against the token at *i + 1*.
    """
    if not predictions:
        return 0.0
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(predictions)
