from __future__ import annotations

import math
from typing import Any

from xaytune.utils.registry import Registry

metric_registry = Registry("metric")

register_metric = metric_registry.register


def _mean_loss(losses: list[float], weights: list[int] | None) -> float | None:
    """The dataset's mean loss, or ``None`` when there is nothing to average.

    A causal LM's batch loss is a mean over that batch's valid next-token
    targets, so averaging batch means gives a batch of 3 targets the weight of
    a batch of 300, and the answer moves with how the same examples were
    batched (issue #36). Weighted by *weights* -- each batch's target count --
    it is the mean over every target in the dataset.

    The contract, checked before anything is averaged:

    - ``weights is None``: the plain mean of *losses*, exactly as before --
      for losses that come with no target count.
    - one weight per loss, none negative; anything else is a caller's bug and
      raises rather than truncating or weighting something undefined.
    - weights summing to zero: nothing to average, so ``None``, which the
      metrics report as ``0.0``, as they always have for no losses.

    Raises:
        ValueError: If the weights do not pair one-to-one with the losses, or
            any is negative.
    """
    if weights is not None:
        if len(weights) != len(losses):
            raise ValueError(f"{len(losses)} losses but {len(weights)} weights")
        negative = [weight for weight in weights if weight < 0]
        if negative:
            raise ValueError(f"weights are target counts and cannot be negative: {negative}")
    if not losses:
        return None
    if weights is None:
        return sum(losses) / len(losses)
    total = sum(weights)
    if total == 0:
        return None
    return sum(loss * weight for loss, weight in zip(losses, weights)) / total


@register_metric("loss")
def compute_loss(
    losses: list[float], *args: Any, weights: list[int] | None = None, **kwargs: Any
) -> float:
    """Mean loss over the dataset: weighted by each batch's target count when given.

    ``0.0`` when there is nothing to average -- no losses, or no targets.
    """
    mean = _mean_loss(losses, weights)
    return 0.0 if mean is None else mean


@register_metric("perplexity")
def compute_perplexity(
    losses: list[float], *args: Any, weights: list[int] | None = None, **kwargs: Any
) -> float:
    """Perplexity: ``exp`` of the mean loss, weighted as :func:`compute_loss` weights it.

    ``0.0`` when there is nothing to average, as :func:`compute_loss` reports.
    """
    mean = _mean_loss(losses, weights)
    return 0.0 if mean is None else math.exp(mean)


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
