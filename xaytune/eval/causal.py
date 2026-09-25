"""What a causal language model is scored on: each position against the next token.

The logits at position *i* predict the token at *i + 1*, so a prediction is
compared with the label one place to the right, and the last position predicts
nothing. Labels equal to ``-100`` are not targets: they count in neither the
numerator nor the denominator.

Shared by :func:`xaytune.eval.evaluate` and the in-training evaluation
callback, which both scored the logits at *i* against the label at *i* until
issue #36 -- a comparison that rewards a model for echoing its input.
"""

from __future__ import annotations

from typing import Any

IGNORE_INDEX = -100

__all__ = ["IGNORE_INDEX", "next_token_pairs", "next_token_targets"]


def next_token_targets(labels: Any) -> int:
    """How many next-token targets *labels* holds: the ones a causal loss averages over."""
    return int((labels[..., 1:] != IGNORE_INDEX).sum())


def next_token_pairs(logits: Any, labels: Any) -> tuple[list[int], list[int]]:
    """The predicted and actual next token at every valid target position.

    Returns:
        ``(predictions, references)``, flattened, in position order.
    """
    import torch

    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    labels = labels.to(logits.device)
    predicted = logits[..., :-1, :].argmax(dim=-1)
    target = labels[..., 1:]
    valid = target != IGNORE_INDEX
    return predicted[valid].cpu().tolist(), target[valid].cpu().tolist()
