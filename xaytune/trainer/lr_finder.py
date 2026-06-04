from __future__ import annotations

import copy
import dataclasses
from itertools import islice
from typing import Any

import torch


@dataclasses.dataclass
class LRFinderResult:
    """Result of an LR range test.

    Attributes:
        lrs: Learning rates tested.
        losses: Raw loss values at each LR.
        suggested_lr: Recommended LR (steepest descent point).
    """

    lrs: list[float]
    losses: list[float]
    suggested_lr: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lrs": self.lrs,
            "losses": self.losses,
            "suggested_lr": self.suggested_lr,
        }


def _cycle(dataloader: Any):
    while True:
        yield from dataloader


def _suggest_lr(lrs: list[float], smooth_losses: list[float]) -> float | None:
    if len(lrs) < 3:
        return lrs[0] if lrs else None
    grads = [smooth_losses[i + 1] - smooth_losses[i] for i in range(len(smooth_losses) - 1)]
    min_idx = min(range(len(grads)), key=lambda i: grads[i])
    safe_idx = max(min_idx - 1, 0)
    return lrs[safe_idx]


def lr_find(
    model: Any,
    train_dataloader: Any,
    *,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    num_iterations: int = 100,
    smoothing_factor: float = 0.05,
    divergence_threshold: float = 4.0,
    loss_fn: Any | None = None,
) -> LRFinderResult:
    """Run an LR range test to find the optimal learning rate.

    Trains with exponentially increasing LR from *start_lr* to *end_lr*,
    tracking loss.  Stops early if loss diverges.  Model weights are
    restored to their original state afterward.

    Args:
        model: The model to test.
        train_dataloader: Training data loader.
        start_lr: Starting learning rate.
        end_lr: Maximum learning rate to test.
        num_iterations: Number of training iterations.
        smoothing_factor: Exponential smoothing for loss curve.
        divergence_threshold: Stop when smoothed loss exceeds this
            multiple of the best smoothed loss.
        loss_fn: Optional custom loss function ``(model, batch, outputs) -> loss``.

    Returns:
        :class:`LRFinderResult` with tested LRs, losses, and a suggestion.
    """
    if not train_dataloader:
        raise ValueError("train_dataloader must not be empty")

    saved_state = copy.deepcopy(model.state_dict())

    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1.0 / num_iterations)

    lrs: list[float] = []
    losses: list[float] = []
    smooth_losses: list[float] = []
    best_smooth = float("inf")

    batches = islice(_cycle(train_dataloader), num_iterations)

    device = next(model.parameters()).device

    for batch in batches:
        current_lr = optimizer.param_groups[0]["lr"]

        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
        else:
            outputs = model(batch)

        if loss_fn is not None:
            loss = loss_fn(model, batch, outputs)
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs

        raw_loss = loss.item()
        if smooth_losses:
            smooth = smoothing_factor * raw_loss + (1 - smoothing_factor) * smooth_losses[-1]
        else:
            smooth = raw_loss

        lrs.append(current_lr)
        losses.append(raw_loss)
        smooth_losses.append(smooth)

        if smooth < best_smooth:
            best_smooth = smooth

        if best_smooth > 0 and smooth > divergence_threshold * best_smooth:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for pg in optimizer.param_groups:
            pg["lr"] *= lr_mult

    model.load_state_dict(saved_state)

    suggested = _suggest_lr(lrs, smooth_losses)
    return LRFinderResult(lrs=lrs, losses=losses, suggested_lr=suggested)
