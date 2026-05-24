from __future__ import annotations

import math
from typing import Any

from torch.optim.lr_scheduler import LambdaLR

_VALID_TYPES = {"cosine", "linear", "constant", "constant_with_warmup"}


def resolve_warmup_steps(
    warmup_steps: int,
    warmup_ratio: float,
    total_steps: int,
) -> int:
    """Return the effective warmup step count from either an absolute count or ratio."""
    if warmup_steps > 0:
        return warmup_steps
    if warmup_ratio > 0.0:
        return int(warmup_ratio * total_steps)
    return 0


def create_scheduler(
    optimizer: Any,
    scheduler_type: str,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    """Create an LR scheduler with optional linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: ``"cosine"``, ``"linear"``, ``"constant"``,
            or ``"constant_with_warmup"``.
        total_steps: Total training steps (for decay calculation).
        warmup_steps: Number of linear warmup steps.

    Raises:
        ValueError: If *scheduler_type* is not recognized.
    """
    if scheduler_type not in _VALID_TYPES:
        raise ValueError(
            f"Unknown scheduler type '{scheduler_type}'. "
            f"Valid options: {', '.join(sorted(_VALID_TYPES))}"
        )

    if scheduler_type == "constant":

        def lr_lambda(current_step: int) -> float:
            return 1.0

    elif scheduler_type == "constant_with_warmup":

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return current_step / warmup_steps
            return 1.0

    elif scheduler_type == "linear":

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return current_step / warmup_steps
            decay_steps = max(total_steps - warmup_steps, 1)
            return max(0.0, 1.0 - (current_step - warmup_steps) / decay_steps)

    elif scheduler_type == "cosine":

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return current_step / warmup_steps
            decay_steps = max(total_steps - warmup_steps, 1)
            progress = (current_step - warmup_steps) / decay_steps
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
