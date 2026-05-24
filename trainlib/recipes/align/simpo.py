from __future__ import annotations

import torch
import torch.nn.functional as F


def simpo_loss(
    *,
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Compute Simple Preference Optimization loss (Meng et al., 2024)."""
    chosen_avg = policy_chosen_logps / chosen_lengths.float()
    rejected_avg = policy_rejected_logps / rejected_lengths.float()

    logits = beta * (chosen_avg - rejected_avg) - gamma

    return -F.logsigmoid(logits).mean()
