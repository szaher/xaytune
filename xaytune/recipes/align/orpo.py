from __future__ import annotations

import torch
import torch.nn.functional as F


def orpo_loss(
    *,
    sft_loss: torch.Tensor,
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    lambda_weight: float = 1.0,
) -> torch.Tensor:
    """Compute Odds Ratio Preference Optimization loss (Hong et al., 2024)."""
    log_odds_ratio = (policy_chosen_logps - policy_rejected_logps) - (
        torch.log1p(-policy_chosen_logps.exp()) - torch.log1p(-policy_rejected_logps.exp())
    )

    or_loss = -F.logsigmoid(log_odds_ratio).mean()

    return sft_loss + lambda_weight * or_loss
