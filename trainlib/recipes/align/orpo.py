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
    chosen_odds = policy_chosen_logps.exp() / (1 - policy_chosen_logps.exp())
    rejected_odds = policy_rejected_logps.exp() / (1 - policy_rejected_logps.exp())

    log_odds_ratio = torch.log(chosen_odds / rejected_odds)

    or_loss = -F.logsigmoid(log_odds_ratio).mean()

    return sft_loss + lambda_weight * or_loss
