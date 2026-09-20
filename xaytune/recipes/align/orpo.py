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
    """Compute Odds Ratio Preference Optimization loss (Hong et al., 2024).

    The odds of a sequence are ``p / (1 - p)``, so the loss degenerates as ``p``
    approaches 1: ``log1p(-1.0)`` is ``-inf`` and the difference of two
    infinities is ``NaN``, which then poisons every subsequent step. ``p == 1``
    is reachable in practice, not just in theory -- a fully-masked sequence has
    log-probabilities summing to exactly 0. The probabilities are therefore
    clamped just below 1.

    The clamp is a no-op for realistic inputs: a sequence log-probability of
    even -10 gives ``p`` around 4.5e-5, nowhere near the bound.
    """
    eps = torch.finfo(policy_chosen_logps.dtype).eps
    chosen_prob = policy_chosen_logps.exp().clamp(max=1.0 - eps)
    rejected_prob = policy_rejected_logps.exp().clamp(max=1.0 - eps)

    log_odds_ratio = (policy_chosen_logps - policy_rejected_logps) - (
        torch.log1p(-chosen_prob) - torch.log1p(-rejected_prob)
    )

    or_loss = -F.logsigmoid(log_odds_ratio).mean()

    return sft_loss + lambda_weight * or_loss
