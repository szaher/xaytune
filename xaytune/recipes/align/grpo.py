from __future__ import annotations

import torch


def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Normalize rewards to zero-mean unit-variance advantages."""
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)

    mean = rewards.mean()
    std = rewards.std()

    if std < 1e-8:
        return torch.zeros_like(rewards)

    return (rewards - mean) / (std + 1e-8)


def grpo_loss(
    *,
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None = None,
    advantages: torch.Tensor,
    kl_coeff: float = 0.04,
) -> torch.Tensor:
    """Compute Group Relative Policy Optimization loss (Shao et al., 2024)."""
    policy_loss = -(logprobs * advantages).mean()

    if ref_logprobs is not None and kl_coeff > 0:
        kl = (logprobs - ref_logprobs).mean()
        return policy_loss + kl_coeff * kl

    return policy_loss
