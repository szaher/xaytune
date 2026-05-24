from __future__ import annotations

import torch


def ppo_clip_loss(
    *,
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Compute PPO clipped surrogate objective (Schulman et al., 2017)."""
    ratio = torch.exp(logprobs - old_logprobs)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    return -torch.min(unclipped, clipped).mean()


def ppo_value_loss(
    *,
    values: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """Compute PPO value function MSE loss."""
    return (values - returns).pow(2).mean()


def reinforce_loss(
    *,
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Compute REINFORCE policy gradient loss."""
    return -(logprobs * advantages).mean()
