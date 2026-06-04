"""Simplified clipped policy gradient losses (PPO-style).

NOTE: This module implements the clipped surrogate objective from PPO
(Schulman et al., 2017) but does NOT include the full PPO training
pipeline (rollout buffer, GAE advantage estimation, value model,
multiple optimization epochs). For a complete PPO implementation,
consider using TRL's PPOTrainer.
"""

from __future__ import annotations

import torch


def ppo_clip_loss(
    *,
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Compute the clipped surrogate policy gradient objective.

    This implements only the clipped loss term from PPO (Schulman et al., 2017).
    It does NOT include rollout buffers, GAE, value model training, or multiple
    optimization epochs. See module docstring for details.
    """
    ratio = torch.exp(logprobs - old_logprobs)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    return -torch.min(unclipped, clipped).mean()


def ppo_value_loss(
    *,
    values: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """Compute value function MSE loss (used alongside the clipped policy gradient)."""
    return (values - returns).pow(2).mean()


def reinforce_loss(
    *,
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Compute REINFORCE policy gradient loss."""
    return -(logprobs * advantages).mean()
