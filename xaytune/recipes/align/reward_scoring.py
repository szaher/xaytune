from __future__ import annotations

from typing import Any

import torch

from xaytune.recipes.align.grpo import compute_group_advantages
from xaytune.recipes.align.rewards import reward_registry


def score_completions(
    prompts: list[str],
    responses: list[str],
    reward_name: str = "default",
    reward_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Score prompt-response pairs using a registered reward function.

    Returns a 1-D float tensor of shape ``(len(prompts),)``.
    """
    fn = reward_registry.get(reward_name)
    kwargs = reward_kwargs or {}
    scores = [fn(p, r, **kwargs) for p, r in zip(prompts, responses)]
    return torch.tensor(scores, dtype=torch.float32)


def compute_advantages_from_rewards(
    rewards: torch.Tensor,
    group_size: int = 1,
) -> torch.Tensor:
    """Convert raw rewards to normalized advantages.

    For GRPO (``group_size > 1``), normalizes within each group using
    :func:`compute_group_advantages`. For single-sample methods
    (PPO, REINFORCE), normalizes across the batch.
    """
    if group_size > 1:
        n_prompts = rewards.shape[0] // group_size
        grouped = rewards.view(n_prompts, group_size)
        advantages = torch.stack([compute_group_advantages(g) for g in grouped])
        return advantages.view(-1)

    return compute_group_advantages(rewards)
