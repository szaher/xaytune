import xaytune.recipes.align.agent_rewards  # register agent rewards  # noqa: F401
from xaytune.recipes.align.align import align
from xaytune.recipes.align.dpo import dpo_loss
from xaytune.recipes.align.generation import GenerationResult, generate_completions
from xaytune.recipes.align.grpo import compute_group_advantages, grpo_loss
from xaytune.recipes.align.logprobs import (
    get_model_logps,
    get_per_token_logps,
    get_sequence_logps,
)
from xaytune.recipes.align.loss_dispatch import (
    create_alignment_loss_fn,
    is_alignment_method,
)
from xaytune.recipes.align.online_step import OnlineRLStep
from xaytune.recipes.align.orpo import orpo_loss
from xaytune.recipes.align.ppo import ppo_clip_loss, ppo_value_loss, reinforce_loss
from xaytune.recipes.align.reward_scoring import (
    compute_advantages_from_rewards,
    score_completions,
)
from xaytune.recipes.align.rewards import register_reward, reward_registry
from xaytune.recipes.align.simpo import simpo_loss

__all__ = [
    "GenerationResult",
    "OnlineRLStep",
    "align",
    "compute_advantages_from_rewards",
    "compute_group_advantages",
    "create_alignment_loss_fn",
    "dpo_loss",
    "generate_completions",
    "get_model_logps",
    "get_per_token_logps",
    "get_sequence_logps",
    "grpo_loss",
    "is_alignment_method",
    "orpo_loss",
    "ppo_clip_loss",
    "ppo_value_loss",
    "register_reward",
    "reinforce_loss",
    "reward_registry",
    "score_completions",
    "simpo_loss",
]
