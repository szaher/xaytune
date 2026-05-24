import torch

from xaytune.recipes.align.reward_scoring import (
    compute_advantages_from_rewards,
    score_completions,
)
from xaytune.recipes.align.rewards import register_reward


class TestScoreCompletions:
    def test_returns_tensor(self):
        scores = score_completions(
            prompts=["hello"],
            responses=["world"],
            reward_name="default",
        )
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (1,)

    def test_default_reward_is_zero(self):
        scores = score_completions(
            prompts=["a", "b"],
            responses=["x", "y"],
            reward_name="default",
        )
        assert torch.allclose(scores, torch.zeros(2))

    def test_length_penalty_reward(self):
        scores = score_completions(
            prompts=["p"],
            responses=["short"],
            reward_name="length_penalty",
            reward_kwargs={"target_length": 5, "penalty_scale": 1.0},
        )
        assert scores[0].item() == 0.0

        scores_long = score_completions(
            prompts=["p"],
            responses=["this is longer than 5"],
            reward_name="length_penalty",
            reward_kwargs={"target_length": 5, "penalty_scale": 0.1},
        )
        assert scores_long[0].item() < 0.0

    def test_format_check_reward(self):
        scores = score_completions(
            prompts=["p"],
            responses=["## Title\n```code```"],
            reward_name="format_check",
            reward_kwargs={"required_markers": ["##", "```"]},
        )
        assert scores[0].item() == 1.0

    def test_custom_reward(self):
        @register_reward("_test_constant")
        def _const(prompt, response):
            return 42.0

        scores = score_completions(
            prompts=["a"],
            responses=["b"],
            reward_name="_test_constant",
        )
        assert scores[0].item() == 42.0

    def test_batch_scoring(self):
        scores = score_completions(
            prompts=["a", "b", "c"],
            responses=["x", "y", "z"],
            reward_name="default",
        )
        assert scores.shape == (3,)


class TestComputeAdvantagesFromRewards:
    def test_single_sample_normalization(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        advantages = compute_advantages_from_rewards(rewards, group_size=1)
        assert advantages.shape == (4,)
        assert abs(advantages.mean().item()) < 1e-5

    def test_group_normalization(self):
        rewards = torch.tensor([1.0, 3.0, 5.0, 7.0, 2.0, 4.0])
        advantages = compute_advantages_from_rewards(rewards, group_size=3)
        assert advantages.shape == (6,)
        # Each group of 3 should be zero-mean
        group1 = advantages[:3]
        group2 = advantages[3:]
        assert abs(group1.mean().item()) < 1e-5
        assert abs(group2.mean().item()) < 1e-5

    def test_equal_rewards_give_zero_advantages(self):
        rewards = torch.tensor([5.0, 5.0, 5.0])
        advantages = compute_advantages_from_rewards(rewards, group_size=3)
        assert torch.allclose(advantages, torch.zeros(3))

    def test_single_element_gives_zero(self):
        rewards = torch.tensor([3.0])
        advantages = compute_advantages_from_rewards(rewards, group_size=1)
        assert advantages.item() == 0.0
