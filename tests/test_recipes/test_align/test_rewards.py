import pytest

from xaytune.recipes.align.rewards import (
    composite_reward,
    format_check_reward,
    length_penalty_reward,
    register_reward,
    reward_registry,
)


class TestRewardRegistry:
    def test_register_and_get(self):
        @register_reward("test-reward")
        def my_reward(prompt: str, response: str) -> float:
            return 1.0

        assert reward_registry.has("test-reward")
        fn = reward_registry.get("test-reward")
        assert fn("hello", "world") == 1.0

    def test_register_returns_original(self):
        @register_reward("identity-test")
        def my_fn(prompt: str, response: str) -> float:
            return 0.5

        assert my_fn("a", "b") == 0.5

    def test_unknown_reward_raises(self):
        with pytest.raises(KeyError, match="not found"):
            reward_registry.get("nonexistent-reward")

    def test_list_rewards(self):
        @register_reward("list-test-reward")
        def r(prompt: str, response: str) -> float:
            return 0.0

        rewards = reward_registry.list()
        assert "list-test-reward" in rewards

    def test_default_reward_registered(self):
        assert reward_registry.has("default")
        fn = reward_registry.get("default")
        assert fn("prompt", "response") == 0.0


class TestLengthPenaltyReward:
    def test_exact_target_returns_zero(self):
        response = "x" * 200
        assert length_penalty_reward("p", response, target_length=200) == 0.0

    def test_shorter_response_penalizes(self):
        reward = length_penalty_reward("p", "short", target_length=200)
        assert reward < 0.0

    def test_longer_response_penalizes(self):
        reward = length_penalty_reward("p", "x" * 400, target_length=200)
        assert reward < 0.0

    def test_penalty_scale(self):
        r1 = length_penalty_reward("p", "x" * 100, penalty_scale=0.001)
        r2 = length_penalty_reward("p", "x" * 100, penalty_scale=0.01)
        assert abs(r2) > abs(r1)

    def test_registered(self):
        assert reward_registry.has("length_penalty")


class TestFormatCheckReward:
    def test_all_markers_present(self):
        response = "## Step 1\n```python\ncode\n```"
        reward = format_check_reward("p", response, required_markers=["##", "```"])
        assert reward == 1.0

    def test_no_markers_present(self):
        reward = format_check_reward("p", "plain text", required_markers=["##", "```"])
        assert reward == 0.0

    def test_partial_markers(self):
        reward = format_check_reward("p", "## heading only", required_markers=["##", "```"])
        assert reward == 0.5

    def test_empty_markers_returns_zero(self):
        assert format_check_reward("p", "anything") == 0.0

    def test_registered(self):
        assert reward_registry.has("format_check")


class TestCompositeReward:
    def test_combines_rewards(self):
        reward = composite_reward(
            "p",
            "x" * 200,
            reward_names=["default", "length_penalty"],
            weights=[1.0, 1.0],
        )
        assert reward == 0.0

    def test_weights_applied(self):
        reward = composite_reward(
            "p",
            "short",
            reward_names=["length_penalty"],
            weights=[2.0],
        )
        single = length_penalty_reward("p", "short")
        assert abs(reward - 2.0 * single) < 1e-9

    def test_empty_names_returns_zero(self):
        assert composite_reward("p", "r") == 0.0

    def test_registered(self):
        assert reward_registry.has("composite")
