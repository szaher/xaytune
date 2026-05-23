import pytest

from trainlib.recipes.align.rewards import register_reward, reward_registry


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
