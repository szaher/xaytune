from __future__ import annotations

from trainlib.utils.registry import Registry

reward_registry = Registry("reward")

register_reward = reward_registry.register


@register_reward("default")
def default_reward(prompt: str, response: str) -> float:
    return 0.0
