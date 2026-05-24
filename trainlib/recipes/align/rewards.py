from __future__ import annotations

from trainlib.utils.registry import Registry

reward_registry = Registry("reward")

register_reward = reward_registry.register


@register_reward("default")
def default_reward(prompt: str, response: str) -> float:
    """Baseline reward that always returns 0."""
    return 0.0


@register_reward("length_penalty")
def length_penalty_reward(
    prompt: str,
    response: str,
    *,
    target_length: int = 200,
    penalty_scale: float = 0.001,
) -> float:
    """Penalize responses that deviate from *target_length* characters."""
    diff = abs(len(response) - target_length)
    return -penalty_scale * diff


@register_reward("format_check")
def format_check_reward(
    prompt: str,
    response: str,
    *,
    required_markers: list[str] | None = None,
) -> float:
    """Reward based on the fraction of *required_markers* present in the response."""
    if required_markers is None:
        required_markers = []
    if not required_markers:
        return 0.0
    matched = sum(1 for m in required_markers if m in response)
    return matched / len(required_markers)


@register_reward("composite")
def composite_reward(
    prompt: str,
    response: str,
    *,
    reward_names: list[str] | None = None,
    weights: list[float] | None = None,
) -> float:
    """Weighted combination of multiple registered reward functions."""
    if not reward_names:
        return 0.0
    if weights is None:
        weights = [1.0] * len(reward_names)
    total = 0.0
    for name, weight in zip(reward_names, weights):
        fn = reward_registry.get(name)
        total += weight * fn(prompt, response)
    return total
