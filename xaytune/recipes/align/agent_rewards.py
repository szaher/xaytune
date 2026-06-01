from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from xaytune.recipes.align.rewards import register_reward


@dataclass
class ParsedToolCall:
    """Represents a parsed tool call from agent output."""

    name: str
    arguments: dict[str, Any]


def parse_tool_calls(
    text: str,
    parser: Callable[[str], list[ParsedToolCall]] | None = None,
) -> list[ParsedToolCall]:
    """Parse tool calls from text containing <tool_call> tags.

    Args:
        text: Text potentially containing tool calls
        parser: Optional custom parser function

    Returns:
        List of ParsedToolCall objects
    """
    if parser is not None:
        return parser(text)

    calls: list[ParsedToolCall] = []
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"

    for match in re.finditer(pattern, text, re.DOTALL):
        try:
            data = json.loads(match.group(1))
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            calls.append(ParsedToolCall(name=name, arguments=arguments))
        except (json.JSONDecodeError, AttributeError):
            continue

    return calls


@register_reward("tool_use_quality")
def tool_use_quality_reward(
    prompt: str,
    response: str,
    *,
    expected_tools: list[str] | None = None,
    required_args: dict[str, list[str]] | None = None,
    parser: Callable | None = None,
) -> float:
    """Reward based on using the expected tools with required arguments.

    Args:
        prompt: The input prompt
        response: The agent's response
        expected_tools: List of tool names that should be used
        required_args: Dict mapping tool names to lists of required argument names
        parser: Optional custom parser

    Returns:
        Score from 0.0 to 1.0
    """
    calls = parse_tool_calls(response, parser=parser)

    if not calls:
        return 1.0 if not expected_tools else 0.0

    if not expected_tools:
        return 1.0

    called_names = {call.name for call in calls}
    matched = sum(1 for t in expected_tools if t in called_names)
    score_parts = [float(matched)]
    total_parts = len(expected_tools)

    if required_args:
        for call in calls:
            if call.name in required_args:
                required = required_args[call.name]
                present = sum(1 for a in required if a in call.arguments)
                total_parts += len(required)
                score_parts.append(float(present))

    if total_parts == 0:
        return 1.0

    return sum(score_parts) / total_parts


@register_reward("task_completion")
def task_completion_reward(
    prompt: str,
    response: str,
    *,
    success_markers: list[str] | None = None,
    failure_markers: list[str] | None = None,
    parser: Callable | None = None,
) -> float:
    """Reward based on task completion indicators.

    Args:
        prompt: The input prompt
        response: The agent's response
        success_markers: Phrases indicating successful completion
        failure_markers: Phrases indicating failure
        parser: Optional custom parser (not used here)

    Returns:
        Score from 0.0 to 1.0
    """
    if failure_markers:
        for marker in failure_markers:
            if marker.lower() in response.lower():
                return 0.0

    if success_markers:
        matched = sum(1 for m in success_markers if m.lower() in response.lower())
        return matched / len(success_markers)

    last_result = response.rfind("</tool_result>")
    if last_result == -1:
        return 1.0 if response.strip() else 0.0

    after_tools = response[last_result + len("</tool_result>") :].strip()
    return 1.0 if after_tools else 0.0


@register_reward("efficiency")
def efficiency_reward(
    prompt: str,
    response: str,
    *,
    max_steps: int = 10,
    optimal_steps: int | None = None,
    parser: Callable | None = None,
) -> float:
    """Reward based on efficiency (fewer tool calls is better).

    Args:
        prompt: The input prompt
        response: The agent's response
        max_steps: Maximum acceptable number of tool calls
        optimal_steps: Optimal number of tool calls (if known)
        parser: Optional custom parser

    Returns:
        Score from 0.0 to 1.0
    """
    calls = parse_tool_calls(response, parser=parser)
    num_calls = len(calls)

    if num_calls == 0:
        return 1.0 if response.strip() else 0.0

    if optimal_steps is not None:
        diff = abs(num_calls - optimal_steps)
        return max(0.0, 1.0 - diff / max_steps)

    return max(0.0, 1.0 - num_calls / max_steps)


@register_reward("agent_composite")
def agent_composite_reward(
    prompt: str,
    response: str,
    *,
    quality_weight: float = 0.4,
    completion_weight: float = 0.4,
    efficiency_weight: float = 0.2,
    parser: Callable | None = None,
    expected_tools: list[str] | None = None,
    required_args: dict[str, list[str]] | None = None,
    success_markers: list[str] | None = None,
    failure_markers: list[str] | None = None,
    max_steps: int = 10,
    optimal_steps: int | None = None,
) -> float:
    """Weighted combination of tool_use_quality, task_completion, and efficiency."""
    quality = tool_use_quality_reward(
        prompt,
        response,
        expected_tools=expected_tools,
        required_args=required_args,
        parser=parser,
    )
    completion = task_completion_reward(
        prompt,
        response,
        success_markers=success_markers,
        failure_markers=failure_markers,
        parser=parser,
    )
    eff = efficiency_reward(
        prompt,
        response,
        max_steps=max_steps,
        optimal_steps=optimal_steps,
        parser=parser,
    )
    return quality_weight * quality + completion_weight * completion + efficiency_weight * eff
