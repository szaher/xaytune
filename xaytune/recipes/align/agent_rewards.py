from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


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

    tool_names = [call.name for call in calls]
    correct_tools = sum(1 for name in tool_names if name in expected_tools)
    tool_score = correct_tools / len(calls)

    if not required_args:
        return tool_score

    arg_scores = []
    for call in calls:
        if call.name not in required_args:
            continue
        required = set(required_args[call.name])
        provided = set(call.arguments.keys())
        if required:
            arg_scores.append(len(required & provided) / len(required))

    if not arg_scores:
        return tool_score

    return (tool_score + sum(arg_scores) / len(arg_scores)) / 2.0


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
    if success_markers is None:
        success_markers = ["done", "Done", "completed", "Completed", "finished", "Finished"]
    if failure_markers is None:
        failure_markers = ["error", "Error", "failed", "Failed", "cannot", "unable"]

    has_success = any(marker in response for marker in success_markers)
    has_failure = any(marker in response for marker in failure_markers)

    if has_success and not has_failure:
        return 1.0
    if has_failure and not has_success:
        return 0.0
    if has_success and has_failure:
        return 0.5
    return 0.0


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
        return 1.0

    if optimal_steps is not None:
        if num_calls <= optimal_steps:
            return 1.0
        if num_calls >= max_steps:
            return 0.0
        excess = num_calls - optimal_steps
        max_excess = max_steps - optimal_steps
        return 1.0 - (excess / max_excess)

    if num_calls >= max_steps:
        return 0.0

    return 1.0 - (num_calls / max_steps)
