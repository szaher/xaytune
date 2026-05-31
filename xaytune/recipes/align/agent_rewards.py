from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from xaytune.recipes.align.rewards import register_reward

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESULT_PATTERN = re.compile(r"<tool_result>\s*(.*?)\s*</tool_result>", re.DOTALL)


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]


def parse_tool_calls(
    response: str,
    parser: Callable[[str], list[ParsedToolCall]] | None = None,
) -> list[ParsedToolCall]:
    if parser is not None:
        return parser(response)

    calls: list[ParsedToolCall] = []
    for match in _TOOL_CALL_PATTERN.finditer(response):
        try:
            data = json.loads(match.group(1))
            calls.append(
                ParsedToolCall(
                    name=data.get("name", ""),
                    arguments=data.get("arguments", {}),
                )
            )
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
    calls = parse_tool_calls(response, parser=parser)

    if not calls:
        return 0.0

    if expected_tools is None and required_args is None:
        return 1.0

    score_parts: list[float] = []
    total_parts = 0

    if expected_tools is not None:
        called_names = {c.name for c in calls}
        matched = sum(1 for t in expected_tools if t in called_names)
        total_parts += len(expected_tools)
        score_parts.append(float(matched))

    if required_args is not None:
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
