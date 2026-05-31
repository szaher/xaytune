from __future__ import annotations

from collections.abc import Callable
from typing import Any

from xaytune.eval.metrics import register_metric
from xaytune.recipes.align.agent_rewards import (
    efficiency_reward,
    parse_tool_calls,
    task_completion_reward,
    tool_use_quality_reward,
)


@register_metric("tool_use_accuracy")
def compute_tool_use_accuracy(
    responses: list[dict[str, str]],
    *args: Any,
    expected_tools: list[str] | None = None,
    required_args: dict[str, list[str]] | None = None,
    parser: Callable | None = None,
    **kwargs: Any,
) -> float:
    if not responses:
        return 0.0
    scores = []
    for item in responses:
        score = tool_use_quality_reward(
            item.get("prompt", ""),
            item.get("response", ""),
            expected_tools=expected_tools,
            required_args=required_args,
            parser=parser,
        )
        scores.append(score)
    return sum(scores) / len(scores)


@register_metric("task_success_rate")
def compute_task_success_rate(
    responses: list[dict[str, str]],
    *args: Any,
    success_markers: list[str] | None = None,
    failure_markers: list[str] | None = None,
    parser: Callable | None = None,
    **kwargs: Any,
) -> float:
    if not responses:
        return 0.0
    successes = 0
    for item in responses:
        score = task_completion_reward(
            item.get("prompt", ""),
            item.get("response", ""),
            success_markers=success_markers,
            failure_markers=failure_markers,
            parser=parser,
        )
        if score >= 0.5:
            successes += 1
    return successes / len(responses)


@register_metric("step_efficiency")
def compute_step_efficiency(
    responses: list[dict[str, str]],
    *args: Any,
    max_steps: int = 10,
    optimal_steps: int | None = None,
    parser: Callable | None = None,
    **kwargs: Any,
) -> float:
    if not responses:
        return 0.0
    scores = []
    for item in responses:
        score = efficiency_reward(
            item.get("prompt", ""),
            item.get("response", ""),
            max_steps=max_steps,
            optimal_steps=optimal_steps,
            parser=parser,
        )
        scores.append(score)
    return sum(scores) / len(scores)


@register_metric("error_recovery_rate")
def compute_error_recovery_rate(
    responses: list[dict[str, str]],
    *args: Any,
    error_indicators: list[str] | None = None,
    parser: Callable | None = None,
    **kwargs: Any,
) -> float:
    if not responses:
        return 0.0
    if error_indicators is None:
        error_indicators = ["error", "Error", "ERROR", "failed", "Failed", "exception"]

    recovered = 0
    total_with_errors = 0

    for item in responses:
        response = item.get("response", "")
        has_error = any(indicator in response for indicator in error_indicators)
        if not has_error:
            continue
        total_with_errors += 1
        calls = parse_tool_calls(response, parser=parser)
        if not calls:
            continue
        last_error_pos = max(response.rfind(ind) for ind in error_indicators if ind in response)
        last_call_pos = response.rfind("</tool_call>")
        has_final_answer = response.rfind("</tool_result>")
        text_after = (
            response[has_final_answer + len("</tool_result>") :].strip()
            if has_final_answer != -1
            else ""
        )

        if last_call_pos > last_error_pos or text_after:
            recovered += 1

    if total_with_errors == 0:
        return 1.0
    return recovered / total_with_errors


def evaluate_agent(
    responses: list[dict[str, str]],
    *,
    expected_tools: list[str] | None = None,
    required_args: dict[str, list[str]] | None = None,
    success_markers: list[str] | None = None,
    failure_markers: list[str] | None = None,
    max_steps: int = 10,
    optimal_steps: int | None = None,
    error_indicators: list[str] | None = None,
    parser: Callable | None = None,
) -> dict[str, float]:
    return {
        "tool_use_accuracy": compute_tool_use_accuracy(
            responses,
            expected_tools=expected_tools,
            required_args=required_args,
            parser=parser,
        ),
        "task_success_rate": compute_task_success_rate(
            responses,
            success_markers=success_markers,
            failure_markers=failure_markers,
            parser=parser,
        ),
        "step_efficiency": compute_step_efficiency(
            responses,
            max_steps=max_steps,
            optimal_steps=optimal_steps,
            parser=parser,
        ),
        "error_recovery_rate": compute_error_recovery_rate(
            responses,
            error_indicators=error_indicators,
            parser=parser,
        ),
    }
