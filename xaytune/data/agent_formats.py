from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from xaytune.data.registry import format_registry


@dataclass
class AgentMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    trainable: bool


def _serialize_tool_call(tool_call: dict[str, Any]) -> str:
    func = tool_call.get("function", tool_call)
    name = func.get("name", "")
    args = func.get("arguments", "{}")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    call_obj = {"name": name, "arguments": args}
    return f"<tool_call>\n{json.dumps(call_obj, ensure_ascii=False)}\n</tool_call>"


def _format_tools_schema(tools: list[dict[str, Any]]) -> str:
    lines = ["Available tools:"]
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        lines.append(f"\n{name}: {desc}")
        lines.append(f"  Parameters: {json.dumps(params, ensure_ascii=False)}")
    return "\n".join(lines)


@format_registry.register("function_calling")
def format_function_calling(sample: dict[str, Any]) -> list[AgentMessage]:
    messages_raw = sample.get("messages", [])
    tools = sample.get("tools", [])
    result: list[AgentMessage] = []

    if tools:
        schema_text = _format_tools_schema(tools)
        result.append(AgentMessage(role="system", content=schema_text, trainable=False))

    for msg in messages_raw:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])

        if role == "assistant":
            if tool_calls:
                parts = []
                if content:
                    parts.append(content)
                for tc in tool_calls:
                    parts.append(_serialize_tool_call(tc))
                result.append(
                    AgentMessage(
                        role="assistant",
                        content="\n".join(parts),
                        trainable=True,
                    )
                )
            else:
                result.append(
                    AgentMessage(
                        role="assistant",
                        content=content,
                        trainable=True,
                    )
                )
        elif role == "tool":
            result.append(
                AgentMessage(
                    role="tool",
                    content=f"<tool_result>\n{content}\n</tool_result>",
                    trainable=False,
                )
            )
        else:
            result.append(
                AgentMessage(
                    role=role,
                    content=content,
                    trainable=False,
                )
            )

    return result


@format_registry.register("react")
def format_react(sample: dict[str, Any]) -> list[AgentMessage]:
    task = sample.get("task", "")
    steps = sample.get("steps", [])
    result: list[AgentMessage] = []

    result.append(AgentMessage(role="user", content=task, trainable=False))

    for step in steps:
        thought = step.get("thought", "")
        action = step.get("action", "")
        action_input = step.get("action_input", "")
        observation = step.get("observation", None)

        assistant_text = f"Thought: {thought}\nAction: {action}\nAction Input: {action_input}"
        result.append(
            AgentMessage(
                role="assistant",
                content=assistant_text,
                trainable=True,
            )
        )

        if observation is not None:
            result.append(
                AgentMessage(
                    role="tool",
                    content=f"Observation: {observation}",
                    trainable=False,
                )
            )

    return result


@format_registry.register("trajectory")
def format_trajectory(sample: dict[str, Any]) -> list[AgentMessage]:
    system = sample.get("system", None)
    goal = sample.get("goal", "")
    turns = sample.get("turns", [])
    result: list[AgentMessage] = []

    if system:
        result.append(AgentMessage(role="system", content=system, trainable=False))

    result.append(AgentMessage(role="user", content=goal, trainable=False))

    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content", "") or ""
        tool_calls = turn.get("tool_calls", [])

        if role == "assistant":
            if tool_calls:
                parts = []
                if content:
                    parts.append(content)
                for tc in tool_calls:
                    parts.append(_serialize_tool_call(tc))
                result.append(
                    AgentMessage(
                        role="assistant",
                        content="\n".join(parts),
                        trainable=True,
                    )
                )
            else:
                result.append(
                    AgentMessage(
                        role="assistant",
                        content=content,
                        trainable=True,
                    )
                )
        elif role == "tool":
            result.append(
                AgentMessage(
                    role="tool",
                    content=f"<tool_result>\n{content}\n</tool_result>",
                    trainable=False,
                )
            )

    return result
