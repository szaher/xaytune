"""Tests for 7e (synthetic agent data generation) and 7f (multi-agent conversations)."""

from __future__ import annotations

import importlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xaytune.data.agent_formats import (
    AgentMessage,
    format_multi_agent,
)
from xaytune.data.agent_tokenizer import tokenize_agent_dataset

_gen_module = importlib.import_module("xaytune.data.prep.generate")
generate = _gen_module.generate


# ---------------------------------------------------------------------------
# 7f: Multi-Agent Format
# ---------------------------------------------------------------------------


class TestMultiAgentFormat:
    def test_basic(self):
        sample = {
            "goal": "Research and write about AI",
            "turns": [
                {"agent": "researcher", "role": "assistant", "content": "I'll search for info."},
                {
                    "agent": "writer",
                    "role": "assistant",
                    "content": "Based on the research, here's a summary.",
                },
            ],
        }
        messages = format_multi_agent(sample)
        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[0].trainable is False
        assert messages[1].role == "assistant"
        assert messages[1].name == "researcher"
        assert messages[1].trainable is True
        assert messages[2].name == "writer"
        assert messages[2].trainable is True

    def test_with_system(self):
        sample = {
            "system": "You are coordinating agents.",
            "goal": "Do the task",
            "turns": [
                {"agent": "agent1", "role": "assistant", "content": "Done."},
            ],
        }
        messages = format_multi_agent(sample)
        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[0].trainable is False

    def test_with_tool_calls(self):
        sample = {
            "goal": "Search and summarize",
            "turns": [
                {
                    "agent": "searcher",
                    "role": "assistant",
                    "content": "Searching...",
                    "tool_calls": [{"name": "search", "arguments": {"q": "AI"}}],
                },
                {"role": "tool", "content": "Found results."},
                {"agent": "summarizer", "role": "assistant", "content": "Here's the summary."},
            ],
        }
        messages = format_multi_agent(sample)
        assert len(messages) == 4
        assert messages[1].name == "searcher"
        assert "<tool_call>" in messages[1].content
        assert messages[2].role == "tool"
        assert messages[2].trainable is False
        assert messages[3].name == "summarizer"

    def test_tool_messages_have_no_name(self):
        sample = {
            "goal": "Do it",
            "turns": [
                {"agent": "a1", "role": "assistant", "content": "Calling tool."},
                {"role": "tool", "content": "Result."},
            ],
        }
        messages = format_multi_agent(sample)
        assert messages[2].name is None

    def test_registered(self):
        from xaytune.data.registry import format_registry

        assert format_registry.has("multi_agent")

    def test_tokenize_multi_agent(self):
        sample = {
            "goal": "Collaborate",
            "turns": [
                {"agent": "a1", "role": "assistant", "content": "Step one done"},
                {"agent": "a2", "role": "assistant", "content": "Step two done"},
            ],
        }
        messages = format_multi_agent(sample)
        tok = MagicMock()
        tok.model_max_length = 512

        def tokenize(text, **kwargs):
            return {"input_ids": list(range(len(text.split())))}

        tok.side_effect = tokenize
        result = tokenize_agent_dataset([messages], tok)
        assert len(result) == 1
        assert any(v != -100 for v in result[0]["labels"])
        assert any(v == -100 for v in result[0]["labels"])


class TestAgentMessageName:
    def test_name_default_none(self):
        msg = AgentMessage(role="assistant", content="hi", trainable=True)
        assert msg.name is None

    def test_name_set(self):
        msg = AgentMessage(role="assistant", content="hi", trainable=True, name="agent1")
        assert msg.name == "agent1"

    def test_existing_formats_have_no_name(self):
        from xaytune.data.agent_formats import format_function_calling

        sample = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        }
        messages = format_function_calling(sample)
        for msg in messages:
            assert msg.name is None


# ---------------------------------------------------------------------------
# 7e: Synthetic Agent Data Generation
# ---------------------------------------------------------------------------


class TestAgentDistill:
    @patch.object(_gen_module, "_get_client", return_value=MagicMock())
    @patch.object(_gen_module, "_call_llm_sync")
    def test_agent_distill_basic(self, mock_call, _mock_client):
        mock_call.return_value = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Search for cats"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"q": "cats"}'},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "Found cats"},
                    {"role": "assistant", "content": "I found information about cats."},
                ]
            }
        )
        result = generate(
            mode="agent_distill",
            topic="searching for animals",
            n=2,
            format="function_calling",
            model="test-model",
            api_key="test-key",
        )
        assert len(result.dataset) == 2
        assert mock_call.call_count == 2
        assert "messages" in result.dataset[0]

    @patch.object(_gen_module, "_get_client", return_value=MagicMock())
    @patch.object(_gen_module, "_call_llm_sync")
    def test_agent_distill_with_tools(self, mock_call, _mock_client):
        mock_call.return_value = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "It's sunny."},
                ]
            }
        )
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            }
        ]
        result = generate(
            mode="agent_distill",
            topic="weather queries",
            tools=tools,
            n=1,
            format="function_calling",
            model="test",
            api_key="k",
        )
        assert len(result.dataset) == 1
        prompt_used = mock_call.call_args[0][3]
        assert "get_weather" in prompt_used

    def test_agent_distill_requires_topic(self):
        with pytest.raises(ValueError, match="topic"):
            generate(
                mode="agent_distill",
                n=1,
                format="function_calling",
                model="test",
                api_key="k",
            )


class TestAgentAugment:
    @patch.object(_gen_module, "_get_client", return_value=MagicMock())
    @patch.object(_gen_module, "_call_llm_sync")
    def test_agent_augment_basic(self, mock_call, _mock_client):
        mock_call.return_value = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "New question"},
                    {"role": "assistant", "content": "New answer"},
                ]
            }
        )
        seeds = [
            {
                "messages": [
                    {"role": "user", "content": "Original question"},
                    {"role": "assistant", "content": "Original answer"},
                ]
            }
        ]
        result = generate(
            mode="agent_augment",
            seed=seeds,
            n=3,
            format="function_calling",
            model="test",
            api_key="k",
        )
        assert len(result.dataset) == 3
        assert mock_call.call_count == 3

    def test_agent_augment_requires_seed(self):
        with pytest.raises(ValueError, match="seed"):
            generate(
                mode="agent_augment",
                n=1,
                format="function_calling",
                model="test",
                api_key="k",
            )

    @patch.object(_gen_module, "_get_client", return_value=MagicMock())
    @patch.object(_gen_module, "_call_llm_sync")
    def test_agent_augment_from_file(self, mock_call, _mock_client):
        mock_call.return_value = json.dumps(
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
        )
        seed_data = {
            "messages": [
                {"role": "user", "content": "Original"},
                {"role": "assistant", "content": "Response"},
            ]
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seeds.jsonl"
            path.write_text(json.dumps(seed_data) + "\n")
            result = generate(
                mode="agent_augment",
                seed=str(path),
                n=1,
                format="function_calling",
                model="test",
                api_key="k",
            )
            assert len(result.dataset) == 1


class TestAgentDistillReport:
    @patch.object(_gen_module, "_get_client", return_value=MagicMock())
    @patch.object(_gen_module, "_call_llm_sync")
    def test_report_counts(self, mock_call, _mock_client):
        mock_call.return_value = json.dumps(
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
        )
        result = generate(
            mode="agent_distill",
            topic="test",
            n=3,
            format="function_calling",
            model="test",
            api_key="k",
        )
        assert result.report.input_rows == 0
        assert result.report.output_rows == 3
        assert result.report.steps[0].details["mode"] == "agent_distill"


class TestMultiAgentFromFile:
    def test_load_multi_agent_jsonl(self):
        from xaytune.data.loader import load_dataset

        sample = {
            "goal": "Collaborate",
            "turns": [
                {"agent": "a1", "role": "assistant", "content": "Step one"},
                {"agent": "a2", "role": "assistant", "content": "Step two"},
            ],
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            path.write_text(json.dumps(sample) + "\n")
            data = load_dataset(str(path), format="multi_agent")
            assert len(data) == 1
            assert isinstance(data[0], list)
            assert isinstance(data[0][0], AgentMessage)
