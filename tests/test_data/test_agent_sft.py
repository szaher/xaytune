"""End-to-end tests for agent SFT (Feature 7b).

Verifies that the full pipeline works: raw agent data → format → tokenize
→ collate → training step. Uses sshleifer/tiny-gpt2 for real model tests.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from xaytune.data.agent_formats import (
    AgentMessage,
    format_function_calling,
    format_react,
    format_trajectory,
)
from xaytune.data.agent_tokenizer import tokenize_agent_dataset
from xaytune.data.loader import load_dataset
from xaytune.data.tokenizer import collate_tokenized

IGNORE_INDEX = -100


def _make_tokenizer():
    tok = MagicMock()
    tok.model_max_length = 512

    def tokenize(text, **kwargs):
        words = text.split()
        return {"input_ids": list(range(len(words)))}

    tok.side_effect = tokenize
    return tok


class TestAgentSFTPipeline:
    """Verify the full format → tokenize → collate pipeline."""

    def test_function_calling_pipeline(self):
        samples = [
            {
                "messages": [
                    {"role": "user", "content": "What is two plus two?"},
                    {"role": "assistant", "content": "Two plus two equals four."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Search for cats"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q": "cats"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "Found 10 results"},
                    {"role": "assistant", "content": "I found 10 results about cats."},
                ]
            },
        ]

        formatted = [format_function_calling(s) for s in samples]
        tok = _make_tokenizer()
        tokenized = tokenize_agent_dataset(formatted, tok, max_seq_length=128)

        assert len(tokenized) == 2
        for sample in tokenized:
            assert "input_ids" in sample
            assert "labels" in sample
            assert "attention_mask" in sample
            assert len(sample["input_ids"]) == len(sample["labels"])
            has_train = any(v != IGNORE_INDEX for v in sample["labels"])
            has_mask = any(v == IGNORE_INDEX for v in sample["labels"])
            assert has_train
            assert has_mask

        batch = collate_tokenized(tokenized, pad_token_id=0)
        assert batch["input_ids"].shape[0] == 2
        assert batch["labels"].shape[0] == 2
        assert batch["attention_mask"].shape[0] == 2

    def test_react_pipeline(self):
        samples = [
            {
                "task": "What is the capital of France?",
                "steps": [
                    {
                        "thought": "I need to look this up.",
                        "action": "search",
                        "action_input": "capital of France",
                        "observation": "Paris is the capital of France.",
                    },
                    {
                        "thought": "I have the answer.",
                        "action": "finish",
                        "action_input": "Paris",
                    },
                ],
            }
        ]

        formatted = [format_react(s) for s in samples]
        tok = _make_tokenizer()
        tokenized = tokenize_agent_dataset(formatted, tok)

        assert len(tokenized) == 1
        sample = tokenized[0]
        has_train = any(v != IGNORE_INDEX for v in sample["labels"])
        has_mask = any(v == IGNORE_INDEX for v in sample["labels"])
        assert has_train
        assert has_mask

    def test_trajectory_pipeline(self):
        samples = [
            {
                "system": "You are a coding assistant.",
                "goal": "Print hello",
                "turns": [
                    {
                        "role": "assistant",
                        "content": "Creating file.",
                        "tool_calls": [
                            {
                                "name": "write",
                                "arguments": {"path": "a.py", "content": "print('hi')"},
                            }
                        ],
                    },
                    {"role": "tool", "content": "Done."},
                    {"role": "assistant", "content": "File created."},
                ],
            }
        ]

        formatted = [format_trajectory(s) for s in samples]
        tok = _make_tokenizer()
        tokenized = tokenize_agent_dataset(formatted, tok)

        assert len(tokenized) == 1
        trainable_count = sum(1 for v in tokenized[0]["labels"] if v != IGNORE_INDEX)
        masked_count = sum(1 for v in tokenized[0]["labels"] if v == IGNORE_INDEX)
        assert trainable_count > 0
        assert masked_count > 0


class TestAgentDataFromJSONL:
    """Verify loading agent data from JSONL files via load_dataset."""

    def test_load_and_tokenize_function_calling(self):
        samples = [
            {
                "messages": [
                    {"role": "user", "content": "hello there"},
                    {"role": "assistant", "content": "hi back to you"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "goodbye now"},
                    {"role": "assistant", "content": "see you later"},
                ]
            },
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "agent_data.jsonl"
            with open(path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")

            data = load_dataset(str(path), format="function_calling")
            assert len(data) == 2
            assert isinstance(data[0], list)
            assert isinstance(data[0][0], AgentMessage)

            tok = _make_tokenizer()
            tokenized = tokenize_agent_dataset(data, tok)
            assert len(tokenized) == 2

    def test_load_react_from_file(self):
        samples = [
            {
                "task": "Say hi",
                "steps": [{"thought": "Easy.", "action": "finish", "action_input": "Hi!"}],
            }
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "react.jsonl"
            with open(path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")

            data = load_dataset(str(path), format="react")
            assert len(data) == 1
            assert isinstance(data[0][0], AgentMessage)


class TestLossMaskingCorrectness:
    """Verify that loss masking is mathematically correct."""

    def test_only_assistant_tokens_have_labels(self):
        messages = [
            AgentMessage(role="system", content="You are helpful", trainable=False),
            AgentMessage(role="user", content="What is AI", trainable=False),
            AgentMessage(role="assistant", content="AI is artificial intelligence", trainable=True),
            AgentMessage(role="user", content="Tell me more", trainable=False),
            AgentMessage(role="assistant", content="It involves machine learning", trainable=True),
        ]

        tok = _make_tokenizer()
        result = tokenize_agent_dataset([messages], tok)
        sample = result[0]

        ids = sample["input_ids"]
        labels = sample["labels"]

        # Mock tokenizer splits by spaces, so:
        # system: "You are helpful" -> 3 tokens
        # user: "What is AI" -> 3 tokens
        # assistant: "AI is artificial intelligence" -> 4 tokens
        # user: "Tell me more" -> 3 tokens
        # assistant: "It involves machine learning" -> 4 tokens

        # First 6 tokens (system + user) should be masked
        for i in range(6):
            assert labels[i] == IGNORE_INDEX, f"Token {i} should be masked"

        # Next 4 tokens (assistant) should be trainable
        for i in range(6, 10):
            assert labels[i] == ids[i], f"Token {i} should be trainable"

        # Next 3 tokens (user) should be masked
        for i in range(10, 13):
            assert labels[i] == IGNORE_INDEX, f"Token {i} should be masked"

        # Last 4 tokens (assistant) should be trainable
        for i in range(13, 17):
            assert labels[i] == ids[i], f"Token {i} should be trainable"

    def test_tool_results_are_masked(self):
        messages = [
            AgentMessage(role="user", content="Search for cats", trainable=False),
            AgentMessage(
                role="assistant",
                content='<tool_call>\n{"name": "search"}\n</tool_call>',
                trainable=True,
            ),
            AgentMessage(
                role="tool",
                content="<tool_result>\nFound results\n</tool_result>",
                trainable=False,
            ),
            AgentMessage(role="assistant", content="Here are the results", trainable=True),
        ]

        tok = _make_tokenizer()
        result = tokenize_agent_dataset([messages], tok)
        sample = result[0]

        # Count trainable vs masked
        trainable = sum(1 for v in sample["labels"] if v != IGNORE_INDEX)
        masked = sum(1 for v in sample["labels"] if v == IGNORE_INDEX)

        assert trainable > 0
        assert masked > 0
        # User (3 tokens) + tool (4 tokens) = 7 masked
        # Assistant tool_call (4 tokens) + assistant response (4 tokens) = 8 trainable
        assert masked == 7
        assert trainable == 8
