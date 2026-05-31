import json
import tempfile
from pathlib import Path

from xaytune.data.agent_formats import AgentMessage, format_function_calling
from xaytune.data.agent_tokenizer import tokenize_agent_dataset
from xaytune.data.loader import load_dataset


class TestFormatRegistration:
    def test_function_calling_registered(self):
        from xaytune.data.registry import format_registry
        assert format_registry.has("function_calling")

    def test_react_registered(self):
        from xaytune.data.registry import format_registry
        assert format_registry.has("react")

    def test_trajectory_registered(self):
        from xaytune.data.registry import format_registry
        assert format_registry.has("trajectory")


class TestLoadDatasetWithAgentFormat:
    def test_load_function_calling_jsonl(self):
        sample = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            path.write_text(json.dumps(sample) + "\n")
            data = load_dataset(str(path), format="function_calling")
            assert len(data) == 1
            assert isinstance(data[0], list)
            assert isinstance(data[0][0], AgentMessage)


class TestEndToEnd:
    def test_format_then_tokenize(self):
        from unittest.mock import MagicMock

        sample = {
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "Artificial intelligence."},
                {"role": "user", "content": "Tell me more."},
                {"role": "assistant", "content": "It's about making machines think."},
            ]
        }
        messages = format_function_calling(sample)
        assert isinstance(messages, list)
        assert all(isinstance(m, AgentMessage) for m in messages)

        tok = MagicMock()
        tok.model_max_length = 1024
        def tokenize_side_effect(text, **kwargs):
            tokens = list(range(len(text.split())))
            result = {"input_ids": tokens}
            if kwargs.get("return_attention_mask", True):
                result["attention_mask"] = [1] * len(tokens)
            return result
        tok.side_effect = tokenize_side_effect

        result = tokenize_agent_dataset([messages], tok)
        assert len(result) == 1
        sample_out = result[0]
        assert len(sample_out["input_ids"]) == len(sample_out["labels"])
        has_trainable = any(l != -100 for l in sample_out["labels"])
        has_masked = any(l == -100 for l in sample_out["labels"])
        assert has_trainable
        assert has_masked


class TestAgentDetection:
    def test_is_agent_data(self):
        data = [
            [AgentMessage(role="user", content="hi", trainable=False),
             AgentMessage(role="assistant", content="hello", trainable=True)]
        ]
        is_agent = (
            isinstance(data, list)
            and data
            and isinstance(data[0], list)
            and data[0]
            and hasattr(data[0][0], "trainable")
        )
        assert is_agent

    def test_regular_data_not_agent(self):
        data = [{"text": "hello world"}]
        is_agent = (
            isinstance(data, list)
            and data
            and isinstance(data[0], list)
            and data[0]
            and isinstance(data[0][0], AgentMessage)
        )
        assert not is_agent
