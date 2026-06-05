from unittest.mock import MagicMock

from xaytune.data.formats import (
    apply_chat_template,
    format_alpaca,
    format_chat,
    format_sharegpt,
    format_text,
)
from xaytune.data.registry import format_registry


class TestAlpacaFormat:
    def test_with_input(self):
        sample = {"instruction": "Translate", "input": "Hello", "output": "Hola"}
        result = format_alpaca(sample)
        assert "Translate" in result["text"]
        assert "Hello" in result["text"]
        assert "Hola" in result["text"]

    def test_without_input(self):
        sample = {"instruction": "Say hi", "input": "", "output": "Hello!"}
        result = format_alpaca(sample)
        assert "Say hi" in result["text"]
        assert "Hello!" in result["text"]

    def test_registered(self):
        assert format_registry.has("alpaca")


class TestShareGPTFormat:
    def test_single_turn(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        result = format_sharegpt(sample)
        assert "turns" in result
        assert result["turns"][0]["content"] == "Hi"
        assert result["turns"][1]["content"] == "Hello!"

    def test_multi_turn(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
                {"from": "human", "value": "How are you?"},
                {"from": "gpt", "value": "Fine, thanks!"},
            ]
        }
        result = format_sharegpt(sample)
        assert len(result["turns"]) == 4
        assert result["turns"][2]["content"] == "How are you?"
        assert result["turns"][3]["content"] == "Fine, thanks!"

    def test_normalizes_roles(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        result = format_sharegpt(sample)
        assert result["turns"][0]["role"] == "user"
        assert result["turns"][1]["role"] == "assistant"

    def test_registered(self):
        assert format_registry.has("sharegpt")


class TestChatFormat:
    def test_openai_format(self):
        sample = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        result = format_chat(sample)
        assert "turns" in result
        assert result["turns"][0]["content"] == "You are helpful."
        assert result["turns"][1]["content"] == "Hi"
        assert result["turns"][2]["content"] == "Hello!"

    def test_registered(self):
        assert format_registry.has("chat")


class TestTextFormat:
    def test_text_field(self):
        sample = {"text": "Hello world"}
        result = format_text(sample)
        assert result["text"] == "Hello world"

    def test_content_field(self):
        sample = {"content": "Hello world"}
        result = format_text(sample)
        assert result["text"] == "Hello world"

    def test_registered(self):
        assert format_registry.has("text")


class TestApplyChatTemplate:
    def test_chat_format_returns_turns(self):
        tokenizer = MagicMock()
        sample = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        result = apply_chat_template(sample, tokenizer, format="chat")
        assert "turns" in result
        assert result["_use_chat_template"] is True
        assert result["turns"][0]["role"] == "user"
        assert result["turns"][1]["role"] == "assistant"

    def test_sharegpt_format_converts_roles(self):
        tokenizer = MagicMock()
        sample = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        result = apply_chat_template(sample, tokenizer, format="sharegpt")
        assert "turns" in result
        assert result["turns"][0]["role"] == "user"
        assert result["turns"][0]["content"] == "Hi"
        assert result["turns"][1]["role"] == "assistant"
        assert result["turns"][1]["content"] == "Hello!"

    def test_sharegpt_preserves_system_role(self):
        tokenizer = MagicMock()
        sample = {
            "conversations": [
                {"from": "system", "value": "Be helpful"},
                {"from": "human", "value": "Hi"},
            ]
        }
        result = apply_chat_template(sample, tokenizer, format="sharegpt")
        assert result["turns"][0]["role"] == "system"
