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
        assert "Hi" in result["text"]
        assert "Hello!" in result["text"]

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
        assert "How are you?" in result["text"]
        assert "Fine, thanks!" in result["text"]

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
        assert "You are helpful." in result["text"]
        assert "Hi" in result["text"]
        assert "Hello!" in result["text"]

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
    def test_chat_format_uses_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<s>[INST] Hi [/INST] Hello!</s>"
        sample = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        result = apply_chat_template(sample, tokenizer, format="chat")
        tokenizer.apply_chat_template.assert_called_once_with(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        assert result["text"] == "<s>[INST] Hi [/INST] Hello!</s>"

    def test_sharegpt_format_converts_roles(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<s>User: Hi\nAssistant: Hello!</s>"
        sample = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        result = apply_chat_template(sample, tokenizer, format="sharegpt")
        call_messages = tokenizer.apply_chat_template.call_args[0][0]
        assert call_messages[0]["role"] == "user"
        assert call_messages[0]["content"] == "Hi"
        assert call_messages[1]["role"] == "assistant"
        assert call_messages[1]["content"] == "Hello!"
        assert result["text"] == "<s>User: Hi\nAssistant: Hello!</s>"

    def test_sharegpt_preserves_system_role(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "templated"
        sample = {
            "conversations": [
                {"from": "system", "value": "Be helpful"},
                {"from": "human", "value": "Hi"},
            ]
        }
        apply_chat_template(sample, tokenizer, format="sharegpt")
        call_messages = tokenizer.apply_chat_template.call_args[0][0]
        assert call_messages[0]["role"] == "system"
