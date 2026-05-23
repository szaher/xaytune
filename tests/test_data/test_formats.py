from trainlib.data.formats import format_alpaca, format_chat, format_sharegpt, format_text
from trainlib.data.registry import format_registry


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
