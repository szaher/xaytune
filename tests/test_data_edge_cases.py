import json
import random
import tempfile
import warnings
from unittest.mock import MagicMock

from xaytune.data.agent_formats import AgentMessage
from xaytune.data.agent_tokenizer import IGNORE_INDEX, tokenize_agent_dataset
from xaytune.data.formats import _warned_text_keys, format_text
from xaytune.data.preferences import load_preference_dataset


class TestFormatTextWarning:
    def test_warns_on_unknown_keys(self):
        _warned_text_keys.clear()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = format_text({"body": "hello"})
            assert result["text"] == ""
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1
            assert "body" in str(user_warnings[0].message)

    def test_no_warning_for_text_key(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = format_text({"text": "hello"})
            assert result["text"] == "hello"
            format_warnings = [
                x for x in w
                if "key" in str(x.message).lower() and "text" in str(x.message).lower()
            ]
            assert len(format_warnings) == 0

    def test_no_warning_for_content_key(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = format_text({"content": "hello"})
            assert result["text"] == "hello"
            format_warnings = [
                x for x in w
                if "no 'text'" in str(x.message).lower()
            ]
            assert len(format_warnings) == 0

    def test_warns_only_once_per_keyset(self):
        _warned_text_keys.clear()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            format_text({"body": "a"})
            format_text({"body": "b"})
            key_warnings = [
                x for x in w
                if "body" in str(x.message)
            ]
            assert len(key_warnings) == 1


class TestPreferenceShuffle:
    def test_eval_split_shuffles_before_split(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(100):
                line = json.dumps({
                    "prompt": f"prompt-{i}",
                    "chosen": f"chosen-{i}",
                    "rejected": f"rejected-{i}",
                })
                f.write(line + "\n")
            path = f.name

        train, eval_set = load_preference_dataset(path, eval_split=0.2)

        # If no shuffle, eval would be the last 20 items (prompt-80..prompt-99).
        # With shuffle (seed=42), it should NOT be exactly the last 20.
        eval_prompts = [s["prompt"] for s in eval_set]
        last_20 = [f"prompt-{i}" for i in range(80, 100)]
        assert eval_prompts != last_20

    def test_shuffle_is_deterministic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(50):
                line = json.dumps({
                    "prompt": f"p-{i}",
                    "chosen": f"c-{i}",
                    "rejected": f"r-{i}",
                })
                f.write(line + "\n")
            path = f.name

        train1, eval1 = load_preference_dataset(path, eval_split=0.2)
        train2, eval2 = load_preference_dataset(path, eval_split=0.2)

        assert [s["prompt"] for s in eval1] == [s["prompt"] for s in eval2]


class TestAgentBOS:
    def _make_tokenizer(self):
        tokenizer = MagicMock()

        def mock_call(text, **kwargs):
            add_special = kwargs.get("add_special_tokens", True)
            ids = [ord(c) for c in text]
            if add_special:
                ids = [1] + ids  # BOS token
            return {"input_ids": ids}

        tokenizer.side_effect = mock_call
        tokenizer.model_max_length = 4096
        return tokenizer

    def test_first_message_gets_special_tokens(self):
        tokenizer = self._make_tokenizer()
        messages = [
            AgentMessage(role="user", content="hi", trainable=False),
            AgentMessage(role="assistant", content="ok", trainable=True),
        ]
        tokenize_agent_dataset([messages], tokenizer)

        first_call = tokenizer.call_args_list[0]
        assert first_call[1]["add_special_tokens"] is True

    def test_subsequent_messages_no_special_tokens(self):
        tokenizer = self._make_tokenizer()
        messages = [
            AgentMessage(role="user", content="hi", trainable=False),
            AgentMessage(role="assistant", content="ok", trainable=True),
            AgentMessage(role="user", content="bye", trainable=False),
        ]
        tokenize_agent_dataset([messages], tokenizer)

        for call in tokenizer.call_args_list[1:]:
            assert call[1]["add_special_tokens"] is False

    def test_trainable_labels_not_masked(self):
        tokenizer = self._make_tokenizer()
        messages = [
            AgentMessage(role="user", content="a", trainable=False),
            AgentMessage(role="assistant", content="b", trainable=True),
        ]
        result = tokenize_agent_dataset([messages], tokenizer)

        assert len(result) == 1
        labels = result[0]["labels"]
        # First message (user, non-trainable) should be masked
        # Second message (assistant, trainable) should have real IDs
        user_len = 2  # BOS + ord('a')
        assert all(l == IGNORE_INDEX for l in labels[:user_len])
        assert all(l != IGNORE_INDEX for l in labels[user_len:])
