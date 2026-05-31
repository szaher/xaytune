from unittest.mock import MagicMock

import torch

from xaytune.data.agent_formats import AgentMessage
from xaytune.data.agent_tokenizer import tokenize_agent_dataset

IGNORE_INDEX = -100


def _make_tokenizer():
    tok = MagicMock()
    tok.model_max_length = 1024
    call_count = [0]

    def tokenize(text, **kwargs):
        tokens = list(range(call_count[0] * 100, call_count[0] * 100 + len(text.split())))
        call_count[0] += 1
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

    tok.side_effect = tokenize
    tok.__call__ = tokenize
    return tok


class TestLossMasking:
    def test_non_trainable_masked(self):
        messages = [
            AgentMessage(role="user", content="hello world", trainable=False),
            AgentMessage(role="assistant", content="hi there friend", trainable=True),
        ]
        tok = _make_tokenizer()
        result = tokenize_agent_dataset([messages], tok)
        assert len(result) == 1
        sample = result[0]
        ids = sample["input_ids"]
        labels = sample["labels"]
        assert len(ids) == len(labels)
        for label in labels[:2]:
            assert label == IGNORE_INDEX
        for i in range(2, len(labels)):
            assert labels[i] == ids[i]

    def test_all_trainable(self):
        messages = [
            AgentMessage(role="assistant", content="hello world", trainable=True),
        ]
        tok = _make_tokenizer()
        result = tokenize_agent_dataset([messages], tok)
        sample = result[0]
        for i, label in enumerate(sample["labels"]):
            assert label == sample["input_ids"][i]

    def test_all_non_trainable(self):
        messages = [
            AgentMessage(role="user", content="hello world", trainable=False),
        ]
        tok = _make_tokenizer()
        result = tokenize_agent_dataset([messages], tok)
        sample = result[0]
        for label in sample["labels"]:
            assert label == IGNORE_INDEX

    def test_alternating_roles(self):
        messages = [
            AgentMessage(role="user", content="one two", trainable=False),
            AgentMessage(role="assistant", content="three four", trainable=True),
            AgentMessage(role="tool", content="five six", trainable=False),
            AgentMessage(role="assistant", content="seven eight", trainable=True),
        ]
        tok = _make_tokenizer()
        result = tokenize_agent_dataset([messages], tok)
        sample = result[0]
        assert len(sample["input_ids"]) == 8
        expected_trainable = [False, False, True, True, False, False, True, True]
        for i, trainable in enumerate(expected_trainable):
            if trainable:
                assert sample["labels"][i] == sample["input_ids"][i]
            else:
                assert sample["labels"][i] == IGNORE_INDEX

    def test_attention_mask_all_ones(self):
        messages = [
            AgentMessage(role="user", content="a b", trainable=False),
            AgentMessage(role="assistant", content="c d", trainable=True),
        ]
        tok = _make_tokenizer()
        result = tokenize_agent_dataset([messages], tok)
        assert all(m == 1 for m in result[0]["attention_mask"])


class TestTruncation:
    def test_max_seq_length(self):
        messages = [
            AgentMessage(
                role="user",
                content=" ".join(f"w{i}" for i in range(50)),
                trainable=False,
            ),
            AgentMessage(
                role="assistant",
                content=" ".join(f"a{i}" for i in range(50)),
                trainable=True,
            ),
        ]
        tok = _make_tokenizer()
        result = tokenize_agent_dataset([messages], tok, max_seq_length=20)
        assert len(result[0]["input_ids"]) == 20
        assert len(result[0]["labels"]) == 20


class TestMultipleSamples:
    def test_batch(self):
        sample_a = [
            AgentMessage(role="user", content="hello", trainable=False),
            AgentMessage(role="assistant", content="hi", trainable=True),
        ]
        sample_b = [
            AgentMessage(role="user", content="one two three", trainable=False),
            AgentMessage(role="assistant", content="four five", trainable=True),
        ]
        tok = _make_tokenizer()
        result = tokenize_agent_dataset([sample_a, sample_b], tok)
        assert len(result) == 2


class TestCollateCompat:
    def test_works_with_collate_tokenized(self):
        from xaytune.data.tokenizer import collate_tokenized

        messages = [
            AgentMessage(role="user", content="a b", trainable=False),
            AgentMessage(role="assistant", content="c", trainable=True),
        ]
        tok = _make_tokenizer()
        result = tokenize_agent_dataset([messages, messages], tok)
        batch = collate_tokenized(result, pad_token_id=0)
        assert "input_ids" in batch
        assert "labels" in batch
        assert "attention_mask" in batch
        assert isinstance(batch["input_ids"], torch.Tensor)
