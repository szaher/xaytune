from __future__ import annotations

from unittest.mock import MagicMock

import torch

from xaytune.data.tokenizer import (
    StreamingTokenizedDataset,
    collate_tokenized,
    tokenize_sample,
)


class TestTokenizeSample:
    def _make_tokenizer(self):
        tok = MagicMock()
        tok.model_max_length = 512
        tok.pad_token_id = 0

        def side_effect(text, **kwargs):
            ids = list(range(1, min(len(text.split()) + 1, kwargs.get("max_length", 512) + 1)))
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        tok.side_effect = side_effect
        tok.__call__ = side_effect
        return tok

    def test_tokenizes_text_sample(self):
        tok = self._make_tokenizer()
        result = tokenize_sample({"text": "hello world"}, tok)
        assert result is not None
        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result
        assert result["labels"] == result["input_ids"]

    def test_returns_none_for_empty_text(self):
        tok = self._make_tokenizer()
        assert tokenize_sample({"text": ""}, tok) is None

    def test_passes_through_pretokenized(self):
        tok = self._make_tokenizer()
        sample = {"input_ids": [1, 2, 3]}
        result = tokenize_sample(sample, tok)
        assert result is sample

    def test_respects_max_seq_length(self):
        tok = self._make_tokenizer()
        result = tokenize_sample({"text": "a b c d e"}, tok, max_seq_length=3)
        assert result is not None
        assert len(result["input_ids"]) <= 3


class TestStreamingTokenizedDataset:
    def _make_tokenizer(self):
        tok = MagicMock()
        tok.model_max_length = 512

        def side_effect(text, **kwargs):
            ids = list(range(1, len(text.split()) + 1))
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        tok.__call__ = side_effect
        return tok

    def test_iterates_and_tokenizes(self):
        samples = [{"text": "hello world"}, {"text": "foo bar baz"}]
        tok = self._make_tokenizer()
        ds = StreamingTokenizedDataset(iter(samples), tok)
        results = list(ds)
        assert len(results) == 2
        assert all("input_ids" in r for r in results)

    def test_skips_empty_text(self):
        samples = [{"text": "hello"}, {"text": ""}, {"text": "world"}]
        tok = self._make_tokenizer()
        ds = StreamingTokenizedDataset(iter(samples), tok)
        results = list(ds)
        assert len(results) == 2

    def test_works_with_dataloader(self):
        samples = [
            {"text": "a b"},
            {"text": "c d e"},
            {"text": "f"},
            {"text": "g h i j"},
        ]
        tok = self._make_tokenizer()
        ds = StreamingTokenizedDataset(iter(samples), tok)

        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=2,
            collate_fn=lambda batch: collate_tokenized(batch, pad_token_id=0),
        )
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0]["input_ids"].shape[0] == 2
        assert batches[0]["input_ids"].dtype == torch.long
