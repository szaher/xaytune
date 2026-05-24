from __future__ import annotations

from unittest.mock import MagicMock

import torch

from xaytune.data.tokenizer import (
    collate_preference,
    collate_tokenized,
    tokenize_dataset,
    tokenize_preference_dataset,
)


def _make_tokenizer(vocab_size: int = 100, max_length: int = 512) -> MagicMock:
    def _tokenize(
        text,
        *,
        truncation=True,
        max_length=512,
        padding=False,
        return_attention_mask=True,
    ):
        ids = list(range(1, min(len(text.split()) + 1, max_length + 1)))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    tok = MagicMock(side_effect=_tokenize)
    tok.model_max_length = max_length
    return tok


class TestTokenizeDataset:
    def test_tokenize_text_samples(self):
        data = [{"text": "hello world"}]
        tok = _make_tokenizer()
        result = tokenize_dataset(data, tok)
        assert len(result) == 1
        assert "input_ids" in result[0]
        assert "labels" in result[0]
        assert "attention_mask" in result[0]

    def test_tokenize_already_tokenized(self):
        data = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3], "attention_mask": [1, 1, 1]}]
        tok = _make_tokenizer()
        result = tokenize_dataset(data, tok)
        assert result is data

    def test_tokenize_respects_max_seq_length(self):
        def _tokenize(
            text,
            *,
            truncation=True,
            max_length=512,
            padding=False,
            return_attention_mask=True,
        ):
            ids = list(range(1, min(11, max_length + 1)))
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        tok = MagicMock(side_effect=_tokenize)
        tok.model_max_length = 1024
        data = [{"text": "a b c d e f g h i j"}]
        result = tokenize_dataset(data, tok, max_seq_length=5)
        assert len(result[0]["input_ids"]) <= 5

    def test_tokenize_skips_empty(self):
        data = [{"text": ""}]
        tok = _make_tokenizer()
        result = tokenize_dataset(data, tok)
        assert result == []

    def test_tokenize_labels_equal_input_ids(self):
        data = [{"text": "hello world foo"}]
        tok = _make_tokenizer()
        result = tokenize_dataset(data, tok)
        assert result[0]["labels"] == result[0]["input_ids"]

    def test_tokenize_empty_data(self):
        tok = _make_tokenizer()
        result = tokenize_dataset([], tok)
        assert result == []

    def test_tokenize_uses_model_max_length_when_no_max_seq(self):
        call_args = {}

        def _tokenize(
            text,
            *,
            truncation=True,
            max_length=512,
            padding=False,
            return_attention_mask=True,
        ):
            call_args["max_length"] = max_length
            ids = list(range(1, 4))
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        tok = MagicMock(side_effect=_tokenize)
        tok.model_max_length = 8
        tokenize_dataset([{"text": "hello"}], tok, max_seq_length=0)
        assert call_args["max_length"] == 8

    def test_tokenize_filters_empty_encoding(self):
        def _tokenize(
            text,
            *,
            truncation=True,
            max_length=512,
            padding=False,
            return_attention_mask=True,
        ):
            return {"input_ids": [], "attention_mask": []}

        tok = MagicMock(side_effect=_tokenize)
        tok.model_max_length = 512
        result = tokenize_dataset([{"text": "x"}], tok)
        assert result == []


class TestCollateTokenized:
    def test_collate_pads_to_longest(self):
        batch = [
            {"input_ids": [1, 2], "labels": [1, 2], "attention_mask": [1, 1]},
            {"input_ids": [3, 4, 5], "labels": [3, 4, 5], "attention_mask": [1, 1, 1]},
        ]
        result = collate_tokenized(batch)
        assert result["input_ids"].shape == (2, 3)
        assert result["input_ids"][0].tolist() == [1, 2, 0]

    def test_collate_labels_padded_with_ignore_index(self):
        batch = [
            {"input_ids": [1, 2], "labels": [1, 2], "attention_mask": [1, 1]},
            {"input_ids": [3, 4, 5], "labels": [3, 4, 5], "attention_mask": [1, 1, 1]},
        ]
        result = collate_tokenized(batch)
        assert result["labels"][0].tolist() == [1, 2, -100]

    def test_collate_returns_tensors(self):
        batch = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "attention_mask": [1, 1, 1]},
        ]
        result = collate_tokenized(batch)
        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["labels"], torch.Tensor)
        assert isinstance(result["attention_mask"], torch.Tensor)

    def test_collate_attention_mask_zeros_for_padding(self):
        batch = [
            {"input_ids": [1], "labels": [1], "attention_mask": [1]},
            {"input_ids": [2, 3], "labels": [2, 3], "attention_mask": [1, 1]},
        ]
        result = collate_tokenized(batch)
        assert result["attention_mask"][0].tolist() == [1, 0]

    def test_collate_custom_pad_token_id(self):
        batch = [
            {"input_ids": [1, 2], "labels": [1, 2], "attention_mask": [1, 1]},
            {"input_ids": [3, 4, 5], "labels": [3, 4, 5], "attention_mask": [1, 1, 1]},
        ]
        result = collate_tokenized(batch, pad_token_id=99)
        assert result["input_ids"][0].tolist() == [1, 2, 99]

    def test_collate_dtype_is_long(self):
        batch = [
            {"input_ids": [1], "labels": [1], "attention_mask": [1]},
        ]
        result = collate_tokenized(batch)
        assert result["input_ids"].dtype == torch.long
        assert result["labels"].dtype == torch.long


class TestTokenizePreferenceDataset:
    def test_tokenize_preference_samples(self):
        data = [{"prompt": "Question? ", "chosen": "Good answer", "rejected": "Bad answer"}]
        tok = _make_tokenizer()
        result = tokenize_preference_dataset(data, tok)
        assert len(result) == 1
        assert "chosen_input_ids" in result[0]
        assert "chosen_attention_mask" in result[0]
        assert "rejected_input_ids" in result[0]
        assert "rejected_attention_mask" in result[0]
        assert "input_ids" not in result[0]

    def test_tokenize_preference_already_tokenized(self):
        data = [{"chosen_input_ids": [1, 2], "rejected_input_ids": [3, 4]}]
        tok = _make_tokenizer()
        result = tokenize_preference_dataset(data, tok)
        assert result is data

    def test_tokenize_preference_concatenates_prompt(self):
        call_texts = []

        def _tokenize(
            text,
            *,
            truncation=True,
            max_length=512,
            padding=False,
            return_attention_mask=True,
        ):
            call_texts.append(text)
            ids = list(range(1, len(text.split()) + 1))
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        tok = MagicMock(side_effect=_tokenize)
        tok.model_max_length = 512
        data = [{"prompt": "Q: ", "chosen": "A", "rejected": "B"}]
        tokenize_preference_dataset(data, tok)
        assert call_texts[0] == "Q: A"
        assert call_texts[1] == "Q: B"

    def test_tokenize_preference_skips_empty_chosen(self):
        data = [{"prompt": "Q", "chosen": "", "rejected": "B"}]
        tok = _make_tokenizer()
        result = tokenize_preference_dataset(data, tok)
        assert result == []

    def test_tokenize_preference_skips_empty_rejected(self):
        data = [{"prompt": "Q", "chosen": "A", "rejected": ""}]
        tok = _make_tokenizer()
        result = tokenize_preference_dataset(data, tok)
        assert result == []

    def test_tokenize_preference_empty_data(self):
        tok = _make_tokenizer()
        result = tokenize_preference_dataset([], tok)
        assert result == []

    def test_tokenize_preference_no_prompt(self):
        call_texts = []

        def _tokenize(
            text,
            *,
            truncation=True,
            max_length=512,
            padding=False,
            return_attention_mask=True,
        ):
            call_texts.append(text)
            ids = list(range(1, len(text.split()) + 1))
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        tok = MagicMock(side_effect=_tokenize)
        tok.model_max_length = 512
        data = [{"prompt": "", "chosen": "Good", "rejected": "Bad"}]
        tokenize_preference_dataset(data, tok)
        assert call_texts[0] == "Good"
        assert call_texts[1] == "Bad"


class TestCollatePreference:
    def test_collate_preference_pads_both(self):
        batch = [
            {
                "chosen_input_ids": [1, 2],
                "chosen_attention_mask": [1, 1],
                "rejected_input_ids": [3, 4, 5],
                "rejected_attention_mask": [1, 1, 1],
            },
            {
                "chosen_input_ids": [6, 7, 8],
                "chosen_attention_mask": [1, 1, 1],
                "rejected_input_ids": [9],
                "rejected_attention_mask": [1],
            },
        ]
        result = collate_preference(batch)
        assert result["chosen_input_ids"].shape == (2, 3)
        assert result["rejected_input_ids"].shape == (2, 3)
        assert result["chosen_input_ids"][0].tolist() == [1, 2, 0]
        assert result["rejected_input_ids"][1].tolist() == [9, 0, 0]

    def test_collate_preference_attention_mask(self):
        batch = [
            {
                "chosen_input_ids": [1],
                "chosen_attention_mask": [1],
                "rejected_input_ids": [2, 3],
                "rejected_attention_mask": [1, 1],
            },
            {
                "chosen_input_ids": [4, 5],
                "chosen_attention_mask": [1, 1],
                "rejected_input_ids": [6],
                "rejected_attention_mask": [1],
            },
        ]
        result = collate_preference(batch)
        assert result["chosen_attention_mask"][0].tolist() == [1, 0]
        assert result["rejected_attention_mask"][1].tolist() == [1, 0]

    def test_collate_preference_returns_tensors(self):
        batch = [
            {
                "chosen_input_ids": [1, 2],
                "chosen_attention_mask": [1, 1],
                "rejected_input_ids": [3],
                "rejected_attention_mask": [1],
            },
        ]
        result = collate_preference(batch)
        for key in result:
            assert isinstance(result[key], torch.Tensor)
            assert result[key].dtype == torch.long

    def test_collate_preference_custom_pad_token(self):
        batch = [
            {
                "chosen_input_ids": [1],
                "chosen_attention_mask": [1],
                "rejected_input_ids": [2, 3],
                "rejected_attention_mask": [1, 1],
            },
            {
                "chosen_input_ids": [4, 5],
                "chosen_attention_mask": [1, 1],
                "rejected_input_ids": [6],
                "rejected_attention_mask": [1],
            },
        ]
        result = collate_preference(batch, pad_token_id=99)
        assert result["chosen_input_ids"][0].tolist() == [1, 99]
        assert result["rejected_input_ids"][1].tolist() == [6, 99]

    def test_collate_preference_has_four_keys(self):
        batch = [
            {
                "chosen_input_ids": [1],
                "chosen_attention_mask": [1],
                "rejected_input_ids": [2],
                "rejected_attention_mask": [1],
            },
        ]
        result = collate_preference(batch)
        assert set(result.keys()) == {
            "chosen_input_ids",
            "chosen_attention_mask",
            "rejected_input_ids",
            "rejected_attention_mask",
        }
