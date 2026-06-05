"""Tests for SFT prompt masking and multi-turn conversation masking.

Verifies BUG-031 / GAP-014 / TASK-025 / FEAT-001:
- Alpaca format masks prompt tokens (labels=-100), trains on output only
- Chat/ShareGPT format masks all non-assistant turns per-turn
- Text format trains on everything (no masking)
- Multi-turn: every assistant turn is trainable, every user/system turn is masked
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import torch

from xaytune.data.formats import (
    apply_chat_template,
    format_alpaca,
    format_chat,
    format_sharegpt,
    format_text,
)
from xaytune.data.tokenizer import (
    IGNORE_INDEX,
    tokenize_dataset,
    tokenize_multiturn,
    tokenize_sample,
)


def _make_tokenizer(bos_token_id: int = 1) -> MagicMock:
    """Create a mock tokenizer that assigns one token per word."""

    def _tokenize(
        text,
        *,
        truncation=True,
        max_length=512,
        padding=False,
        return_attention_mask=True,
        add_special_tokens=True,
    ):
        words = text.split()
        ids = [bos_token_id] + [hash(w) % 1000 + 2 for w in words] if add_special_tokens else [hash(w) % 1000 + 2 for w in words]
        ids = ids[:max_length]
        result = {"input_ids": ids}
        if return_attention_mask:
            result["attention_mask"] = [1] * len(ids)
        return result

    tok = MagicMock(side_effect=_tokenize)
    tok.model_max_length = 512
    return tok


# --- Format function tests ---


class TestAlpacaFormatMasking:
    def test_returns_prompt_text(self):
        sample = {"instruction": "Translate", "input": "Hello", "output": "Hola"}
        result = format_alpaca(sample)
        assert "prompt_text" in result
        assert "Hola" not in result["prompt_text"]
        assert "Hola" in result["text"]

    def test_prompt_text_ends_with_response_header(self):
        sample = {"instruction": "Say hi", "input": "", "output": "Hello!"}
        result = format_alpaca(sample)
        assert result["prompt_text"].endswith("### Response:\n")

    def test_prompt_text_excludes_output(self):
        sample = {"instruction": "Do X", "input": "Y", "output": "Z"}
        result = format_alpaca(sample)
        assert result["text"] == result["prompt_text"] + "Z"


class TestShareGPTFormatMasking:
    def test_returns_turns_not_text(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        result = format_sharegpt(sample)
        assert "turns" in result
        assert "text" not in result

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

    def test_multi_turn(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
                {"from": "human", "value": "Bye"},
                {"from": "gpt", "value": "Goodbye!"},
            ]
        }
        result = format_sharegpt(sample)
        assert len(result["turns"]) == 4


class TestChatFormatMasking:
    def test_returns_turns(self):
        sample = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        result = format_chat(sample)
        assert "turns" in result
        assert len(result["turns"]) == 2

    def test_preserves_system_role(self):
        sample = {
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        result = format_chat(sample)
        assert result["turns"][0]["role"] == "system"


class TestTextFormatNoMasking:
    def test_no_prompt_text(self):
        sample = {"text": "Hello world"}
        result = format_text(sample)
        assert "prompt_text" not in result
        assert "turns" not in result

    def test_warns_on_unknown_keys(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from xaytune.data.formats import _warned_text_keys
            _warned_text_keys.clear()
            result = format_text({"body": "Hello"})
            assert result["text"] == ""
            assert len(w) == 1
            assert "body" in str(w[0].message)


class TestApplyChatTemplateMasking:
    def test_returns_turns(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "templated"
        sample = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        result = apply_chat_template(sample, tokenizer, format="chat")
        assert "turns" in result
        assert result.get("_use_chat_template") is True


# --- Tokenizer masking tests ---


class TestTokenizeDatasetMasking:
    def test_alpaca_masks_prompt(self):
        sample = format_alpaca(
            {"instruction": "Say hi", "input": "", "output": "Hello!"}
        )
        tok = _make_tokenizer()
        result = tokenize_dataset([sample], tok)
        assert len(result) == 1

        labels = result[0]["labels"]
        input_ids = result[0]["input_ids"]
        assert len(labels) == len(input_ids)

        masked_count = sum(1 for l in labels if l == IGNORE_INDEX)
        trainable_count = sum(1 for l in labels if l != IGNORE_INDEX)
        assert masked_count > 0, "Prompt tokens should be masked"
        assert trainable_count > 0, "Output tokens should be trainable"

    def test_text_format_no_masking(self):
        sample = format_text({"text": "Hello world foo bar"})
        tok = _make_tokenizer()
        result = tokenize_dataset([sample], tok)
        labels = result[0]["labels"]
        assert all(l != IGNORE_INDEX for l in labels), "Text format should have no masking"
        assert labels == result[0]["input_ids"]

    def test_already_tokenized_passthrough(self):
        data = [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}]
        tok = _make_tokenizer()
        result = tokenize_dataset(data, tok)
        assert result is data


class TestTokenizeSampleMasking:
    def test_alpaca_masks_prompt(self):
        sample = format_alpaca(
            {"instruction": "Translate", "input": "cat", "output": "gato"}
        )
        tok = _make_tokenizer()
        result = tokenize_sample(sample, tok)
        assert result is not None

        labels = result["labels"]
        masked = [i for i, l in enumerate(labels) if l == IGNORE_INDEX]
        trainable = [i for i, l in enumerate(labels) if l != IGNORE_INDEX]
        assert len(masked) > 0
        assert len(trainable) > 0
        assert max(masked) < min(trainable), "Masked tokens should come before trainable"

    def test_text_format_no_masking(self):
        sample = {"text": "Hello world"}
        tok = _make_tokenizer()
        result = tokenize_sample(sample, tok)
        assert result is not None
        assert all(l != IGNORE_INDEX for l in result["labels"])


# --- Multi-turn masking tests ---


class TestTokenizeMultiturn:
    def test_single_turn_masks_user(self):
        data = [
            {
                "turns": [
                    {"role": "user", "content": "Hi there"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        ]
        tok = _make_tokenizer()
        result = tokenize_multiturn(data, tok)
        assert len(result) == 1

        labels = result[0]["labels"]
        input_ids = result[0]["input_ids"]
        assert len(labels) == len(input_ids)

        masked = sum(1 for l in labels if l == IGNORE_INDEX)
        trainable = sum(1 for l in labels if l != IGNORE_INDEX)
        assert masked > 0, "User turn should be masked"
        assert trainable > 0, "Assistant turn should be trainable"

    def test_multi_turn_all_assistants_trainable(self):
        data = [
            {
                "turns": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "How are you"},
                    {"role": "assistant", "content": "Fine thanks"},
                ]
            }
        ]
        tok = _make_tokenizer()
        result = tokenize_multiturn(data, tok)
        labels = result[0]["labels"]

        trainable_count = sum(1 for l in labels if l != IGNORE_INDEX)
        assert trainable_count > 2, "Both assistant turns should contribute trainable tokens"

    def test_system_turn_masked(self):
        data = [
            {
                "turns": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        ]
        tok = _make_tokenizer()
        result = tokenize_multiturn(data, tok)
        labels = result[0]["labels"]

        masked_count = sum(1 for l in labels if l == IGNORE_INDEX)
        trainable_count = sum(1 for l in labels if l != IGNORE_INDEX)
        assert masked_count > trainable_count, "System + user should have more masked tokens than assistant trainable"

    def test_no_assistant_all_masked(self):
        data = [
            {
                "turns": [
                    {"role": "user", "content": "Hi"},
                    {"role": "user", "content": "Hello"},
                ]
            }
        ]
        tok = _make_tokenizer()
        result = tokenize_multiturn(data, tok)
        labels = result[0]["labels"]
        assert all(l == IGNORE_INDEX for l in labels), "No assistant turns = all masked"

    def test_empty_turns_skipped(self):
        data = [{"turns": []}]
        tok = _make_tokenizer()
        result = tokenize_multiturn(data, tok)
        assert result == []

    def test_empty_data(self):
        tok = _make_tokenizer()
        result = tokenize_multiturn([], tok)
        assert result == []

    def test_already_tokenized_passthrough(self):
        data = [{"input_ids": [1, 2], "labels": [-100, 2], "turns": []}]
        tok = _make_tokenizer()
        result = tokenize_multiturn(data, tok)
        assert result is data

    def test_attention_mask_all_ones(self):
        data = [
            {
                "turns": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        ]
        tok = _make_tokenizer()
        result = tokenize_multiturn(data, tok)
        mask = result[0]["attention_mask"]
        assert all(m == 1 for m in mask)
        assert len(mask) == len(result[0]["input_ids"])


# --- Integration: format → tokenize pipeline ---


class TestFormatToTokenizePipeline:
    def test_sharegpt_multi_turn_pipeline(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "What is 2+2"},
                {"from": "gpt", "value": "4"},
                {"from": "human", "value": "And 3+3"},
                {"from": "gpt", "value": "6"},
            ]
        }
        formatted = format_sharegpt(sample)
        assert "turns" in formatted

        tok = _make_tokenizer()
        result = tokenize_multiturn([formatted], tok)
        assert len(result) == 1

        labels = result[0]["labels"]
        trainable = sum(1 for l in labels if l != IGNORE_INDEX)
        assert trainable > 0, "Assistant responses should be trainable"

    def test_chat_multi_turn_pipeline(self):
        sample = {
            "messages": [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Bye"},
                {"role": "assistant", "content": "Goodbye"},
            ]
        }
        formatted = format_chat(sample)
        assert "turns" in formatted

        tok = _make_tokenizer()
        result = tokenize_multiturn([formatted], tok)
        labels = result[0]["labels"]

        masked = sum(1 for l in labels if l == IGNORE_INDEX)
        trainable = sum(1 for l in labels if l != IGNORE_INDEX)
        assert masked > 0, "System + user turns should be masked"
        assert trainable > 0, "Assistant turns should be trainable"

    def test_alpaca_single_turn_pipeline(self):
        sample = {"instruction": "Say hello", "input": "", "output": "Hello!"}
        formatted = format_alpaca(sample)
        assert "prompt_text" in formatted

        tok = _make_tokenizer()
        result = tokenize_dataset([formatted], tok)
        labels = result[0]["labels"]

        masked = sum(1 for l in labels if l == IGNORE_INDEX)
        trainable = sum(1 for l in labels if l != IGNORE_INDEX)
        assert masked > 0
        assert trainable > 0
