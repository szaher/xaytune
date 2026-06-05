"""Tests for preference prompt masking.

Covers:
- BUG-032: get_sequence_logps with prompt_length excludes prompt tokens.
- tokenize_preference_dataset includes prompt_length in output.
"""

import torch
from unittest.mock import MagicMock

from xaytune.recipes.align.logprobs import get_sequence_logps
from xaytune.data.tokenizer import tokenize_preference_dataset


class TestPreferencePromptMasking:
    def test_prompt_length_zero_no_masking(self):
        """prompt_length=0 should behave identically to no prompt_length."""
        logits = torch.randn(1, 10, 100)
        labels = torch.randint(0, 100, (1, 10))
        result_masked = get_sequence_logps(logits, labels, prompt_length=0)
        result_normal = get_sequence_logps(logits, labels)
        assert torch.allclose(result_masked, result_normal)

    def test_prompt_length_excludes_tokens(self):
        """Masking prompt tokens should change the result."""
        logits = torch.randn(1, 10, 100)
        labels = torch.randint(0, 100, (1, 10))
        full = get_sequence_logps(logits, labels, prompt_length=0)
        masked = get_sequence_logps(logits, labels, prompt_length=5)
        # Masked zeros out first 5 positions, so the sum should differ
        # (unless those positions happened to contribute exactly 0, which is
        # astronomically unlikely with random logits).
        assert not torch.allclose(full, masked), (
            "Masking 5 prompt tokens should change the sequence log probability"
        )

    def test_prompt_length_reduces_magnitude(self):
        """With fewer contributing tokens, absolute value should generally be smaller."""
        torch.manual_seed(42)
        logits = torch.randn(1, 20, 50)
        labels = torch.randint(0, 50, (1, 20))
        full = get_sequence_logps(logits, labels, prompt_length=0)
        masked = get_sequence_logps(logits, labels, prompt_length=10)
        # Log probs are negative, so sum of fewer negatives -> closer to zero
        assert masked > full or masked.abs() < full.abs()

    def test_prompt_length_tensor(self):
        """prompt_length as a tensor should mask different lengths per batch item."""
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        prompt_lengths = torch.tensor([3, 5])
        result = get_sequence_logps(logits, labels, prompt_length=prompt_lengths)
        assert result.shape == (2,)
        assert torch.isfinite(result).all()

    def test_prompt_length_tensor_per_item(self):
        """Each batch item should be masked independently."""
        torch.manual_seed(0)
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))

        # Mask nothing for item 0, mask 5 tokens for item 1
        prompt_lengths = torch.tensor([0, 5])
        result = get_sequence_logps(logits, labels, prompt_length=prompt_lengths)

        # Compare item 0 with an unmasked single-item computation
        full_0 = get_sequence_logps(logits[0:1], labels[0:1], prompt_length=0)
        assert torch.allclose(result[0], full_0.squeeze())

    def test_mask_all_tokens_returns_zero(self):
        """Masking all tokens should yield a log probability of zero."""
        logits = torch.randn(1, 6, 50)
        labels = torch.randint(0, 50, (1, 6))
        # get_per_token_logps produces (1, 5) from logits[:, :-1, :] and labels[:, 1:]
        # So prompt_length=5 zeros out all 5 positions.
        result = get_sequence_logps(logits, labels, prompt_length=5)
        assert torch.allclose(result, torch.tensor(0.0))


class TestPreferenceTokenizationPromptLength:
    def _make_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.model_max_length = 512

        def mock_call(text, **kwargs):
            # Simple mock: each character is a token
            ids = list(range(len(text)))
            return {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
            }

        tokenizer.side_effect = mock_call
        tokenizer.__call__ = mock_call
        return tokenizer

    def test_prompt_length_in_output(self):
        """tokenize_preference_dataset should include prompt_length in each output dict."""
        tokenizer = self._make_tokenizer()
        data = [
            {
                "prompt": "What is AI?",
                "chosen": " It is artificial intelligence.",
                "rejected": " I don't know.",
            }
        ]
        result = tokenize_preference_dataset(data, tokenizer)
        assert len(result) == 1
        assert "prompt_length" in result[0]
        assert result[0]["prompt_length"] > 0

    def test_no_prompt_gives_zero_length(self):
        """When prompt is empty, prompt_length should be 0."""
        tokenizer = self._make_tokenizer()
        data = [
            {
                "prompt": "",
                "chosen": "Good answer.",
                "rejected": "Bad answer.",
            }
        ]
        result = tokenize_preference_dataset(data, tokenizer)
        assert len(result) == 1
        assert result[0]["prompt_length"] == 0

    def test_all_preference_keys_present(self):
        """Output should contain all required keys for preference training."""
        tokenizer = self._make_tokenizer()
        data = [
            {
                "prompt": "Hello",
                "chosen": " world",
                "rejected": " there",
            }
        ]
        result = tokenize_preference_dataset(data, tokenizer)
        assert len(result) == 1
        item = result[0]
        expected_keys = {
            "chosen_input_ids",
            "chosen_attention_mask",
            "rejected_input_ids",
            "rejected_attention_mask",
            "prompt_length",
        }
        assert expected_keys.issubset(set(item.keys()))
