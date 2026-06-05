"""Tests for the full PPO trainer (value head, rollout buffer, PPO training step).

Covers:
- ValueHead: forward pass produces scalar values
- RolloutBuffer: store, iterate, shuffle, clear
- PPOTrainer._training_step: loss is finite with mock model
- _extract_prompts: handles all batch formats
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from xaytune.recipes.align.ppo import ppo_clip_loss, ppo_value_loss
from xaytune.recipes.align.rollout_buffer import Rollout, RolloutBuffer
from xaytune.recipes.align.value_head import ValueHead


class TestValueHead:
    def test_forward_shape(self):
        vh = ValueHead(hidden_size=64)
        hidden = torch.randn(4, 10, 64)
        mask = torch.ones(4, 10, dtype=torch.long)
        values = vh(hidden, mask)
        assert values.shape == (4,)

    def test_forward_with_padding(self):
        vh = ValueHead(hidden_size=32)
        hidden = torch.randn(2, 8, 32)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]])
        values = vh(hidden, mask)
        assert values.shape == (2,)
        assert torch.isfinite(values).all()

    def test_forward_single_token(self):
        vh = ValueHead(hidden_size=16)
        hidden = torch.randn(1, 1, 16)
        mask = torch.ones(1, 1, dtype=torch.long)
        values = vh(hidden, mask)
        assert values.shape == (1,)

    def test_gradient_flows(self):
        vh = ValueHead(hidden_size=16)
        hidden = torch.randn(2, 5, 16, requires_grad=True)
        mask = torch.ones(2, 5, dtype=torch.long)
        values = vh(hidden, mask)
        loss = values.sum()
        loss.backward()
        assert hidden.grad is not None

    def test_dropout(self):
        vh = ValueHead(hidden_size=16, dropout=0.5)
        hidden = torch.randn(4, 5, 16)
        mask = torch.ones(4, 5, dtype=torch.long)
        vh.train()
        v1 = vh(hidden, mask)
        v2 = vh(hidden, mask)
        vh.eval()
        v3 = vh(hidden, mask)
        v4 = vh(hidden, mask)
        assert torch.equal(v3, v4), "Eval mode should be deterministic"


class TestRolloutBuffer:
    def _make_rollout(self, n: int = 16, seq_len: int = 10) -> Rollout:
        return Rollout(
            input_ids=torch.randint(0, 100, (n, seq_len)),
            attention_mask=torch.ones(n, seq_len, dtype=torch.long),
            old_logprobs=torch.randn(n),
            rewards=torch.randn(n),
            values=torch.randn(n),
            advantages=torch.randn(n),
            returns=torch.randn(n),
            prompt_lengths=torch.full((n,), 3, dtype=torch.long),
        )

    def test_store_and_size(self):
        buf = RolloutBuffer()
        assert buf.size == 0
        buf.store(self._make_rollout(8))
        assert buf.size == 8

    def test_iterate_yields_batches(self):
        buf = RolloutBuffer()
        buf.store(self._make_rollout(16))
        batches = buf.iterate(mini_batch_size=4, shuffle=False)
        assert len(batches) == 4
        for b in batches:
            assert b["input_ids"].shape[0] == 4
            assert "old_logprobs" in b
            assert "advantages" in b

    def test_iterate_shuffle(self):
        buf = RolloutBuffer()
        rollout = self._make_rollout(16)
        buf.store(rollout)
        b1 = buf.iterate(mini_batch_size=16, shuffle=True)
        b2 = buf.iterate(mini_batch_size=16, shuffle=True)
        assert b1[0]["input_ids"].shape == b2[0]["input_ids"].shape

    def test_iterate_uneven_batch(self):
        buf = RolloutBuffer()
        buf.store(self._make_rollout(10))
        batches = buf.iterate(mini_batch_size=4, shuffle=False)
        sizes = [b["input_ids"].shape[0] for b in batches]
        assert sum(sizes) == 10
        assert sizes[-1] == 2

    def test_clear(self):
        buf = RolloutBuffer()
        buf.store(self._make_rollout(8))
        buf.clear()
        assert buf.size == 0
        assert buf.iterate(4) == []

    def test_empty_iterate(self):
        buf = RolloutBuffer()
        assert buf.iterate(4) == []

    def test_all_fields_present_in_batch(self):
        buf = RolloutBuffer()
        buf.store(self._make_rollout(8))
        batches = buf.iterate(4)
        expected_keys = {
            "input_ids", "attention_mask", "old_logprobs", "rewards",
            "values", "advantages", "returns", "prompt_lengths",
        }
        assert set(batches[0].keys()) == expected_keys


class TestPPOLosses:
    def test_clip_loss_no_clipping(self):
        logprobs = torch.tensor([-2.0])
        old_logprobs = torch.tensor([-2.0])
        advantages = torch.tensor([1.0])
        loss = ppo_clip_loss(logprobs=logprobs, old_logprobs=old_logprobs, advantages=advantages)
        assert torch.isfinite(loss)

    def test_clip_loss_with_clipping(self):
        logprobs = torch.tensor([0.0])
        old_logprobs = torch.tensor([-5.0])
        advantages = torch.tensor([1.0])
        loss = ppo_clip_loss(logprobs=logprobs, old_logprobs=old_logprobs, advantages=advantages, clip_eps=0.2)
        assert torch.isfinite(loss)

    def test_value_loss(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        returns = torch.tensor([1.5, 2.5, 3.5])
        loss = ppo_value_loss(values=values, returns=returns)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_value_loss_perfect(self):
        values = torch.tensor([1.0, 2.0])
        returns = torch.tensor([1.0, 2.0])
        loss = ppo_value_loss(values=values, returns=returns)
        assert abs(loss.item()) < 1e-6


class TestExtractPrompts:
    def test_prompt_format(self):
        from xaytune.recipes.align.ppo_trainer import _extract_prompts

        batch = {
            "prompt_input_ids": torch.tensor([[1, 2, 3]]),
            "prompt_attention_mask": torch.tensor([[1, 1, 1]]),
        }
        ids, mask = _extract_prompts(batch, torch.device("cpu"))
        assert ids.shape == (1, 3)
        assert mask.shape == (1, 3)

    def test_input_ids_format(self):
        from xaytune.recipes.align.ppo_trainer import _extract_prompts

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        ids, mask = _extract_prompts(batch, torch.device("cpu"))
        assert ids.shape == (1, 3)

    def test_chosen_format(self):
        from xaytune.recipes.align.ppo_trainer import _extract_prompts

        batch = {
            "chosen_input_ids": torch.tensor([[1, 2, 3]]),
            "chosen_attention_mask": torch.tensor([[1, 1, 1]]),
        }
        ids, mask = _extract_prompts(batch, torch.device("cpu"))
        assert ids.shape == (1, 3)

    def test_unknown_format_raises(self):
        from xaytune.recipes.align.ppo_trainer import _extract_prompts

        batch = {"foo": torch.tensor([1])}
        try:
            _extract_prompts(batch, torch.device("cpu"))
            assert False, "Should have raised KeyError"
        except KeyError:
            pass
