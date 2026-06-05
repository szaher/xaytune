"""Tests for ORPO end-to-end (BUG-033 / TASK-027).

Verifies that _orpo_step computes its own SFT loss when outputs=None
(the skip_forward path in the training loop passes None for preference
batches). Previously this crashed with TypeError.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from xaytune.recipes.align.loss_dispatch import (
    _orpo_step,
    create_alignment_loss_fn,
)
from xaytune.recipes.align.orpo import orpo_loss


class TestOrpoLoss:
    def test_basic_orpo_loss(self):
        result = orpo_loss(
            sft_loss=torch.tensor(1.0),
            policy_chosen_logps=torch.tensor(-2.0),
            policy_rejected_logps=torch.tensor(-3.0),
        )
        assert torch.isfinite(result)

    def test_orpo_loss_chosen_better(self):
        result = orpo_loss(
            sft_loss=torch.tensor(0.5),
            policy_chosen_logps=torch.tensor(-1.0),
            policy_rejected_logps=torch.tensor(-5.0),
        )
        assert torch.isfinite(result)


class TestOrpoStepWithNoneOutputs:
    def _make_model(self):
        model = MagicMock()
        logits = torch.randn(2, 10, 100)
        loss = torch.tensor(0.5, requires_grad=True)

        def forward(**kwargs):
            out = MagicMock()
            out.logits = logits
            out.loss = loss
            return out

        model.side_effect = forward
        model.__call__ = forward
        return model

    def test_orpo_step_outputs_none_no_crash(self):
        model = self._make_model()
        batch = {
            "chosen_input_ids": torch.randint(0, 100, (2, 10)),
            "chosen_attention_mask": torch.ones(2, 10, dtype=torch.long),
            "rejected_input_ids": torch.randint(0, 100, (2, 10)),
            "rejected_attention_mask": torch.ones(2, 10, dtype=torch.long),
        }
        result = _orpo_step(model, batch, None, lambda_weight=1.0)
        assert torch.isfinite(result), "ORPO should work with outputs=None"

    def test_orpo_step_with_valid_outputs(self):
        model = self._make_model()
        outputs = MagicMock()
        outputs.loss = torch.tensor(0.5)
        batch = {
            "chosen_input_ids": torch.randint(0, 100, (2, 10)),
            "chosen_attention_mask": torch.ones(2, 10, dtype=torch.long),
            "rejected_input_ids": torch.randint(0, 100, (2, 10)),
            "rejected_attention_mask": torch.ones(2, 10, dtype=torch.long),
        }
        result = _orpo_step(model, batch, outputs, lambda_weight=1.0)
        assert torch.isfinite(result)

    def test_orpo_step_with_prompt_length(self):
        model = self._make_model()
        batch = {
            "chosen_input_ids": torch.randint(0, 100, (2, 10)),
            "chosen_attention_mask": torch.ones(2, 10, dtype=torch.long),
            "rejected_input_ids": torch.randint(0, 100, (2, 10)),
            "rejected_attention_mask": torch.ones(2, 10, dtype=torch.long),
            "prompt_length": torch.tensor([3, 4]),
        }
        result = _orpo_step(model, batch, None, lambda_weight=1.0)
        assert torch.isfinite(result)


class TestOrpoViaLossDispatch:
    def test_create_alignment_loss_fn_orpo(self):
        loss_fn = create_alignment_loss_fn(method="orpo")
        assert callable(loss_fn)

    def test_orpo_loss_fn_handles_none_outputs(self):
        loss_fn = create_alignment_loss_fn(method="orpo")
        model = MagicMock()
        logits = torch.randn(2, 10, 100)
        loss = torch.tensor(0.5, requires_grad=True)

        def forward(**kwargs):
            out = MagicMock()
            out.logits = logits
            out.loss = loss
            return out

        model.side_effect = forward
        model.__call__ = forward

        batch = {
            "chosen_input_ids": torch.randint(0, 100, (2, 10)),
            "chosen_attention_mask": torch.ones(2, 10, dtype=torch.long),
            "rejected_input_ids": torch.randint(0, 100, (2, 10)),
            "rejected_attention_mask": torch.ones(2, 10, dtype=torch.long),
        }
        result = loss_fn(model, batch, None)
        assert torch.isfinite(result), "ORPO via loss dispatch should handle None outputs"
