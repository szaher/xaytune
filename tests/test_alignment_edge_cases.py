"""Tests for alignment loss edge cases.

Covers:
- BUG-011: ORPO numerical stability when logps=0 (probability=1).
- BUG-027: SimPO zero-length guard (division by zero).
- GRPO no-ref-model path (OOM fix: ref_logprobs=None skips KL).
"""

import torch

from xaytune.recipes.align.grpo import grpo_loss
from xaytune.recipes.align.orpo import orpo_loss
from xaytune.recipes.align.simpo import simpo_loss


class TestOrpoEdgeCases:
    def test_normal_inputs_finite(self):
        result = orpo_loss(
            sft_loss=torch.tensor(1.0),
            policy_chosen_logps=torch.tensor(-2.0),
            policy_rejected_logps=torch.tensor(-3.0),
        )
        assert torch.isfinite(result)

    def test_logprobs_at_zero_no_nan(self):
        # Edge case: logps=0 means probability=1. Previously caused div-by-zero
        # in log1p(-exp(0)) = log1p(-1) = log(0) = -inf.
        result = orpo_loss(
            sft_loss=torch.tensor(1.0),
            policy_chosen_logps=torch.tensor(0.0),
            policy_rejected_logps=torch.tensor(-5.0),
        )
        assert torch.isfinite(result), "ORPO should not produce NaN/Inf for logps=0"

    def test_both_logprobs_zero(self):
        result = orpo_loss(
            sft_loss=torch.tensor(1.0),
            policy_chosen_logps=torch.tensor(0.0),
            policy_rejected_logps=torch.tensor(0.0),
        )
        assert torch.isfinite(result)

    def test_very_negative_logprobs(self):
        result = orpo_loss(
            sft_loss=torch.tensor(1.0),
            policy_chosen_logps=torch.tensor(-100.0),
            policy_rejected_logps=torch.tensor(-100.0),
        )
        assert torch.isfinite(result)

    def test_chosen_higher_than_rejected(self):
        result = orpo_loss(
            sft_loss=torch.tensor(0.5),
            policy_chosen_logps=torch.tensor(-1.0),
            policy_rejected_logps=torch.tensor(-5.0),
        )
        assert torch.isfinite(result)
        # Loss should be lower when chosen is clearly preferred
        result_equal = orpo_loss(
            sft_loss=torch.tensor(0.5),
            policy_chosen_logps=torch.tensor(-3.0),
            policy_rejected_logps=torch.tensor(-3.0),
        )
        assert result < result_equal

    def test_lambda_weight_zero_returns_sft_loss(self):
        sft = torch.tensor(2.5)
        result = orpo_loss(
            sft_loss=sft,
            policy_chosen_logps=torch.tensor(-1.0),
            policy_rejected_logps=torch.tensor(-5.0),
            lambda_weight=0.0,
        )
        assert torch.allclose(result, sft)


class TestSimpoEdgeCases:
    def test_normal_inputs(self):
        result = simpo_loss(
            policy_chosen_logps=torch.tensor(-2.0),
            policy_rejected_logps=torch.tensor(-3.0),
            chosen_lengths=torch.tensor(10),
            rejected_lengths=torch.tensor(10),
            beta=0.1,
            gamma=0.5,
        )
        assert torch.isfinite(result)

    def test_zero_length_no_crash(self):
        # Previously would divide by zero; clamp(min=1) guards this.
        result = simpo_loss(
            policy_chosen_logps=torch.tensor(-2.0),
            policy_rejected_logps=torch.tensor(-3.0),
            chosen_lengths=torch.tensor(0),
            rejected_lengths=torch.tensor(0),
            beta=0.1,
            gamma=0.5,
        )
        assert torch.isfinite(result), "SimPO should clamp lengths to min=1"

    def test_one_zero_length(self):
        result = simpo_loss(
            policy_chosen_logps=torch.tensor(-2.0),
            policy_rejected_logps=torch.tensor(-3.0),
            chosen_lengths=torch.tensor(0),
            rejected_lengths=torch.tensor(10),
            beta=0.1,
            gamma=0.5,
        )
        assert torch.isfinite(result)

    def test_chosen_preferred_lower_loss(self):
        # When chosen avg logp > rejected avg logp, loss should be lower.
        loss_preferred = simpo_loss(
            policy_chosen_logps=torch.tensor(-1.0),
            policy_rejected_logps=torch.tensor(-5.0),
            chosen_lengths=torch.tensor(5),
            rejected_lengths=torch.tensor(5),
            beta=2.0,
            gamma=0.5,
        )
        loss_equal = simpo_loss(
            policy_chosen_logps=torch.tensor(-3.0),
            policy_rejected_logps=torch.tensor(-3.0),
            chosen_lengths=torch.tensor(5),
            rejected_lengths=torch.tensor(5),
            beta=2.0,
            gamma=0.5,
        )
        assert loss_preferred < loss_equal


class TestGrpoNoRefModel:
    def test_no_ref_model_no_kl(self):
        result = grpo_loss(
            logprobs=torch.tensor(-2.0),
            ref_logprobs=None,
            advantages=torch.tensor(1.0),
            kl_coeff=0.0,
        )
        assert torch.isfinite(result)

    def test_no_ref_model_with_kl_coeff(self):
        # ref_logprobs=None but kl_coeff>0 -- should skip KL term entirely.
        result = grpo_loss(
            logprobs=torch.tensor(-2.0),
            ref_logprobs=None,
            advantages=torch.tensor(1.0),
            kl_coeff=0.04,
        )
        assert torch.isfinite(result)

    def test_no_ref_matches_zero_kl(self):
        # With no ref model, result should equal -(logprobs * advantages).mean()
        logprobs = torch.tensor(-2.0)
        advantages = torch.tensor(1.0)
        result = grpo_loss(
            logprobs=logprobs,
            ref_logprobs=None,
            advantages=advantages,
            kl_coeff=0.04,
        )
        expected = -(logprobs * advantages).mean()
        assert torch.allclose(result, expected)

    def test_with_ref_model(self):
        result = grpo_loss(
            logprobs=torch.tensor(-2.0),
            ref_logprobs=torch.tensor(-2.5),
            advantages=torch.tensor(1.0),
            kl_coeff=0.04,
        )
        assert torch.isfinite(result)

    def test_with_ref_model_includes_kl(self):
        logprobs = torch.tensor(-2.0)
        ref_logprobs = torch.tensor(-2.5)
        advantages = torch.tensor(1.0)
        kl_coeff = 0.04

        result = grpo_loss(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            kl_coeff=kl_coeff,
        )

        policy_loss = -(logprobs * advantages).mean()
        kl = (logprobs - ref_logprobs).mean()
        expected = policy_loss + kl_coeff * kl
        assert torch.allclose(result, expected)
