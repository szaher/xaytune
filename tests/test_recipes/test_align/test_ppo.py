import pytest
import torch
from trainlib.recipes.align.ppo import ppo_clip_loss, reinforce_loss


class TestPPOClipLoss:
    def test_basic_loss_computation(self):
        logprobs = torch.tensor([-1.0, -2.0, -1.5])
        old_logprobs = torch.tensor([-1.1, -2.2, -1.4])
        advantages = torch.tensor([1.0, -0.5, 0.2])

        loss = ppo_clip_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_no_clip_when_ratio_near_one(self):
        logprobs = torch.tensor([-1.0])
        old_logprobs = torch.tensor([-1.0])
        advantages = torch.tensor([1.0])

        loss = ppo_clip_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
        )

        # ratio = 1.0, no clipping, loss = -(1.0 * 1.0) = -1.0
        assert abs(loss.item() - (-1.0)) < 1e-5

    def test_clipping_limits_positive_advantage(self):
        old_logprobs = torch.tensor([-2.0])
        advantages = torch.tensor([1.0])

        loss_small_ratio = ppo_clip_loss(
            logprobs=torch.tensor([-1.8]),
            old_logprobs=old_logprobs,
            advantages=advantages,
            clip_eps=0.2,
        )

        loss_large_ratio = ppo_clip_loss(
            logprobs=torch.tensor([-0.5]),
            old_logprobs=old_logprobs,
            advantages=advantages,
            clip_eps=0.2,
        )

        # Both should produce finite losses
        assert torch.isfinite(loss_large_ratio)
        assert torch.isfinite(loss_small_ratio)

    def test_custom_clip_eps(self):
        logprobs = torch.tensor([-1.0])
        old_logprobs = torch.tensor([-2.0])
        advantages = torch.tensor([1.0])

        loss_tight = ppo_clip_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            clip_eps=0.1,
        )

        loss_loose = ppo_clip_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            clip_eps=0.5,
        )

        assert loss_tight.item() != loss_loose.item()

    def test_negative_advantage_flips_clipping(self):
        logprobs = torch.tensor([-1.0])
        old_logprobs = torch.tensor([-2.0])

        loss_pos_adv = ppo_clip_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=torch.tensor([1.0]),
        )

        loss_neg_adv = ppo_clip_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=torch.tensor([-1.0]),
        )

        assert loss_pos_adv.item() != loss_neg_adv.item()

    def test_batch_dimension(self):
        batch_size = 8
        logprobs = torch.randn(batch_size)
        old_logprobs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)

        loss = ppo_clip_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
        )

        assert loss.ndim == 0

    def test_value_loss(self):
        from trainlib.recipes.align.ppo import ppo_value_loss

        values = torch.tensor([1.0, 2.0, 3.0])
        returns = torch.tensor([1.5, 2.5, 2.0])

        loss = ppo_value_loss(values=values, returns=returns)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        # MSE of [0.5, 0.5, -1.0] = (0.25 + 0.25 + 1.0) / 3 = 0.5
        assert abs(loss.item() - 0.5) < 1e-5


class TestREINFORCELoss:
    def test_basic_loss_computation(self):
        logprobs = torch.tensor([-1.0, -2.0, -1.5])
        advantages = torch.tensor([1.0, -0.5, 0.2])

        loss = reinforce_loss(
            logprobs=logprobs,
            advantages=advantages,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_reinforce_is_negative_weighted_mean(self):
        logprobs = torch.tensor([-1.0, -2.0])
        advantages = torch.tensor([1.0, 1.0])

        loss = reinforce_loss(logprobs=logprobs, advantages=advantages)

        # -((-1.0 * 1.0) + (-2.0 * 1.0)) / 2 = -(-3.0 / 2) = 1.5
        assert abs(loss.item() - 1.5) < 1e-5

    def test_zero_advantage_gives_zero_loss(self):
        logprobs = torch.tensor([-1.0, -2.0])
        advantages = torch.tensor([0.0, 0.0])

        loss = reinforce_loss(logprobs=logprobs, advantages=advantages)

        assert abs(loss.item()) < 1e-5

    def test_batch_dimension(self):
        batch_size = 8
        logprobs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)

        loss = reinforce_loss(logprobs=logprobs, advantages=advantages)

        assert loss.ndim == 0
