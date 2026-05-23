import torch

from trainlib.recipes.align.grpo import compute_group_advantages, grpo_loss


class TestComputeGroupAdvantages:
    def test_basic_advantages(self):
        rewards = torch.tensor([1.0, 3.0, 2.0])
        advantages = compute_group_advantages(rewards)

        assert advantages.shape == rewards.shape
        # Mean-centered: highest reward gets positive advantage
        assert advantages[1] > advantages[0]
        assert advantages[1] > advantages[2]

    def test_advantages_are_normalized(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        advantages = compute_group_advantages(rewards)

        assert abs(advantages.mean().item()) < 1e-5
        assert abs(advantages.std().item() - 1.0) < 0.2

    def test_single_reward(self):
        rewards = torch.tensor([5.0])
        advantages = compute_group_advantages(rewards)

        assert advantages.shape == (1,)
        assert advantages[0].item() == 0.0

    def test_equal_rewards(self):
        rewards = torch.tensor([2.0, 2.0, 2.0])
        advantages = compute_group_advantages(rewards)

        for a in advantages:
            assert abs(a.item()) < 1e-5


class TestGRPOLoss:
    def test_basic_loss(self):
        logprobs = torch.tensor([-1.0, -2.0, -1.5])
        ref_logprobs = torch.tensor([-1.2, -2.1, -1.6])
        advantages = torch.tensor([1.0, -0.5, 0.2])

        loss = grpo_loss(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_kl_penalty_increases_loss(self):
        logprobs = torch.tensor([-1.0, -2.0])
        ref_logprobs = torch.tensor([-3.0, -4.0])
        advantages = torch.tensor([1.0, 1.0])

        loss_no_kl = grpo_loss(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            kl_coeff=0.0,
        )

        loss_with_kl = grpo_loss(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            kl_coeff=0.1,
        )

        assert loss_with_kl.item() != loss_no_kl.item()

    def test_zero_advantages_only_kl(self):
        logprobs = torch.tensor([-1.0, -2.0])
        ref_logprobs = torch.tensor([-1.5, -2.5])
        advantages = torch.tensor([0.0, 0.0])

        loss = grpo_loss(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            kl_coeff=0.1,
        )

        assert loss.item() >= 0.0

    def test_custom_kl_coeff(self):
        logprobs = torch.tensor([-1.0])
        ref_logprobs = torch.tensor([-2.0])
        advantages = torch.tensor([1.0])

        loss_low = grpo_loss(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            kl_coeff=0.01,
        )

        loss_high = grpo_loss(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            kl_coeff=0.5,
        )

        assert loss_low.item() != loss_high.item()
