import torch

from trainlib.recipes.align.dpo import dpo_loss


class TestDPOLoss:
    def test_basic_loss_computation(self):
        policy_chosen_logps = torch.tensor([-1.0, -2.0])
        policy_rejected_logps = torch.tensor([-3.0, -4.0])
        ref_chosen_logps = torch.tensor([-1.5, -2.5])
        ref_rejected_logps = torch.tensor([-3.5, -4.5])

        loss = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_loss_decreases_when_chosen_preferred(self):
        policy_chosen_logps = torch.tensor([-0.5])
        policy_rejected_logps = torch.tensor([-3.0])
        ref_chosen_logps = torch.tensor([-1.0])
        ref_rejected_logps = torch.tensor([-1.0])

        loss = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )

        assert loss.item() < 1.0

    def test_loss_increases_when_rejected_preferred(self):
        policy_chosen_logps = torch.tensor([-3.0])
        policy_rejected_logps = torch.tensor([-0.5])
        ref_chosen_logps = torch.tensor([-1.0])
        ref_rejected_logps = torch.tensor([-1.0])

        loss = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )

        assert loss.item() > 0.5

    def test_custom_beta(self):
        policy_chosen_logps = torch.tensor([-1.0])
        policy_rejected_logps = torch.tensor([-2.0])
        ref_chosen_logps = torch.tensor([-1.2])
        ref_rejected_logps = torch.tensor([-2.5])

        loss_low_beta = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            beta=0.05,
        )

        loss_high_beta = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            beta=0.5,
        )

        assert loss_low_beta.item() != loss_high_beta.item()

    def test_equal_logprobs_gives_log2_loss(self):
        policy_chosen_logps = torch.tensor([-1.0])
        policy_rejected_logps = torch.tensor([-1.0])
        ref_chosen_logps = torch.tensor([-1.0])
        ref_rejected_logps = torch.tensor([-1.0])

        loss = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )

        # When logprob ratios are equal, sigmoid(0) = 0.5, -log(0.5) = log(2)
        assert abs(loss.item() - 0.6931) < 0.01

    def test_batch_dimension(self):
        batch_size = 8
        policy_chosen_logps = torch.randn(batch_size)
        policy_rejected_logps = torch.randn(batch_size)
        ref_chosen_logps = torch.randn(batch_size)
        ref_rejected_logps = torch.randn(batch_size)

        loss = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )

        assert loss.ndim == 0
