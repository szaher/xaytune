import torch

from trainlib.recipes.align.simpo import simpo_loss


class TestSimPOLoss:
    def test_basic_loss_computation(self):
        policy_chosen_logps = torch.tensor([-5.0, -10.0])
        policy_rejected_logps = torch.tensor([-15.0, -20.0])
        chosen_lengths = torch.tensor([5, 10])
        rejected_lengths = torch.tensor([5, 10])

        loss = simpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_chosen_preferred_gives_lower_loss(self):
        chosen_lengths = torch.tensor([5])
        rejected_lengths = torch.tensor([5])

        loss_good = simpo_loss(
            policy_chosen_logps=torch.tensor([-2.0]),
            policy_rejected_logps=torch.tensor([-10.0]),
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
        )

        loss_bad = simpo_loss(
            policy_chosen_logps=torch.tensor([-10.0]),
            policy_rejected_logps=torch.tensor([-2.0]),
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
        )

        assert loss_good.item() < loss_bad.item()

    def test_length_normalization_matters(self):
        loss_same_len = simpo_loss(
            policy_chosen_logps=torch.tensor([-10.0]),
            policy_rejected_logps=torch.tensor([-10.0]),
            chosen_lengths=torch.tensor([10]),
            rejected_lengths=torch.tensor([10]),
        )

        loss_diff_len = simpo_loss(
            policy_chosen_logps=torch.tensor([-10.0]),
            policy_rejected_logps=torch.tensor([-10.0]),
            chosen_lengths=torch.tensor([5]),
            rejected_lengths=torch.tensor([20]),
        )

        assert loss_same_len.item() != loss_diff_len.item()

    def test_custom_beta(self):
        policy_chosen_logps = torch.tensor([-5.0])
        policy_rejected_logps = torch.tensor([-10.0])
        chosen_lengths = torch.tensor([5])
        rejected_lengths = torch.tensor([5])

        loss_low_beta = simpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
            beta=0.5,
        )

        loss_high_beta = simpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
            beta=5.0,
        )

        assert loss_low_beta.item() != loss_high_beta.item()

    def test_custom_gamma(self):
        policy_chosen_logps = torch.tensor([-5.0])
        policy_rejected_logps = torch.tensor([-10.0])
        chosen_lengths = torch.tensor([5])
        rejected_lengths = torch.tensor([5])

        loss_no_gamma = simpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
            gamma=0.0,
        )

        loss_with_gamma = simpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
            gamma=1.0,
        )

        assert loss_with_gamma.item() > loss_no_gamma.item()

    def test_equal_normalized_logps_and_zero_gamma(self):
        policy_chosen_logps = torch.tensor([-5.0])
        policy_rejected_logps = torch.tensor([-10.0])
        chosen_lengths = torch.tensor([5])
        rejected_lengths = torch.tensor([10])

        loss = simpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
            beta=1.0,
            gamma=0.0,
        )

        # chosen_avg = -5/5 = -1.0, rejected_avg = -10/10 = -1.0
        # logits = 1.0 * (-1.0 - (-1.0)) - 0.0 = 0.0
        # loss = -logsigmoid(0) = log(2)
        assert abs(loss.item() - 0.6931) < 0.01

    def test_batch_dimension(self):
        batch_size = 8
        policy_chosen_logps = torch.randn(batch_size) * 5
        policy_rejected_logps = torch.randn(batch_size) * 5
        chosen_lengths = torch.randint(1, 50, (batch_size,))
        rejected_lengths = torch.randint(1, 50, (batch_size,))

        loss = simpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
        )

        assert loss.ndim == 0
