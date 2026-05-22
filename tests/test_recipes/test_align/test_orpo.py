import pytest
import torch
from trainlib.recipes.align.orpo import orpo_loss


class TestORPOLoss:
    def test_basic_loss_computation(self):
        sft_loss = torch.tensor(2.0)
        policy_chosen_logps = torch.tensor([-1.0, -2.0])
        policy_rejected_logps = torch.tensor([-3.0, -4.0])

        loss = orpo_loss(
            sft_loss=sft_loss,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_loss_includes_sft_component(self):
        policy_chosen_logps = torch.tensor([-1.0])
        policy_rejected_logps = torch.tensor([-3.0])

        loss_low_sft = orpo_loss(
            sft_loss=torch.tensor(0.5),
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
        )

        loss_high_sft = orpo_loss(
            sft_loss=torch.tensor(5.0),
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
        )

        assert loss_high_sft.item() > loss_low_sft.item()

    def test_chosen_preferred_gives_lower_or_loss(self):
        sft_loss = torch.tensor(1.0)

        loss_good = orpo_loss(
            sft_loss=sft_loss,
            policy_chosen_logps=torch.tensor([-0.5]),
            policy_rejected_logps=torch.tensor([-3.0]),
        )

        loss_bad = orpo_loss(
            sft_loss=sft_loss,
            policy_chosen_logps=torch.tensor([-3.0]),
            policy_rejected_logps=torch.tensor([-0.5]),
        )

        assert loss_good.item() < loss_bad.item()

    def test_custom_lambda(self):
        sft_loss = torch.tensor(1.0)
        policy_chosen_logps = torch.tensor([-1.0])
        policy_rejected_logps = torch.tensor([-2.0])

        loss_low_lambda = orpo_loss(
            sft_loss=sft_loss,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            lambda_weight=0.1,
        )

        loss_high_lambda = orpo_loss(
            sft_loss=sft_loss,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            lambda_weight=2.0,
        )

        assert loss_low_lambda.item() != loss_high_lambda.item()

    def test_equal_logprobs_or_component_is_log2(self):
        sft_loss = torch.tensor(0.0)
        policy_chosen_logps = torch.tensor([-2.0])
        policy_rejected_logps = torch.tensor([-2.0])

        loss = orpo_loss(
            sft_loss=sft_loss,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            lambda_weight=1.0,
        )

        # When logps are equal, odds ratio = 1, log(1) = 0, sigmoid(0) = 0.5, -log(0.5) = log(2)
        assert abs(loss.item() - 0.6931) < 0.01

    def test_batch_dimension(self):
        sft_loss = torch.tensor(1.0)
        batch_size = 8
        policy_chosen_logps = torch.randn(batch_size)
        policy_rejected_logps = torch.randn(batch_size)

        loss = orpo_loss(
            sft_loss=sft_loss,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
        )

        assert loss.ndim == 0
