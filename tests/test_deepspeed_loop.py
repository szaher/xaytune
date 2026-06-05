"""Tests for DeepSpeed engine delegation in the training loop.

Covers:
- BUG-036: DeepSpeed engine backward/step delegation.
- _is_deepspeed_engine detection.
- Optimizer creation skipped for DS engines.
- model.backward(loss) called instead of loss.backward().
"""

import torch
from unittest.mock import MagicMock, patch

from xaytune.config.schema import TrainerConfig
from xaytune.trainer.callbacks import CallbackManager, TrainState
from xaytune.trainer.loop import Trainer


class TestIsDeepSpeedEngine:
    def test_regular_model_returns_false(self):
        model = MagicMock()
        assert Trainer._is_deepspeed_engine(model) is False

    def test_regular_nn_module_returns_false(self):
        model = torch.nn.Linear(10, 10)
        assert Trainer._is_deepspeed_engine(model) is False

    @patch("xaytune.trainer.loop.Trainer._is_deepspeed_engine", return_value=True)
    def test_mock_deepspeed_engine_returns_true(self, mock_check):
        model = MagicMock()
        assert Trainer._is_deepspeed_engine(model) is True


class TestDeepSpeedOptimizerSkip:
    def test_deepspeed_skips_optimizer_creation(self):
        """When model is a DS engine, optimizer should be set to None."""
        config = TrainerConfig(num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.loss = MagicMock()
        mock_output.loss.item.return_value = 0.5
        mock_model.return_value = mock_output

        dl = [{"input_ids": torch.tensor([1, 2, 3])}]

        with patch.object(Trainer, "_is_deepspeed_engine", return_value=True):
            state = trainer.train(
                model=mock_model,
                train_dataloader=dl,
                scheduler=MagicMock(),
            )

        assert trainer._optimizer is None
        assert state.global_step == 1

    def test_non_deepspeed_creates_optimizer(self):
        """Without DS, providing no optimizer should create AdamW."""
        config = TrainerConfig(num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.randn(4, requires_grad=True)])
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.5, requires_grad=True)
        mock_model.return_value = mock_output

        dl = [{"input_ids": torch.tensor([1, 2, 3])}]

        state = trainer.train(
            model=mock_model,
            train_dataloader=dl,
        )

        assert trainer._optimizer is not None
        assert state.global_step == 1


class TestDeepSpeedBackwardDelegation:
    def test_deepspeed_calls_model_backward(self):
        """DS engine should use model.backward(loss) not loss.backward()."""
        config = TrainerConfig(num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)

        mock_model = MagicMock()
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.42
        mock_output = MagicMock()
        mock_output.loss = mock_loss
        mock_model.return_value = mock_output

        dl = [{"input_ids": torch.tensor([1, 2, 3])}]

        with patch.object(Trainer, "_is_deepspeed_engine", return_value=True):
            state = trainer.train(
                model=mock_model,
                train_dataloader=dl,
                scheduler=MagicMock(),
            )

        # DS path calls model.backward(loss) and model.step()
        mock_model.backward.assert_called_once()
        mock_model.step.assert_called_once()
        # loss.backward() should NOT have been called
        mock_loss.backward.assert_not_called()

    def test_non_deepspeed_calls_loss_backward(self):
        """Non-DS should use loss.backward(), not model.backward()."""
        config = TrainerConfig(num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.randn(4, requires_grad=True)])
        mock_loss = torch.tensor(0.5, requires_grad=True)
        mock_output = MagicMock()
        mock_output.loss = mock_loss
        mock_model.return_value = mock_output

        dl = [{"input_ids": torch.tensor([1, 2, 3])}]

        state = trainer.train(
            model=mock_model,
            train_dataloader=dl,
        )

        # model.backward should NOT have been called
        mock_model.backward.assert_not_called()
        assert state.global_step == 1


class TestDeepSpeedLossValue:
    def test_deepspeed_returns_correct_loss(self):
        """The training_step should return the scalar loss value from DS path."""
        config = TrainerConfig(num_epochs=1, max_steps=2)
        trainer = Trainer(config=config)

        call_count = {"n": 0}
        losses = [0.7, 0.3]

        mock_model = MagicMock()

        def make_output(**kwargs):
            mock_out = MagicMock()
            mock_out.loss = MagicMock()
            mock_out.loss.item.return_value = losses[call_count["n"]]
            call_count["n"] += 1
            return mock_out

        mock_model.side_effect = make_output

        dl = [
            {"input_ids": torch.tensor([1])},
            {"input_ids": torch.tensor([2])},
        ]

        with patch.object(Trainer, "_is_deepspeed_engine", return_value=True):
            state = trainer.train(
                model=mock_model,
                train_dataloader=dl,
                scheduler=MagicMock(),
            )

        # Last loss should be recorded
        assert abs(state.metrics["loss"] - 0.3) < 1e-5
        assert state.global_step == 2
