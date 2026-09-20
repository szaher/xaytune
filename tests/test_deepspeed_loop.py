"""Tests for DeepSpeed engine delegation in the training loop.

Covers:
- BUG-036: DeepSpeed engine backward/step delegation.
- _is_deepspeed_engine detection.
- Optimizer creation skipped for DS engines.
- model.backward(loss) called instead of loss.backward().
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from xaytune.config.schema import TrainerConfig
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


class TestDeepSpeedResume:
    def test_resume_does_not_crash_when_deepspeed_owns_the_optimizer(self, tmp_path, caplog):
        """The DS path sets optimizer to None; restoring state must not crash.

        Regression: train() called optimizer.load_state_dict() unguarded while
        the adjacent scaler and scheduler branches both checked for None, so
        DeepSpeed + resume_checkpoint_dir raised AttributeError.
        """
        config = TrainerConfig(num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)

        mock_model = MagicMock()
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.42
        mock_output = MagicMock()
        mock_output.loss = mock_loss
        mock_model.return_value = mock_output

        # A checkpoint that does contain optimizer state, so the branch is taken.
        torch.save({"state": {}, "param_groups": []}, tmp_path / "optimizer.pt")

        dl = [{"input_ids": torch.tensor([1, 2, 3])}]

        with patch.object(Trainer, "_is_deepspeed_engine", return_value=True):
            with caplog.at_level(logging.WARNING):
                state = trainer.train(
                    model=mock_model,
                    train_dataloader=dl,
                    resume_checkpoint_dir=str(tmp_path),
                )

        assert state.global_step == 1
        assert trainer._optimizer is None
        # The skip is reported rather than silently resuming from a fresh optimizer.
        assert any("DeepSpeed manages optimizer state" in r.message for r in caplog.records)

    def test_non_deepspeed_resume_restores_optimizer_state(self, tmp_path):
        config = TrainerConfig(num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)

        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")

        target = MagicMock()
        target.load_state_dict = MagicMock()
        target.param_groups = optimizer.param_groups
        target.zero_grad = MagicMock()
        target.step = MagicMock()

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.loss = MagicMock()
        mock_output.loss.item.return_value = 0.1
        mock_model.return_value = mock_output

        dl = [{"input_ids": torch.tensor([1, 2, 3])}]

        with patch.object(Trainer, "_is_deepspeed_engine", return_value=False):
            trainer.train(
                model=mock_model,
                train_dataloader=dl,
                optimizer=target,
                scheduler=MagicMock(),
                resume_checkpoint_dir=str(tmp_path),
            )

        target.load_state_dict.assert_called_once()


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
            trainer.train(
                model=mock_model,
                train_dataloader=dl,
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
            )

        # Last loss should be recorded
        assert abs(state.metrics["loss"] - 0.3) < 1e-5
        assert state.global_step == 2


class TestDeepSpeedSchedulerSkip:
    """A DeepSpeed run supplies no scheduler, and none may be built for it.

    Every production entrypoint -- ``recipes.finetune``, ``recipes.pretrain``,
    ``recipes.align`` and ``studio.jobs`` -- calls ``train()`` without a
    ``scheduler``.  Every DeepSpeed test in this module used to pass
    ``scheduler=MagicMock()``, which is what let the trainer-side scheduler
    branch go unexercised on the DeepSpeed path: with the optimizer set to None
    it reached ``LambdaLR(None, ...)`` and raised ``AttributeError: 'NoneType'
    object has no attribute 'param_groups'`` before the first batch.  Those
    injections are gone, so the DeepSpeed tests now exercise the path that
    production actually takes.
    """

    @staticmethod
    def _engine() -> MagicMock:
        model = MagicMock()
        output = MagicMock()
        output.loss = MagicMock()
        output.loss.item.return_value = 0.5
        model.return_value = output
        return model

    def test_deepspeed_train_without_scheduler_does_not_raise(self):
        config = TrainerConfig(num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)
        model = self._engine()
        dl = [{"input_ids": torch.tensor([1, 2, 3])}]

        with patch.object(Trainer, "_is_deepspeed_engine", return_value=True):
            state = trainer.train(model=model, train_dataloader=dl)

        assert trainer._optimizer is None
        assert trainer._scheduler is None
        assert state.global_step == 1
        model.backward.assert_called()
        model.step.assert_called()

    def test_deepspeed_train_without_scheduler_never_builds_one(self):
        """The guard is the absence of the call, not just the absence of a crash."""
        config = TrainerConfig(num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)
        dl = [{"input_ids": torch.tensor([1, 2, 3])}]

        with (
            patch.object(Trainer, "_is_deepspeed_engine", return_value=True),
            patch("xaytune.trainer.loop.create_scheduler") as mock_create,
        ):
            trainer.train(model=self._engine(), train_dataloader=dl)

        mock_create.assert_not_called()

    def test_explicit_scheduler_under_deepspeed_is_refused(self):
        """Retaining a scheduler is not the same as honouring it.

        ``training_step()`` returns early on the DeepSpeed branch, so
        ``self._scheduler.step()`` is unreachable no matter how the scheduler
        got there.  Accepting one and silently never stepping it would leave a
        caller believing their LR schedule was in effect, so it is refused.
        The supported routes are the DeepSpeed config or ``ds.initialize()``.
        """
        config = TrainerConfig(num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)
        dl = [{"input_ids": torch.tensor([1, 2, 3])}]

        with patch.object(Trainer, "_is_deepspeed_engine", return_value=True):
            with pytest.raises(ValueError, match="never be stepped"):
                trainer.train(
                    model=self._engine(),
                    train_dataloader=dl,
                    scheduler=MagicMock(),
                )

    def test_non_deepspeed_without_scheduler_still_builds_one(self):
        """The skip is conditional on DeepSpeed, not a blanket removal."""
        config = TrainerConfig(num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)

        model = MagicMock()
        model.parameters.return_value = iter([torch.randn(4, requires_grad=True)])
        output = MagicMock()
        output.loss = torch.tensor(0.5, requires_grad=True)
        model.return_value = output

        dl = [{"input_ids": torch.tensor([1, 2, 3])}]

        trainer.train(model=model, train_dataloader=dl)

        assert trainer._scheduler is not None
