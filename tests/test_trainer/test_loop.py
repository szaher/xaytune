from unittest.mock import MagicMock

import torch

from trainlib.config.schema import TrainerConfig
from trainlib.trainer.callbacks import CallbackManager
from trainlib.trainer.loop import Trainer


class TestTrainer:
    def _make_trainer(self, **kwargs):
        config = TrainerConfig(**kwargs)
        return Trainer(config=config)

    def test_init_defaults(self):
        trainer = self._make_trainer()
        assert trainer.config.batch_size == 4
        assert trainer.config.learning_rate == 2e-4
        assert trainer.config.num_epochs == 3
        assert trainer.callback_manager is not None

    def test_custom_callback_manager(self):
        cm = CallbackManager()
        trainer = Trainer(config=TrainerConfig(), callback_manager=cm)
        assert trainer.callback_manager is cm

    def test_compute_total_steps(self):
        trainer = self._make_trainer(num_epochs=3)
        total = trainer.compute_total_steps(dataset_size=100, batch_size=4)
        # 100 / 4 = 25 steps per epoch, 25 * 3 = 75 total
        assert total == 75

    def test_compute_total_steps_with_accumulation(self):
        trainer = self._make_trainer(num_epochs=2, gradient_accumulation=4)
        total = trainer.compute_total_steps(dataset_size=80, batch_size=4)
        # 80 / 4 = 20 micro-steps per epoch, 20 / 4 = 5 optimizer steps, 5 * 2 = 10
        assert total == 10

    def test_compute_total_steps_with_max_steps(self):
        trainer = self._make_trainer(num_epochs=100, max_steps=50)
        total = trainer.compute_total_steps(dataset_size=1000, batch_size=4)
        assert total == 50

    def test_train_fires_callbacks(self):
        trainer = self._make_trainer(num_epochs=1, max_steps=2)
        events = []

        @trainer.callback_manager.on("train_start")
        def on_start(state):
            events.append("train_start")

        @trainer.callback_manager.on("step_start")
        def on_step_start(state):
            events.append(f"step_start:{state.global_step}")

        @trainer.callback_manager.on("step_end")
        def on_step_end(state):
            events.append(f"step_end:{state.global_step}")

        @trainer.callback_manager.on("train_end")
        def on_end(state):
            events.append("train_end")

        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = MagicMock()
        mock_model.return_value.loss.item.return_value = 0.5
        mock_model.return_value.loss.backward = MagicMock()

        mock_optimizer = MagicMock()

        mock_dataloader = [
            {"input_ids": MagicMock(), "attention_mask": MagicMock(), "labels": MagicMock()},
            {"input_ids": MagicMock(), "attention_mask": MagicMock(), "labels": MagicMock()},
            {"input_ids": MagicMock(), "attention_mask": MagicMock(), "labels": MagicMock()},
        ]

        trainer.train(model=mock_model, train_dataloader=mock_dataloader, optimizer=mock_optimizer)

        assert "train_start" in events
        assert "train_end" in events
        assert "step_start:0" in events

    def test_early_stopping_via_callback(self):
        trainer = self._make_trainer(num_epochs=100, max_steps=-1)

        @trainer.callback_manager.on("step_end")
        def stop_early(state):
            if state.global_step >= 1:
                state.stop_training()

        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = MagicMock()
        mock_model.return_value.loss.item.return_value = 0.5
        mock_model.return_value.loss.backward = MagicMock()

        mock_optimizer = MagicMock()

        mock_dataloader = [
            {"input_ids": MagicMock(), "attention_mask": MagicMock(), "labels": MagicMock()}
            for _ in range(100)
        ]

        state = trainer.train(
            model=mock_model, train_dataloader=mock_dataloader, optimizer=mock_optimizer
        )
        assert state.should_stop is True
        assert state.global_step <= 2


class TestMixedPrecision:
    def _make_model_and_dataloader(self):
        """Helper to create mock model and dataloader."""
        model = MagicMock()
        model.parameters.return_value = iter([torch.randn(10, requires_grad=True)])
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.5, requires_grad=True)
        model.return_value = mock_output
        model.__call__ = MagicMock(return_value=mock_output)

        dataloader = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
        ]
        return model, dataloader

    def test_fp32_no_autocast(self):
        """fp32 should not use autocast at all -- current behavior preserved."""
        config = TrainerConfig(mixed_precision="fp32", num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader()
        state = trainer.train(model=model, train_dataloader=dl)
        assert state.global_step == 1
        assert trainer._amp_dtype is None
        assert trainer._scaler is None

    def test_bf16_sets_autocast_dtype(self):
        """bf16 should use autocast with bfloat16, no scaler."""
        config = TrainerConfig(mixed_precision="bf16", num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader()
        state = trainer.train(model=model, train_dataloader=dl)
        assert state.global_step == 1
        assert trainer._amp_dtype == torch.bfloat16
        assert trainer._scaler is None

    def test_fp16_on_cpu_no_scaler(self):
        """fp16 on CPU should set autocast dtype but no scaler."""
        config = TrainerConfig(mixed_precision="fp16", num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader()
        state = trainer.train(model=model, train_dataloader=dl)
        assert state.global_step == 1
        assert trainer._amp_dtype == torch.float16
        assert trainer._scaler is None

    def test_training_completes_with_all_precision_modes(self):
        """All three precision modes should complete training successfully."""
        for mode in ("fp16", "bf16", "fp32"):
            config = TrainerConfig(mixed_precision=mode, num_epochs=1, max_steps=2)
            trainer = Trainer(config=config)
            model, dl = self._make_model_and_dataloader()
            # Add a second batch so max_steps=2 is reachable
            dl.append(
                {"input_ids": torch.tensor([4, 5, 6]), "labels": torch.tensor([4, 5, 6])}
            )
            state = trainer.train(model=model, train_dataloader=dl)
            assert state.global_step == 2, f"Failed for {mode}"
