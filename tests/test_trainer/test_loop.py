from unittest.mock import MagicMock

import torch

from trainlib.config.schema import TrainerConfig
from trainlib.trainer.callbacks import CallbackManager, TrainState
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

        trainer.train(
            model=mock_model,
            train_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=MagicMock(),
        )

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
            model=mock_model,
            train_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            scheduler=MagicMock(),
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

    def test_bf16_on_cpu_disables_amp(self):
        """bf16 on CPU should disable AMP (CPU doesn't support autocast)."""
        config = TrainerConfig(mixed_precision="bf16", num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader()
        state = trainer.train(model=model, train_dataloader=dl)
        assert state.global_step == 1
        assert trainer._amp_dtype is None
        assert trainer._scaler is None

    def test_fp16_on_cpu_disables_amp(self):
        """fp16 on CPU should disable AMP (CPU doesn't support autocast)."""
        config = TrainerConfig(mixed_precision="fp16", num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader()
        state = trainer.train(model=model, train_dataloader=dl)
        assert state.global_step == 1
        assert trainer._amp_dtype is None
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


class TestResumeTraining:
    def _make_model_and_dataloader(self, num_samples=10):
        model = MagicMock()
        model.parameters.return_value = [torch.randn(10, requires_grad=True)]
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.5, requires_grad=True)
        model.return_value = mock_output
        model.__call__ = MagicMock(return_value=mock_output)
        dataloader = [
            {"input_ids": torch.tensor([i]), "labels": torch.tensor([i])}
            for i in range(num_samples)
        ]
        return model, dataloader

    def test_optimizer_stored_as_instance_attr(self):
        config = TrainerConfig(num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader(2)
        trainer.train(model=model, train_dataloader=dl)
        assert trainer._optimizer is not None

    def test_optimizer_stored_when_externally_provided(self):
        config = TrainerConfig(num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader(2)
        ext_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer.train(model=model, train_dataloader=dl, optimizer=ext_optimizer)
        assert trainer._optimizer is ext_optimizer

    def test_resume_none_is_default_behavior(self):
        config = TrainerConfig(num_epochs=1, max_steps=3, batch_size=1)
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader(5)
        state = trainer.train(model=model, train_dataloader=dl, resume_state=None)
        assert state.global_step == 3  # max_steps=3

    def test_resume_skips_completed_epochs(self):
        config = TrainerConfig(num_epochs=3, max_steps=-1, batch_size=1)
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader(2)

        resume = TrainState(epoch=1, step=1, global_step=4)
        state = trainer.train(model=model, train_dataloader=dl, resume_state=resume)

        # Should only run epochs 1 and 2 (skipping epoch 0)
        # Epoch 1: steps after step 1 = none (2 samples, steps 0,1 both skipped)
        # Epoch 2: both steps run = global_step goes from 4 to 6
        assert state.global_step == 6
        assert state.epoch == 2

    def test_resume_continues_global_step(self):
        config = TrainerConfig(num_epochs=2, max_steps=-1, batch_size=1)
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader(3)

        resume = TrainState(epoch=0, step=1, global_step=2)
        state = trainer.train(model=model, train_dataloader=dl, resume_state=resume)

        # Epoch 0: skip steps 0,1 -> run step 2 -> global_step 3
        # Epoch 1: run steps 0,1,2 -> global_step 6
        assert state.global_step == 6

    def test_resume_with_max_steps(self):
        config = TrainerConfig(num_epochs=10, max_steps=5, batch_size=1)
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader(10)

        resume = TrainState(epoch=0, step=2, global_step=3)
        state = trainer.train(model=model, train_dataloader=dl, resume_state=resume)

        # Resume from global_step=3, max_steps=5, so only 2 more steps
        assert state.global_step == 5
        assert state.should_stop

    def test_resume_fires_train_start(self):
        config = TrainerConfig(num_epochs=1, max_steps=1, batch_size=1)
        events = []
        cm = CallbackManager()

        @cm.on("train_start")
        def on_start(state):
            events.append("train_start")

        trainer = Trainer(config=config, callback_manager=cm)
        model, dl = self._make_model_and_dataloader(2)
        resume = TrainState(epoch=0, step=0, global_step=0)
        trainer.train(model=model, train_dataloader=dl, resume_state=resume)
        assert "train_start" in events


class TestSchedulerIntegration:
    def _make_model_and_dataloader(self, num_samples=4):
        model = MagicMock()
        model.parameters.return_value = [torch.randn(10, requires_grad=True)]
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.5, requires_grad=True)
        model.return_value = mock_output
        model.__call__ = MagicMock(return_value=mock_output)
        dataloader = [
            {"input_ids": torch.tensor([i]), "labels": torch.tensor([i])}
            for i in range(num_samples)
        ]
        return model, dataloader

    def test_scheduler_created_from_config(self):
        config = TrainerConfig(
            num_epochs=1, max_steps=2, batch_size=1, scheduler="cosine"
        )
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader()
        trainer.train(model=model, train_dataloader=dl)
        assert trainer._scheduler is not None

    def test_scheduler_stepped_each_optimizer_step(self):
        config = TrainerConfig(num_epochs=1, max_steps=3, batch_size=1)
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader(5)

        mock_scheduler = MagicMock()
        trainer.train(
            model=model, train_dataloader=dl, scheduler=mock_scheduler
        )
        assert mock_scheduler.step.call_count == 3

    def test_external_scheduler_used_when_provided(self):
        config = TrainerConfig(num_epochs=1, max_steps=1, batch_size=1)
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader()

        ext_scheduler = MagicMock()
        trainer.train(
            model=model, train_dataloader=dl, scheduler=ext_scheduler
        )
        assert trainer._scheduler is ext_scheduler

    def test_scheduler_with_warmup_steps(self):
        config = TrainerConfig(
            num_epochs=1, max_steps=4, batch_size=1,
            scheduler="cosine", warmup_steps=2,
        )
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader(6)
        state = trainer.train(model=model, train_dataloader=dl)
        assert state.global_step == 4
        assert trainer._scheduler is not None

    def test_scheduler_with_gradient_accumulation(self):
        config = TrainerConfig(
            num_epochs=1, max_steps=4, batch_size=1,
            gradient_accumulation=1, scheduler="linear",
        )
        trainer = Trainer(config=config)
        model, dl = self._make_model_and_dataloader(6)

        mock_scheduler = MagicMock()
        trainer.train(
            model=model, train_dataloader=dl, scheduler=mock_scheduler
        )
        assert mock_scheduler.step.call_count == 4


class TestCustomLossFn:
    def test_custom_loss_fn_overrides_default(self):
        config = TrainerConfig(num_epochs=1, max_steps=2)
        trainer = Trainer(config=config)

        custom_loss = MagicMock(
            return_value=torch.tensor(0.42, requires_grad=True)
        )

        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = torch.tensor(
            999.0, requires_grad=True
        )

        mock_optimizer = MagicMock()
        dl = [
            {"input_ids": torch.tensor([[1, 2]])},
            {"input_ids": torch.tensor([[3, 4]])},
        ]

        state = trainer.train(
            model=mock_model,
            train_dataloader=dl,
            optimizer=mock_optimizer,
            scheduler=MagicMock(),
            loss_fn=custom_loss,
        )

        assert custom_loss.call_count == 2
        assert abs(state.metrics["loss"] - 0.42) < 1e-5

    def test_no_loss_fn_uses_model_loss(self):
        config = TrainerConfig(num_epochs=1, max_steps=1)
        trainer = Trainer(config=config)

        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = torch.tensor(
            1.23, requires_grad=True
        )

        mock_optimizer = MagicMock()
        dl = [{"input_ids": torch.tensor([[1, 2]])}]

        state = trainer.train(
            model=mock_model,
            train_dataloader=dl,
            optimizer=mock_optimizer,
            scheduler=MagicMock(),
        )

        assert abs(state.metrics["loss"] - 1.23) < 1e-5


class TestGradientAccumulation:
    def test_accum_across_epoch_boundary(self):
        config = TrainerConfig(
            num_epochs=2, batch_size=1, gradient_accumulation=3,
        )
        trainer = Trainer(config=config)

        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.randn(4, requires_grad=True)]
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = torch.tensor(0.6, requires_grad=True)

        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()

        # 2 batches per epoch * 2 epochs = 4 micro-steps total
        # With accum=3, optimizer should step once (at micro-step 3) not twice
        dl = [
            {"input_ids": torch.tensor([1])},
            {"input_ids": torch.tensor([2])},
        ]

        trainer.train(
            model=mock_model,
            train_dataloader=dl,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

        assert mock_optimizer.step.call_count == 1

    def test_accum_steps_correctly_within_epoch(self):
        config = TrainerConfig(
            num_epochs=1, batch_size=1, gradient_accumulation=2,
        )
        trainer = Trainer(config=config)

        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.randn(4, requires_grad=True)]
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = torch.tensor(0.5, requires_grad=True)

        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()

        # 6 micro-steps / accum=2 = 3 optimizer steps
        dl = [{"input_ids": torch.tensor([i])} for i in range(6)]

        trainer.train(
            model=mock_model,
            train_dataloader=dl,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

        assert mock_optimizer.step.call_count == 3


class TestSubclassableTrainer:
    def test_custom_training_step(self):
        class MyTrainer(Trainer):
            def training_step(self, model, batch, optimizer, state):
                return 0.42

        config = TrainerConfig(num_epochs=1, max_steps=2)
        trainer = MyTrainer(config=config)

        mock_model = MagicMock()
        dl = [{"input_ids": torch.tensor([1])}, {"input_ids": torch.tensor([2])}]

        state = trainer.train(
            model=mock_model,
            train_dataloader=dl,
            optimizer=MagicMock(),
            scheduler=MagicMock(),
        )

        assert state.global_step == 2
        assert abs(state.metrics["loss"] - 0.42) < 1e-5
        mock_model.assert_not_called()

    def test_subclass_inherits_callbacks(self):
        class MyTrainer(Trainer):
            def training_step(self, model, batch, optimizer, state):
                return 0.1

        config = TrainerConfig(num_epochs=1, max_steps=2)
        events = []
        cm = CallbackManager()

        @cm.on("train_start")
        def on_start(state):
            events.append("start")

        @cm.on("step_end")
        def on_step(state):
            events.append(f"step:{state.global_step}")

        trainer = MyTrainer(config=config, callback_manager=cm)
        dl = [{"x": 1}, {"x": 2}]
        trainer.train(model=MagicMock(), train_dataloader=dl,
                      optimizer=MagicMock(), scheduler=MagicMock())

        assert "start" in events
        assert "step:1" in events
        assert "step:2" in events

    def test_subclass_with_custom_optimizer_logic(self):
        step_count = {"n": 0}

        class MyTrainer(Trainer):
            def training_step(self, model, batch, optimizer, state):
                step_count["n"] += 1
                optimizer.step()
                optimizer.zero_grad()
                return 0.5

        config = TrainerConfig(num_epochs=1, max_steps=3)
        trainer = MyTrainer(config=config)
        mock_opt = MagicMock()
        dl = [{"x": i} for i in range(5)]

        trainer.train(model=MagicMock(), train_dataloader=dl,
                      optimizer=mock_opt, scheduler=MagicMock())

        assert step_count["n"] == 3
        assert mock_opt.step.call_count == 3
