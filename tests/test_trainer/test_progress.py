from __future__ import annotations

from unittest.mock import MagicMock, patch

from xaytune.trainer.callbacks import CallbackManager, TrainState
from xaytune.trainer.progress import register_progress_callbacks


class TestProgressCallbacks:
    def test_progress_created_on_train_start(self):
        cb = CallbackManager()
        with patch("rich.progress.Progress") as mock_cls:
            mock_progress = MagicMock()
            mock_progress.add_task.return_value = 0
            mock_cls.return_value = mock_progress

            register_progress_callbacks(
                callback_manager=cb,
                total_steps=100,
            )
            cb.fire("train_start", TrainState())

            mock_cls.assert_called_once()
            mock_progress.add_task.assert_called_once()
            mock_progress.start.assert_called_once()

    def test_progress_updated_on_step_end(self):
        cb = CallbackManager()
        with patch("rich.progress.Progress") as mock_cls:
            mock_progress = MagicMock()
            mock_progress.add_task.return_value = 0
            mock_cls.return_value = mock_progress

            register_progress_callbacks(
                callback_manager=cb,
                total_steps=100,
            )
            state = TrainState()
            cb.fire("train_start", state)

            state.global_step = 5
            state.metrics["loss"] = 0.5
            state.metrics["learning_rate"] = 1e-4
            cb.fire("step_end", state)

            mock_progress.update.assert_called_once()
            call_kwargs = mock_progress.update.call_args
            assert call_kwargs.kwargs["completed"] == 5
            assert "loss: 0.5000" in call_kwargs.kwargs["status"]
            assert "lr: 1.00e-04" in call_kwargs.kwargs["status"]

    def test_progress_stopped_on_train_end(self):
        cb = CallbackManager()
        with patch("rich.progress.Progress") as mock_cls:
            mock_progress = MagicMock()
            mock_progress.add_task.return_value = 0
            mock_cls.return_value = mock_progress

            register_progress_callbacks(
                callback_manager=cb,
                total_steps=100,
            )
            cb.fire("train_start", TrainState())
            cb.fire("train_end", TrainState())

            mock_progress.stop.assert_called_once()

    def test_skipped_when_not_main_process(self):
        cb = CallbackManager()
        with patch("rich.progress.Progress") as mock_cls:
            register_progress_callbacks(
                callback_manager=cb,
                total_steps=100,
                is_main_process=False,
            )
            cb.fire("train_start", TrainState())

            mock_cls.assert_not_called()

    def test_update_with_loss_only(self):
        cb = CallbackManager()
        with patch("rich.progress.Progress") as mock_cls:
            mock_progress = MagicMock()
            mock_progress.add_task.return_value = 0
            mock_cls.return_value = mock_progress

            register_progress_callbacks(
                callback_manager=cb,
                total_steps=50,
            )
            state = TrainState()
            cb.fire("train_start", state)

            state.global_step = 1
            state.metrics["loss"] = 2.3
            cb.fire("step_end", state)

            status = mock_progress.update.call_args.kwargs["status"]
            assert "loss: 2.3000" in status
            assert "lr:" not in status
