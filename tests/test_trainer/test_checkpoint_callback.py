from __future__ import annotations

from unittest.mock import MagicMock, patch

from xaytune.trainer.callbacks import CallbackManager, TrainState
from xaytune.trainer.checkpoint_callback import register_checkpoint_callbacks


def _make_trainer_mock():
    trainer = MagicMock()
    trainer._optimizer = MagicMock()
    trainer._scaler = None
    return trainer


class TestPeriodicCheckpoint:
    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_saves_at_interval(self, mock_save):
        cb = CallbackManager()
        trainer = _make_trainer_mock()
        model = MagicMock()

        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=trainer,
            model=model,
            output_dir="/out",
            checkpoint_every_n_steps=3,
            save_last=False,
        )

        for step in range(1, 7):
            state = TrainState(global_step=step)
            cb.fire("step_end", state)

        assert mock_save.call_count == 2
        dirs = [c.kwargs["output_dir"] for c in mock_save.call_args_list]
        assert "/out/checkpoint-3" in dirs
        assert "/out/checkpoint-6" in dirs

    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_no_save_at_step_zero(self, mock_save):
        cb = CallbackManager()
        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=_make_trainer_mock(),
            model=MagicMock(),
            output_dir="/out",
            checkpoint_every_n_steps=1,
            save_last=False,
        )

        cb.fire("step_end", TrainState(global_step=0))
        mock_save.assert_not_called()

    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_disabled_when_interval_zero(self, mock_save):
        cb = CallbackManager()
        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=_make_trainer_mock(),
            model=MagicMock(),
            output_dir="/out",
            checkpoint_every_n_steps=0,
            save_last=False,
        )

        cb.fire("step_end", TrainState(global_step=5))
        mock_save.assert_not_called()

    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_not_main_process_skips_save(self, mock_save):
        cb = CallbackManager()
        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=_make_trainer_mock(),
            model=MagicMock(),
            output_dir="/out",
            checkpoint_every_n_steps=1,
            save_last=False,
            is_main_process=False,
        )

        cb.fire("step_end", TrainState(global_step=1))
        mock_save.assert_not_called()

    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_passes_scaler_to_save(self, mock_save):
        cb = CallbackManager()
        trainer = _make_trainer_mock()
        trainer._scaler = MagicMock()
        model = MagicMock()

        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=trainer,
            model=model,
            output_dir="/out",
            checkpoint_every_n_steps=1,
            save_last=False,
        )

        cb.fire("step_end", TrainState(global_step=1))
        assert mock_save.call_args.kwargs["scaler"] is trainer._scaler

    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_passes_scheduler_to_save(self, mock_save):
        cb = CallbackManager()
        trainer = _make_trainer_mock()
        trainer._scheduler = MagicMock()
        model = MagicMock()

        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=trainer,
            model=model,
            output_dir="/out",
            checkpoint_every_n_steps=1,
            save_last=False,
        )

        cb.fire("step_end", TrainState(global_step=1))
        assert mock_save.call_args.kwargs["scheduler"] is trainer._scheduler

    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_scheduler_none_when_not_set(self, mock_save):
        cb = CallbackManager()
        trainer = MagicMock(spec=[])
        trainer._optimizer = MagicMock()

        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=trainer,
            model=MagicMock(),
            output_dir="/out",
            checkpoint_every_n_steps=1,
            save_last=False,
        )

        cb.fire("step_end", TrainState(global_step=1))
        assert mock_save.call_args.kwargs["scheduler"] is None

    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_fires_checkpoint_saved_event(self, mock_save):
        cb = CallbackManager()
        saved_events = []

        @cb.on("checkpoint_saved")
        def _on_saved(state):
            saved_events.append(state.global_step)

        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=_make_trainer_mock(),
            model=MagicMock(),
            output_dir="/out",
            checkpoint_every_n_steps=2,
            save_last=False,
        )

        for step in range(1, 5):
            cb.fire("step_end", TrainState(global_step=step))

        assert saved_events == [2, 4]


class TestFinalCheckpoint:
    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_save_last_on_train_end(self, mock_save):
        cb = CallbackManager()
        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=_make_trainer_mock(),
            model=MagicMock(),
            output_dir="/out",
            checkpoint_every_n_steps=0,
            save_last=True,
        )

        cb.fire("train_end", TrainState(global_step=10))
        mock_save.assert_called_once()
        assert mock_save.call_args.kwargs["output_dir"] == "/out/checkpoint-10"

    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_save_last_skips_if_already_saved(self, mock_save):
        cb = CallbackManager()
        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=_make_trainer_mock(),
            model=MagicMock(),
            output_dir="/out",
            checkpoint_every_n_steps=5,
            save_last=True,
        )

        cb.fire("step_end", TrainState(global_step=5))
        cb.fire("train_end", TrainState(global_step=5))

        assert mock_save.call_count == 1

    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_save_last_false_no_final(self, mock_save):
        cb = CallbackManager()
        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=_make_trainer_mock(),
            model=MagicMock(),
            output_dir="/out",
            checkpoint_every_n_steps=0,
            save_last=False,
        )

        cb.fire("train_end", TrainState(global_step=10))
        mock_save.assert_not_called()

    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_not_main_process_skips_final(self, mock_save):
        cb = CallbackManager()
        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=_make_trainer_mock(),
            model=MagicMock(),
            output_dir="/out",
            checkpoint_every_n_steps=0,
            save_last=True,
            is_main_process=False,
        )

        cb.fire("train_end", TrainState(global_step=10))
        mock_save.assert_not_called()


class TestAsyncSaverIntegration:
    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_async_saver_used_when_provided(self, mock_save):
        cb = CallbackManager()
        async_saver = MagicMock()

        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=_make_trainer_mock(),
            model=MagicMock(),
            output_dir="/out",
            checkpoint_every_n_steps=1,
            save_last=False,
            async_saver=async_saver,
        )

        cb.fire("step_end", TrainState(global_step=1))

        async_saver.save.assert_called_once()
        mock_save.assert_not_called()

    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_sync_save_when_no_async_saver(self, mock_save):
        cb = CallbackManager()

        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=_make_trainer_mock(),
            model=MagicMock(),
            output_dir="/out",
            checkpoint_every_n_steps=1,
            save_last=False,
        )

        cb.fire("step_end", TrainState(global_step=1))
        mock_save.assert_called_once()

    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_final_checkpoint_waits_on_async(self, mock_save):
        cb = CallbackManager()
        async_saver = MagicMock()

        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=_make_trainer_mock(),
            model=MagicMock(),
            output_dir="/out",
            checkpoint_every_n_steps=0,
            save_last=True,
            async_saver=async_saver,
        )

        cb.fire("train_end", TrainState(global_step=5))

        async_saver.save.assert_called_once()
        async_saver.wait.assert_called_once()

    @patch("xaytune.trainer.checkpoint_callback.save_checkpoint")
    def test_wait_called_even_when_save_last_false(self, mock_save):
        cb = CallbackManager()
        async_saver = MagicMock()

        register_checkpoint_callbacks(
            callback_manager=cb,
            trainer=_make_trainer_mock(),
            model=MagicMock(),
            output_dir="/out",
            checkpoint_every_n_steps=1,
            save_last=False,
            async_saver=async_saver,
        )

        cb.fire("step_end", TrainState(global_step=1))
        cb.fire("train_end", TrainState(global_step=1))

        async_saver.wait.assert_called_once()
