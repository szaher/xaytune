from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trainlib.config.schema import LoggingConfig
from trainlib.logging import LoggingBackend, LoggingManager, setup_logging
from trainlib.logging.console import ConsoleBackend
from trainlib.trainer.callbacks import CallbackManager, TrainState


class TestSetupLogging:
    def test_returns_logging_manager(self):
        config = LoggingConfig(backends=["console"])
        cb = CallbackManager()
        manager = setup_logging(config, cb)
        assert isinstance(manager, LoggingManager)

    def test_console_always_added(self):
        config = LoggingConfig(backends=[])
        cb = CallbackManager()
        manager = setup_logging(config, cb)
        assert any(isinstance(b, ConsoleBackend) for b in manager.backends)

    def test_console_not_duplicated(self):
        config = LoggingConfig(backends=["console"])
        cb = CallbackManager()
        manager = setup_logging(config, cb)
        console_count = sum(1 for b in manager.backends if isinstance(b, ConsoleBackend))
        assert console_count == 1

    @patch("trainlib.logging.TensorBoardBackend")
    def test_tensorboard_added(self, mock_tb_cls):
        mock_tb_cls.return_value = MagicMock(spec=LoggingBackend)
        config = LoggingConfig(backends=["console", "tensorboard"])
        cb = CallbackManager()
        setup_logging(config, cb, output_dir="output/test")
        mock_tb_cls.assert_called_once_with(log_dir="output/test/runs")

    @patch("trainlib.logging._create_wandb_backend")
    def test_wandb_added(self, mock_create_wandb):
        mock_create_wandb.return_value = MagicMock(spec=LoggingBackend)
        config = LoggingConfig(backends=["wandb"], project="my-proj", run_name="run-1")
        cb = CallbackManager()
        setup_logging(config, cb)
        mock_create_wandb.assert_called_once_with(project="my-proj", run_name="run-1")

    @patch("trainlib.logging._create_mlflow_backend")
    def test_mlflow_added(self, mock_create_mlflow):
        mock_create_mlflow.return_value = MagicMock(spec=LoggingBackend)
        config = LoggingConfig(backends=["mlflow"], run_name="run-1")
        cb = CallbackManager()
        setup_logging(config, cb)
        mock_create_mlflow.assert_called_once_with(run_name="run-1")

    def test_registers_callbacks(self):
        config = LoggingConfig(backends=["console"], log_every_n_steps=1)
        cb = CallbackManager()
        setup_logging(config, cb)

        state = TrainState(global_step=1)
        state.metrics["loss"] = 0.5
        cb.fire("step_end", state)

    def test_log_every_n_steps_passed(self):
        config = LoggingConfig(backends=["console"], log_every_n_steps=50)
        cb = CallbackManager()
        manager = setup_logging(config, cb)
        assert manager.log_every_n_steps == 50

    def test_unknown_backend_raises(self):
        config = LoggingConfig(backends=["nonexistent"])
        cb = CallbackManager()
        with pytest.raises(ValueError, match="Unknown logging backend"):
            setup_logging(config, cb)

    def test_setup_logging_passes_rank(self):
        config = LoggingConfig(backends=["console"])
        cb = CallbackManager()
        manager = setup_logging(config, cb, rank=5)
        assert manager.rank == 5

    def test_setup_logging_default_rank_is_zero(self):
        config = LoggingConfig(backends=["console"])
        cb = CallbackManager()
        manager = setup_logging(config, cb)
        assert manager.rank == 0


class TestModuleExports:
    def test_logging_backend_importable(self):
        from trainlib.logging import LoggingBackend

        assert LoggingBackend is not None

    def test_logging_manager_importable(self):
        from trainlib.logging import LoggingManager

        assert LoggingManager is not None

    def test_setup_logging_importable(self):
        from trainlib.logging import setup_logging

        assert callable(setup_logging)
