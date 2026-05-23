from unittest.mock import MagicMock

import pytest

from trainlib.logging.base import LoggingBackend, LoggingManager
from trainlib.trainer.callbacks import CallbackManager, TrainState


class FakeBackend(LoggingBackend):
    def __init__(self):
        self.scalars = []
        self.config_logged = None
        self.closed = False

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self.scalars.append((key, value, step))

    def log_config(self, config: dict) -> None:
        self.config_logged = config

    def close(self) -> None:
        self.closed = True


class TestLoggingBackend:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            LoggingBackend()

    def test_subclass_works(self):
        backend = FakeBackend()
        backend.log_scalar("loss", 0.5, 1)
        assert backend.scalars == [("loss", 0.5, 1)]


class TestLoggingManager:
    def test_add_backend(self):
        manager = LoggingManager()
        backend = FakeBackend()
        manager.add_backend(backend)
        assert len(manager.backends) == 1

    def test_log_scalar_delegates(self):
        manager = LoggingManager()
        b1 = FakeBackend()
        b2 = FakeBackend()
        manager.add_backend(b1)
        manager.add_backend(b2)

        manager.log_scalar("loss", 0.5, 10)

        assert b1.scalars == [("loss", 0.5, 10)]
        assert b2.scalars == [("loss", 0.5, 10)]

    def test_log_config_delegates(self):
        manager = LoggingManager()
        backend = FakeBackend()
        manager.add_backend(backend)

        manager.log_config({"lr": 0.001})

        assert backend.config_logged == {"lr": 0.001}

    def test_close_delegates(self):
        manager = LoggingManager()
        b1 = FakeBackend()
        b2 = FakeBackend()
        manager.add_backend(b1)
        manager.add_backend(b2)

        manager.close()

        assert b1.closed is True
        assert b2.closed is True

    def test_register_callbacks(self):
        cb_manager = CallbackManager()
        log_manager = LoggingManager(log_every_n_steps=1)
        backend = FakeBackend()
        log_manager.add_backend(backend)

        log_manager.register_callbacks(cb_manager)

        state = TrainState(global_step=5)
        state.metrics["loss"] = 0.25

        cb_manager.fire("step_end", state)

        assert ("loss", 0.25, 5) in backend.scalars

    def test_log_every_n_steps_skips(self):
        cb_manager = CallbackManager()
        log_manager = LoggingManager(log_every_n_steps=10)
        backend = FakeBackend()
        log_manager.add_backend(backend)
        log_manager.register_callbacks(cb_manager)

        state = TrainState(global_step=3)
        state.metrics["loss"] = 0.5
        cb_manager.fire("step_end", state)

        assert len(backend.scalars) == 0

    def test_log_every_n_steps_fires_on_match(self):
        cb_manager = CallbackManager()
        log_manager = LoggingManager(log_every_n_steps=10)
        backend = FakeBackend()
        log_manager.add_backend(backend)
        log_manager.register_callbacks(cb_manager)

        state = TrainState(global_step=10)
        state.metrics["loss"] = 0.3
        cb_manager.fire("step_end", state)

        assert ("loss", 0.3, 10) in backend.scalars

    def test_train_end_closes(self):
        cb_manager = CallbackManager()
        log_manager = LoggingManager()
        backend = FakeBackend()
        log_manager.add_backend(backend)
        log_manager.register_callbacks(cb_manager)

        cb_manager.fire("train_end", TrainState())

        assert backend.closed is True


class TestLoggingManagerRankGating:
    def test_rank0_logs_normally(self):
        manager = LoggingManager(rank=0)
        backend = MagicMock(spec=LoggingBackend)
        manager.add_backend(backend)
        manager.log_scalar("loss", 0.5, 1)
        backend.log_scalar.assert_called_once_with("loss", 0.5, 1)

    def test_rank0_logs_config(self):
        manager = LoggingManager(rank=0)
        backend = MagicMock(spec=LoggingBackend)
        manager.add_backend(backend)
        manager.log_config({"lr": 0.001})
        backend.log_config.assert_called_once_with({"lr": 0.001})

    def test_non_rank0_suppresses_log_scalar(self):
        manager = LoggingManager(rank=1)
        backend = MagicMock(spec=LoggingBackend)
        manager.add_backend(backend)
        manager.log_scalar("loss", 0.5, 1)
        backend.log_scalar.assert_not_called()

    def test_non_rank0_suppresses_log_config(self):
        manager = LoggingManager(rank=2)
        backend = MagicMock(spec=LoggingBackend)
        manager.add_backend(backend)
        manager.log_config({"lr": 0.001})
        backend.log_config.assert_not_called()

    def test_non_rank0_still_closes(self):
        manager = LoggingManager(rank=3)
        backend = MagicMock(spec=LoggingBackend)
        manager.add_backend(backend)
        manager.close()
        backend.close.assert_called_once()

    def test_default_rank_is_zero(self):
        manager = LoggingManager()
        assert manager.rank == 0
