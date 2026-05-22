from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from trainlib.trainer.callbacks import CallbackManager, TrainState


class LoggingBackend(ABC):
    @abstractmethod
    def log_scalar(self, key: str, value: float, step: int) -> None: ...

    @abstractmethod
    def log_config(self, config: dict[str, Any]) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class LoggingManager:
    def __init__(self, log_every_n_steps: int = 10) -> None:
        self.backends: list[LoggingBackend] = []
        self.log_every_n_steps = log_every_n_steps

    def add_backend(self, backend: LoggingBackend) -> None:
        self.backends.append(backend)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        for backend in self.backends:
            backend.log_scalar(key, value, step)

    def log_config(self, config: dict[str, Any]) -> None:
        for backend in self.backends:
            backend.log_config(config)

    def close(self) -> None:
        for backend in self.backends:
            backend.close()

    def register_callbacks(self, callback_manager: CallbackManager) -> None:
        @callback_manager.on("step_end")
        def _on_step_end(state: TrainState) -> None:
            if state.global_step % self.log_every_n_steps != 0:
                return
            for key, value in state.metrics.items():
                if isinstance(value, (int, float)):
                    self.log_scalar(key, float(value), state.global_step)

        @callback_manager.on("train_end")
        def _on_train_end(state: TrainState) -> None:
            self.close()
