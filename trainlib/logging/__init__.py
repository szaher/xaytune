from __future__ import annotations

from typing import Any

from trainlib.config.schema import LoggingConfig
from trainlib.logging.base import LoggingBackend, LoggingManager
from trainlib.logging.console import ConsoleBackend
from trainlib.logging.tensorboard import TensorBoardBackend
from trainlib.trainer.callbacks import CallbackManager

_BACKEND_NAMES = {"console", "tensorboard", "wandb", "mlflow"}


def _create_wandb_backend(project: str, run_name: str | None) -> LoggingBackend:
    from trainlib.logging.wandb import WandbBackend
    return WandbBackend(project=project, run_name=run_name)


def _create_mlflow_backend(run_name: str | None) -> LoggingBackend:
    from trainlib.logging.mlflow import MLflowBackend
    return MLflowBackend(run_name=run_name)


def setup_logging(
    config: LoggingConfig,
    callback_manager: CallbackManager,
    *,
    output_dir: str = "output",
) -> LoggingManager:
    manager = LoggingManager(log_every_n_steps=config.log_every_n_steps)

    manager.add_backend(ConsoleBackend())

    for name in config.backends:
        if name == "console":
            continue
        if name == "tensorboard":
            manager.add_backend(TensorBoardBackend(log_dir=f"{output_dir}/runs"))
        elif name == "wandb":
            manager.add_backend(_create_wandb_backend(
                project=config.project or "trainlib",
                run_name=config.run_name,
            ))
        elif name == "mlflow":
            manager.add_backend(_create_mlflow_backend(run_name=config.run_name))
        else:
            raise ValueError(
                f"Unknown logging backend: '{name}'. "
                f"Available: {', '.join(sorted(_BACKEND_NAMES))}"
            )

    manager.register_callbacks(callback_manager)
    return manager


__all__ = [
    "ConsoleBackend",
    "LoggingBackend",
    "LoggingManager",
    "setup_logging",
    "TensorBoardBackend",
]
