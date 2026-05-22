from __future__ import annotations

from typing import Any

import wandb

from trainlib.logging.base import LoggingBackend


class WandbBackend(LoggingBackend):
    def __init__(
        self,
        project: str = "trainlib",
        run_name: str | None = None,
    ) -> None:
        wandb.init(project=project, name=run_name)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        wandb.log({key: value}, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        wandb.config.update(config)

    def close(self) -> None:
        wandb.finish()
