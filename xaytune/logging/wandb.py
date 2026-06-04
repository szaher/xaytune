from __future__ import annotations

from typing import Any

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

from xaytune.logging.base import LoggingBackend


class WandbBackend(LoggingBackend):
    def __init__(
        self,
        project: str = "xaytune",
        run_name: str | None = None,
    ) -> None:
        if wandb is None:
            raise ImportError(
                "wandb is required for WandbBackend. "
                "Install it with: pip install wandb or pip install xaytune[wandb]"
            )
        wandb.init(project=project, name=run_name)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        wandb.log({key: value}, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        wandb.config.update(config)

    def close(self) -> None:
        wandb.finish()
