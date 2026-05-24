from __future__ import annotations

import json
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from xaytune.logging.base import LoggingBackend


class TensorBoardBackend(LoggingBackend):
    def __init__(self, log_dir: str = "runs") -> None:
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self.writer.add_scalar(key, value, step)

    def log_config(self, config: dict[str, Any]) -> None:
        self.writer.add_text("config", json.dumps(config, indent=2))

    def close(self) -> None:
        self.writer.close()
