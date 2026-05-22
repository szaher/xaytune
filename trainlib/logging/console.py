from __future__ import annotations

from typing import Any

from trainlib.logging.base import LoggingBackend


class ConsoleBackend(LoggingBackend):
    def log_scalar(self, key: str, value: float, step: int) -> None:
        print(f"[step {step}] {key}: {value:.4f}")

    def log_config(self, config: dict[str, Any]) -> None:
        print("Training config:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    def close(self) -> None:
        pass
