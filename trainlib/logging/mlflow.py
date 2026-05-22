from __future__ import annotations

from typing import Any

import mlflow

from trainlib.logging.base import LoggingBackend


class MLflowBackend(LoggingBackend):
    def __init__(self, run_name: str | None = None) -> None:
        mlflow.start_run(run_name=run_name)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        mlflow.log_params(config)

    def close(self) -> None:
        mlflow.end_run()
