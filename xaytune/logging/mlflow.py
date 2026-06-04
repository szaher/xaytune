from __future__ import annotations

from typing import Any

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment]

from xaytune.logging.base import LoggingBackend


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, key))
        else:
            flat[key] = str(v)
    return flat


class MLflowBackend(LoggingBackend):
    def __init__(self, run_name: str | None = None) -> None:
        if mlflow is None:
            raise ImportError(
                "mlflow is required for MLflowBackend. "
                "Install it with: pip install mlflow or pip install xaytune[mlflow]"
            )
        mlflow.start_run(run_name=run_name)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        mlflow.log_params(_flatten_dict(config))

    def close(self) -> None:
        mlflow.end_run()
