from __future__ import annotations

import math
from typing import Any

from trainlib.trainer.callbacks import CallbackManager, TrainState


def _infer_mode(metric: str) -> str:
    if any(k in metric for k in ("loss", "perplexity")):
        return "min"
    return "max"


def register_early_stopping_callbacks(
    *,
    callback_manager: CallbackManager,
    patience: int,
    metric: str,
    min_delta: float = 0.0,
) -> None:
    """Register an ``eval_end`` callback that stops training if *metric* doesn't improve."""
    mode = _infer_mode(metric)
    best: dict[str, Any] = {
        "value": math.inf if mode == "min" else -math.inf,
        "wait": 0,
    }

    @callback_manager.on("eval_end")
    def _check_early_stopping(state: TrainState) -> None:
        current = state.metrics.get(metric)
        if current is None:
            return

        if mode == "min":
            improved = current < best["value"] - min_delta
        else:
            improved = current > best["value"] + min_delta

        if improved:
            best["value"] = current
            best["wait"] = 0
        else:
            best["wait"] += 1
            if best["wait"] >= patience:
                state.stop_training()
