from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

VALID_EVENTS = {
    "train_start",
    "train_end",
    "epoch_start",
    "epoch_end",
    "step_start",
    "step_end",
    "eval_start",
    "eval_end",
    "checkpoint_saved",
    "error",
}


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    global_step: int = 0
    num_epochs: int = 0
    max_steps: int = -1
    metrics: dict[str, Any] = field(default_factory=dict)
    should_stop: bool = False

    def stop_training(self) -> None:
        self.should_stop = True

    def to_dict(self) -> dict[str, Any]:
        import dataclasses as _dc

        d = _dc.asdict(self)
        safe_metrics: dict[str, Any] = {}
        for k, v in d["metrics"].items():
            safe_metrics[k] = v.item() if hasattr(v, "item") else v
        d["metrics"] = safe_metrics
        return d


class CallbackManager:
    def __init__(self) -> None:
        self._callbacks: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self._callbacks[event].append(fn)
            return fn

        return decorator

    def fire(self, event: str, state: TrainState) -> None:
        for callback in self._callbacks.get(event, []):
            callback(state)
