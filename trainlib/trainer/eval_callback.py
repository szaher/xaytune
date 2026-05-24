from __future__ import annotations

from typing import Any

import torch

from trainlib.eval.metrics import metric_registry
from trainlib.trainer.callbacks import CallbackManager, TrainState


def register_eval_callbacks(
    *,
    callback_manager: CallbackManager,
    model: Any,
    eval_dataloader: Any,
    every_n_steps: int,
    metrics: list[str],
    is_main_process: bool = True,
) -> None:
    """Register a ``step_end`` callback that runs evaluation every N steps."""
    @callback_manager.on("step_end")
    def _periodic_eval(state: TrainState) -> None:
        if not is_main_process:
            return
        if every_n_steps <= 0:
            return
        if state.global_step <= 0:
            return
        if state.global_step % every_n_steps != 0:
            return

        callback_manager.fire("eval_start", state)

        was_training = model.training if hasattr(model, "training") else False
        if hasattr(model, "eval"):
            model.eval()

        losses: list[float] = []
        try:
            device = next(iter(model.parameters())).device
        except (StopIteration, AttributeError, TypeError):
            device = torch.device("cpu")
        with torch.no_grad():
            for batch in eval_dataloader:
                if isinstance(batch, dict):
                    batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    outputs = model(**batch)
                else:
                    outputs = model(batch)
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    raw = outputs.loss
                    loss_val = raw.item() if hasattr(raw, "item") else float(raw)
                    losses.append(loss_val)

        for metric_name in metrics:
            compute_fn = metric_registry.get(metric_name)
            if metric_name in ("loss", "perplexity"):
                state.metrics[f"eval_{metric_name}"] = compute_fn(losses)
            else:
                state.metrics[f"eval_{metric_name}"] = compute_fn([], [])

        if was_training and hasattr(model, "train"):
            model.train()

        callback_manager.fire("eval_end", state)
