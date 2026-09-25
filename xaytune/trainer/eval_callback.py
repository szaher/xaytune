from __future__ import annotations

from typing import Any

import torch

from xaytune.trainer.callbacks import CallbackManager, TrainState


def _metric_registry() -> Any:
    """Resolve the metric registry lazily.

    ``xaytune.eval`` imports ``xaytune.recipes``, which imports back into
    ``xaytune.trainer``. Importing it at module scope makes ``import
    xaytune.trainer`` a circular import; the package ``__init__`` used to mask
    that by importing ``xaytune.eval`` first.
    """
    from xaytune.eval.metrics import metric_registry

    return metric_registry


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

        # Lazily, for the reason _metric_registry() is.
        from xaytune.eval.causal import next_token_pairs, next_token_targets

        # Loss and perplexity come from the model's own loss; only other metrics
        # need the predictions, and an argmax over the vocabulary for metrics
        # nobody asked for is wasted work.
        scores_tokens = any(name not in ("loss", "perplexity") for name in metrics)
        losses: list[float] = []
        # Each loss's target count, so the mean does not depend on batching
        # (issue #36); None once a loss arrives with no labels to count.
        weights: list[int] | None = []
        all_preds: list[Any] = []
        all_refs: list[Any] = []
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
                labels = batch.get("labels") if isinstance(batch, dict) else None
                if labels is not None and not isinstance(labels, torch.Tensor):
                    labels = torch.as_tensor(labels)
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    targets = next_token_targets(labels) if labels is not None else None
                    if targets == 0:
                        continue  # a mean over no targets is not a loss
                    raw = outputs.loss
                    losses.append(raw.item() if hasattr(raw, "item") else float(raw))
                    if weights is not None and targets is not None:
                        weights.append(targets)
                    else:
                        weights = None
                # Next-token pairs: the logits at i against the label at i + 1.
                if scores_tokens and hasattr(outputs, "logits") and labels is not None:
                    preds, refs = next_token_pairs(outputs.logits, labels)
                    all_preds.extend(preds)
                    all_refs.extend(refs)

        for metric_name in metrics:
            compute_fn = _metric_registry().get(metric_name)
            if metric_name in ("loss", "perplexity"):
                state.metrics[f"eval_{metric_name}"] = compute_fn(losses, weights=weights)
            else:
                state.metrics[f"eval_{metric_name}"] = compute_fn(all_preds, all_refs)

        if was_training and hasattr(model, "train"):
            model.train()

        callback_manager.fire("eval_end", state)
