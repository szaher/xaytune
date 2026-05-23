from __future__ import annotations

from typing import Any

from trainlib.trainer.callbacks import CallbackManager, TrainState


def register_progress_callbacks(
    *,
    callback_manager: CallbackManager,
    total_steps: int,
    is_main_process: bool = True,
) -> None:
    if not is_main_process:
        return

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )

    holder: dict[str, Any] = {}

    @callback_manager.on("train_start")
    def _start_progress(state: TrainState) -> None:
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[status]}"),
        )
        task_id = progress.add_task("Training", total=total_steps, status="")
        holder["progress"] = progress
        holder["task_id"] = task_id
        progress.start()

    @callback_manager.on("step_end")
    def _update_progress(state: TrainState) -> None:
        progress = holder.get("progress")
        if progress is None:
            return
        parts = []
        if "loss" in state.metrics:
            parts.append(f"loss: {state.metrics['loss']:.4f}")
        if "learning_rate" in state.metrics:
            parts.append(f"lr: {state.metrics['learning_rate']:.2e}")
        progress.update(
            holder["task_id"],
            completed=state.global_step,
            status=" | ".join(parts),
        )

    @callback_manager.on("train_end")
    def _stop_progress(state: TrainState) -> None:
        progress = holder.get("progress")
        if progress is not None:
            progress.stop()
