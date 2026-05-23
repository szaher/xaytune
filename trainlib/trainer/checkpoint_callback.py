from __future__ import annotations

from typing import Any

from trainlib.trainer.callbacks import CallbackManager, TrainState
from trainlib.trainer.checkpointing import save_checkpoint


def register_checkpoint_callbacks(
    *,
    callback_manager: CallbackManager,
    trainer: Any,
    model: Any,
    output_dir: str,
    checkpoint_every_n_steps: int,
    save_last: bool,
    is_main_process: bool = True,
    async_saver: Any | None = None,
) -> None:
    last_saved_step: dict[str, int] = {"step": -1}

    def _get_optimizer() -> Any:
        return getattr(trainer, "_optimizer", None)

    def _get_scaler() -> Any:
        return getattr(trainer, "_scaler", None)

    def _get_scheduler() -> Any:
        return getattr(trainer, "_scheduler", None)

    def _do_save(ckpt_dir: str, state: TrainState) -> None:
        kwargs = dict(
            output_dir=ckpt_dir,
            model=model,
            optimizer=_get_optimizer(),
            state=state,
            scaler=_get_scaler(),
            scheduler=_get_scheduler(),
        )
        if async_saver is not None:
            async_saver.save(**kwargs)
        else:
            save_checkpoint(**kwargs)

    @callback_manager.on("step_end")
    def _periodic_checkpoint(state: TrainState) -> None:
        if not is_main_process:
            return
        if checkpoint_every_n_steps <= 0:
            return
        if state.global_step > 0 and state.global_step % checkpoint_every_n_steps == 0:
            ckpt_dir = f"{output_dir}/checkpoint-{state.global_step}"
            _do_save(ckpt_dir, state)
            last_saved_step["step"] = state.global_step
            callback_manager.fire("checkpoint_saved", state)

    @callback_manager.on("train_end")
    def _final_checkpoint(state: TrainState) -> None:
        if not is_main_process:
            return
        if not save_last:
            if async_saver is not None:
                async_saver.wait()
            return
        if state.global_step == last_saved_step["step"]:
            if async_saver is not None:
                async_saver.wait()
            return
        ckpt_dir = f"{output_dir}/checkpoint-{state.global_step}"
        _do_save(ckpt_dir, state)
        last_saved_step["step"] = state.global_step
        if async_saver is not None:
            async_saver.wait()
        callback_manager.fire("checkpoint_saved", state)
