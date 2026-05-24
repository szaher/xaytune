from __future__ import annotations

import threading
from typing import Any

import torch

from xaytune.trainer.callbacks import TrainState
from xaytune.trainer.checkpointing import save_checkpoint


class AsyncCheckpointSaver:
    """Write checkpoints in a background thread to avoid blocking training."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def save(
        self,
        *,
        output_dir: str,
        model: Any,
        optimizer: Any,
        state: TrainState,
        scheduler: Any | None = None,
        scaler: Any | None = None,
    ) -> None:
        self.wait()

        model_sd = _snapshot_state_dict(model)
        opt_sd = _snapshot_state_dict(optimizer)
        sched_sd = _snapshot_state_dict(scheduler) if scheduler is not None else None
        scaler_sd = _snapshot_state_dict(scaler) if scaler is not None else None

        state_copy = TrainState(
            step=state.step,
            epoch=state.epoch,
            global_step=state.global_step,
            num_epochs=state.num_epochs,
            max_steps=state.max_steps,
            metrics=dict(state.metrics),
        )

        self._error = None
        self._thread = threading.Thread(
            target=self._write_to_disk,
            args=(output_dir, model_sd, opt_sd, sched_sd, scaler_sd, state_copy),
            daemon=True,
        )
        self._thread.start()

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            err = self._error
            self._error = None
            raise err

    def _write_to_disk(
        self,
        output_dir: str,
        model_sd: dict,
        opt_sd: dict,
        sched_sd: dict | None,
        scaler_sd: dict | None,
        state: TrainState,
    ) -> None:
        try:
            model_proxy = _StateDictProxy(model_sd)
            opt_proxy = _StateDictProxy(opt_sd)
            sched_proxy = _StateDictProxy(sched_sd) if sched_sd is not None else None
            scaler_proxy = _StateDictProxy(scaler_sd) if scaler_sd is not None else None

            save_checkpoint(
                output_dir=output_dir,
                model=model_proxy,
                optimizer=opt_proxy,
                state=state,
                scheduler=sched_proxy,
                scaler=scaler_proxy,
            )
        except Exception as exc:
            self._error = exc


def _snapshot_state_dict(obj: Any) -> dict:
    if obj is None or not hasattr(obj, "state_dict"):
        return {}
    sd = obj.state_dict()
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sd.items()}


class _StateDictProxy:
    def __init__(self, sd: dict) -> None:
        self._sd = sd

    def state_dict(self) -> dict:
        return self._sd

    def load_state_dict(self, state: Any) -> None:
        pass
