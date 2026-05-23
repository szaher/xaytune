from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from trainlib.trainer.callbacks import TrainState


def save_checkpoint(
    *,
    output_dir: str,
    model: Any,
    optimizer: Any,
    state: TrainState,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> None:
    ckpt_path = Path(output_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    model_state = model.state_dict() if hasattr(model, "state_dict") else {}
    torch.save(model_state, ckpt_path / "model.pt")

    optimizer_state = optimizer.state_dict() if hasattr(optimizer, "state_dict") else {}
    torch.save(optimizer_state, ckpt_path / "optimizer.pt")

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        torch.save(scheduler.state_dict(), ckpt_path / "scheduler.pt")

    if scaler is not None and hasattr(scaler, "state_dict"):
        torch.save(scaler.state_dict(), ckpt_path / "scaler.pt")

    metadata = {
        "global_step": state.global_step,
        "epoch": state.epoch,
        "step": state.step,
        "metrics": state.metrics,
    }
    (ckpt_path / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_checkpoint(
    *,
    checkpoint_dir: str,
    model: Any,
    optimizer: Any,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> TrainState:
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    model_path = ckpt_path / "model.pt"
    if model_path.exists():
        model_state = torch.load(model_path, weights_only=True)
        if model_state and hasattr(model, "load_state_dict"):
            model.load_state_dict(model_state)

    optimizer_path = ckpt_path / "optimizer.pt"
    if optimizer_path.exists():
        opt_state = torch.load(optimizer_path, weights_only=True)
        if opt_state and hasattr(optimizer, "load_state_dict"):
            optimizer.load_state_dict(opt_state)

    scheduler_path = ckpt_path / "scheduler.pt"
    if scheduler is not None and scheduler_path.exists():
        sched_state = torch.load(scheduler_path, weights_only=True)
        if sched_state and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(sched_state)

    scaler_path = ckpt_path / "scaler.pt"
    if scaler is not None and scaler_path.exists():
        scaler_state = torch.load(scaler_path, weights_only=True)
        if scaler_state and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(scaler_state)

    metadata_path = ckpt_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    return TrainState(
        global_step=metadata.get("global_step", 0),
        epoch=metadata.get("epoch", 0),
        step=metadata.get("step", 0),
        metrics=metadata.get("metrics", {}),
    )


def find_latest_checkpoint(output_dir: str) -> str | None:
    base = Path(output_dir)
    if not base.exists():
        return None

    checkpoints = []
    for d in base.iterdir():
        if d.is_dir() and (d / "metadata.json").exists():
            meta = json.loads((d / "metadata.json").read_text())
            checkpoints.append((meta.get("global_step", 0), str(d)))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]
