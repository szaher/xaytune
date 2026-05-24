from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from xaytune.trainer.async_checkpoint import AsyncCheckpointSaver
from xaytune.trainer.callbacks import TrainState


def _make_model():
    model = MagicMock()
    model.state_dict.return_value = {"weight": torch.randn(4, 4)}
    return model


def _make_optimizer():
    opt = MagicMock()
    opt.state_dict.return_value = {"lr": 0.001}
    return opt


def _make_state(**kwargs):
    defaults = {"global_step": 10, "epoch": 1, "step": 5}
    defaults.update(kwargs)
    return TrainState(**defaults)


class TestAsyncCheckpointSaver:
    def test_save_writes_checkpoint_files(self, tmp_path):
        saver = AsyncCheckpointSaver()
        ckpt_dir = str(tmp_path / "checkpoint-10")

        saver.save(
            output_dir=ckpt_dir,
            model=_make_model(),
            optimizer=_make_optimizer(),
            state=_make_state(),
        )
        saver.wait()

        assert (Path(ckpt_dir) / "model.pt").exists()
        assert (Path(ckpt_dir) / "optimizer.pt").exists()
        assert (Path(ckpt_dir) / "metadata.json").exists()

        meta = json.loads((Path(ckpt_dir) / "metadata.json").read_text())
        assert meta["global_step"] == 10

    def test_save_does_not_block_caller(self, tmp_path):
        saver = AsyncCheckpointSaver()
        model = MagicMock()
        model.state_dict.return_value = {"w": torch.randn(100, 100)}

        original_save = torch.save

        def slow_save(*args, **kwargs):
            time.sleep(0.3)
            return original_save(*args, **kwargs)

        ckpt_dir = str(tmp_path / "checkpoint-1")

        with patch("xaytune.trainer.checkpointing.torch.save", side_effect=slow_save):
            start = time.monotonic()
            saver.save(
                output_dir=ckpt_dir,
                model=model,
                optimizer=_make_optimizer(),
                state=_make_state(),
            )
            elapsed = time.monotonic() - start

        assert elapsed < 0.2
        saver.wait()

    def test_sequential_saves_complete(self, tmp_path):
        saver = AsyncCheckpointSaver()

        for step in (5, 10):
            ckpt_dir = str(tmp_path / f"checkpoint-{step}")
            saver.save(
                output_dir=ckpt_dir,
                model=_make_model(),
                optimizer=_make_optimizer(),
                state=_make_state(global_step=step),
            )

        saver.wait()

        for step in (5, 10):
            meta_path = tmp_path / f"checkpoint-{step}" / "metadata.json"
            assert meta_path.exists()
            meta = json.loads(meta_path.read_text())
            assert meta["global_step"] == step

    def test_wait_reraises_background_error(self, tmp_path):
        saver = AsyncCheckpointSaver()

        with patch(
            "xaytune.trainer.async_checkpoint.save_checkpoint",
            side_effect=RuntimeError("disk full"),
        ):
            saver.save(
                output_dir=str(tmp_path / "ckpt"),
                model=_make_model(),
                optimizer=_make_optimizer(),
                state=_make_state(),
            )

            with pytest.raises(RuntimeError, match="disk full"):
                saver.wait()

    def test_snapshot_isolation(self, tmp_path):
        model = MagicMock()
        weight = torch.tensor([1.0, 2.0, 3.0])
        model.state_dict.return_value = {"weight": weight}

        saver = AsyncCheckpointSaver()
        ckpt_dir = str(tmp_path / "checkpoint-1")

        saver.save(
            output_dir=ckpt_dir,
            model=model,
            optimizer=_make_optimizer(),
            state=_make_state(),
        )

        weight.fill_(99.0)

        saver.wait()

        saved = torch.load(Path(ckpt_dir) / "model.pt", weights_only=True)
        assert torch.equal(saved["weight"], torch.tensor([1.0, 2.0, 3.0]))

    def test_save_with_scaler(self, tmp_path):
        saver = AsyncCheckpointSaver()
        scaler = MagicMock()
        scaler.state_dict.return_value = {"scale": 1.0}

        ckpt_dir = str(tmp_path / "checkpoint-1")
        saver.save(
            output_dir=ckpt_dir,
            model=_make_model(),
            optimizer=_make_optimizer(),
            state=_make_state(),
            scaler=scaler,
        )
        saver.wait()

        assert (Path(ckpt_dir) / "scaler.pt").exists()
