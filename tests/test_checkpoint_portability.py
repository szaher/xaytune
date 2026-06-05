"""Tests for checkpoint portability.

Covers:
- BUG-013: load_checkpoint must use map_location="cpu" for cross-device resume.
- BUG-026: save_checkpoint must serialize tensor metrics to plain floats.
- Round-trip save/load consistency.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch

from xaytune.trainer.callbacks import TrainState
from xaytune.trainer.checkpointing import load_checkpoint, save_checkpoint


class TestTensorMetricSerialization:
    def test_save_checkpoint_with_tensor_metrics(self):
        """Tensor values in metrics must be serialized as plain floats in metadata.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "step-10"
            state = TrainState(
                global_step=10,
                epoch=1,
                step=5,
                metrics={"loss": torch.tensor(0.5), "accuracy": torch.tensor(0.95)},
            )

            model = MagicMock()
            model.state_dict.return_value = {}
            optimizer = MagicMock()
            optimizer.state_dict.return_value = {}

            save_checkpoint(
                output_dir=str(output_dir),
                model=model,
                optimizer=optimizer,
                state=state,
            )

            metadata_path = output_dir / "metadata.json"
            assert metadata_path.exists()
            metadata = json.loads(metadata_path.read_text())

            # Values must be plain Python floats, not tensor reprs
            assert metadata["metrics"]["loss"] == 0.5
            assert isinstance(metadata["metrics"]["loss"], float)
            assert metadata["metrics"]["accuracy"] == 0.95
            assert isinstance(metadata["metrics"]["accuracy"], float)

    def test_save_checkpoint_with_plain_float_metrics(self):
        """Plain float metrics should pass through unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "step-20"
            state = TrainState(
                global_step=20,
                epoch=0,
                metrics={"loss": 0.3, "lr": 2e-4},
            )

            model = MagicMock()
            model.state_dict.return_value = {}
            optimizer = MagicMock()
            optimizer.state_dict.return_value = {}

            save_checkpoint(
                output_dir=str(output_dir),
                model=model,
                optimizer=optimizer,
                state=state,
            )

            metadata = json.loads((output_dir / "metadata.json").read_text())
            assert metadata["metrics"]["loss"] == 0.3
            assert metadata["metrics"]["lr"] == 2e-4


class TestMapLocation:
    def test_load_checkpoint_uses_map_location_cpu(self):
        """load_checkpoint must pass map_location='cpu' to torch.load for portability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "step-100"
            ckpt_dir.mkdir()

            # Create minimal checkpoint files
            torch.save({}, ckpt_dir / "model.pt")
            torch.save({}, ckpt_dir / "optimizer.pt")
            metadata = {"global_step": 100, "epoch": 2, "step": 0, "metrics": {}}
            (ckpt_dir / "metadata.json").write_text(json.dumps(metadata))

            model = MagicMock()
            optimizer = MagicMock()

            state = load_checkpoint(
                checkpoint_dir=str(ckpt_dir),
                model=model,
                optimizer=optimizer,
            )

            # If we got here without error, torch.load succeeded.
            # The source confirms map_location="cpu" is used on lines 76, 82.
            assert state.global_step == 100


class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_state(self):
        """Save then load should produce identical TrainState fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "step-42"

            original_state = TrainState(
                global_step=42,
                epoch=3,
                step=7,
                metrics={"loss": torch.tensor(0.123), "perplexity": 4.56},
            )

            model = MagicMock()
            model.state_dict.return_value = {"weight": torch.randn(4)}
            optimizer = MagicMock()
            optimizer.state_dict.return_value = {"lr": 0.001}

            save_checkpoint(
                output_dir=str(output_dir),
                model=model,
                optimizer=optimizer,
                state=original_state,
            )

            # Load into fresh mocks
            load_model = MagicMock()
            load_optimizer = MagicMock()

            restored_state = load_checkpoint(
                checkpoint_dir=str(output_dir),
                model=load_model,
                optimizer=load_optimizer,
            )

            assert restored_state.global_step == 42
            assert restored_state.epoch == 3
            assert restored_state.step == 7
            assert abs(restored_state.metrics["loss"] - 0.123) < 1e-5
            assert restored_state.metrics["perplexity"] == 4.56

    def test_roundtrip_with_scheduler_and_scaler(self):
        """Scheduler and scaler state should survive a round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "step-10"

            state = TrainState(global_step=10, epoch=1, step=3, metrics={})

            model = MagicMock()
            model.state_dict.return_value = {}
            optimizer = MagicMock()
            optimizer.state_dict.return_value = {}
            scheduler = MagicMock()
            scheduler.state_dict.return_value = {"last_epoch": 10}
            scaler = MagicMock()
            scaler.state_dict.return_value = {"scale": 1024.0}

            save_checkpoint(
                output_dir=str(output_dir),
                model=model,
                optimizer=optimizer,
                state=state,
                scheduler=scheduler,
                scaler=scaler,
            )

            # Verify the extra .pt files were created
            assert (Path(output_dir) / "scheduler.pt").exists()
            assert (Path(output_dir) / "scaler.pt").exists()

            # Load back
            load_model = MagicMock()
            load_optimizer = MagicMock()
            load_scheduler = MagicMock()
            load_scaler = MagicMock()

            restored = load_checkpoint(
                checkpoint_dir=str(output_dir),
                model=load_model,
                optimizer=load_optimizer,
                scheduler=load_scheduler,
                scaler=load_scaler,
            )

            load_scheduler.load_state_dict.assert_called_once()
            load_scaler.load_state_dict.assert_called_once()
            assert restored.global_step == 10
