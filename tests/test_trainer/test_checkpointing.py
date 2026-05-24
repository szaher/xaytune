import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xaytune.trainer.callbacks import TrainState
from xaytune.trainer.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint


class TestSaveCheckpoint:
    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "checkpoints" / "step-100"
            state = TrainState(global_step=100, epoch=2)

            with patch("xaytune.trainer.checkpointing.torch"):
                save_checkpoint(
                    output_dir=str(output_dir),
                    model=MagicMock(),
                    optimizer=MagicMock(),
                    state=state,
                )

            assert output_dir.exists()

    def test_save_writes_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "step-100"
            state = TrainState(global_step=100, epoch=2, metrics={"loss": 0.5})

            with patch("xaytune.trainer.checkpointing.torch"):
                save_checkpoint(
                    output_dir=str(output_dir),
                    model=MagicMock(),
                    optimizer=MagicMock(),
                    state=state,
                )

            metadata_path = output_dir / "metadata.json"
            assert metadata_path.exists()
            metadata = json.loads(metadata_path.read_text())
            assert metadata["global_step"] == 100
            assert metadata["epoch"] == 2
            assert metadata["metrics"]["loss"] == 0.5

    @patch("xaytune.trainer.checkpointing.torch")
    def test_save_calls_torch_save(self, mock_torch):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "step-50"
            state = TrainState(global_step=50)
            mock_model = MagicMock()
            mock_optimizer = MagicMock()

            save_checkpoint(
                output_dir=str(output_dir),
                model=mock_model,
                optimizer=mock_optimizer,
                state=state,
            )

            assert mock_torch.save.call_count >= 1


class TestLoadCheckpoint:
    @patch("xaytune.trainer.checkpointing.torch")
    def test_load_restores_state(self, mock_torch):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "step-100"
            ckpt_dir.mkdir()
            metadata = {"global_step": 100, "epoch": 2, "metrics": {"loss": 0.3}}
            (ckpt_dir / "metadata.json").write_text(json.dumps(metadata))

            mock_model = MagicMock()
            mock_optimizer = MagicMock()
            mock_torch.load.return_value = {}

            state = load_checkpoint(
                checkpoint_dir=str(ckpt_dir),
                model=mock_model,
                optimizer=mock_optimizer,
            )

            assert state.global_step == 100
            assert state.epoch == 2
            assert state.metrics["loss"] == 0.3

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(
                checkpoint_dir="nonexistent",
                model=MagicMock(),
                optimizer=MagicMock(),
            )


class TestFindLatestCheckpoint:
    def test_find_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for step in [10, 50, 30]:
                d = Path(tmpdir) / f"step-{step}"
                d.mkdir()
                meta = {"global_step": step, "epoch": 0, "metrics": {}}
                (d / "metadata.json").write_text(json.dumps(meta))

            latest = find_latest_checkpoint(tmpdir)
            assert latest is not None
            assert "step-50" in str(latest)

    def test_no_checkpoints_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            latest = find_latest_checkpoint(tmpdir)
            assert latest is None


class TestScalerCheckpointing:
    @patch("xaytune.trainer.checkpointing.torch")
    def test_save_checkpoint_with_scaler(self, mock_torch):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "step-100"
            state = TrainState(global_step=100, epoch=2)
            mock_scaler = MagicMock()

            save_checkpoint(
                output_dir=str(output_dir),
                model=MagicMock(),
                optimizer=MagicMock(),
                state=state,
                scaler=mock_scaler,
            )

            # Verify torch.save was called with scaler state_dict and scaler.pt path
            scaler_save_calls = [
                call
                for call in mock_torch.save.call_args_list
                if str(call[0][1]).endswith("scaler.pt")
            ]
            assert len(scaler_save_calls) == 1
            mock_scaler.state_dict.assert_called_once()

    @patch("xaytune.trainer.checkpointing.torch")
    def test_save_checkpoint_without_scaler(self, mock_torch):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "step-100"
            state = TrainState(global_step=100, epoch=2)

            save_checkpoint(
                output_dir=str(output_dir),
                model=MagicMock(),
                optimizer=MagicMock(),
                state=state,
            )

            # Verify no scaler.pt was saved
            scaler_save_calls = [
                call
                for call in mock_torch.save.call_args_list
                if str(call[0][1]).endswith("scaler.pt")
            ]
            assert len(scaler_save_calls) == 0

    @patch("xaytune.trainer.checkpointing.torch")
    def test_load_checkpoint_with_scaler(self, mock_torch):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "step-100"
            ckpt_dir.mkdir()
            metadata = {"global_step": 100, "epoch": 2, "step": 0, "metrics": {}}
            (ckpt_dir / "metadata.json").write_text(json.dumps(metadata))
            # Create scaler.pt so the file exists check passes
            (ckpt_dir / "scaler.pt").write_bytes(b"fake")

            mock_scaler = MagicMock()
            mock_torch.load.return_value = {"scale": 1024.0}

            load_checkpoint(
                checkpoint_dir=str(ckpt_dir),
                model=MagicMock(),
                optimizer=MagicMock(),
                scaler=mock_scaler,
            )

            mock_scaler.load_state_dict.assert_called_once_with({"scale": 1024.0})

    @patch("xaytune.trainer.checkpointing.torch")
    def test_load_checkpoint_no_scaler_file(self, mock_torch):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "step-100"
            ckpt_dir.mkdir()
            metadata = {"global_step": 100, "epoch": 2, "step": 0, "metrics": {}}
            (ckpt_dir / "metadata.json").write_text(json.dumps(metadata))
            # No scaler.pt file created

            mock_scaler = MagicMock()
            mock_torch.load.return_value = {}

            # Should not raise, and should not call load_state_dict on scaler
            state = load_checkpoint(
                checkpoint_dir=str(ckpt_dir),
                model=MagicMock(),
                optimizer=MagicMock(),
                scaler=mock_scaler,
            )

            mock_scaler.load_state_dict.assert_not_called()
            assert state.global_step == 100


class TestStepMetadata:
    def test_metadata_includes_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "step-5"
            state = TrainState(global_step=50, epoch=1, step=5)

            with patch("xaytune.trainer.checkpointing.torch"):
                save_checkpoint(
                    output_dir=str(output_dir),
                    model=MagicMock(),
                    optimizer=MagicMock(),
                    state=state,
                )

            metadata_path = output_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            assert metadata["step"] == 5

    @patch("xaytune.trainer.checkpointing.torch")
    def test_load_restores_step(self, mock_torch):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "step-5"
            ckpt_dir.mkdir()
            metadata = {"global_step": 50, "epoch": 1, "step": 5, "metrics": {}}
            (ckpt_dir / "metadata.json").write_text(json.dumps(metadata))

            mock_torch.load.return_value = {}

            state = load_checkpoint(
                checkpoint_dir=str(ckpt_dir),
                model=MagicMock(),
                optimizer=MagicMock(),
            )

            assert state.step == 5
