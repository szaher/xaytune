import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_tensorboard_module():
    """Mock tensorboard modules before they're imported."""
    mock_summary_writer = MagicMock()
    mock_tb = MagicMock()
    mock_tb.SummaryWriter = mock_summary_writer

    sys.modules["tensorboard"] = MagicMock()
    sys.modules["torch.utils.tensorboard"] = mock_tb
    yield mock_summary_writer
    for mod in [
        "tensorboard",
        "torch.utils.tensorboard",
        "xaytune.logging.tensorboard",
    ]:
        sys.modules.pop(mod, None)


class TestTensorBoardBackend:
    def test_creates_writer(self, mock_tensorboard_module):
        from xaytune.logging.tensorboard import TensorBoardBackend

        TensorBoardBackend(log_dir="runs/test")
        mock_tensorboard_module.assert_called_once_with(log_dir="runs/test")

    def test_log_scalar(self, mock_tensorboard_module):
        from xaytune.logging.tensorboard import TensorBoardBackend

        mock_writer = MagicMock()
        mock_tensorboard_module.return_value = mock_writer

        backend = TensorBoardBackend(log_dir="runs/test")
        backend.log_scalar("loss", 0.5, 10)

        mock_writer.add_scalar.assert_called_once_with("loss", 0.5, 10)

    def test_log_config(self, mock_tensorboard_module):
        from xaytune.logging.tensorboard import TensorBoardBackend

        mock_writer = MagicMock()
        mock_tensorboard_module.return_value = mock_writer

        backend = TensorBoardBackend(log_dir="runs/test")
        backend.log_config({"lr": 0.001})

        mock_writer.add_text.assert_called_once()
        call_args = mock_writer.add_text.call_args
        assert call_args[0][0] == "config"
        assert "lr" in call_args[0][1]

    def test_close(self, mock_tensorboard_module):
        from xaytune.logging.tensorboard import TensorBoardBackend

        mock_writer = MagicMock()
        mock_tensorboard_module.return_value = mock_writer

        backend = TensorBoardBackend(log_dir="runs/test")
        backend.close()

        mock_writer.close.assert_called_once()

    def test_is_logging_backend(self, mock_tensorboard_module):
        from xaytune.logging.base import LoggingBackend
        from xaytune.logging.tensorboard import TensorBoardBackend

        backend = TensorBoardBackend(log_dir="runs/test")
        assert isinstance(backend, LoggingBackend)

    def test_default_log_dir(self, mock_tensorboard_module):
        from xaytune.logging.tensorboard import TensorBoardBackend

        TensorBoardBackend()
        mock_tensorboard_module.assert_called_once_with(log_dir="runs")
