import pytest
from unittest.mock import patch, MagicMock
from trainlib.logging.tensorboard import TensorBoardBackend


class TestTensorBoardBackend:
    @patch("trainlib.logging.tensorboard.SummaryWriter")
    def test_creates_writer(self, mock_writer_cls):
        backend = TensorBoardBackend(log_dir="runs/test")
        mock_writer_cls.assert_called_once_with(log_dir="runs/test")

    @patch("trainlib.logging.tensorboard.SummaryWriter")
    def test_log_scalar(self, mock_writer_cls):
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        backend = TensorBoardBackend(log_dir="runs/test")
        backend.log_scalar("loss", 0.5, 10)

        mock_writer.add_scalar.assert_called_once_with("loss", 0.5, 10)

    @patch("trainlib.logging.tensorboard.SummaryWriter")
    def test_log_config(self, mock_writer_cls):
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        backend = TensorBoardBackend(log_dir="runs/test")
        backend.log_config({"lr": 0.001})

        mock_writer.add_text.assert_called_once()
        call_args = mock_writer.add_text.call_args
        assert call_args[0][0] == "config"
        assert "lr" in call_args[0][1]

    @patch("trainlib.logging.tensorboard.SummaryWriter")
    def test_close(self, mock_writer_cls):
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        backend = TensorBoardBackend(log_dir="runs/test")
        backend.close()

        mock_writer.close.assert_called_once()

    @patch("trainlib.logging.tensorboard.SummaryWriter")
    def test_is_logging_backend(self, mock_writer_cls):
        from trainlib.logging.base import LoggingBackend
        backend = TensorBoardBackend(log_dir="runs/test")
        assert isinstance(backend, LoggingBackend)

    @patch("trainlib.logging.tensorboard.SummaryWriter")
    def test_default_log_dir(self, mock_writer_cls):
        backend = TensorBoardBackend()
        mock_writer_cls.assert_called_once_with(log_dir="runs")
