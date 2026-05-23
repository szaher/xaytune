import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_wandb_module():
    """Mock the wandb module before it's imported."""
    # Create a proper module mock with __spec__ but allow all attributes
    mock_wandb = MagicMock()
    mock_wandb.__spec__ = MagicMock()
    mock_wandb.__spec__.name = "wandb"
    sys.modules["wandb"] = mock_wandb
    yield mock_wandb
    # Clean up after each test
    if "wandb" in sys.modules:
        del sys.modules["wandb"]
    if "trainlib.logging.wandb" in sys.modules:
        del sys.modules["trainlib.logging.wandb"]


class TestWandbBackend:
    def test_init_calls_wandb_init(self, mock_wandb_module):
        from trainlib.logging.wandb import WandbBackend

        WandbBackend(project="my-project", run_name="run-1")
        mock_wandb_module.init.assert_called_once_with(project="my-project", name="run-1")

    def test_log_scalar(self, mock_wandb_module):
        from trainlib.logging.wandb import WandbBackend

        backend = WandbBackend(project="test")
        backend.log_scalar("loss", 0.5, 10)

        mock_wandb_module.log.assert_called_once_with({"loss": 0.5}, step=10)

    def test_log_config(self, mock_wandb_module):
        from trainlib.logging.wandb import WandbBackend

        mock_wandb_module.config = MagicMock()
        backend = WandbBackend(project="test")
        backend.log_config({"lr": 0.001, "epochs": 3})

        mock_wandb_module.config.update.assert_called_once_with({"lr": 0.001, "epochs": 3})

    def test_close_calls_finish(self, mock_wandb_module):
        from trainlib.logging.wandb import WandbBackend

        backend = WandbBackend(project="test")
        backend.close()

        mock_wandb_module.finish.assert_called_once()

    def test_is_logging_backend(self, mock_wandb_module):
        from trainlib.logging.base import LoggingBackend
        from trainlib.logging.wandb import WandbBackend

        backend = WandbBackend(project="test")
        assert isinstance(backend, LoggingBackend)

    def test_default_project(self, mock_wandb_module):
        from trainlib.logging.wandb import WandbBackend

        WandbBackend()
        mock_wandb_module.init.assert_called_once_with(project="trainlib", name=None)
