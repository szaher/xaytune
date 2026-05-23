from unittest.mock import patch

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m not slow')"
    )


@pytest.fixture(autouse=True)
def _no_checkpoint_io():
    """Prevent checkpoint saves from hitting disk in tests with mock models."""
    with patch("trainlib.trainer.checkpoint_callback.save_checkpoint"):
        yield
