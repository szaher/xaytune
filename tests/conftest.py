from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_checkpoint_io():
    """Prevent checkpoint saves from hitting disk in tests with mock models."""
    with patch("trainlib.trainer.checkpoint_callback.save_checkpoint"):
        yield
