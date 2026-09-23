import os
from unittest.mock import patch

import pytest

REQUIRE_TRL_ENV = "XAYTUNE_REQUIRE_TRL"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m not slow')")
    config.addinivalue_line(
        "markers",
        f"trl: needs the optional trl extra; skipped without it, failed if {REQUIRE_TRL_ENV}=1",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Under ``XAYTUNE_REQUIRE_TRL=1`` a skipped TRL test is a failed one.

    TRL is optional, so its tests skip when it is absent -- and a CI job that
    lost the extra would then pass having tested none of it. CI sets the
    variable, so there a skip can only mean the suite did not run.
    """
    outcome = yield
    report = outcome.get_result()
    if (
        report.skipped
        and os.environ.get(REQUIRE_TRL_ENV) == "1"
        and item.get_closest_marker("trl") is not None
    ):
        report.outcome = "failed"
        report.longrepr = (
            f"{item.nodeid} is part of the TRL suite and was skipped, but "
            f"{REQUIRE_TRL_ENV}=1 requires it to run: {report.longrepr}"
        )


@pytest.fixture(autouse=True)
def _no_checkpoint_io():
    """Prevent checkpoint saves from hitting disk in tests with mock models."""
    with patch("xaytune.trainer.checkpoint_callback.save_checkpoint"):
        yield
