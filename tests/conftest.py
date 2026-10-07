import os
from unittest.mock import patch

import pytest

REQUIRE_TRL_ENV = "XAYTUNE_REQUIRE_TRL"
REQUIRE_RAY_ENV = "XAYTUNE_REQUIRE_RAY"
REQUIRE_KUBERAY_ENV = "XAYTUNE_REQUIRE_KUBERAY"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m not slow')")
    config.addinivalue_line(
        "markers",
        f"trl: needs the optional trl extra; skipped without it, failed if {REQUIRE_TRL_ENV}=1",
    )
    config.addinivalue_line(
        "markers",
        "ray: needs a real local Ray head (the ray extra); skipped without it, "
        f"failed if {REQUIRE_RAY_ENV}=1",
    )
    config.addinivalue_line(
        "markers",
        "kuberay: needs a Kubernetes context with the KubeRay operator (XAYTUNE_KUBERAY_CONTEXT) "
        f"and the kuberay extra; skipped without them, failed if {REQUIRE_KUBERAY_ENV}=1",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Under ``XAYTUNE_REQUIRE_TRL=1`` a skipped TRL test is a failed one; Ray and KubeRay likewise.

    TRL is optional, so its tests skip when it is absent -- and a CI job that
    lost the extra would then pass having tested none of it. CI sets the
    variable, so there a skip can only mean the suite did not run.
    """
    outcome = yield
    report = outcome.get_result()
    for marker, variable, suite in (
        ("trl", REQUIRE_TRL_ENV, "TRL"),
        ("ray", REQUIRE_RAY_ENV, "Ray"),
        ("kuberay", REQUIRE_KUBERAY_ENV, "KubeRay"),
    ):
        if (
            report.skipped
            and os.environ.get(variable) == "1"
            and item.get_closest_marker(marker) is not None
        ):
            report.outcome = "failed"
            report.longrepr = (
                f"{item.nodeid} is part of the {suite} suite and was skipped, but "
                f"{variable}=1 requires it to run: {report.longrepr}"
            )


@pytest.fixture(autouse=True)
def _no_checkpoint_io():
    """Prevent checkpoint saves from hitting disk in tests with mock models."""
    with patch("xaytune.trainer.checkpoint_callback.save_checkpoint"):
        yield
