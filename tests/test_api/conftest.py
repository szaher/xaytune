"""A real state database for the Studio API tests: the spec 18 adaptive scenario, recorded.

No mocked records (ADR-023 acceptance 7). One controller host carries one
experiment through training with a CUDA OOM and a recovered attempt,
evaluation, a decision, a planner branch and the child's training, into a
file database. The host is closed before any API reads it.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from tests.test_experiment.adaptive_fixtures import adaptive_spec  # noqa: E402
from tests.test_experiment.test_adaptive_mvp import _host, _submit_and_wait, _world  # noqa: E402
from xaytune.api import create_app  # noqa: E402

BASE_URL = "http://127.0.0.1"


@dataclass(frozen=True)
class Recorded:
    """The recorded scenario: its database and the ids a test needs."""

    state: Path
    experiment_id: str
    status: str


def _record(directory: Path) -> Recorded:
    world = _world(directory)

    async def scenario() -> tuple[str, Any]:
        host = _host(directory, world)
        try:
            handle, result = await _submit_and_wait(host, adaptive_spec(directory))
            return str(handle.experiment_id), result
        finally:
            await host.close()

    experiment_id, result = asyncio.run(scenario())
    return Recorded(
        state=directory / "state.db", experiment_id=experiment_id, status=result.status.value
    )


@pytest.fixture(scope="session")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> Recorded:
    return _record(tmp_path_factory.mktemp("studio-api-record"))


@pytest.fixture
def state_copy(recorded: Recorded, tmp_path: Path) -> Path:
    """A private copy of the recorded database, for tests that change the file around it."""
    copy = tmp_path / "state.db"
    shutil.copyfile(recorded.state, copy)
    return copy


def client(state: Path | str, *, web_client: Path | None = None, **options: Any) -> TestClient:
    """The API over *state*; without the web client unless one is named, so a
    locally built bundle never changes what these tests see."""
    return TestClient(create_app(state, web_client=web_client), base_url=BASE_URL, **options)


@pytest.fixture
def api(recorded: Recorded) -> TestClient:
    return client(recorded.state)
