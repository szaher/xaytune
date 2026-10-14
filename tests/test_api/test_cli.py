"""``xaytune studio-web``: refuses before it listens (ADR-023 §S10)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_api.conftest import Recorded
from xaytune.api.server import ApiExposureError, is_loopback, serve
from xaytune.cli import main


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "[::1]", "localhost"])
def test_loopback_hosts(host: str) -> None:
    assert is_loopback(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.5", "studio.example", ""])
def test_other_hosts_are_refused_before_anything_listens(
    recorded: Recorded, host: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("uvicorn must not start")

    monkeypatch.setattr("uvicorn.run", never)
    assert not is_loopback(host)
    with pytest.raises(ApiExposureError):
        serve(recorded.state, host=host)


def test_the_command_serves_on_loopback(
    recorded: Recorded, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: dict[str, object] = {}

    def run(app: object, **options: object) -> None:
        served.update(options)

    monkeypatch.setattr("uvicorn.run", run)
    assert main(["studio-web", "--state", str(recorded.state), "--port", "9999"]) == 0
    assert served["host"] == "127.0.0.1"
    assert served["port"] == 9999
    assert served["proxy_headers"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["studio-web"],
        ["studio-web", "--state", "/nonexistent/state.db"],
        ["studio-web", "--host", "0.0.0.0"],
    ],
)
def test_the_command_refuses_with_exit_code_2(
    recorded: Recorded,
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("XAYTUNE_STATE", raising=False)
    if "--host" in arguments:
        arguments = [*arguments, "--state", str(recorded.state)]
    assert main(arguments) == 2
    assert capsys.readouterr().err.startswith("Error:")


def test_a_missing_state_database_is_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "state.db"
    assert main(["studio-web", "--state", str(missing)]) == 2
    assert not missing.exists()
