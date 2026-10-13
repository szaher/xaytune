"""Studio's network exposure and code execution (UX-00).

Studio has no authentication, so whatever it serves is available to whoever
can reach it. These tests need no Gradio: they run in every CI job.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import xaytune.studio
from xaytune.cli import _build_parser, main
from xaytune.studio import server
from xaytune.studio.server import (
    SHARE_DISABLED,
    StudioExposureError,
    is_loopback_host,
    launch,
)

STUDIO_DIR = Path(xaytune.studio.__file__).parent


@pytest.fixture
def fake_gradio(monkeypatch, tmp_path):
    """Run ``launch`` with Gradio and the app replaced, recording the calls."""
    gradio = MagicMock(name="gradio")
    app_module = types.ModuleType("xaytune.studio.app")
    app_module.create_app = MagicMock(name="create_app")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gradio", gradio)
    monkeypatch.setitem(sys.modules, "xaytune.studio.app", app_module)
    monkeypatch.setattr("xaytune.studio.jobs.JobManager", MagicMock(name="JobManager"))
    monkeypatch.setenv("HOME", str(tmp_path))
    return app_module.create_app


class TestDefaults:
    def test_cli_binds_to_loopback_by_default(self):
        args = _build_parser().parse_args(["studio"])
        assert args.host == "127.0.0.1"
        assert args.share is False

    def test_server_binds_to_loopback_by_default(self, fake_gradio):
        launch()
        app = fake_gradio.return_value
        app.launch.assert_called_once_with(server_name="127.0.0.1", server_port=7860, share=False)
        assert fake_gradio.call_args.kwargs["allow_remote_code"] is True

    def test_cli_passes_the_default_host_to_the_server(self):
        with patch.object(server, "launch") as launched:
            assert main(["studio"]) == 0
        launched.assert_called_once_with(host="127.0.0.1", port=7860, share=False)


class TestLoopback:
    @pytest.mark.parametrize(
        "host", ["127.0.0.1", "127.1.2.3", "localhost", "LOCALHOST", "::1", "[::1]"]
    )
    def test_loopback_hosts(self, host):
        assert is_loopback_host(host)

    @pytest.mark.parametrize(
        "host", ["0.0.0.0", "::", "192.168.1.10", "10.0.0.5", "example.com", "", "localhost.evil"]
    )
    def test_everything_else_is_reachable_from_the_network(self, host):
        assert not is_loopback_host(host)


class TestRemoteBinding:
    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::"])
    def test_remote_host_disables_remote_code(self, fake_gradio, host, caplog):
        with caplog.at_level(logging.WARNING, logger="xaytune.studio.server"):
            launch(host=host)
        assert fake_gradio.call_args.kwargs["allow_remote_code"] is False
        fake_gradio.return_value.launch.assert_called_once_with(
            server_name=host, server_port=7860, share=False
        )
        assert "no authentication" in caplog.text

    def test_explicit_loopback_host_keeps_remote_code(self, fake_gradio):
        launch(host="localhost", port=8080)
        assert fake_gradio.call_args.kwargs["allow_remote_code"] is True


class TestPublicSharing:
    def test_server_refuses_share_before_starting(self, monkeypatch):
        # Gradio is never imported: the refusal comes first.
        monkeypatch.setitem(sys.modules, "gradio", None)
        with pytest.raises(StudioExposureError, match="Public sharing is disabled"):
            launch(share=True)

    def test_server_refuses_share_on_any_host(self, fake_gradio):
        for host in ("127.0.0.1", "0.0.0.0"):
            with pytest.raises(StudioExposureError):
                launch(host=host, share=True)
        fake_gradio.assert_not_called()

    def test_cli_refuses_share(self, capsys):
        with patch.object(server, "launch") as launched:
            assert main(["studio", "--share"]) == 2
        launched.assert_not_called()
        assert SHARE_DISABLED in capsys.readouterr().err


class TestNoCodeExecution:
    def test_the_code_runner_is_gone(self):
        assert importlib.util.find_spec("xaytune.studio.code_runner") is None

    @pytest.mark.parametrize("path", sorted(STUDIO_DIR.glob("*.py")), ids=lambda p: p.name)
    def test_studio_never_executes_source(self, path):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"exec", "eval", "compile"}, (
                    f"{path.name}:{node.lineno} calls {node.func.id}()"
                )
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "code_runner" not in node.module, f"{path.name} imports {node.module}"
