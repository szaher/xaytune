"""The daemon commands' wiring, without a daemon (PR-029).

The commands themselves run against a real daemon process in
:mod:`tests.test_daemon.test_daemon_cli`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from xaytune.cli import _build_parser, main
from xaytune.cli_control import CONTROL_COMMANDS


def test_every_daemon_command_is_a_subcommand() -> None:
    parser = _build_parser()
    for command in CONTROL_COMMANDS:
        args = parser.parse_args(
            [command, "act_1" if command in ("approve", "reject") else "exp_1"]
            + (["--reason", "why"] if command in ("approve", "reject") else [])
        )
        assert args.command == command


def test_a_command_needs_a_state_database(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("XAYTUNE_STATE", raising=False)
    assert main(["status", "exp_1"]) == 2
    assert "--state is required" in capsys.readouterr().err


def test_the_state_database_defaults_to_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XAYTUNE_STATE", str(tmp_path / "state.db"))
    assert main(["status", "exp_missing"]) == 1
    assert "exp_missing" in capsys.readouterr().err
    assert (tmp_path / "state.db").exists()


def test_importing_the_cli_does_not_import_the_control_plane() -> None:
    code = (
        "import sys, xaytune.cli_control; "
        "print(any(m.startswith(('xaytune.daemon', 'xaytune.storage', 'xaytune.experiment')) "
        "for m in sys.modules))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "False"
