"""The control-plane examples run against the public API as it is.

``scripts/wheel_smoke.py`` runs the same examples against an installed wheel
in CI. This runs them against the source tree, so a change that breaks one
fails where it was made.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "control_plane"
SCRIPTS = sorted(EXAMPLES.glob("[0-9][0-9]_*.py"))


def test_there_are_examples() -> None:
    assert [s.name for s in SCRIPTS][:1] == ["01_compile_a_candidate.py"]


def test_compiling_a_candidate_needs_no_model() -> None:
    """The example promises it runs anywhere; run it as a user would."""
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS[0])], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert "refused by native" in completed.stdout
    assert "submit request digest: sha256:" in completed.stdout


@pytest.mark.parametrize("script", SCRIPTS[1:], ids=lambda p: p.name)
def test_each_runnable_example_imports(script: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Import-only: running them trains a model, which the end-to-end tests do."""
    monkeypatch.syspath_prepend(str(EXAMPLES))
    namespace = runpy.run_path(str(script), run_name="__example__")
    assert "EmbeddedControllerHost" in namespace
