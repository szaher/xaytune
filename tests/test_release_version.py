"""The release gate: a tag publishes only the version it names.

``scripts/check_release_version.py`` runs before a GitHub release uploads to
PyPI. These tests hold it to refusing the drift that already happened once --
the ``v0.1.0`` release uploaded ``0.6.0`` -- and to passing the tag the
current version would be released under.
"""

from __future__ import annotations

import ast
import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import pytest

import xaytune
import xaytune._version

ROOT = Path(__file__).resolve().parent.parent


def _gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_release_version", ROOT / "scripts" / "check_release_version.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _gate()


def _copy_of_the_version_files(tmp_path: Path, *, runtime: str | None = None) -> Path:
    (tmp_path / "xaytune").mkdir()
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    source = (ROOT / "xaytune" / "_version.py").read_text(encoding="utf-8")
    if runtime is not None:
        source = source.replace(
            f'__version__ = "{xaytune.__version__}"', f'__version__ = "{runtime}"'
        )
    (tmp_path / "xaytune" / "_version.py").write_text(source, encoding="utf-8")
    return tmp_path


def test_the_runtime_version_is_the_packaged_one() -> None:
    """What ``xaytune.__version__`` reports is what ``uv build`` builds."""
    assert xaytune.__version__ is xaytune._version.__version__
    assert gate.runtime_version() == gate.project_version()


def _hardcoded_xaytune_versions() -> list[str]:
    """Every ``xaytune_version=`` in the package given as a literal, not ``__version__``."""
    found = []
    for path in sorted((ROOT / "xaytune").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.keyword) and node.arg == "xaytune_version":
                if isinstance(node.value, ast.Constant):
                    found.append(f"{path.relative_to(ROOT)}:{node.value.lineno}")
    return found


def test_no_built_in_plugin_states_a_version_of_its_own() -> None:
    """A descriptor's ``xaytune_version`` is the build it ships in, so it cannot drift.

    A literal would still say 0.6.0 in the release after it, claiming every
    plugin was built against a version it was not.
    """
    assert _hardcoded_xaytune_versions() == []


def test_the_built_in_plugins_record_the_running_version() -> None:
    from xaytune.compilation.native import NativeCompiler
    from xaytune.compilation.trl import TRLCompiler
    from xaytune.runtimes.local import LocalRuntime

    for plugin in (NativeCompiler, TRLCompiler, LocalRuntime):
        assert plugin.descriptor.xaytune_version == gate.project_version()


def test_the_current_version_releases_under_its_own_tag() -> None:
    assert gate.problems(f"v{gate.project_version()}") == []
    assert gate.main(["check", f"v{gate.project_version()}"]) == 0


@pytest.mark.parametrize(
    "tag",
    ["v0.1.0", "0.6.0", "v0.6", "v1.0.0a1", "release-0.6.0", ""],
    ids=["old-tag", "no-v", "short", "future", "prefix", "empty"],
)
def test_any_other_tag_is_refused(tag: str) -> None:
    if tag == f"v{gate.project_version()}":
        pytest.skip("this tag is the current version's")
    assert gate.main(["check", tag]) != 0
    if tag:
        [problem] = gate.problems(tag)
        assert "does not name the version being built" in problem


def test_a_runtime_version_that_disagrees_is_refused(tmp_path: Path) -> None:
    root = _copy_of_the_version_files(tmp_path, runtime="9.9.9")
    version = gate.project_version(root)
    [problem] = gate.problems(f"v{version}", root)
    assert "xaytune/_version.py says '9.9.9'" in problem


def test_an_unreadable_runtime_version_is_an_error(tmp_path: Path) -> None:
    root = _copy_of_the_version_files(tmp_path)
    module = root / "xaytune" / "_version.py"
    module.write_text(
        module.read_text(encoding="utf-8").replace("__version__ =", "_version ="), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="__version__"):
        gate.runtime_version(root)
