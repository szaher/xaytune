"""The sdist check refuses local state, stray files and a package with parts missing.

``scripts/check_sdist.py`` runs on the real archive in CI and before every
upload. These tests hold its rules to the failures it exists for, on member
lists shaped like a real sdist's.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pytest itself requires tomli before 3.11
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "xaytune-0.6.0/"


def _check() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_sdist", ROOT / "scripts" / "check_sdist.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _check()


def _members() -> list[str]:
    """What a correct sdist of this tree holds."""
    package = [
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "xaytune").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    ]
    top = ["PKG-INFO", ".gitignore", "pyproject.toml", "README.md", "CHANGELOG.md", "LICENSE"]
    return [PREFIX + name for name in top + package]


def test_a_correct_sdist_passes() -> None:
    assert check.problems(_members()) == []


@pytest.mark.parametrize(
    "stray",
    [
        ".venv310/bin/python",
        ".venv/lib/python3.10/site-packages/torch/__init__.py",
        ".codegraph/index.db",
        ".serena/cache/symbols.json",
        ".saad-agent/notes.md",
        ".git/HEAD",
        "xaytune/__pycache__/cli.cpython-310.pyc",
        "xaytune/state.db",
        "dist/xaytune-0.5.0-py3-none-any.whl",
    ],
)
def test_local_state_is_refused(stray: str) -> None:
    [problem] = check.problems([*_members(), PREFIX + stray])
    assert problem.startswith(stray) and "must not ship" in problem


@pytest.mark.parametrize("stray", ["tests/conftest.py", "RESEARCH_PROMPT.md", "uv.lock"])
def test_anything_off_the_allow_list_is_refused(stray: str) -> None:
    [problem] = check.problems([*_members(), PREFIX + stray])
    assert "not on the sdist allow-list" in problem


@pytest.mark.parametrize(
    "missing",
    [
        "xaytune/_version.py",
        "xaytune/storage/migrations/006_evaluation_lifecycle.sql",
        "README.md",
        "LICENSE",
    ],
)
def test_a_missing_part_of_the_package_is_refused(missing: str) -> None:
    members = [m for m in _members() if m != PREFIX + missing]
    [problem] = check.problems(members)
    assert problem.startswith(missing) and "missing" in problem


@pytest.mark.parametrize("declared", ["MIT", None, "Apache-2.0 OR MIT"])
def test_a_license_other_than_the_text_shipped_is_refused(declared: object) -> None:
    [problem] = check.problems(_members(), declared_license=declared)
    assert "declares license" in problem


def test_the_package_declares_the_license_it_ships() -> None:
    """Presence and the SPDX identifier; the text itself is Apache's, verbatim."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        assert tomllib.load(handle)["project"]["license"] == check.LICENSE_EXPRESSION
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text and "Version 2.0, January 2004" in text


def test_the_check_allows_exactly_what_pyproject_includes() -> None:
    """The two allow-lists are one decision; neither may widen alone."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        include = tomllib.load(handle)["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    declared = {entry.lstrip("/") for entry in include}
    allowed = (set(check.ALLOWED_FILES) - {"PKG-INFO", ".gitignore"}) | set(
        check.ALLOWED_DIRECTORIES
    )
    assert declared == allowed
