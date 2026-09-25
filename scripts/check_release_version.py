"""Refuse to publish a release whose tag does not name the version being built.

PyPI publishing runs when a GitHub release is published, and builds whatever
``pyproject.toml`` says. Nothing tied the two together, and they have already
drifted: the GitHub release ``v0.1.0`` is what uploaded ``0.6.0`` to PyPI. A
release named for one version that uploads another cannot be traced from
either side, and a PyPI upload cannot be taken back.

So the release job runs this first::

    python scripts/check_release_version.py "$TAG"

It passes only when all three agree::

    release tag  ==  v + pyproject.toml version  ==  v + xaytune/_version.py

``xaytune/_version.py`` is what the package reports at runtime --
``xaytune.__version__``, and the ``xaytune_version`` every built-in plugin
descriptor records -- so a user sees the version they installed. The files are
read as text and nothing is imported from the package, so this runs before
any dependency is installed.

The TestPyPI dispatch does not run it: a manual test upload has no tag to check.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # the tests run this on 3.10 too, where pytest brings tomli
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent


def project_version(root: Path = ROOT) -> str:
    """The version ``pyproject.toml`` declares, which is the one ``uv build`` builds."""
    with (root / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def runtime_version(root: Path = ROOT) -> str:
    """``xaytune/_version.py``'s ``__version__``, read without importing it."""
    tree = ast.parse((root / "xaytune" / "_version.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise ValueError("xaytune/_version.py assigns no string literal to __version__")


def problems(tag: str, root: Path = ROOT) -> list[str]:
    """Every reason *tag* may not publish the package at *root*; empty if it may."""
    version = project_version(root)
    found: list[str] = []
    if tag != f"v{version}":
        found.append(
            f"release tag {tag!r} does not name the version being built: pyproject.toml "
            f"says {version!r}, so the tag must be 'v{version}'"
        )
    runtime = runtime_version(root)
    if runtime != version:
        found.append(
            f"xaytune/_version.py says {runtime!r} but pyproject.toml says {version!r}; "
            f"the installed package would report a version it is not"
        )
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not argv[1]:
        print("usage: check_release_version.py <release-tag>", file=sys.stderr)
        return 2
    found = problems(argv[1])
    for problem in found:
        print(f"error: {problem}", file=sys.stderr)
    if found:
        return 1
    print(f"release tag {argv[1]} matches version {project_version()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
