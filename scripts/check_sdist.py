"""Does the source distribution contain the package, and nothing else?

    python scripts/check_sdist.py dist/xaytune-*.tar.gz

An sdist is built from a working tree, and a working tree accumulates things:
virtualenvs, editor and agent indexes, caches, local databases. Whatever the
build picks up is uploaded to PyPI for good. ``pyproject.toml`` restricts the
sdist to an allow-list; this checks that the archive built from it really is
that, and names every file that is not.

It also checks the other direction: everything the wheel needs is present --
the package, its version module, every migration -- and the license text the
package declares. A migration left out would still build, and fail only when a
user's store was created; a missing LICENSE would publish Apache-2.0 code
without the terms Apache-2.0 requires to travel with it.

Standard library only, so it runs before anything is installed.
"""

from __future__ import annotations

import sys
import tarfile

if sys.version_info >= (3, 11):
    import tomllib
else:  # the tests run this on 3.10 too, where pytest brings tomli
    import tomli as tomllib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent

ALLOWED_FILES = frozenset(
    # PKG-INFO is the build's metadata. Hatch adds .gitignore itself, so a build
    # from the sdist applies the same exclusions as a build from the tree.
    {"PKG-INFO", ".gitignore", "pyproject.toml", "README.md", "CHANGELOG.md", "LICENSE"}
)
ALLOWED_DIRECTORIES = ("xaytune",)

FORBIDDEN_PARTS = (
    ".git",
    ".github",
    ".venv",
    ".codegraph",
    ".serena",
    ".saad-agent",
    ".playwright-cli",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "site",
    "output",
)
"""Named explicitly, so a failure says *why* a file must not ship, not only that
it is off the list. ``.venv`` matches ``.venv310`` and the like too."""

LICENSE_EXPRESSION = "Apache-2.0"
"""The SPDX identifier ``pyproject.toml`` must declare, for the text in LICENSE."""

FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".db", ".sqlite", ".sqlite3", ".egg-info")


def _forbidden(part: str) -> bool:
    return part in FORBIDDEN_PARTS or part.startswith(".venv") or part.endswith(FORBIDDEN_SUFFIXES)


def _relative(name: str) -> PurePosixPath | None:
    """*name* without the ``xaytune-<version>/`` directory every member sits under."""
    parts = PurePosixPath(name).parts
    return PurePosixPath(*parts[1:]) if len(parts) > 1 else None


def problems(
    members: list[str], root: Path = ROOT, *, declared_license: object = LICENSE_EXPRESSION
) -> list[str]:
    """Every reason an sdist with these member names must not be published.

    *declared_license* is ``project.license`` from the archive's own
    ``pyproject.toml``.
    """
    found: list[str] = []
    if declared_license != LICENSE_EXPRESSION:
        found.append(
            f"pyproject.toml declares license {declared_license!r}, not "
            f"{LICENSE_EXPRESSION!r}, which is the text LICENSE carries"
        )
    files: set[PurePosixPath] = set()
    for name in members:
        path = _relative(name)
        if path is None:
            continue
        files.add(path)
        forbidden = [part for part in path.parts if _forbidden(part)]
        if forbidden:
            found.append(f"{path}: local or generated state ({forbidden[0]}) must not ship")
        elif str(path) not in ALLOWED_FILES and path.parts[0] not in ALLOWED_DIRECTORIES:
            found.append(f"{path}: not on the sdist allow-list in pyproject.toml")

    required = {PurePosixPath(f) for f in ("pyproject.toml", "README.md", "PKG-INFO", "LICENSE")}
    required |= {
        PurePosixPath("xaytune", p.name) for p in (root / "xaytune").glob("*.py") if p.is_file()
    }
    required |= {
        PurePosixPath("xaytune/storage/migrations", p.name)
        for p in (root / "xaytune" / "storage" / "migrations").glob("*.sql")
    }
    for missing in sorted(required - files):
        found.append(f"{missing}: missing, and the wheel is built from this archive")
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_sdist.py <sdist.tar.gz>", file=sys.stderr)
        return 2
    with tarfile.open(argv[1]) as archive:
        files = [m for m in archive.getmembers() if m.isfile()]
        members = [m.name for m in files]
        pyproject = next(
            (m for m in files if _relative(m.name) == PurePosixPath("pyproject.toml")), None
        )
        declared = None
        if pyproject is not None:
            handle = archive.extractfile(pyproject)
            assert handle is not None
            declared = tomllib.loads(handle.read().decode("utf-8"))["project"].get("license")
    found = problems(members, declared_license=declared)
    for problem in found:
        print(f"error: {problem}", file=sys.stderr)
    if found:
        return 1
    print(f"{Path(argv[1]).name}: {len(members)} files, all on the allow-list")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
