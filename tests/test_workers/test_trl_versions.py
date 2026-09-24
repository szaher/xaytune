"""The TRL worker runs only on the releases it was classified against (PR-012b).

```text
pyproject.toml  trl extra        ─┐
xaytune.workers.trl              ─┼─ the same ranges
  SUPPORTED_VERSIONS             ─┘
uv.lock                          ─── resolves inside them
the environment under test       ─── is inside them (TRL suite)
```

Three places name the supported releases, and "CI green" means something
only if they agree: CI once installed transformers 5.17 from the loose
pyproject while ``uv sync`` installed the locked 5.9, on which the worker
refuses every run. Each agreement is checked here from the files themselves,
parsed as TOML, so the check depends neither on their formatting nor on what
happens to be installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

if sys.version_info >= (3, 11):
    import tomllib
else:  # pytest itself requires tomli before 3.11
    import tomli as tomllib

from xaytune.workers.trl import (
    SUPPORTED_VERSIONS,
    UnsupportedTrainerVersionError,
    verify_versions,
)

_REPOSITORY = Path(__file__).resolve().parents[2]
_SUPPORTED = {"trl": "1.13.0", "transformers": "5.17.0"}


def _toml(name: str) -> dict[str, Any]:
    with (_REPOSITORY / name).open("rb") as file:
        return tomllib.load(file)


def _trl_extra() -> dict[str, SpecifierSet]:
    """The ``trl`` extra's requirements, read from pyproject.toml.

    Not from installed metadata: an editable install's metadata is as old as
    the install, and would hide a pyproject edit nobody reinstalled after.
    """
    extras = _toml("pyproject.toml")["project"]["optional-dependencies"]
    requirements = [Requirement(r) for r in extras["trl"]]
    return {r.name: r.specifier for r in requirements}


def _locked(name: str) -> str:
    versions = [p["version"] for p in _toml("uv.lock")["package"] if p["name"] == name]
    assert len(versions) == 1, f"uv.lock should resolve {name} to exactly one version: {versions}"
    return versions[0]


# ---- the three places agree ------------------------------------------------------


def test_the_trl_extra_pins_exactly_the_supported_releases() -> None:
    assert _trl_extra() == {name: SpecifierSet(s) for name, s in SUPPORTED_VERSIONS.items()}


@pytest.mark.parametrize("name", sorted(SUPPORTED_VERSIONS))
def test_the_lock_resolves_a_supported_release(name: str) -> None:
    locked = _locked(name)
    assert SpecifierSet(SUPPORTED_VERSIONS[name]).contains(Version(locked), prereleases=False), (
        f"uv.lock resolves {name} {locked}, outside the supported "
        f"{SUPPORTED_VERSIONS[name]}: `uv sync` would install a release the worker refuses"
    )


@pytest.mark.trl
def test_the_environment_under_test_is_a_supported_one() -> None:
    """In the TRL suite, so a CI job cannot pass having tested other releases."""
    pytest.importorskip("trl")
    verify_versions()


# ---- and anything else is refused, clearly ----------------------------------------


def test_the_supported_releases_are_accepted() -> None:
    verify_versions(_SUPPORTED)


def test_an_unclassified_transformers_is_refused_naming_both_versions() -> None:
    """The failure this PR exists for: a lock that resolved transformers 5.9."""
    with pytest.raises(UnsupportedTrainerVersionError) as refused:
        verify_versions({**_SUPPORTED, "transformers": "5.9.0"})
    message = str(refused.value)
    assert "transformers 5.9.0 is installed (supported: >=5.17,<5.18)" in message
    assert "trl" not in message.split("transformers 5.9.0")[0], "only the wrong one is named"
    assert "uv sync --locked --extra trl" in message


@pytest.mark.parametrize(
    "installed",
    [
        {**_SUPPORTED, "transformers": "5.18.0"},
        {**_SUPPORTED, "trl": "1.14.0"},
        {**_SUPPORTED, "trl": "1.12.3"},
        {**_SUPPORTED, "transformers": "5.17.1rc1"},
    ],
    ids=["transformers-newer-minor", "trl-newer-minor", "trl-older-minor", "prerelease"],
)
def test_any_other_release_is_refused(installed: dict[str, str]) -> None:
    with pytest.raises(UnsupportedTrainerVersionError):
        verify_versions(installed)


def test_a_missing_package_is_refused() -> None:
    with pytest.raises(UnsupportedTrainerVersionError, match="trl is not installed"):
        verify_versions({"transformers": "5.17.0", "trl": None})


def test_a_patch_release_is_accepted_and_left_to_the_classification() -> None:
    """Within the minor, the field-by-field check is what catches a moved default."""
    verify_versions({"trl": "1.13.4", "transformers": "5.17.3"})
