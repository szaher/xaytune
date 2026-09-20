"""The top-level API resolves lazily without changing what it exposes.

``xaytune.core`` must import without an ML runtime, and importing
``xaytune.core`` executes the package ``__init__`` first — so the public API
has to resolve on access rather than at import time (ADR-010).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import xaytune

PUBLIC_NAMES = [
    "align",
    "evaluate",
    "finetune",
    "JobManager",
    "lr_find",
    "pipeline",
    "pretrain",
]


class TestPublicApiPreserved:
    @pytest.mark.parametrize("name", PUBLIC_NAMES)
    def test_public_name_resolves(self, name):
        assert callable(getattr(xaytune, name))

    def test_all_names_resolve(self):
        for name in xaytune.__all__:
            assert hasattr(xaytune, name)

    def test_discover_plugins_is_still_reachable(self):
        """Not in __all__, but it was importable from the package before."""
        assert callable(xaytune.discover_plugins)

    def test_dir_lists_the_lazy_names(self):
        listed = dir(xaytune)
        for name in PUBLIC_NAMES:
            assert name in listed

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="no attribute 'nope'"):
            xaytune.nope

    def test_lazy_attribute_is_cached(self):
        assert xaytune.finetune is xaytune.finetune

    def test_identities_match_the_underlying_callables(self):
        from xaytune.pipeline import run_pipeline
        from xaytune.recipes.finetune import finetune

        assert xaytune.finetune is finetune
        assert xaytune.pipeline is run_pipeline


class TestSubmoduleAccess:
    @pytest.mark.parametrize("name", ["trainer", "config", "data", "eval", "recipes"])
    def test_submodule_attribute_access_still_works(self, name):
        """Submodules used to be registered by the eager imports."""
        assert getattr(xaytune, name).__name__ == f"xaytune.{name}"

    def test_submodule_can_be_imported_first(self):
        """Regression: xaytune.trainer <-> xaytune.eval was a latent cycle.

        The old eager ``__init__`` masked it by importing ``xaytune.eval``
        before ``xaytune.trainer``.
        """
        result = subprocess.run(
            [sys.executable, "-c", "import xaytune.trainer; print(xaytune.trainer.Trainer)"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
