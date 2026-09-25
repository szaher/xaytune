"""The package version, in one module that imports nothing.

``xaytune/__init__.py`` re-exports it as ``xaytune.__version__``, and every
built-in plugin descriptor states it as the ``xaytune_version`` it was built
with. They import it from here rather than from ``xaytune``, whose
``__init__`` is the public API: a compiler or runtime reading the version
must not depend on what that module imports, or be caught in a cycle
through it.

Kept equal to ``pyproject.toml``'s ``version``. Nothing derives one from the
other, so the release gate (``scripts/check_release_version.py``) refuses to
publish when they disagree, and ``tests/test_release_version.py`` fails first.
"""

__version__ = "1.0.0a1"
