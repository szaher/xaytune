"""xaytune — An opinionated LLM training and fine-tuning library."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# `pipeline` is the one public name that collides with a submodule
# (xaytune/pipeline.py). Importing that submodule anywhere makes Python set it
# as an attribute on this package, which shadows the lazy lookup below --
# `__getattr__` only runs when normal attribute lookup fails. Binding it here
# keeps `xaytune.pipeline` pointing at the callable regardless of import order,
# as it did before. The module is pydantic-only, so this pulls in no ML stack.
from xaytune.pipeline import run_pipeline as pipeline

__version__ = "0.6.0"

# Public names are resolved lazily (PEP 562) so that importing a submodule such
# as ``xaytune.core`` does not drag in torch/transformers through the recipes.
# ``xaytune.core`` is the control-plane core and must stay importable without an
# ML runtime installed (ADR-010).
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "align": ("xaytune.recipes.align", "align"),
    "discover_plugins": ("xaytune.plugins", "discover_plugins"),
    "evaluate": ("xaytune.eval", "evaluate"),
    "finetune": ("xaytune.recipes.finetune", "finetune"),
    "JobManager": ("xaytune.studio.jobs", "JobManager"),
    "lr_find": ("xaytune.trainer.lr_finder", "lr_find"),
    "pretrain": ("xaytune.recipes.pretrain", "pretrain"),
}

if TYPE_CHECKING:  # pragma: no cover - import shapes for type checkers only
    from xaytune.eval.evaluate import evaluate
    from xaytune.plugins import discover_plugins as discover_plugins
    from xaytune.recipes.align.align import align
    from xaytune.recipes.finetune import finetune
    from xaytune.recipes.pretrain import pretrain
    from xaytune.studio.jobs import JobManager
    from xaytune.trainer.lr_finder import lr_find


def __getattr__(name: str) -> Any:
    """Resolve a public attribute on first access.

    Plugin discovery runs here rather than at import time: it can import the
    recipe/model/format/metric registries, which pull in the ML stack.
    """
    if name not in _LAZY_ATTRS:
        # Submodules used to become attributes as a side effect of the eager
        # imports this module no longer does, so `xaytune.trainer` and friends
        # must still resolve.
        if name.startswith("_"):
            raise AttributeError(f"module 'xaytune' has no attribute '{name}'")
        try:
            return importlib.import_module(f"xaytune.{name}")
        except ModuleNotFoundError as exc:
            if exc.name == f"xaytune.{name}":
                raise AttributeError(f"module 'xaytune' has no attribute '{name}'") from None
            raise  # a real missing dependency inside that submodule

    module_name, attr = _LAZY_ATTRS[name]

    if name != "discover_plugins":
        from xaytune.plugins import discover_plugins as _discover

        _discover()

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # cache so __getattr__ runs once per name
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = [
    "__version__",
    "align",
    "evaluate",
    "finetune",
    "JobManager",
    "lr_find",
    "pipeline",
    "pretrain",
]
