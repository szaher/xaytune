from __future__ import annotations

import logging
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUPS = {
    "xaytune.recipes": lambda: _get_registry("xaytune.recipes", "recipe"),
    "xaytune.models": lambda: _get_registry("xaytune.models", "model"),
    "xaytune.formats": lambda: _get_registry("xaytune.formats", "format"),
    "xaytune.metrics": lambda: _get_registry("xaytune.metrics", "metric"),
}

_discovered = False


def _get_registry(group: str, name: str):
    if name == "recipe":
        from xaytune.recipes import recipe_registry

        return recipe_registry
    if name == "model":
        from xaytune.models.registry import model_registry

        return model_registry
    if name == "format":
        from xaytune.data.registry import format_registry

        return format_registry
    if name == "metric":
        from xaytune.eval.metrics import metric_registry

        return metric_registry
    return None


def discover_plugins() -> None:
    """Scan installed packages for xaytune entry points and register them.

    Looks for entry points in four groups: ``xaytune.recipes``,
    ``xaytune.models``, ``xaytune.formats``, and ``xaytune.metrics``.
    Each discovered entry point is loaded and registered with the
    appropriate registry.  Safe to call multiple times.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True

    for group, get_registry in _ENTRY_POINT_GROUPS.items():
        eps = entry_points(group=group)
        if not eps:
            continue

        registry = get_registry()
        if registry is None:
            continue

        for ep in eps:
            try:
                obj = ep.load()
                if not registry.has(ep.name):
                    registry.register(ep.name)(obj)
            except Exception:
                logger.warning("Failed to load plugin %s from %s", ep.name, group)
