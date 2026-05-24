from __future__ import annotations

import logging
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUPS = {
    "trainlib.recipes": lambda: _get_registry("trainlib.recipes", "recipe"),
    "trainlib.models": lambda: _get_registry("trainlib.models", "model"),
    "trainlib.formats": lambda: _get_registry("trainlib.formats", "format"),
    "trainlib.metrics": lambda: _get_registry("trainlib.metrics", "metric"),
}

_discovered = False


def _get_registry(group: str, name: str):
    if name == "recipe":
        from trainlib.recipes import recipe_registry
        return recipe_registry
    if name == "model":
        from trainlib.models.registry import model_registry
        return model_registry
    if name == "format":
        from trainlib.data.registry import format_registry
        return format_registry
    if name == "metric":
        from trainlib.eval.metrics import metric_registry
        return metric_registry
    return None


def discover_plugins() -> None:
    """Scan installed packages for trainlib entry points and register them.

    Looks for entry points in four groups: ``trainlib.recipes``,
    ``trainlib.models``, ``trainlib.formats``, and ``trainlib.metrics``.
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
