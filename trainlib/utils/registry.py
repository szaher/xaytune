from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    """A generic registry that maps string names to objects (functions, classes, etc.)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Any] = {}

    def register(self, key: str, override: bool = False) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if key in self._items and not override:
                raise ValueError(
                    f"'{key}' is already registered in {self.name}. "
                    f"Use override=True to replace it."
                )
            self._items[key] = obj
            return obj

        return decorator

    def get(self, key: str) -> Any:
        if key not in self._items:
            raise KeyError(
                f"'{key}' not found in {self.name} registry. "
                f"Available: {', '.join(self._items) or '(none)'}"
            )
        return self._items[key]

    def has(self, key: str) -> bool:
        return key in self._items

    def list(self) -> list[str]:
        return sorted(self._items.keys())
