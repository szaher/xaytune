"""Deep immutability for domain records.

Pydantic's ``frozen=True`` blocks attribute assignment but not mutation of the
values behind those attributes. A ``dict`` field on a frozen model stays fully
mutable, and the model keeps a reference to whatever the caller passed in, so a
"frozen" scientific record can be changed after construction from either side::

    snapshot.payload["optimizer"]["lr"] = 7   # succeeds
    caller_dict["optimizer"]["lr"] = 7        # also changes the snapshot

That is an identity hole: a fingerprint computed at construction would no longer
describe the record's contents. The helpers here close it by recursively
converting mappings to :class:`FrozenDict` and sequences to tuples at validation
time, which also severs the caller's reference.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Annotated, Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

__all__ = ["FrozenDict", "FrozenJson", "deep_freeze", "thaw"]


class FrozenDict(Mapping[str, Any]):
    """An immutable mapping that rejects every mutating operation."""

    __slots__ = ("_data",)

    _data: dict[str, Any]

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        object.__setattr__(self, "_data", dict(data or {}))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenDict):
            return bool(self._data == other._data)
        if isinstance(other, Mapping):
            return bool(self._data == dict(other))
        return NotImplemented

    def __hash__(self) -> int:
        # Values may themselves be unhashable containers; frozen ones are not.
        return hash(tuple(sorted((k, _hashable(v)) for k, v in self._data.items())))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            _validate_frozen_dict,
            serialization=core_schema.plain_serializer_function_ser_schema(
                thaw, when_used="always"
            ),
        )

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("FrozenDict is immutable")

    def __delitem__(self, key: str) -> None:
        raise TypeError("FrozenDict is immutable")

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("FrozenDict is immutable")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrozenDict is immutable")

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("FrozenDict is immutable")

    def popitem(self) -> Any:
        raise TypeError("FrozenDict is immutable")

    def clear(self) -> None:
        raise TypeError("FrozenDict is immutable")

    def setdefault(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("FrozenDict is immutable")


def _hashable(value: Any) -> Any:
    if isinstance(value, FrozenDict):
        return hash(value)
    if isinstance(value, tuple):
        return tuple(_hashable(v) for v in value)
    return value


def deep_freeze(value: Any) -> Any:
    """Recursively convert mappings to :class:`FrozenDict` and sequences to tuples.

    Strings and bytes are sequences but are already immutable, so they pass
    through. The conversion copies, which is what severs any reference the
    caller retained.
    """
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict({k: deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (set, frozenset)):
        return frozenset(deep_freeze(v) for v in value)
    if isinstance(value, Sequence):
        return tuple(deep_freeze(v) for v in value)
    return value


def thaw(value: Any) -> Any:
    """Return a plain mutable copy, for callers that need to build on a record."""
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (tuple, list, set, frozenset)):
        return [thaw(v) for v in value]
    return value


FrozenJson = Annotated[
    Any,
    core_schema.no_info_after_validator_function(deep_freeze, core_schema.any_schema()),
]
"""An arbitrary JSON-shaped value, recursively frozen at validation time."""


def _validate_frozen_dict(value: Any) -> FrozenDict:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping, got {type(value).__name__}")
    frozen = deep_freeze(value)
    assert isinstance(frozen, FrozenDict)
    return frozen
