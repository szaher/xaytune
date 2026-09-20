"""Deep immutability and a canonical value contract for domain records.

Pydantic's ``frozen=True`` blocks attribute assignment but not mutation of the
values behind those attributes. A ``dict`` field on a frozen model stays fully
mutable, and the model keeps a reference to whatever the caller passed in, so a
"frozen" scientific record can be changed after construction from either side::

    snapshot.payload["optimizer"]["lr"] = 7   # would succeed
    caller_dict["optimizer"]["lr"] = 7        # would also change the snapshot

That is an identity hole: a fingerprint computed at construction would no longer
describe the record's contents.

:class:`FrozenDict` closes it by freezing recursively **at construction**, so
every instance is deeply immutable by class invariant rather than by how it
happened to be built. Its backing store is a :class:`~types.MappingProxyType`,
so there is no mutable dictionary to reach through.

Freezing also enforces a canonical value contract. Only JSON-shaped data with
string keys is accepted:

.. code-block:: text

    null | bool | int | finite float | str
    Mapping[str, ...] -> FrozenDict
    Sequence[...]     -> tuple

Sets are rejected because they have no stable iteration order, which would make
serialization -- and therefore fingerprints -- vary between processes. NaN and
infinity are rejected because they have no JSON representation. Arbitrary
objects are rejected because they can neither round-trip nor be frozen.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler
from pydantic_core import core_schema
from typing_extensions import Self

from xaytune.core.errors import InvalidDomainValueError

__all__ = ["AggregateModel", "FrozenDict", "FrozenDomainModel", "deep_freeze", "thaw"]


class FrozenDict(Mapping[str, Any]):
    """An immutable, recursively frozen, string-keyed mapping.

    Construction freezes: every value is passed through :func:`deep_freeze`, so
    an instance can never contain a mutable container. That invariant is what
    lets :func:`deep_freeze` treat an existing ``FrozenDict`` as already safe.
    """

    __slots__ = ("_data",)

    _data: Mapping[str, Any]

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        frozen: dict[str, Any] = {}
        for key, value in (data or {}).items():
            if not isinstance(key, str):
                raise InvalidDomainValueError(
                    f"domain mapping keys must be strings, got {type(key).__name__} ({key!r})"
                )
            frozen[key] = deep_freeze(value)
        # MappingProxyType so there is no mutable dict to reach through.
        object.__setattr__(self, "_data", MappingProxyType(frozen))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({dict(self._data)!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenDict):
            return bool(dict(self._data) == dict(other._data))
        if isinstance(other, Mapping):
            return bool(dict(self._data) == dict(other))
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._data.items(), key=lambda kv: kv[0])))

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("FrozenDict is immutable")

    def __delitem__(self, key: str) -> None:
        raise TypeError("FrozenDict is immutable")

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("FrozenDict is immutable")

    def __delattr__(self, name: str) -> None:
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

    def __copy__(self) -> FrozenDict:
        # Immutable, so a copy can safely be the same object. This also avoids
        # deepcopy failing on the MappingProxyType backing store.
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        # MappingProxyType is not picklable; rebuild from a plain dict instead.
        return (FrozenDict, (dict(self._data),))

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


def deep_freeze(value: Any) -> Any:
    """Recursively freeze *value*, rejecting anything not canonically persistable.

    Raises:
        InvalidDomainValueError: If *value* contains a set, a non-string mapping
            key, NaN, infinity, or an object with no JSON representation.
    """
    # Safe by class invariant: FrozenDict freezes its contents on construction.
    if isinstance(value, FrozenDict):
        return value

    if value is None or isinstance(value, (bool, str, int)):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise InvalidDomainValueError(f"domain values must be finite numbers, got {value!r}")
        return value

    if isinstance(value, Mapping):
        return FrozenDict(value)

    if isinstance(value, (set, frozenset)):
        raise InvalidDomainValueError(
            "sets are not allowed in domain records: they have no stable "
            "iteration order, so serialization and fingerprints would vary "
            "between processes. Use a sorted sequence instead."
        )

    if isinstance(value, (bytes, bytearray)):
        raise InvalidDomainValueError(
            "bytes are not allowed in domain records; encode them as a string"
        )

    if isinstance(value, Sequence):
        return tuple(deep_freeze(item) for item in value)

    raise InvalidDomainValueError(
        f"{type(value).__name__} cannot be stored in a domain record: it has no "
        f"canonical JSON representation and cannot be frozen"
    )


def thaw(value: Any) -> Any:
    """Return a plain mutable copy, for callers that need to build on a record."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (tuple, list)):
        return [thaw(item) for item in value]
    return value


def _validate_frozen_dict(value: Any) -> FrozenDict:
    if isinstance(value, FrozenDict):
        return value
    if not isinstance(value, Mapping):
        raise InvalidDomainValueError(f"expected a mapping, got {type(value).__name__}")
    return FrozenDict(value)


class FrozenDomainModel(BaseModel):
    """Base for immutable domain records, with a validating ``model_copy``.

        Pydantic's ``model_copy(update=...)`` assigns the update values **without
        validating them**, which defeats every guarantee the field types provide::

            snapshot.model_copy(update={"payload": {"a": {"b": 1}}})
            # payload is now a plain dict again, and mutable

            experiment.model_copy(update={"active_node_ids": ["not-an-id"]})
            # no longer a typed id, and never validated

    That is not an obscure corner: ``with_status`` is built on it, and the docs
    recommend it for deriving one record from another. So this class re-validates
    instead of assigning, which makes the copy slower than Pydantic's and correct.

    ``extra="forbid"`` is deliberate too: silently dropping an unknown field would
    lose provenance rather than surface a schema mismatch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def _validated_copy(self, update: Mapping[str, Any]) -> Self:
        """Return a copy with *update* applied and re-validated.

        Internal: aggregates expose this only through their transition methods,
        which enforce the domain rules that validation alone cannot.
        """
        data = self.model_dump(mode="python", round_trip=True)
        data.update(update)
        return type(self).model_validate(data)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Return a copy, re-validating any updated fields.

        Raises:
            ValidationError: If an updated value is not valid for its field.
        """
        if not update:
            return super().model_copy(deep=deep)
        return self._validated_copy(update)


class AggregateModel(FrozenDomainModel):
    """Base for aggregates, where a validated copy is still not enough.

        Re-validating an update checks the *schema*. Aggregates also have domain
        rules that no field type can express::

            experiment.model_copy(update={"status": ExperimentStatus.SUCCEEDED})
            # schema-valid, and skips the state machine entirely

            attempt.model_copy(update={"status": RunAttemptStatus.SUCCEEDED})
            # succeeded, with started_at=None, ended_at=None, revision=0 --
            # a valid Pydantic object and an impossible domain object

            node.model_copy(update={"candidate_fingerprint": "sha256:other"})
            # rewrites scientific identity with no new node, event or lineage

    Rule 7 forbids direct status mutation, and an unrestricted ``model_copy`` is
    that mutation with extra steps. Updates are therefore refused here; aggregates
    change through their own transition methods, which validate the transition,
    bump the revision and stamp the timestamps together.

    A field with no transition method yet cannot be updated at all. That is
    deliberate: the next person needs an explicit, named operation rather than a
    generic escape hatch.
    """

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Return an unchanged copy.

        Raises:
            TypeError: If *update* is given. Use the aggregate's transition
                methods, which enforce the domain rules validation cannot.
        """
        if update:
            raise TypeError(
                f"{type(self).__name__} cannot be updated through model_copy: "
                f"an update would skip state-machine validation, revision and "
                f"timestamp semantics. Use a transition method such as "
                f"with_status() instead. Fields: {sorted(update)}"
            )
        return super().model_copy(deep=deep)
