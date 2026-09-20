"""Typed, sortable identifiers for control-plane aggregates.

Every aggregate gets its own ``str`` subclass carrying a stable prefix, so that
passing a :class:`RunId` where an :class:`ExperimentId` is expected fails both
type checking and runtime validation.

The body after the prefix is ULID-shaped: 10 Crockford base32 characters of
millisecond timestamp followed by 16 of randomness. Lexicographic order is
therefore creation order, which lets event streams and listings sort on the id
alone. Generation is monotonic within a process, so two ids minted in the same
millisecond still sort in the order they were created.

IDs are stable and never recycled.
"""

from __future__ import annotations

import secrets
import threading
import time
from typing import Any, ClassVar, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from xaytune.core.errors import InvalidIdError

__all__ = [
    "ActionId",
    "ArtifactId",
    "CheckpointId",
    "DecisionId",
    "EvaluationId",
    "EventId",
    "ExperimentId",
    "ExperimentNodeId",
    "IncidentId",
    "RunAttemptId",
    "RunId",
    "TypedId",
]

# Crockford base32: no I, L, O or U, so ids survive being read aloud or typed.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ALPHABET_SET = frozenset(_ALPHABET)

_TIME_CHARS = 10
_RANDOM_CHARS = 16
_BODY_LEN = _TIME_CHARS + _RANDOM_CHARS

# One bit of headroom below the full width, so the monotonic counter below has
# room to increment without wrapping inside a millisecond.
_RANDOM_BITS = _RANDOM_CHARS * 5
_RANDOM_MAX = (1 << _RANDOM_BITS) - 1
_RANDOM_SEED_BITS = _RANDOM_BITS - 1

_lock = threading.Lock()
_last_ms = -1
_last_random = 0


def _encode(value: int, length: int) -> str:
    chars = [""] * length
    for i in range(length - 1, -1, -1):
        chars[i] = _ALPHABET[value & 0x1F]
        value >>= 5
    return "".join(chars)


def _next_body() -> str:
    """Return the next monotonically increasing id body."""
    global _last_ms, _last_random

    with _lock:
        now = max(int(time.time() * 1000), _last_ms)
        if now == _last_ms:
            _last_random += 1
            if _last_random > _RANDOM_MAX:
                # Exhausting 2**79 ids inside one millisecond is not reachable
                # in practice; borrow from the next millisecond rather than
                # emit a non-monotonic id.
                _last_ms = now = now + 1
                _last_random = secrets.randbits(_RANDOM_SEED_BITS)
        else:
            _last_ms = now
            _last_random = secrets.randbits(_RANDOM_SEED_BITS)
        return _encode(_last_ms, _TIME_CHARS) + _encode(_last_random, _RANDOM_CHARS)


_T = TypeVar("_T", bound="TypedId")


class TypedId(str):
    """Base class for prefixed, sortable identifiers.

    Subclasses set :attr:`prefix`. Instances are ``str``, so they serialize to
    plain JSON strings and can be used directly as dict keys or SQL parameters.
    """

    __slots__ = ()

    prefix: ClassVar[str] = ""

    @classmethod
    def generate(cls: type[_T]) -> _T:
        """Mint a new identifier."""
        return cls(f"{cls.prefix}{_next_body()}")

    @classmethod
    def validate(cls: type[_T], value: object) -> _T:
        """Coerce *value* to this id type, rejecting anything malformed.

        Raises:
            InvalidIdError: If *value* is not a string, lacks this type's
                prefix, or has a malformed body.
        """
        if not isinstance(value, str):
            raise InvalidIdError(f"{cls.__name__} must be a string, got {type(value).__name__}")
        if not value.startswith(cls.prefix):
            raise InvalidIdError(f"{cls.__name__} must start with {cls.prefix!r}, got {value!r}")
        body = value[len(cls.prefix) :]
        if len(body) != _BODY_LEN:
            raise InvalidIdError(
                f"{cls.__name__} body must be {_BODY_LEN} characters, got {len(body)} in {value!r}"
            )
        if not _ALPHABET_SET.issuperset(body):
            raise InvalidIdError(f"{cls.__name__} body is not Crockford base32: {value!r}")
        return cls(value)

    @property
    def created_at_ms(self) -> int:
        """Milliseconds since the Unix epoch encoded in this id."""
        body = self[len(self.prefix) :]
        value = 0
        for char in body[:_TIME_CHARS]:
            value = (value << 5) | _ALPHABET.index(char)
        return value

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )


class ExperimentId(TypedId):
    """Identifies an :class:`~xaytune.core.domain.Experiment`."""

    __slots__ = ()
    prefix = "exp_"


class ExperimentNodeId(TypedId):
    """Identifies an :class:`~xaytune.core.domain.ExperimentNode`."""

    __slots__ = ()
    prefix = "node_"


class RunId(TypedId):
    """Identifies a :class:`~xaytune.core.domain.Run`."""

    __slots__ = ()
    prefix = "run_"


class RunAttemptId(TypedId):
    """Identifies a :class:`~xaytune.core.domain.RunAttempt`."""

    __slots__ = ()
    prefix = "attempt_"


class ActionId(TypedId):
    """Identifies a typed action."""

    __slots__ = ()
    prefix = "act_"


class IncidentId(TypedId):
    """Identifies an incident."""

    __slots__ = ()
    prefix = "inc_"


class EvaluationId(TypedId):
    """Identifies an evaluation result."""

    __slots__ = ()
    prefix = "eval_"


class ArtifactId(TypedId):
    """Identifies a produced artifact."""

    __slots__ = ()
    prefix = "artifact_"


class CheckpointId(TypedId):
    """Identifies a checkpoint."""

    __slots__ = ()
    prefix = "ckpt_"


class DecisionId(TypedId):
    """Identifies a decision."""

    __slots__ = ()
    prefix = "decision_"


class EventId(TypedId):
    """Identifies a durable event."""

    __slots__ = ()
    prefix = "event_"
