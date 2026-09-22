"""Canonical typed encoding, and the fingerprints built on it (ADR-006, ADR-011).

A fingerprint is an **identity**, not a reproduction recipe. It answers *have we
seen this before?* — and for that to work, two processes looking at the same
record must produce the same bytes, forever.

Three ordinary Python behaviours break that, which is why the encoding here is
deliberate rather than a `model_dump_json()`:

* **Mapping order.** Two equal dicts can differ in insertion order, so a naive
  dump may emit their keys in different orders.
* **``hash()`` is salted per process.** It cannot appear anywhere in a persisted
  fingerprint: the same record would hash differently after a restart.
* **Python equality is untyped.** ``True == 1`` and ``1 == 1.0``, so ``{"x": True}``
  compares equal to ``{"x": 1}`` while their JSON forms — ``{"x":true}`` and
  ``{"x":1}`` — describe different records and must fingerprint differently.

So values are encoded with their type distinguished, mappings are emitted in
sorted key order, and the digest is SHA-256 over UTF-8 bytes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from xaytune.core.errors import InvalidDomainValueError

__all__ = ["canonical_encode", "fingerprint"]

_PREFIX = "sha256:"


def fingerprint(value: Any) -> str:
    """Return a stable ``sha256:`` digest of *value*.

    Stable across processes, machines and interpreter versions, which
    ``hash()`` is not.
    """
    digest = hashlib.sha256(canonical_encode(value).encode("utf-8")).hexdigest()
    return f"{_PREFIX}{digest}"


def canonical_encode(value: Any) -> str:
    """Return the canonical typed encoding of *value*.

    Every value carries a type tag, so values that Python considers equal but
    that mean different things encode differently::

        canonical_encode(True)  != canonical_encode(1)
        canonical_encode(1)     != canonical_encode(1.0)

    Raises:
        InvalidDomainValueError: If *value* contains a set (no stable iteration
            order), a non-string mapping key, a non-finite float, or anything
            with no canonical representation.
    """
    return _encode(value)


def _encode(value: Any) -> str:
    # bool before int: bool IS an int in Python, and checking int first would
    # tag True as an integer and collapse the distinction above.
    if value is None:
        return "n"
    if isinstance(value, bool):
        return f"b:{'1' if value else '0'}"
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float):
        return f"f:{_encode_float(value)}"
    if isinstance(value, str):
        # Length-prefixed so that ["a", "bc"] and ["ab", "c"] cannot collide
        # once concatenated.
        return f"s:{len(value)}:{value}"

    if isinstance(value, BaseModel):
        # by_alias so a field renamed with a serialization alias fingerprints
        # under the name it persists as, not its Python spelling.
        #
        # Hashing a model directly makes the *current schema* define identity,
        # which is fine for a transient digest and wrong for a durable one:
        # adding a field with a default later changes the fingerprint of every
        # record already stored, though nobody changed anything. Durable
        # identities therefore hash an explicit versioned projection instead
        # (see candidate_identity_v1), and reach this branch only for values
        # nested inside one.
        return _encode(value.model_dump(mode="json", by_alias=True))

    if isinstance(value, Mapping):
        # Validate every key BEFORE sorting. A mixed mapping such as
        # {"a": 1, 2: "b"} raises Python's own comparison TypeError inside
        # sorted() otherwise, and the caller gets an error about int and str
        # not being orderable rather than the one that explains the rule.
        for key in value:
            if not isinstance(key, str):
                raise InvalidDomainValueError(
                    f"fingerprint mapping keys must be strings, got {type(key).__name__} ({key!r})"
                )
        items = [f"{_encode(key)}={_encode(value[key])}" for key in sorted(value)]
        return "m:{" + ",".join(items) + "}"

    if isinstance(value, (set, frozenset)):
        raise InvalidDomainValueError(
            "sets cannot be fingerprinted: they have no stable iteration order, "
            "so the same record would encode differently between processes. "
            "Use a sorted sequence."
        )

    if isinstance(value, (bytes, bytearray)):
        raise InvalidDomainValueError(
            "bytes cannot be fingerprinted; encode them as a string first"
        )

    if isinstance(value, Sequence):
        return "l:[" + ",".join(_encode(item) for item in value) + "]"

    raise InvalidDomainValueError(
        f"{type(value).__name__} cannot be fingerprinted: it has no canonical representation"
    )


def _encode_float(value: float) -> str:
    """Encode a float unambiguously.

    ``repr`` round-trips exactly in Python, and integral floats keep their
    ``.0`` so ``1.0`` never encodes as ``1`` -- which, combined with the ``f:``
    tag, is what keeps it distinct from the integer.

    Raises:
        InvalidDomainValueError: For NaN or infinity, which have no JSON form.
    """
    if value != value or value in (float("inf"), float("-inf")):
        raise InvalidDomainValueError(f"fingerprint values must be finite numbers, got {value!r}")
    return repr(value)
