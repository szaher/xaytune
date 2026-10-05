"""The response schemas an agent model may be asked for, and how answers are checked.

A response schema is xaytune's closed response-schema subset, written in JSON
Schema vocabulary: not a standards-complete JSON Schema validator, and not a
drop-in for one. Every keyword in the subset is fully implemented, annotations
(``title``, ``description``) included -- they must be strings. A keyword
outside it is refused when the request is built, never ignored when the
answer is checked: an ignored ``pattern`` or ``format`` would let through an
answer the caller believed was constrained, and that is the failure
structured output exists to prevent.

```text
every node     type | anyOf, enum, const, title, description
object         properties, required, additionalProperties (absent or false)
array          items (required), minItems, maxItems
string         minLength, maxLength
integer/number minimum, maximum, exclusiveMinimum, exclusiveMaximum
boolean, null  --
```

The root is an object. Objects are **closed**: a key the schema does not name
is an error, as if ``additionalProperties: false`` were always written --
``true`` or a schema there is refused. ``type`` names one type; a nullable
value is ``anyOf`` with ``{"type": "null"}``. There are no ``$ref``, no
``pattern`` (Python and ECMA regular expressions differ), no ``format``.

Checking is typed, as fingerprints are (:mod:`xaytune.core.fingerprint`), and
stricter than JSON Schema's numeric equivalence: ``true`` is not an integer,
``1.0`` is a number but not an integer, and ``enum``/``const`` compare by
canonical encoding, so ``1`` does not match ``true``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from xaytune.core.fingerprint import canonical_encode

__all__ = [
    "UnsupportedResponseSchemaError",
    "check_response_schema",
    "schema_violations",
]

_TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})
_EVERY_NODE = frozenset({"type", "anyOf", "enum", "const", "title", "description"})
_BOUNDS = frozenset({"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"})
_BY_TYPE: Mapping[str, frozenset[str]] = {
    "object": frozenset({"properties", "required", "additionalProperties"}),
    "array": frozenset({"items", "minItems", "maxItems"}),
    "string": frozenset({"minLength", "maxLength"}),
    "integer": _BOUNDS,
    "number": _BOUNDS,
    "boolean": frozenset(),
    "null": frozenset(),
}


class UnsupportedResponseSchemaError(ValueError):
    """A response schema uses something outside the supported subset. Carries every reason."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("unsupported response schema: " + "; ".join(reasons))


def check_response_schema(schema: Mapping[str, Any]) -> None:
    """Refuse *schema* unless every part of it is in the supported subset.

    Raises:
        UnsupportedResponseSchemaError: Naming each offending keyword and where.
    """
    reasons: list[str] = []
    if schema.get("type") != "object":
        reasons.append('$: the root must be {"type": "object"}')
    _check_node(schema, "$", reasons)
    if reasons:
        raise UnsupportedResponseSchemaError(tuple(reasons))


def schema_violations(schema: Mapping[str, Any], value: Any) -> tuple[str, ...]:
    """Every way *value* fails *schema*, by path; ``()`` when it conforms.

    *schema* must already have passed :func:`check_response_schema`.
    """
    violations: list[str] = []
    _check_value(schema, value, "$", violations)
    return tuple(violations)


def _check_node(node: Any, path: str, reasons: list[str]) -> None:
    local: list[str] = []
    _check_shape(node, path, local)
    if not local:
        # Only a well-formed node can say whether its literals satisfy it.
        _check_literals(node, path, local)
    reasons.extend(local)


def _check_shape(node: Any, path: str, reasons: list[str]) -> None:
    if not isinstance(node, Mapping):
        reasons.append(f"{path}: a schema must be an object, got {type(node).__name__}")
        return
    for annotation in ("title", "description"):
        if annotation in node and not isinstance(node[annotation], str):
            reasons.append(f"{path}.{annotation}: must be a string")
    kind = node.get("type")
    if "anyOf" in node:
        if kind is not None:
            reasons.append(f"{path}: use either type or anyOf, not both")
        alternatives = node["anyOf"]
        if not _is_list(alternatives) or not alternatives:
            reasons.append(f"{path}.anyOf: must be a non-empty list of schemas")
        else:
            for index, alternative in enumerate(alternatives):
                _check_node(alternative, f"{path}.anyOf[{index}]", reasons)
        allowed = _EVERY_NODE
    elif kind is None:
        reasons.append(f"{path}: every schema names a type (or anyOf)")
        return
    elif not isinstance(kind, str) or kind not in _TYPES:
        reasons.append(f"{path}.type: {kind!r} is not one of {sorted(_TYPES)}")
        return
    else:
        allowed = _EVERY_NODE | _BY_TYPE[kind]
    for keyword in sorted(set(node) - allowed):
        reasons.append(f"{path}: keyword {keyword!r} is not supported here")
    if kind == "object":
        _check_object(node, path, reasons)
    elif kind == "array":
        _check_array(node, path, reasons)
    elif kind == "string":
        _check_sizes(node, path, ("minLength", "maxLength"), reasons)
    elif kind in ("integer", "number"):
        for bound in sorted(_BOUNDS & set(node)):
            if not _is_number(node[bound]):
                reasons.append(f"{path}.{bound}: must be a number")


def _check_object(node: Mapping[str, Any], path: str, reasons: list[str]) -> None:
    properties = node.get("properties", {})
    if not isinstance(properties, Mapping):
        reasons.append(f"{path}.properties: must be an object")
        properties = {}
    for name, child in properties.items():
        _check_node(child, f"{path}.{name}", reasons)
    required = node.get("required", ())
    if not _is_list(required) or not all(isinstance(name, str) for name in required):
        reasons.append(f"{path}.required: must be a list of property names")
    else:
        if len(set(required)) != len(required):
            reasons.append(f"{path}.required: names a property more than once")
        for name in required:
            if name not in properties:
                reasons.append(f"{path}.required: {name!r} is not a declared property")
    if node.get("additionalProperties", False) is not False:
        reasons.append(f"{path}.additionalProperties: objects are closed; only false is allowed")


def _check_array(node: Mapping[str, Any], path: str, reasons: list[str]) -> None:
    if "items" not in node:
        reasons.append(f"{path}: an array declares its items")
    else:
        _check_node(node["items"], f"{path}[]", reasons)
    _check_sizes(node, path, ("minItems", "maxItems"), reasons)


def _check_sizes(
    node: Mapping[str, Any], path: str, names: tuple[str, str], reasons: list[str]
) -> None:
    for name in names:
        if name in node and not _is_size(node[name]):
            reasons.append(f"{path}.{name}: must be a non-negative integer")
    low: Any = node.get(names[0])
    high: Any = node.get(names[1])
    if _is_size(low) and _is_size(high) and low > high:
        reasons.append(f"{path}: {names[0]} exceeds {names[1]}")


def _check_literals(node: Mapping[str, Any], path: str, reasons: list[str]) -> None:
    """``enum`` and ``const`` must be satisfiable by the node they sit on."""
    if "enum" in node and (not _is_list(node["enum"]) or not node["enum"]):
        reasons.append(f"{path}.enum: must be a non-empty list")
        return
    shape = {key: value for key, value in node.items() if key not in ("enum", "const")}
    literals = [*node.get("enum", ()), *([node["const"]] if "const" in node else [])]
    for literal in literals:
        if schema_violations(shape, literal):
            reasons.append(f"{path}: literal {literal!r} does not satisfy the schema it sits on")


def _check_value(schema: Mapping[str, Any], value: Any, path: str, out: list[str]) -> None:
    if "anyOf" in schema:
        if not any(not schema_violations(option, value) for option in schema["anyOf"]):
            _explain_alternatives(schema["anyOf"], value, path, out)
            return
    elif not _has_type(schema["type"], value):
        out.append(f"{path}: expected {schema['type']}, got {_json_type(value)}")
        return
    if "const" in schema and canonical_encode(value) != canonical_encode(schema["const"]):
        out.append(f"{path}: must be {schema['const']!r}")
    if "enum" in schema and canonical_encode(value) not in {
        canonical_encode(option) for option in schema["enum"]
    }:
        out.append(f"{path}: must be one of {list(schema['enum'])!r}")
    kind = schema.get("type")
    if kind == "object":
        properties = schema.get("properties", {})
        for name in schema.get("required", ()):
            if name not in value:
                out.append(f"{path}: missing required {name!r}")
        for name in sorted(value):
            if name not in properties:
                out.append(f"{path}: unexpected {name!r}")
            else:
                _check_value(properties[name], value[name], f"{path}.{name}", out)
    elif kind == "array":
        _check_size(len(value), schema, ("minItems", "maxItems"), "items", path, out)
        for index, item in enumerate(value):
            _check_value(schema["items"], item, f"{path}[{index}]", out)
    elif kind == "string":
        _check_size(len(value), schema, ("minLength", "maxLength"), "characters", path, out)
    elif kind in ("integer", "number"):
        _check_bounds(value, schema, path, out)


def _explain_alternatives(
    alternatives: Sequence[Mapping[str, Any]], value: Any, path: str, out: list[str]
) -> None:
    """Why *value* matches no alternative: the one it was evidently meant as, if there is one.

    An alternative is *meant* when the value has its type and, for an object,
    agrees with every ``const`` property it carries -- a discriminator such as
    an action's ``type``. With exactly one such alternative, its own
    violations are the useful answer; otherwise the value matches none.
    """
    meant = [option for option in alternatives if _could_be(option, value)]
    if len(meant) == 1:
        _check_value(meant[0], value, path, out)
    else:
        out.append(f"{path}: matches none of the {len(alternatives)} alternatives")


def _could_be(option: Mapping[str, Any], value: Any) -> bool:
    if "anyOf" in option:
        return any(_could_be(inner, value) for inner in option["anyOf"])
    if not _has_type(option["type"], value):
        return False
    if option["type"] != "object":
        return True
    return all(
        name in value and canonical_encode(value[name]) == canonical_encode(child["const"])
        for name, child in option.get("properties", {}).items()
        if "const" in child
    )


def _check_size(
    size: int,
    schema: Mapping[str, Any],
    names: tuple[str, str],
    unit: str,
    path: str,
    out: list[str],
) -> None:
    low, high = (schema.get(name) for name in names)
    if low is not None and size < low:
        out.append(f"{path}: at least {low} {unit}, got {size}")
    if high is not None and size > high:
        out.append(f"{path}: at most {high} {unit}, got {size}")


def _check_bounds(value: float, schema: Mapping[str, Any], path: str, out: list[str]) -> None:
    if "minimum" in schema and value < schema["minimum"]:
        out.append(f"{path}: must be >= {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        out.append(f"{path}: must be <= {schema['maximum']}")
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        out.append(f"{path}: must be > {schema['exclusiveMinimum']}")
    if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
        out.append(f"{path}: must be < {schema['exclusiveMaximum']}")


def _has_type(kind: str, value: Any) -> bool:
    if kind == "object":
        return isinstance(value, Mapping)
    if kind == "array":
        return _is_list(value)
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return _is_number(value)
    if kind == "boolean":
        return isinstance(value, bool)
    return value is None


def _json_type(value: Any) -> str:
    for kind in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _has_type(kind, value):
            return kind
    return type(value).__name__


def _is_list(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_size(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
