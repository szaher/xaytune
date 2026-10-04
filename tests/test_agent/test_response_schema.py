"""The response-schema subset: refused when unsupported, checked exactly and typed."""

from __future__ import annotations

from typing import Any

import pytest

from xaytune.agent.schema import (
    UnsupportedResponseSchemaError,
    check_response_schema,
    schema_violations,
)
from xaytune.core.immutable import FrozenDict


def obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


@pytest.mark.parametrize(
    ("schema", "reason"),
    [
        ({"type": "array", "items": {"type": "string"}}, "the root must be"),
        (obj({"a": {"type": "string", "pattern": "^x"}}), "'pattern' is not supported"),
        (obj({"a": {"type": "string", "format": "uri"}}), "'format' is not supported"),
        (obj({"a": {"$ref": "#/$defs/x"}}), "every schema names a type"),
        (obj({"a": {"type": ["string", "null"]}}), "is not one of"),
        (obj({"a": {"type": "string", "minimum": 1}}), "'minimum' is not supported"),
        ({**obj({}), "additionalProperties": True}, "objects are closed"),
        ({**obj({}), "additionalProperties": {"type": "string"}}, "objects are closed"),
        (obj({}, ["missing"]), "'missing' is not a declared property"),
        (obj({"a": {"type": "string"}}, ["a", "a"]), "more than once"),
        (obj({"a": {"type": "array"}}), "declares its items"),
        (obj({"a": {"type": "array", "items": {}, "minItems": -1}}), "every schema names"),
        (obj({"a": {"type": "string", "minLength": 3, "maxLength": 2}}), "exceeds"),
        (obj({"a": {"type": "string", "maxLength": True}}), "non-negative integer"),
        (obj({"a": {"type": "integer", "maximum": "9"}}), "must be a number"),
        (obj({"a": {"type": "string", "enum": []}}), "non-empty list"),
        (obj({"a": {"type": "string", "enum": ["x", 1]}}), "literal 1 does not satisfy"),
        (obj({"a": {"type": "integer", "const": True}}), "literal True does not satisfy"),
        (obj({"a": {"type": "string", "anyOf": [{"type": "null"}]}}), "not both"),
        (obj({"a": {"anyOf": []}}), "non-empty list"),
        (obj({"a": "string"}), "a schema must be an object"),
        ({**obj({}), "title": 123}, "$.title: must be a string"),
        ({**obj({}), "description": {"not": "a string"}}, "$.description: must be a string"),
        (obj({"choice": {"type": "string", "description": 42}}), "$.choice.description: must be"),
        (obj({"a": {"anyOf": [{"type": "null", "title": None}]}}), "$.a.anyOf[0].title: must be"),
    ],
)
def test_a_schema_outside_the_subset_is_refused(schema: dict[str, Any], reason: str) -> None:
    with pytest.raises(UnsupportedResponseSchemaError) as caught:
        check_response_schema(FrozenDict(schema))
    assert any(reason in line for line in caught.value.reasons), caught.value.reasons


def test_every_unsupported_part_is_named() -> None:
    schema = obj({"a": {"type": "string", "pattern": "x"}, "b": {"type": "number", "format": "f"}})
    with pytest.raises(UnsupportedResponseSchemaError) as caught:
        check_response_schema(schema)
    assert caught.value.reasons == (
        "$.a: keyword 'pattern' is not supported here",
        "$.b: keyword 'format' is not supported here",
    )


RICH = obj(
    {
        "kind": {"type": "string", "const": "proposal", "title": "Kind", "description": "d"},
        "rank": {"type": "integer", "minimum": 1, "maximum": 64},
        "scale": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
        "note": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        "flag": {"type": "boolean"},
        "nested": obj({"x": {"type": "integer", "enum": [1, 2]}}, ["x"]),
    },
    ["kind", "rank"],
)


def test_the_supported_subset_is_accepted() -> None:
    check_response_schema(FrozenDict(RICH))


def test_a_conforming_value_has_no_violations() -> None:
    value = {
        "kind": "proposal",
        "rank": 16,
        "scale": 0.5,
        "tags": ("a",),
        "note": None,
        "flag": False,
        "nested": {"x": 2},
    }
    assert schema_violations(FrozenDict(RICH), FrozenDict(value)) == ()


@pytest.mark.parametrize(
    ("value", "violation"),
    [
        ({"kind": "other", "rank": 1}, "$.kind: must be 'proposal'"),
        ({"kind": "proposal", "rank": 0}, "$.rank: must be >= 1"),
        ({"kind": "proposal", "rank": 65}, "$.rank: must be <= 64"),
        ({"kind": "proposal", "rank": True}, "$.rank: expected integer, got boolean"),
        ({"kind": "proposal", "rank": 1.0}, "$.rank: expected integer, got number"),
        ({"kind": "proposal", "rank": 1, "scale": 0}, "$.scale: must be > 0"),
        ({"kind": "proposal", "rank": 1, "scale": 1}, "$.scale: must be < 1"),
        ({"kind": "proposal", "rank": 1, "tags": []}, "$.tags: at least 1 items, got 0"),
        ({"kind": "proposal", "rank": 1, "tags": ["a", "b", "c"]}, "at most 2 items"),
        ({"kind": "proposal", "rank": 1, "tags": ["a", 2]}, "$.tags[1]: expected string"),
        ({"kind": "proposal", "rank": 1, "tags": "a"}, "$.tags: expected array, got string"),
        ({"kind": "proposal", "rank": 1, "note": ""}, "$.note: at least 1 characters, got 0"),
        ({"kind": "proposal", "rank": 1, "flag": 0}, "$.flag: expected boolean, got integer"),
        ({"kind": "proposal", "rank": 1, "nested": {"x": True}}, "$.nested.x: expected integer"),
        ({"kind": "proposal", "rank": 1, "nested": {"x": 3}}, "$.nested.x: must be one of"),
        ({"kind": "proposal", "rank": 1, "nested": {}}, "$.nested: missing required 'x'"),
        ({"kind": "proposal", "rank": 1, "nested": {"x": 1, "y": 1}}, "$.nested: unexpected 'y'"),
        ({"rank": 1}, "$: missing required 'kind'"),
    ],
)
def test_each_violation_is_found_and_located(value: dict[str, Any], violation: str) -> None:
    found = schema_violations(FrozenDict(RICH), FrozenDict(value))
    assert any(violation in line for line in found), found


def test_enum_and_const_compare_by_type_not_python_equality() -> None:
    """``True == 1`` in Python; an answer of ``true`` is not the integer 1."""
    either = {"anyOf": [{"type": "integer"}, {"type": "boolean"}], "enum": [1]}
    schema = FrozenDict(obj({"a": either}))
    assert schema_violations(schema, FrozenDict({"a": 1})) == ()
    assert schema_violations(schema, FrozenDict({"a": True})) == ("$.a: must be one of [1]",)


UNION = obj(
    {
        "pick": {
            "anyOf": [
                {"type": "null"},
                obj({"kind": {"type": "string", "const": "a"}, "n": {"type": "integer"}}, ["kind"]),
                obj({"kind": {"type": "string", "const": "b"}, "s": {"type": "string"}}, ["kind"]),
            ]
        }
    },
    ["pick"],
)


@pytest.mark.parametrize(
    ("pick", "violations"),
    [
        # the const discriminator names one alternative: its own violations are reported
        ({"kind": "a", "n": "x"}, ("$.pick.n: expected integer, got string",)),
        (
            {"kind": "b", "s": 1, "n": 2},
            ("$.pick: unexpected 'n'", "$.pick.s: expected string, got integer"),
        ),
        # nothing to go on, or several candidates: the value matches none
        ({"kind": "c"}, ("$.pick: matches none of the 3 alternatives",)),
        ("text", ("$.pick: matches none of the 3 alternatives",)),
    ],
)
def test_a_failed_alternative_is_explained_when_the_value_says_which_it_meant(
    pick: Any, violations: tuple[str, ...]
) -> None:
    assert schema_violations(FrozenDict(UNION), FrozenDict({"pick": pick})) == violations


def test_a_value_meant_as_null_or_a_typed_scalar_is_explained_too() -> None:
    schema = FrozenDict(
        obj({"n": {"anyOf": [{"type": "null"}, {"type": "integer", "minimum": 1}]}})
    )
    assert schema_violations(schema, FrozenDict({"n": 0})) == ("$.n: must be >= 1",)
