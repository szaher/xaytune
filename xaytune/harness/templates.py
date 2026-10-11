"""Rendering a harness's task template, in the syntaxes the reference path supports.

``literal`` is the text as written. ``python-format`` is ``str.format`` syntax
restricted to plain ``{name}`` fields: no attribute or index access, no format
spec, no conversion. Those reach into objects or change a value's rendering,
which the declared variable list cannot describe. ``{{`` and ``}}`` are the
literal braces they always are.

Rendering is by substitution, never by calling ``str.format``. A field is
replaced only by the string the task supplies for it.
"""

from __future__ import annotations

import re
import string
from collections.abc import Mapping

__all__ = ["RENDERABLE_SYNTAXES", "render_template", "template_refusals"]

RENDERABLE_SYNTAXES: tuple[str, ...] = ("literal", "python-format")
"""``jinja2`` is not here: rendering it safely needs a sandboxed environment
this path does not have, so it is refused rather than rendered unsafely."""

_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def template_refusals(syntax: str, text: str, variables: tuple[str, ...]) -> tuple[str, ...]:
    """Every reason *text* cannot be rendered under *syntax* with exactly *variables*."""
    if syntax not in RENDERABLE_SYNTAXES:
        return (
            f"template syntax {syntax!r} is not supported; supported: "
            f"{', '.join(RENDERABLE_SYNTAXES)}",
        )
    if syntax == "literal":
        return ()
    try:
        parsed = list(string.Formatter().parse(text))
    except ValueError:
        return ("the python-format template is malformed (an unmatched brace)",)
    reasons: list[str] = []
    fields: set[str] = set()
    for _, name, spec, conversion in parsed:
        if name is None:
            continue
        if not _FIELD.match(name):
            reasons.append(
                "python-format fields must be plain names: no positional, attribute or index fields"
            )
        elif spec or conversion:
            reasons.append("python-format fields take no format spec or conversion")
        else:
            fields.add(name)
    if not reasons and tuple(sorted(fields)) != variables:
        reasons.append("the template's fields are not exactly its declared variables")
    return tuple(dict.fromkeys(reasons))


def render_template(syntax: str, text: str, values: Mapping[str, str]) -> str:
    """Render *text* with *values*. The template must have passed :func:`template_refusals`."""
    if syntax == "literal":
        return text
    if syntax != "python-format":
        raise ValueError(f"template syntax {syntax!r} cannot be rendered")
    parts: list[str] = []
    for literal, name, _, _ in string.Formatter().parse(text):
        parts.append(literal)
        if name is not None:
            parts.append(values[name])
    return "".join(parts)
