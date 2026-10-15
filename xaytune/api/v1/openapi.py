"""The committed OpenAPI document: the contract between the API and its clients (ADR-023 §S5).

``python -m xaytune.api.v1.openapi`` rewrites ``openapi.json`` beside this
module from the application. A test fails when the two differ, so a change
to the wire is always a reviewed change to this file too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["DOCUMENT", "generate"]

DOCUMENT = Path(__file__).with_name("openapi.json")


def generate() -> dict[str, Any]:
    """The OpenAPI document the application serves."""
    from xaytune.api.app import application

    return application().openapi()


def render() -> str:
    return json.dumps(generate(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    DOCUMENT.write_text(render())
