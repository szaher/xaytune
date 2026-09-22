"""Decoding stored aggregate payloads.

One place, so a payload this code cannot read reports the same thing however
it was reached. Before this existed, ``get_node()`` raised
:class:`IncompatiblePayloadError` while ``nodes_for_experiment()`` and every
graph traversal raised a Pydantic ``ValidationError`` about missing fields --
the same database, the same cause, three different stories.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from xaytune.storage.errors import IncompatiblePayloadError

__all__ = ["decode_node_payload", "decode_payload"]

ModelT = TypeVar("ModelT", bound=BaseModel)


def decode_payload(
    payload_json: str, model: type[ModelT], aggregate_id: str | None = None
) -> ModelT:
    """Decode a stored payload, naming a format change when that is the cause.

    Raises:
        IncompatiblePayloadError: If the payload has a pre-PR-007 node body.
        ValidationError: For any other invalid payload -- a genuine schema
            error still surfaces as one rather than being reported as a
            version problem.
    """
    try:
        return model.model_validate_json(payload_json)
    except ValidationError as error:
        if _looks_pre_candidate(payload_json):
            raise IncompatiblePayloadError(
                model.__name__, aggregate_id or "?", "pre-CandidateSpec node body"
            ) from error
        raise


def decode_node_payload(
    payload_json: str, model: type[ModelT], aggregate_id: str | None = None
) -> ModelT:
    """Decode an :class:`ExperimentNode` payload."""
    return decode_payload(payload_json, model, aggregate_id)


def _looks_pre_candidate(payload: str) -> bool:
    """Whether this payload has the band B node shape."""
    try:
        body = json.loads(payload)
    except json.JSONDecodeError:
        return False
    return isinstance(body, dict) and "training_spec" in body and "candidate" not in body
