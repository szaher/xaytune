"""Decoding stored aggregate payloads.

One place, so a payload this code cannot read reports the same thing however
it was reached. Before this existed, ``get_node()`` raised
:class:`IncompatiblePayloadError` while ``nodes_for_experiment()`` and every
graph traversal raised a Pydantic ``ValidationError`` about missing fields --
the same database, the same cause, three different stories.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError

from xaytune.storage.errors import IncompatiblePayloadError, NodeRecordMismatchError

if TYPE_CHECKING:
    from xaytune.core.domain.experiment import ExperimentNode

__all__ = ["NODE_COLUMNS", "decode_node_row", "decode_payload"]

NODE_COLUMNS = "payload_json, candidate_kind, candidate_fingerprint"
"""What every node read selects: the payload, and the columns it must agree with."""

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


def decode_node_row(row: sqlite3.Row) -> ExperimentNode:
    """Decode a node from its row, refusing one whose columns and payload disagree.

    The second of two integrity layers. Migration 021's triggers keep the
    columns equal to what the payload *declares*, and refuse a snapshot
    version they do not know, but SQL cannot compute a fingerprint. This
    checks the declaration against the candidate itself.

    The row must hold :data:`NODE_COLUMNS`. Checked on every read:

    - ``candidate_kind`` is the kind the payload's candidate holds;
    - ``candidate_fingerprint`` is the payload's, and is one of the candidate's
      own identities (training's v2 or, for an older node, v1), so a payload
      whose candidate was replaced is caught even when both fingerprints were
      copied over.

    Raises:
        NodeRecordMismatchError: Naming every disagreement.
        IncompatiblePayloadError: If the payload has a pre-PR-007 node body.
    """
    from xaytune.core.domain.experiment import ExperimentNode

    node = decode_payload(row["payload_json"], ExperimentNode)
    problems = []
    if node.candidate_kind.value != row["candidate_kind"]:
        problems.append(
            f"the candidate_kind column says {row['candidate_kind']!r}, the payload holds "
            f"{node.candidate_kind.value!r}"
        )
    if node.candidate_fingerprint != row["candidate_fingerprint"]:
        problems.append("the candidate_fingerprint column is not the payload's")
    identities = node.candidate.experiment_candidate().fingerprints_for_lookup()
    if row["candidate_fingerprint"] not in identities:
        # Any identity the candidate may be recorded under: a training node
        # recorded before candidate fingerprint v2 holds its v1.
        problems.append("the candidate_fingerprint column is not the candidate's identity")
    if problems:
        raise NodeRecordMismatchError(str(node.id), problems)
    return node


def _looks_pre_candidate(payload: str) -> bool:
    """Whether this payload has the band B node shape."""
    try:
        body = json.loads(payload)
    except json.JSONDecodeError:
        return False
    return isinstance(body, dict) and "training_spec" in body and "candidate" not in body
