"""Durable payload compatibility across the `candidate_fingerprint` rename.

Written in PR-004 against stand-in models, because the rename had not happened
yet. It has now, so these assert the real thing: a payload written before the
rename still loads, and every write since carries the new key.

The hazard was never the SQL column -- that was named for its destination from
the start. It was ``payload_json``: :class:`FrozenDomainModel` sets
``extra="forbid"``, so a naive rename fails to load an older payload with two
errors at once, the old key unexpected and the new one missing.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from xaytune.core import Actor, ExperimentNodeStatus
from xaytune.core.domain.experiment import ExperimentNode
from xaytune.storage import ControlPlaneRepository, write_transaction

ACTOR = Actor(type="system", id="controller")


def _legacy_payload(node: ExperimentNode) -> str:
    """The same node as a PR-004-era database would have stored it."""
    payload = node.model_dump(mode="json", by_alias=True)
    payload["training_fingerprint"] = payload.pop("candidate_fingerprint")
    return json.dumps(payload, sort_keys=True)


def test_a_payload_written_before_the_rename_still_loads(seeded: dict[str, Any]) -> None:
    """The compatibility this whole module exists for."""
    node = seeded["node"]

    restored = ExperimentNode.model_validate_json(_legacy_payload(node))

    assert restored.candidate_fingerprint == node.candidate_fingerprint
    assert restored == node


def test_a_payload_missing_both_spellings_is_refused(seeded: dict[str, Any]) -> None:
    """Accepting either name must not become accepting neither."""
    payload = json.loads(_legacy_payload(seeded["node"]))
    del payload["training_fingerprint"]

    with pytest.raises(Exception, match="candidate_fingerprint"):
        ExperimentNode.model_validate_json(json.dumps(payload))


def test_an_unknown_key_is_still_refused(seeded: dict[str, Any]) -> None:
    """``extra="forbid"`` is what made the rename a hard failure rather than a
    silently dropped field. It is still in force."""
    payload = json.loads(_legacy_payload(seeded["node"]))
    payload["fingerprint"] = "sha256:wrong-name"

    with pytest.raises(Exception, match="fingerprint"):
        ExperimentNode.model_validate_json(json.dumps(payload))


def test_writes_carry_only_the_new_key(
    connection: sqlite3.Connection, seeded: dict[str, Any]
) -> None:
    """Every payload the repository writes uses the new spelling."""
    for table, key in (("experiment_nodes", "node"), ("runs", "run")):
        row = connection.execute(
            f"SELECT payload_json FROM {table} WHERE id = ?",  # noqa: S608
            (str(seeded[key].id),),
        ).fetchone()
        assert "candidate_fingerprint" in row["payload_json"]
        assert "training_fingerprint" not in row["payload_json"]


def test_a_legacy_row_converges_on_the_next_write(
    connection: sqlite3.Connection, seeded: dict[str, Any]
) -> None:
    """No backfill migration: a row adopts the new key when it is next written.

    This is what makes naming the column for its destination pay off rather
    than merely defer the cost.
    """
    node = seeded["node"]
    repo = ControlPlaneRepository(connection)

    # Rewind the stored payload to its pre-rename form.
    with write_transaction(connection):
        connection.execute(
            "UPDATE experiment_nodes SET payload_json = ? WHERE id = ?",
            (_legacy_payload(node), str(node.id)),
        )
    before = connection.execute(
        "SELECT payload_json FROM experiment_nodes WHERE id = ?", (str(node.id),)
    ).fetchone()["payload_json"]
    assert "training_fingerprint" in before

    # A perfectly ordinary transition, through the real repository.
    repo.transition_node(
        node.id,
        expected_revision=node.revision,
        new_status=ExperimentNodeStatus.PLANNED,
        actor=ACTOR,
    )

    rewritten = connection.execute(
        "SELECT payload_json FROM experiment_nodes WHERE id = ?", (str(node.id),)
    ).fetchone()["payload_json"]
    assert "candidate_fingerprint" in rewritten
    assert "training_fingerprint" not in rewritten


def test_the_column_never_needed_renaming(connection: sqlite3.Connection) -> None:
    """Why it was named for the destination in the first place."""
    for table in ("experiment_nodes", "runs"):
        columns = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        assert "candidate_fingerprint" in columns
        assert "training_fingerprint" not in columns


def test_the_payload_and_its_column_agree(
    connection: sqlite3.Connection, seeded: dict[str, Any]
) -> None:
    """A row whose index says one thing and whose body says another is worse
    than either being wrong alone."""
    row = connection.execute(
        "SELECT candidate_fingerprint, payload_json FROM experiment_nodes WHERE id = ?",
        (str(seeded["node"].id),),
    ).fetchone()

    assert (
        json.loads(row["payload_json"])["candidate_fingerprint"] == (row["candidate_fingerprint"])
    )


def test_unencodable_values_are_refused() -> None:
    """The snapshot is fingerprinted, so it must stay canonically encodable.

    ``FrozenDict`` rejects the set at construction, so it surfaces as a
    validation error rather than later at fingerprint time.
    """
    from pydantic import ValidationError

    from xaytune.core.domain.candidate import TrainingKind, TrainingSpec

    with pytest.raises(ValidationError, match="sets are not allowed"):
        TrainingSpec(kind=TrainingKind.SFT, metadata={"seen": {1, 2}})
