"""Durable payload compatibility across a domain field rename.

The aggregate body is stored as JSON in ``payload_json``, which is what lets a
new domain field ship without a SQL migration. It does **not** make renames
free, and the reason is easy to miss: :class:`FrozenDomainModel` sets
``extra="forbid"``, so a payload written before a rename fails to load
afterwards with two errors at once -- the old key is unexpected, and the new one
is missing.

PR-007 renames ``training_fingerprint`` to ``candidate_fingerprint`` when
``CandidateSpec`` lands, and by then PR-004 databases exist. These tests pin the
pattern that makes that rename safe, and they are written to fail if it is done
naively.

The pattern: accept both keys on the way in, write only the new one on the way
out. A database converges on the new spelling as its aggregates are next
written, and no backfill migration is needed.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest
from pydantic import AliasChoices, Field, create_model

from xaytune.core import ExperimentNodeStatus
from xaytune.core.domain.experiment import ExperimentNode
from xaytune.storage import AggregateStore, write_transaction

# ---------------------------------------------------------------------------
# Stand-ins for the PR-007-era model. Built here rather than imported because
# the rename has not happened yet -- the point is to cross the boundary now,
# while the fix is still cheap.
# ---------------------------------------------------------------------------


def _renamed_model(*, with_alias: bool) -> type[ExperimentNode]:
    """Return ``ExperimentNode`` with the PR-007 field name.

    ``with_alias=False`` is the naive rename: the one a coding agent would write
    by search-and-replace.
    """
    fields: dict[str, Any] = {
        name: (info.annotation, info) for name, info in ExperimentNode.model_fields.items()
    }
    del fields["training_fingerprint"]

    if with_alias:
        fields["candidate_fingerprint"] = (
            str,
            Field(
                validation_alias=AliasChoices("candidate_fingerprint", "training_fingerprint"),
                serialization_alias="candidate_fingerprint",
            ),
        )
    else:
        fields["candidate_fingerprint"] = (str, ...)

    return create_model(  # type: ignore[no-any-return, call-overload]
        "RenamedExperimentNode",
        __base__=ExperimentNode.__mro__[1],
        **fields,
    )


def test_a_naive_rename_cannot_read_a_pr004_payload(seeded: dict[str, Any]) -> None:
    """The failure this whole module exists to prevent.

    ``extra="forbid"`` turns a rename into a hard read failure, not a silently
    dropped field -- which is the better of the two, but only if someone has
    planned for it.
    """
    payload = json.dumps(seeded["node"].model_dump(mode="json"))
    naive = _renamed_model(with_alias=False)

    with pytest.raises(Exception) as caught:
        naive.model_validate_json(payload)

    message = str(caught.value)
    assert "training_fingerprint" in message
    assert "candidate_fingerprint" in message


def test_an_aliased_rename_reads_a_pr004_payload(seeded: dict[str, Any]) -> None:
    node = seeded["node"]
    payload = json.dumps(node.model_dump(mode="json"))

    renamed = _renamed_model(with_alias=True).model_validate_json(payload)

    assert renamed.candidate_fingerprint == node.training_fingerprint  # type: ignore[attr-defined]


def test_a_rewritten_payload_converges_on_the_new_key(
    store: AggregateStore, connection: sqlite3.Connection, seeded: dict[str, Any]
) -> None:
    """PR-004 writes it, PR-007 reads it, transitions it, and writes it back.

    The assertion that matters is the last one: the rewritten payload carries
    only the new key, so a database converges on the new spelling as its rows
    are next written rather than needing a backfill.

    The write below is direct SQL rather than a repository call. The stand-in
    PR-007 model is not one of the aggregate types the store maps, so it cannot
    go through the typed writer -- and the property under test is the payload
    encoding, not the write path. Naming this "through the real store" would
    have claimed a path it does not take.
    """
    node_id = str(seeded["node"].id)
    model = _renamed_model(with_alias=True)

    # 1. Read the PR-004-era row with the PR-007-era model.
    row = connection.execute(
        "SELECT payload_json FROM experiment_nodes WHERE id = ?", (node_id,)
    ).fetchone()
    assert "training_fingerprint" in row["payload_json"]
    loaded = model.model_validate_json(row["payload_json"])

    # 2. Transition it, as any controller would. Done by hand because the
    #    stand-in model is built from the aggregate base and so has no
    #    with_status(); the real PR-007 ExperimentNode will keep its own.
    planned = model.model_validate(
        {
            **loaded.model_dump(mode="python", by_alias=True),
            "status": ExperimentNodeStatus.PLANNED,
            "revision": loaded.revision + 1,
        }
    )

    # 3. Persist it back through the real writer.
    with write_transaction(connection):
        connection.execute(
            "UPDATE experiment_nodes SET status = ?, revision = ?, payload_json = ? "
            "WHERE id = ? AND revision = ?",
            (
                planned.status.value,
                planned.revision,
                json.dumps(planned.model_dump(mode="json", by_alias=True), sort_keys=True),
                node_id,
                loaded.revision,
            ),
        )

    # 4. The stored payload has converged on the new key.
    rewritten = connection.execute(
        "SELECT payload_json FROM experiment_nodes WHERE id = ?", (node_id,)
    ).fetchone()["payload_json"]

    assert "candidate_fingerprint" in rewritten
    assert "training_fingerprint" not in rewritten


def test_the_sql_column_never_needed_the_rename(
    connection: sqlite3.Connection, seeded: dict[str, Any]
) -> None:
    """Why the column was named for the destination rather than the origin."""
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(experiment_nodes)").fetchall()
    }
    assert "candidate_fingerprint" in columns
    assert "training_fingerprint" not in columns
