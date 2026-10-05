"""Row-level persistence for agent invocations (PR-032, migration 019).

Rows are read here and written only through
:class:`~xaytune.storage.ControlPlaneRepository`, inside its fenced
transactions. :class:`RepositoryAgentInvocationJournal` is how a host hands a
model-backed planner that repository as an
:class:`~xaytune.core.domain.agent_invocation.AgentInvocationJournal`.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

from xaytune.core.domain.agent_invocation import (
    AgentInvocation,
    AgentInvocationFailure,
    AgentInvocationIntent,
    AgentInvocationStatus,
    StampingAgentInvocationJournal,
)
from xaytune.core.errors import ConcurrentModificationError
from xaytune.core.ids import AgentInvocationId
from xaytune.core.immutable import thaw
from xaytune.storage.journal import _require_transaction

if TYPE_CHECKING:
    from xaytune.storage.control_plane import ControlPlaneRepository

__all__ = ["AgentInvocationStore", "RepositoryAgentInvocationJournal"]

_COLUMNS = (
    "id, attempt, intent_json, status, revision, response_json, failure_json, proposal_json, "
    "proposal_fingerprint, created_at, settled_at"
)


def _json(value: object) -> str | None:
    return None if value is None else json.dumps(thaw(value), sort_keys=True)


def _load(row: sqlite3.Row) -> AgentInvocation:
    return AgentInvocation(
        id=row["id"],
        attempt=row["attempt"],
        intent=AgentInvocationIntent.model_validate_json(row["intent_json"]),
        status=AgentInvocationStatus(row["status"]),
        revision=row["revision"],
        response=None if row["response_json"] is None else json.loads(row["response_json"]),
        failure=None
        if row["failure_json"] is None
        else AgentInvocationFailure.model_validate_json(row["failure_json"]),
        proposal=None if row["proposal_json"] is None else json.loads(row["proposal_json"]),
        proposal_fingerprint=row["proposal_fingerprint"],
        created_at=datetime.fromisoformat(row["created_at"]),
        settled_at=None if row["settled_at"] is None else datetime.fromisoformat(row["settled_at"]),
    )


class AgentInvocationStore:
    """Reads, and guarded writes, of agent invocations."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, invocation_id: AgentInvocationId | str) -> AgentInvocation | None:
        row = self._connection.execute(
            f"SELECT {_COLUMNS} FROM agent_invocations WHERE id = ?", (str(invocation_id),)
        ).fetchone()
        return None if row is None else _load(row)

    def for_experiment(self, experiment_id: str) -> tuple[AgentInvocation, ...]:
        """Every invocation in an experiment, oldest first."""
        rows = self._connection.execute(
            f"SELECT {_COLUMNS} FROM agent_invocations WHERE experiment_id = ? "
            "ORDER BY created_at, id",
            (experiment_id,),
        ).fetchall()
        return tuple(_load(row) for row in rows)

    def latest_for_round(
        self, experiment_id: str, planner_spec_fingerprint: str, context_fingerprint: str
    ) -> AgentInvocation | None:
        """The round's highest attempt, or ``None`` if it was never begun."""
        row = self._connection.execute(
            f"SELECT {_COLUMNS} FROM agent_invocations WHERE experiment_id = ? "
            "AND planner_spec_fingerprint = ? AND context_fingerprint = ? "
            "ORDER BY attempt DESC LIMIT 1",
            (experiment_id, planner_spec_fingerprint, context_fingerprint),
        ).fetchone()
        return None if row is None else _load(row)

    def _insert(self, invocation: AgentInvocation) -> None:
        _require_transaction(self._connection, "agent_invocations")
        intent = invocation.intent
        self._connection.execute(
            "INSERT INTO agent_invocations (id, experiment_id, planner_spec_fingerprint, "
            "context_fingerprint, attempt, request_fingerprint, intent_json, status, revision, "
            "response_json, failure_json, proposal_json, proposal_fingerprint, created_at, "
            "settled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(invocation.id),
                str(intent.experiment_id),
                intent.planner_spec_fingerprint,
                intent.context_fingerprint,
                invocation.attempt,
                intent.request_fingerprint,
                json.dumps(intent.model_dump(mode="json"), sort_keys=True),
                invocation.status.value,
                invocation.revision,
                _json(invocation.response),
                None if invocation.failure is None else invocation.failure.model_dump_json(),
                _json(invocation.proposal),
                invocation.proposal_fingerprint,
                invocation.created_at.isoformat(),
                None if invocation.settled_at is None else invocation.settled_at.isoformat(),
            ),
        )

    def _update(self, invocation: AgentInvocation) -> None:
        """Write a moved invocation back, guarded on the revision it moved from."""
        _require_transaction(self._connection, "agent_invocations")
        cursor = self._connection.execute(
            "UPDATE agent_invocations SET status = ?, revision = ?, response_json = ?, "
            "failure_json = ?, proposal_json = ?, proposal_fingerprint = ?, settled_at = ? "
            "WHERE id = ? AND revision = ?",
            (
                invocation.status.value,
                invocation.revision,
                _json(invocation.response),
                None if invocation.failure is None else invocation.failure.model_dump_json(),
                _json(invocation.proposal),
                invocation.proposal_fingerprint,
                None if invocation.settled_at is None else invocation.settled_at.isoformat(),
                str(invocation.id),
                invocation.revision - 1,
            ),
        )
        if cursor.rowcount != 1:
            raise ConcurrentModificationError(
                "AgentInvocation", str(invocation.id), invocation.revision - 1
            )


class RepositoryAgentInvocationJournal(StampingAgentInvocationJournal):
    """An :class:`~xaytune.core.domain.agent_invocation.AgentInvocationJournal` over the record.

    Every write goes through the repository's fenced transactions.
    """

    def __init__(self, repository: ControlPlaneRepository) -> None:
        super().__init__()
        self._repository = repository

    def begin(self, intent: AgentInvocationIntent) -> AgentInvocation:
        return self._repository.begin_agent_invocation(intent)

    def get(self, invocation_id: AgentInvocationId | str) -> AgentInvocation | None:
        return self._repository.agent_invocations.get(invocation_id)

    def _write(self, moved: AgentInvocation) -> AgentInvocation:
        return self._repository.settle_agent_invocation(moved)
