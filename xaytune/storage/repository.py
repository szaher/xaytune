"""Revision-guarded aggregate persistence (ADR-005 §2).

Scope, and why it is drawn here
-------------------------------

ADR-005 §3 requires that a state transition, its domain event and its outbox
record commit together, and the ADR's consequences are explicit that callers
must not be *able* to write state on its own::

    Callers cannot write state without an event, or request an external effect
    without durable intent, because the API does not offer those operations
    separately.

PR-004 builds the tables and the revision semantics; PR-005 adds the events and
the outbox. So this module deliberately exposes **no public write method**. The
row-level writers are private, they refuse to run outside a write transaction,
and PR-005 composes them with the event and outbox writes into the single public
operation the contract describes.

Shipping a public ``save_experiment()`` now would be the easier path and would
bake in the exact violation the ADR was expanded to prevent -- and once callers
exist, removing it is a breaking change rather than a design decision.

Concurrency
-----------

Every update is a compare-and-swap on ``revision``::

    UPDATE ... SET ... WHERE id = ? AND revision = ?

A zero-row result means someone else committed first, and raises
:class:`~xaytune.core.errors.ConcurrentModificationError`. The caller re-reads
and retries; it never writes with a stale revision. Aggregates bump their own
revision in ``with_status()``, so the expected value is always
``aggregate.revision - 1`` -- the revision the row held before this transition.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, TypeVar

from pydantic import ValidationError

from xaytune.core.clock import utc_now
from xaytune.core.domain.experiment import Experiment, ExperimentNode
from xaytune.core.domain.run import Run, RunAttempt
from xaytune.core.errors import ConcurrentModificationError
from xaytune.core.immutable import AggregateModel
from xaytune.storage.errors import AggregateNotFoundError, IncompatiblePayloadError

__all__ = ["AggregateStore"]

AggregateT = TypeVar("AggregateT", bound=AggregateModel)


class AggregateStore:
    """Row-level reads and revision-guarded writes for the core aggregates.

    Reads are public. Writes are not: see the module docstring.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    # ---- reads ---------------------------------------------------------

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Return the experiment, or ``None`` if it does not exist."""
        return self._get("experiments", experiment_id, Experiment)

    def get_node(self, node_id: str) -> ExperimentNode | None:
        """Return the node, or ``None`` if it does not exist."""
        return self._get("experiment_nodes", node_id, ExperimentNode)

    def get_run(self, run_id: str) -> Run | None:
        """Return the run, or ``None`` if it does not exist."""
        return self._get("runs", run_id, Run)

    def get_attempt(self, attempt_id: str) -> RunAttempt | None:
        """Return the attempt, or ``None`` if it does not exist."""
        return self._get("run_attempts", attempt_id, RunAttempt)

    def load_experiment(self, experiment_id: str) -> Experiment:
        """Return the experiment.

        Raises:
            AggregateNotFoundError: If it does not exist.
        """
        return self._require(self.get_experiment(experiment_id), "Experiment", experiment_id)

    def load_node(self, node_id: str) -> ExperimentNode:
        """Return the node.

        Raises:
            AggregateNotFoundError: If it does not exist.
        """
        return self._require(self.get_node(node_id), "ExperimentNode", node_id)

    def load_run(self, run_id: str) -> Run:
        """Return the run.

        Raises:
            AggregateNotFoundError: If it does not exist.
        """
        return self._require(self.get_run(run_id), "Run", run_id)

    def load_attempt(self, attempt_id: str) -> RunAttempt:
        """Return the attempt.

        Raises:
            AggregateNotFoundError: If it does not exist.
        """
        return self._require(self.get_attempt(attempt_id), "RunAttempt", attempt_id)

    def nodes_for_experiment(self, experiment_id: str) -> tuple[ExperimentNode, ...]:
        """Return the experiment's nodes in creation order."""
        rows = self._connection.execute(
            "SELECT payload_json FROM experiment_nodes "
            "WHERE experiment_id = ? ORDER BY created_at, id",
            (experiment_id,),
        ).fetchall()
        return tuple(ExperimentNode.model_validate_json(row["payload_json"]) for row in rows)

    def runs_for_node(self, node_id: str) -> tuple[Run, ...]:
        """Return the node's runs in creation order."""
        rows = self._connection.execute(
            "SELECT payload_json FROM runs WHERE node_id = ? ORDER BY created_at, id",
            (node_id,),
        ).fetchall()
        return tuple(Run.model_validate_json(row["payload_json"]) for row in rows)

    def attempts_for_run(self, run_id: str) -> tuple[RunAttempt, ...]:
        """Return the run's attempts in attempt-number order."""
        rows = self._connection.execute(
            "SELECT payload_json FROM run_attempts WHERE run_id = ? ORDER BY attempt_number",
            (run_id,),
        ).fetchall()
        return tuple(RunAttempt.model_validate_json(row["payload_json"]) for row in rows)

    def unresolved_attempts(self) -> tuple[RunAttempt, ...]:
        """Return every attempt that is not in a terminal state.

        Reconciliation's first question after a restart. Ordered oldest-first,
        because the attempt that has been unresolved longest is the one most
        likely to have been orphaned.
        """
        rows = self._connection.execute(
            "SELECT payload_json FROM run_attempts "
            "WHERE status NOT IN ('succeeded', 'failed', 'cancelled', 'preempted') "
            "ORDER BY updated_at, id"
        ).fetchall()
        return tuple(RunAttempt.model_validate_json(row["payload_json"]) for row in rows)

    # ---- writes (private; see the module docstring) ---------------------

    def _insert_experiment(self, experiment: Experiment) -> None:
        self._insert(
            "experiments",
            experiment,
            {"status": experiment.status.value},
        )

    def _insert_node(self, node: ExperimentNode) -> None:
        self._insert(
            "experiment_nodes",
            node,
            {
                "experiment_id": str(node.experiment_id),
                "status": node.status.value,
                "candidate_fingerprint": node.candidate_fingerprint,
            },
        )
        for parent_id in node.parent_ids:
            self._connection.execute(
                # Plain INSERT: duplicate parents are refused by
                # ExperimentGraph.validate_parents, so OR IGNORE would only
                # mask a lineage the payload and the edge table disagree about.
                "INSERT INTO experiment_edges "
                "(parent_id, child_id, reason, payload_json) VALUES (?, ?, ?, ?)",
                (str(parent_id), str(node.id), node.reason, "{}"),
            )

    def _insert_run(self, run: Run) -> None:
        self._insert(
            "runs",
            run,
            {
                "experiment_id": str(run.experiment_id),
                "node_id": str(run.node_id),
                "status": run.status.value,
                "candidate_fingerprint": run.candidate_fingerprint,
            },
        )

    def _insert_attempt(self, attempt: RunAttempt) -> None:
        self._insert(
            "run_attempts",
            attempt,
            {
                "run_id": str(attempt.run_id),
                "attempt_number": attempt.attempt_number,
                "status": attempt.status.value,
                # ADR-014 §1a. The aggregate does not carry this field yet; it
                # arrives with the telemetry work, and the column defaults to
                # generation 0 until then.
                "telemetry_generation": 0,
            },
            # RunAttempt tracks started_at/ended_at, which are lifecycle facts
            # about the workload and are both None at creation. The row's
            # timestamps are about the record, so they come from the clock --
            # otherwise every pending attempt would sort under an empty string.
            created_at=utc_now(),
            updated_at=utc_now(),
        )

    def _update_experiment(self, experiment: Experiment) -> None:
        self._update("experiments", experiment, {"status": experiment.status.value})

    def _update_node(self, node: ExperimentNode) -> None:
        self._update(
            "experiment_nodes",
            node,
            {
                "status": node.status.value,
                "candidate_fingerprint": node.candidate_fingerprint,
            },
        )

    def _update_run(self, run: Run) -> None:
        self._update(
            "runs",
            run,
            {
                "status": run.status.value,
                "candidate_fingerprint": run.candidate_fingerprint,
            },
        )

    def _update_attempt(self, attempt: RunAttempt) -> None:
        self._update(
            "run_attempts",
            attempt,
            {"status": attempt.status.value},
            updated_at=utc_now(),
        )

    # ---- machinery ------------------------------------------------------

    def _get(self, table: str, aggregate_id: str, model: type[AggregateT]) -> AggregateT | None:
        row = self._connection.execute(
            f"SELECT payload_json FROM {table} WHERE id = ?",  # noqa: S608 - table is a literal
            (aggregate_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            return model.model_validate_json(row["payload_json"])
        except ValidationError as error:
            # A payload from before a format change reads as a pile of missing
            # and unexpected fields. Saying so plainly beats letting the caller
            # infer it from a validation error two frames down.
            if _looks_pre_candidate(row["payload_json"]):
                raise IncompatiblePayloadError(
                    model.__name__, aggregate_id, "pre-CandidateSpec node body"
                ) from error
            raise

    @staticmethod
    def _require(aggregate: AggregateT | None, name: str, aggregate_id: str) -> AggregateT:
        if aggregate is None:
            raise AggregateNotFoundError(name, aggregate_id)
        return aggregate

    def _insert(
        self,
        table: str,
        aggregate: AggregateModel,
        columns: dict[str, Any],
        *,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        self._require_transaction()
        values: dict[str, Any] = {
            "id": str(aggregate.id),  # type: ignore[attr-defined]
            **columns,
            "revision": aggregate.revision,  # type: ignore[attr-defined]
            "payload_json": _dump(aggregate),
            "created_at": _stamp(created_at or getattr(aggregate, "created_at", None)),
            "updated_at": _stamp(updated_at or getattr(aggregate, "updated_at", None)),
        }
        names = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        self._connection.execute(
            f"INSERT INTO {table} ({names}) VALUES ({placeholders})",  # noqa: S608
            tuple(values.values()),
        )

    def _update(
        self,
        table: str,
        aggregate: AggregateModel,
        columns: dict[str, Any],
        *,
        updated_at: datetime | None = None,
    ) -> None:
        """Write *aggregate* back, guarded on the revision it was read at.

        Raises:
            ConcurrentModificationError: If no row matched, meaning another
                writer committed a transition to this aggregate first.
        """
        self._require_transaction()
        aggregate_id = str(aggregate.id)  # type: ignore[attr-defined]
        revision: int = aggregate.revision  # type: ignore[attr-defined]
        # The aggregate bumped its own revision in with_status(), so the row
        # still holds the previous one.
        expected = revision - 1

        values: dict[str, Any] = {
            **columns,
            "revision": revision,
            "payload_json": _dump(aggregate),
            "updated_at": _stamp(updated_at or getattr(aggregate, "updated_at", None)),
        }
        assignments = ", ".join(f"{name} = ?" for name in values)
        cursor = self._connection.execute(
            f"UPDATE {table} SET {assignments} WHERE id = ? AND revision = ?",  # noqa: S608
            (*values.values(), aggregate_id, expected),
        )
        if cursor.rowcount == 0:
            self._explain_failed_cas(table, type(aggregate).__name__, aggregate_id, expected)

    def _explain_failed_cas(
        self, table: str, aggregate: str, aggregate_id: str, expected: int
    ) -> None:
        """Turn a zero-row update into the error that actually describes it.

        A failed CAS has two causes and they need different responses. Another
        writer getting there first is a race, and the caller should re-read and
        retry. A caller applying several transitions in memory and persisting
        once is a bug, and retrying will not help: the row is *behind* the
        aggregate, and the transitions in between would have no events, which
        ADR-005 §3 does not allow.

        Raises:
            ConcurrentModificationError: If the row has moved on, or vanished.
            ValueError: If the aggregate is more than one transition ahead.
        """
        row = self._connection.execute(
            f"SELECT revision FROM {table} WHERE id = ?",  # noqa: S608
            (aggregate_id,),
        ).fetchone()

        if row is not None and int(row["revision"]) < expected:
            raise ValueError(
                f"{aggregate} {aggregate_id} is {expected - int(row['revision']) + 1} "
                f"transitions ahead of the stored row (stored revision "
                f"{row['revision']}, writing revision {expected + 1}). Each "
                f"transition commits separately with its own event (ADR-005 §3), "
                f"so persist them one at a time rather than batching them."
            )

        raise ConcurrentModificationError(aggregate, aggregate_id, expected)

    def _require_transaction(self) -> None:
        """Refuse a write outside a transaction.

        Without this, a write issued outside ``write_transaction()`` would
        autocommit on its own -- and a transition that commits without its event
        is precisely the split ADR-005 §3 forbids. Catching it here makes that a
        loud failure rather than a silently weaker guarantee.
        """
        if not self._connection.in_transaction:
            raise sqlite3.ProgrammingError(
                "aggregate writes must run inside write_transaction(): an "
                "autocommitted transition would commit state without its event "
                "and outbox record (ADR-005 §3)."
            )


def _looks_pre_candidate(payload: str) -> bool:
    """Whether this payload has the band B node shape."""
    try:
        body = json.loads(payload)
    except json.JSONDecodeError:
        return False
    return isinstance(body, dict) and "training_spec" in body and "candidate" not in body


def _dump(aggregate: AggregateModel) -> str:
    """Serialize an aggregate to its durable JSON payload.

    Sorted keys so that two processes writing the same aggregate produce the
    same bytes, which keeps payloads diffable and comparable.
    """
    return json.dumps(
        # by_alias so a renamed field persists under the name the column uses.
        # Without it, PR-007's rename would write training_fingerprint into a
        # payload whose column says candidate_fingerprint.
        aggregate.model_dump(mode="json", by_alias=True),
        sort_keys=True,
        separators=(",", ":"),
    )


def _stamp(value: datetime | None) -> str:
    """Render a timestamp for storage.

    ISO-8601 in UTC, so lexicographic order is chronological order and the
    ``ORDER BY created_at`` queries above mean what they say. ``None`` falls
    back to now rather than to an empty string, which would sort before every
    real timestamp and quietly corrupt those orderings.
    """
    return (value or utc_now()).isoformat()
