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

from xaytune.core.clock import utc_now
from xaytune.core.domain.evaluation import EvaluationAttempt, EvaluationResult, EvaluationRun
from xaytune.core.domain.experiment import Experiment, ExperimentNode
from xaytune.core.domain.run import Run, RunAttempt
from xaytune.core.errors import ConcurrentModificationError
from xaytune.core.immutable import AggregateModel
from xaytune.core.telemetry import EvaluationCompletedPayload
from xaytune.storage.errors import AggregateNotFoundError
from xaytune.storage.payloads import decode_node_payload, decode_payload

__all__ = ["AggregateStore"]

AggregateT = TypeVar("AggregateT", bound=AggregateModel)

_ATTEMPT_TABLES: dict[str, str] = {
    "training-attempt": "run_attempts",
    "evaluation-attempt": "evaluation_attempts",
}
"""Where each kind of attempt keeps its telemetry cursor. Both tables carry
the same (generation, sequence) pair with the same meaning (005, 006)."""


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

    def get_evaluation_run(self, run_id: str) -> EvaluationRun | None:
        """Return the evaluation run, or ``None`` if it does not exist."""
        return self._get("evaluation_runs", run_id, EvaluationRun)

    def get_evaluation_attempt(self, attempt_id: str) -> EvaluationAttempt | None:
        """Return the evaluation attempt, or ``None`` if it does not exist."""
        return self._get("evaluation_attempts", attempt_id, EvaluationAttempt)

    def load_evaluation_run(self, run_id: str) -> EvaluationRun:
        """Return the evaluation run.

        Raises:
            AggregateNotFoundError: If it does not exist.
        """
        return self._require(self.get_evaluation_run(run_id), "EvaluationRun", run_id)

    def load_evaluation_attempt(self, attempt_id: str) -> EvaluationAttempt:
        """Return the evaluation attempt.

        Raises:
            AggregateNotFoundError: If it does not exist.
        """
        return self._require(
            self.get_evaluation_attempt(attempt_id), "EvaluationAttempt", attempt_id
        )

    def evaluation_runs_for_node(
        self, node_id: str, *, cycle: int | None = None
    ) -> tuple[EvaluationRun, ...]:
        """Return the node's evaluation runs in creation order -- of one cycle, if given."""
        if cycle is None:
            rows = self._connection.execute(
                "SELECT payload_json FROM evaluation_runs WHERE node_id = ? "
                "ORDER BY created_at, id",
                (node_id,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT payload_json FROM evaluation_runs "
                "WHERE node_id = ? AND evaluation_cycle = ? ORDER BY created_at, id",
                (node_id, cycle),
            ).fetchall()
        return tuple(EvaluationRun.model_validate_json(row["payload_json"]) for row in rows)

    def evaluation_attempts_for_run(self, run_id: str) -> tuple[EvaluationAttempt, ...]:
        """Return the evaluation run's attempts in attempt-number order."""
        rows = self._connection.execute(
            "SELECT payload_json FROM evaluation_attempts "
            "WHERE evaluation_run_id = ? ORDER BY attempt_number",
            (run_id,),
        ).fetchall()
        return tuple(EvaluationAttempt.model_validate_json(row["payload_json"]) for row in rows)

    def evaluation_result_for_run(self, run_id: str) -> EvaluationResult | None:
        """Return the result the evaluation run produced, if it produced one."""
        row = self._connection.execute(
            "SELECT payload_json FROM evaluation_results WHERE evaluation_run_id = ?",
            (run_id,),
        ).fetchone()
        return None if row is None else EvaluationResult.model_validate_json(row["payload_json"])

    def evaluation_results_for_node(self, node_id: str) -> tuple[EvaluationResult, ...]:
        """Return every result recorded for the node, oldest first."""
        rows = self._connection.execute(
            "SELECT payload_json FROM evaluation_results WHERE node_id = ? ORDER BY created_at, id",
            (node_id,),
        ).fetchall()
        return tuple(EvaluationResult.model_validate_json(row["payload_json"]) for row in rows)

    def pending_completion(
        self, attempt_id: str
    ) -> tuple[EvaluationCompletedPayload, tuple[int, int]] | None:
        """The completion an evaluation attempt has received and not yet settled, if any.

        With the ``(generation, sequence)`` it arrived at. Kept on the attempt
        rather than in the stream, so it survives the stream (see migration
        006).

        Raises:
            AggregateNotFoundError: If no such attempt exists.
        """
        row = self._connection.execute(
            "SELECT pending_completion_json FROM evaluation_attempts WHERE id = ?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise AggregateNotFoundError("EvaluationAttempt", attempt_id)
        if row["pending_completion_json"] is None:
            return None
        held = json.loads(row["pending_completion_json"])
        completion = EvaluationCompletedPayload.model_validate(held["completion"])
        generation, sequence = held["position"]
        return completion, (int(generation), int(sequence))

    def _hold_completion(
        self,
        attempt_id: str,
        completion: EvaluationCompletedPayload,
        position: tuple[int, int],
    ) -> None:
        self._require_transaction()
        self._connection.execute(
            "UPDATE evaluation_attempts SET pending_completion_json = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "completion": completion.model_dump(mode="json"),
                        "position": list(position),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                attempt_id,
            ),
        )

    def nodes_for_experiment(self, experiment_id: str) -> tuple[ExperimentNode, ...]:
        """Return the experiment's nodes in creation order."""
        rows = self._connection.execute(
            "SELECT payload_json FROM experiment_nodes "
            "WHERE experiment_id = ? ORDER BY created_at, id",
            (experiment_id,),
        ).fetchall()
        return tuple(decode_node_payload(row["payload_json"], ExperimentNode) for row in rows)

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

    def _insert_evaluation_run(self, run: EvaluationRun) -> None:
        self._insert(
            "evaluation_runs",
            run,
            {
                "experiment_id": str(run.experiment_id),
                "node_id": str(run.node_id),
                "evaluation_cycle": run.evaluation_cycle,
                "subject_artifact_id": str(run.subject.id),
                "subject_digest": run.subject.digest,
                "evaluation_fingerprint": run.evaluation_fingerprint,
                "seed": run.seed,
                "replicate": run.replicate,
                "status": run.status.value,
            },
        )

    def _insert_evaluation_attempt(self, attempt: EvaluationAttempt) -> None:
        self._insert(
            "evaluation_attempts",
            attempt,
            {
                "evaluation_run_id": str(attempt.evaluation_run_id),
                "attempt_number": attempt.attempt_number,
                "status": attempt.status.value,
            },
            created_at=utc_now(),
            updated_at=utc_now(),
        )

    def _insert_evaluation_result(self, result: EvaluationResult) -> None:
        """Write a result. There is no update: a result is a historical fact."""
        self._require_transaction()
        self._connection.execute(
            "INSERT INTO evaluation_results (id, evaluation_run_id, node_id, "
            "evaluation_fingerprint, subject_artifact_id, subject_digest, payload_json, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(result.id),
                str(result.evaluation_run_id),
                str(result.node_id),
                result.evaluation_fingerprint,
                str(result.subject.id),
                result.subject.digest,
                json.dumps(
                    result.model_dump(mode="json", by_alias=True),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                _stamp(result.created_at),
            ),
        )

    def _update_evaluation_run(self, run: EvaluationRun) -> None:
        self._update("evaluation_runs", run, {"status": run.status.value})

    def _update_evaluation_attempt(self, attempt: EvaluationAttempt) -> None:
        self._update(
            "evaluation_attempts", attempt, {"status": attempt.status.value}, updated_at=utc_now()
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

    def telemetry_position(
        self, attempt_id: str, *, kind: str = "training-attempt"
    ) -> tuple[int, int]:
        """The attempt's durable telemetry cursor, as ``(generation, sequence)``.

        The position of the last telemetry event whose consequences are
        recorded (ADR-014 §4). ``(0, -1)`` for an attempt nothing has been
        recorded from. *kind* is the attempt's operation target kind; training
        and evaluation attempts keep the same cursor in their own tables.

        Raises:
            AggregateNotFoundError: If no such attempt exists.
        """
        table = _ATTEMPT_TABLES[kind]
        row = self._connection.execute(
            f"SELECT telemetry_generation, telemetry_sequence FROM {table} WHERE id = ?",  # noqa: S608
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise AggregateNotFoundError(kind, attempt_id)
        return int(row["telemetry_generation"]), int(row["telemetry_sequence"])

    def _advance_telemetry(
        self, attempt_id: str, position: tuple[int, int], *, kind: str = "training-attempt"
    ) -> None:
        """Move the cursor forward to *position*, or leave it if already past.

        Only forward: a replayed or late event re-applied after a restart must
        not wind the cursor back over events whose effects are recorded.
        """
        self._require_transaction()
        table = _ATTEMPT_TABLES[kind]
        generation, sequence = position
        self._connection.execute(
            f"UPDATE {table} SET telemetry_generation = ?, telemetry_sequence = ? "  # noqa: S608
            "WHERE id = ? AND (telemetry_generation < ? "
            "OR (telemetry_generation = ? AND telemetry_sequence < ?))",
            (generation, sequence, attempt_id, generation, generation, sequence),
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
        return decode_payload(row["payload_json"], model, aggregate_id)

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
