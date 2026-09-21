"""The public write surface (ADR-005 §3–§5).

PR-004 deliberately shipped no public write method, because a transition and its
event have to commit together and there were no events yet. This module is the
other half: every operation here is one transaction, and the units are the ones
the ADR names.

```text
transition(aggregate, ...)                  state + event + outbox        §3
create_attempt_with_submit_intent(...)      attempt + INTENDED operation  §4
confirm_operation / mark_operation_sent / fail_operation
request_cancellation(...)                   Action + cancel operation     §5
resolve_cancellation(...)                   Action settled with an outcome
```

There is still no ``save_experiment()``. The row writers stay private on
:class:`~xaytune.storage.repository.AggregateStore` and
:mod:`xaytune.storage.journal`, and are composed here rather than exposed, so a
caller cannot write state without its event or request an effect without durable
intent -- the API simply does not offer those operations separately.

The asymmetry this buys, from ADR-005 §9: it is always safe to hold durable
intent with no effect, and never safe to have an effect with no durable intent.
Every boundary below is arranged so only the first can happen.
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import Literal

from xaytune.core.clock import utc_now
from xaytune.core.domain.action import (
    Action,
    ActionOutcome,
    ActionStatus,
    ActionTarget,
)
from xaytune.core.domain.event import DomainEvent, OutboxRecord
from xaytune.core.domain.experiment import Experiment, ExperimentNode
from xaytune.core.domain.operation import (
    RuntimeOperation,
    RuntimeOperationTarget,
)
from xaytune.core.domain.run import Run, RunAttempt
from xaytune.core.ids import ActionId, EventId, ExperimentId, OperationId
from xaytune.core.immutable import AggregateModel, FrozenDict
from xaytune.core.refs import Actor, RuntimeRef
from xaytune.storage.actions import ActionStore
from xaytune.storage.database import write_transaction
from xaytune.storage.errors import AggregateNotFoundError, StorageError
from xaytune.storage.journal import EventJournal, OperationJournal
from xaytune.storage.repository import AggregateStore

__all__ = ["ControlPlaneRepository", "UnknownOperationTargetError"]

_AGGREGATE_TABLES: dict[str, str] = {
    "Experiment": "experiments",
    "ExperimentNode": "experiment_nodes",
    "Run": "runs",
    "RunAttempt": "run_attempts",
}

_ATTEMPT_KINDS: frozenset[str] = frozenset({"training-attempt", "evaluation-attempt"})
"""The target kinds a single cancel operation can address.

Run- and experiment-level cancellation fans out over descendants (ADR-013 §6),
so it is a saga rather than one effect, and PR-006a refuses it rather than
writing intent nothing will carry out.
"""

_CANCEL_TYPE_FOR: dict[str, str] = {
    "experiment": "cancel-experiment",
    "run": "cancel-run",
    "training-attempt": "cancel-attempt",
    "evaluation-attempt": "cancel-attempt",
}
"""Which cancellation action a target kind implies.

``cancel-attempt`` covers both attempt kinds: cancelling is the same operation
on the same kind of subject, and only the table differs.
"""

_TARGET_TABLES: dict[str, str] = {
    "training-attempt": "run_attempts",
    # ADR-015's tables arrive with the evaluation lifecycle; until then an
    # evaluation target has nothing to resolve against and is refused rather
    # than accepted unchecked.
    "evaluation-attempt": "evaluation_attempts",
}


class CancellationSagaRequiredError(StorageError):
    """Run- and experiment-level cancellation is a saga, not one operation.

    ADR-013 §6: cancelling an experiment fans out to its descendants, and the
    invariant is that it must not reach ``CANCELLED`` while any descendant is
    unresolved. That coordination is a controller concern and is not built yet.

    Refused rather than half-implemented. Writing the `Action` and no operation
    would leave durable intent that nothing carries out and nothing retries --
    the exact state ADR-005 §5 exists to prevent -- and minting a single
    operation against a run id would ask a runtime to cancel something it has
    no handle on.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        super().__init__(
            f"cancelling a {kind} requires the descendant saga of ADR-013 §6, "
            f"which is not implemented. Cancel its attempts individually."
        )


class UnknownOperationTargetError(StorageError):
    """An operation names a target that does not exist.

    SQLite cannot enforce this: the journal's target is typed rather than a
    foreign key, because training and evaluation attempts live in different
    tables (ADR-013). So the repository enforces it instead -- ADR-005 §10.1.
    """

    def __init__(self, kind: str, target_id: str) -> None:
        self.kind = kind
        self.target_id = target_id
        super().__init__(f"no {kind} exists with id {target_id}")


class ControlPlaneRepository:
    """Atomic writes over the control-plane aggregates and their journals."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self.aggregates = AggregateStore(connection)
        self.events = EventJournal(connection)
        self.operations = OperationJournal(connection)
        self.actions = ActionStore(connection)

    # ---- ADR-005 §3 ----------------------------------------------------

    def create_experiment(
        self,
        experiment: Experiment,
        *,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> Experiment:
        """Create an experiment, its creation event and any outbox records."""
        with write_transaction(self._connection):
            self.aggregates._insert_experiment(experiment)
            self._emit(experiment, "ExperimentCreated", str(experiment.id), actor, destinations)
        return experiment

    def create_node(
        self,
        node: ExperimentNode,
        *,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> ExperimentNode:
        """Create a node, its creation event and any outbox records."""
        with write_transaction(self._connection):
            self.aggregates._insert_node(node)
            self._emit(node, "NodeCreated", str(node.experiment_id), actor, destinations)
        return node

    def create_run(self, run: Run, *, actor: Actor, destinations: tuple[str, ...] = ()) -> Run:
        """Create a run, its creation event and any outbox records."""
        with write_transaction(self._connection):
            self.aggregates._insert_run(run)
            self._emit(run, "RunCreated", str(run.experiment_id), actor, destinations)
        return run

    def transition(
        self,
        aggregate: AggregateModel,
        *,
        actor: Actor,
        event_type: str | None = None,
        destinations: tuple[str, ...] = (),
    ) -> None:
        """Persist a transitioned aggregate with its event, atomically.

        *aggregate* is the post-transition value, produced by the aggregate's
        own ``with_status()`` -- so the state machine has already validated the
        edge and bumped the revision. This writes it under a revision guard and
        records what happened.

        The owning experiment is **derived**, never supplied. A caller-provided
        id can disagree with the aggregate's own ownership, and the result is
        silent: the state change lands on one experiment while its event is
        filed under another's history. Nothing errors, and the provenance record
        is wrong in a way no later query can detect.

        Raises:
            ConcurrentModificationError: If another writer moved it first.
            ValueError: If the aggregate is more than one transition ahead of
                the stored row, which would leave the intervening transitions
                with no events.
        """
        with write_transaction(self._connection):
            self._update(aggregate)
            self._emit(
                aggregate,
                event_type or f"{type(aggregate).__name__}StatusChanged",
                self._owning_experiment(aggregate),
                actor,
                destinations,
            )

    # ---- ADR-005 §4 ----------------------------------------------------

    def create_attempt_with_submit_intent(
        self,
        attempt: RunAttempt,
        *,
        request_digest: str,
        actor: Actor,
        operation_id: OperationId | None = None,
        destinations: tuple[str, ...] = (),
    ) -> tuple[RunAttempt, RuntimeOperation]:
        """Create an attempt and its ``INTENDED`` submit operation in one commit.

        The runtime call happens **after** this returns and outside any
        transaction. A crash in between leaves the operation ``INTENDED``, which
        ADR-013 defines as *reconcile*, not *re-issue blindly* -- so the effect
        may exist, but never without a record that it was intended.

        **Get-or-create.** This is the method a crash retries, so retrying it
        with the same ids and request returns what is already recorded and
        writes nothing new. Idempotency in the journal alone would not be
        enough: the attempt insert runs first, so a naive retry would hit the
        attempt's primary key before the journal ever checked the operation id
        -- which is precisely the crash-and-retry case ADR-013 exists to make
        safe.

        Returns:
            The attempt and the operation, so the caller has the id to pass to
            ``submit_or_get``.

        Raises:
            IdempotencyConflictError: If the operation id exists against a
                different target, type or request digest.
        """
        operation = RuntimeOperation(
            id=operation_id or OperationId.generate(),
            target=RuntimeOperationTarget(kind="training-attempt", id=str(attempt.id)),
            type="submit",
            request_digest=request_digest,
        )

        with write_transaction(self._connection):
            existing = self.operations.get(str(operation.id))
            if existing is not None:
                self.operations._assert_same_request(existing, operation)
                stored_attempt = self.aggregates.get_attempt(existing.target.id)
                if stored_attempt is None:
                    raise StorageError(
                        f"operation {existing.id} references attempt "
                        f"{existing.target.id}, which does not exist: the two "
                        f"are written in one transaction, so this is a "
                        f"corrupted record rather than a retry"
                    )
                return stored_attempt, existing

            experiment_id = self._experiment_of_run(str(attempt.run_id))
            self.aggregates._insert_attempt(attempt)
            stored = self.operations._insert(operation)
            self._emit(attempt, "RunAttemptCreated", experiment_id, actor, destinations)
            self._emit_operation(stored, "RuntimeOperationIntended", experiment_id, actor)

        return attempt, stored

    def confirm_operation(
        self,
        operation: RuntimeOperation,
        *,
        actor: Actor,
        runtime_ref: RuntimeRef | None = None,
    ) -> RuntimeOperation:
        """Record that an operation took effect, with the runtime's reference."""
        return self._settle(operation, "confirmed", actor, runtime_ref=runtime_ref)

    def mark_operation_sent(self, operation: RuntimeOperation, *, actor: Actor) -> RuntimeOperation:
        """Record that the request was issued but its outcome is not yet known."""
        return self._settle(operation, "sent", actor)

    def fail_operation(self, operation: RuntimeOperation, *, actor: Actor) -> RuntimeOperation:
        """Record that the request definitively did not take effect.

        Only for a **known** negative outcome. A lost response is not a failure:
        it stays ``INTENDED`` or ``SENT`` so the reconciler keeps looking, which
        is the difference between "it did not happen" and "we do not know"
        (ADR-005 §6).
        """
        return self._settle(operation, "failed", actor)

    # ---- ADR-005 §5 ----------------------------------------------------

    def request_cancellation(
        self,
        target: ActionTarget,
        *,
        reason: str,
        actor: Actor,
        request_digest: str,
        action_id: ActionId | None = None,
        operation_id: OperationId | None = None,
        destinations: tuple[str, ...] = (),
    ) -> tuple[Action, RuntimeOperation | None]:
        """Record intent to cancel, and the effect it causes, in one commit.

        This is the §5 unit, and it could not exist before PR-006a: the
        ``Action`` owns the durable intent while the ``RuntimeOperation`` carries
        the external effect, and splitting them produces two states ADR-013's
        model cannot represent -- an intent nothing will ever act on, or an
        effect with no recorded cause.

        Cancellation is intent plus an observed terminal state, never a status
        the target occupies (ADR-013 §4). A workload that finishes first is not
        a failure: the action `SUCCEEDED` with outcome `SUPERSEDED`, because it
        did what it was asked and the answer was that there was nothing left to
        stop.

        Only attempt targets are accepted. Run- and experiment-level
        cancellation is a saga over descendants (ADR-013 §6) and is refused
        here rather than half-built.

        Returns:
            The action, and the operation it caused -- or ``None`` when the
            target was already terminal, in which case no external effect is
            requested and the action resolves immediately.

        Raises:
            CancellationSagaRequiredError: If the target is a run or experiment.
            UnknownOperationTargetError: If the target does not exist.
        """
        if target.kind not in _ATTEMPT_KINDS:
            raise CancellationSagaRequiredError(target.kind)

        action = Action(
            id=action_id or ActionId.generate(),
            experiment_id=ExperimentId(self._experiment_of_action_target(target)),
            type=_CANCEL_TYPE_FOR[target.kind],
            target=target,
            proposed_by=actor,
            reason=reason,
        )

        with write_transaction(self._connection):
            experiment_id = str(action.experiment_id)

            # Validation happens before any effect is requested, so an action
            # that cannot be carried out never mints an operation.
            validating = action.with_status(ActionStatus.VALIDATING)
            validated = validating.with_status(ActionStatus.VALIDATED)

            if self._target_is_terminal(target):
                # Nothing to stop. Resolve without an external effect rather
                # than asking a runtime to cancel a workload that has ended.
                executing = validated.with_status(ActionStatus.EXECUTING)
                settled = executing.with_status(
                    ActionStatus.SUCCEEDED, outcome=ActionOutcome.SUPERSEDED
                )
                self.actions._insert(settled)
                self._emit_action(settled, "CancellationSuperseded", actor, destinations)
                return settled, None

            executing = validated.with_status(ActionStatus.EXECUTING)
            self.actions._insert(executing)

            operation_target = RuntimeOperationTarget.model_validate(
                {"kind": target.kind, "id": target.id}
            )
            operation = RuntimeOperation(
                id=operation_id or OperationId.generate(),
                target=operation_target,
                type="cancel",
                request_digest=request_digest,
                caused_by_action_id=executing.id,
            )
            stored = self.operations._insert(operation)

            self._emit_action(executing, "CancellationRequested", actor, destinations)
            self._emit_operation(stored, "RuntimeOperationIntended", experiment_id, actor)

        return executing, stored

    def resolve_cancellation(
        self,
        action: Action,
        *,
        outcome: ActionOutcome,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> Action:
        """Settle a cancellation action once its outcome is observed.

        ``APPLIED`` when the workload stopped because of it, ``SUPERSEDED``
        when it had already ended, ``NOOP`` when there was nothing to do.
        """
        settled = action.with_status(ActionStatus.SUCCEEDED, outcome=outcome)
        with write_transaction(self._connection):
            self.actions._update(settled)
            self._emit_action(
                settled, f"Cancellation{outcome.value.capitalize()}", actor, destinations
            )
        return settled

    # ---- machinery ------------------------------------------------------

    def _settle(
        self,
        operation: RuntimeOperation,
        state: Literal["sent", "confirmed", "failed"],
        actor: Actor,
        *,
        runtime_ref: RuntimeRef | None = None,
    ) -> RuntimeOperation:
        moved = operation.with_state(state, runtime_ref=runtime_ref)
        with write_transaction(self._connection):
            experiment_id = self._experiment_of_operation(moved)
            self.operations._update(moved)
            self._emit_operation(
                moved, f"RuntimeOperation{state.capitalize()}", experiment_id, actor
            )
        return moved

    def _update(self, aggregate: AggregateModel) -> None:
        name = type(aggregate).__name__
        if name not in _AGGREGATE_TABLES:
            raise StorageError(f"{name} has no persistence mapping")
        writer = {
            "Experiment": self.aggregates._update_experiment,
            "ExperimentNode": self.aggregates._update_node,
            "Run": self.aggregates._update_run,
            "RunAttempt": self.aggregates._update_attempt,
        }[name]
        writer(aggregate)  # type: ignore[operator, arg-type]

    def _experiment_of_action_target(self, target: ActionTarget) -> str:
        """Resolve the experiment an action's target belongs to."""
        if target.kind == "experiment":
            if self.aggregates.get_experiment(target.id) is None:
                raise UnknownOperationTargetError(target.kind, target.id)
            return target.id
        if target.kind == "run":
            return self._experiment_of_run(target.id)
        if target.kind == "training-attempt":
            attempt = self.aggregates.get_attempt(target.id)
            if attempt is None:
                raise UnknownOperationTargetError(target.kind, target.id)
            return self._experiment_of_run(str(attempt.run_id))
        # node / evaluation-run / evaluation-attempt: the first has no
        # cancellation type yet, the others wait for ADR-015's tables.
        raise UnknownOperationTargetError(target.kind, target.id)

    def _target_is_terminal(self, target: ActionTarget) -> bool:
        """Whether there is anything left to cancel.

        Observed, not assumed. ADR-013 §5: a workload that reached a terminal
        state before the cancel was issued wins, and the action is superseded
        rather than failed.
        """
        if target.kind == "training-attempt":
            attempt = self.aggregates.load_attempt(target.id)
            return attempt.is_terminal
        if target.kind == "run":
            return self.aggregates.load_run(target.id).is_terminal
        if target.kind == "experiment":
            return self.aggregates.load_experiment(target.id).is_terminal
        raise UnknownOperationTargetError(target.kind, target.id)

    def _emit_action(
        self,
        action: Action,
        event_type: str,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> DomainEvent:
        event = DomainEvent(
            id=EventId.generate(),
            experiment_id=str(action.experiment_id),
            aggregate_type="Action",
            aggregate_id=str(action.id),
            aggregate_revision=action.revision,
            event_type=event_type,
            actor=actor,
            payload=FrozenDict(
                {
                    "type": action.type,
                    "status": action.status.value,
                    "outcome": action.outcome.value if action.outcome else None,
                    "target_kind": action.target.kind,
                    "target_id": action.target.id,
                }
            ),
        )
        return self._write_event(event, destinations)

    def _owning_experiment(self, aggregate: AggregateModel) -> str:
        """Return the experiment an aggregate belongs to, from the record itself.

        Derived rather than accepted from the caller, so an event cannot be
        filed under an experiment that does not own the thing it describes.
        """
        if isinstance(aggregate, Experiment):
            return str(aggregate.id)
        if isinstance(aggregate, (ExperimentNode, Run)):
            return str(aggregate.experiment_id)
        if isinstance(aggregate, RunAttempt):
            return self._experiment_of_run(str(aggregate.run_id))
        raise StorageError(f"{type(aggregate).__name__} has no owning experiment")

    def _experiment_of_run(self, run_id: str) -> str:
        """Resolve a run's experiment, which an attempt does not carry."""
        row = self._connection.execute(
            "SELECT experiment_id FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise AggregateNotFoundError("Run", run_id)
        return str(row["experiment_id"])

    def _experiment_of_operation(self, operation: RuntimeOperation) -> str:
        """Resolve an operation's experiment through the attempt it targets."""
        table = _TARGET_TABLES.get(operation.target.kind)
        if table != "run_attempts":
            raise UnknownOperationTargetError(operation.target.kind, operation.target.id)
        row = self._connection.execute(
            "SELECT run_id FROM run_attempts WHERE id = ?", (operation.target.id,)
        ).fetchone()
        if row is None:
            raise UnknownOperationTargetError(operation.target.kind, operation.target.id)
        return self._experiment_of_run(str(row["run_id"]))

    def _require_target(self, target: RuntimeOperationTarget) -> None:
        """Enforce ADR-005 §10.1, which SQLite cannot."""
        table = _TARGET_TABLES[target.kind]
        try:
            row = self._connection.execute(
                f"SELECT 1 FROM {table} WHERE id = ?",  # noqa: S608 - table from a literal map
                (target.id,),
            ).fetchone()
        except sqlite3.OperationalError as error:
            # The table is not in the schema yet (evaluation_attempts, ADR-015).
            raise UnknownOperationTargetError(target.kind, target.id) from error
        if row is None:
            raise UnknownOperationTargetError(target.kind, target.id)

    def _emit(
        self,
        aggregate: AggregateModel,
        event_type: str,
        experiment_id: str,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> DomainEvent:
        event = DomainEvent(
            id=EventId.generate(),
            experiment_id=experiment_id,
            aggregate_type=type(aggregate).__name__,
            aggregate_id=str(aggregate.id),  # type: ignore[attr-defined]
            aggregate_revision=aggregate.revision,  # type: ignore[attr-defined]
            event_type=event_type,
            actor=actor,
            payload=FrozenDict(
                {"status": getattr(getattr(aggregate, "status", None), "value", None)}
            ),
        )
        return self._write_event(event, destinations)

    def _emit_operation(
        self,
        operation: RuntimeOperation,
        event_type: str,
        experiment_id: str,
        actor: Actor,
        destinations: tuple[str, ...] = (),
    ) -> DomainEvent:
        event = DomainEvent(
            id=EventId.generate(),
            experiment_id=experiment_id,
            aggregate_type="RuntimeOperation",
            aggregate_id=str(operation.id),
            aggregate_revision=operation.revision,
            event_type=event_type,
            actor=actor,
            payload=FrozenDict(
                {
                    "state": operation.state,
                    "type": operation.type,
                    "target_kind": operation.target.kind,
                    "target_id": operation.target.id,
                }
            ),
        )
        return self._write_event(event, destinations)

    def _write_event(self, event: DomainEvent, destinations: tuple[str, ...]) -> DomainEvent:
        self.events._append(event)
        now = utc_now()
        for destination in destinations:
            self.events._enqueue(
                OutboxRecord(
                    id=f"obx_{uuid.uuid4().hex}",
                    event_id=event.id,
                    destination=destination,
                    created_at=now,
                    updated_at=now,
                )
            )
        return event
