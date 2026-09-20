# Persistence, Events, Outbox, and Concurrency

## 1. Design

Do not implement independent state writes and event writes.

Use a single repository transaction for local storage.

SQLite is authoritative for the MVP.

## 2. Tables

Recommended initial schema:

```text
experiments
experiment_nodes
runs
run_attempts
actions
incidents
decisions
evaluations
artifacts
checkpoints
budget_ledger
training_interventions
intervention_applications
runtime_operations
events
outbox
controller_leases
```

Each mutable aggregate table has:

```text
id
state/status
revision
created_at
updated_at
payload_json
```

## 3. Atomic transition

Example:

```text
BEGIN IMMEDIATE;

SELECT revision FROM run_attempts WHERE id = ?;

validate expected revision
validate transition

UPDATE run_attempts
SET status = ?,
    revision = revision + 1,
    payload_json = ?
WHERE id = ? AND revision = ?;

INSERT INTO events (...);

INSERT INTO outbox (...);

COMMIT;
```

If the revision update affects zero rows, raise `ConcurrentModificationError`.

## 4. Repository API

```python
class ExperimentRepository(Protocol):
    def create_experiment(...) -> Experiment:
        ...

    def transition_experiment(
        self,
        id: ExperimentId,
        expected_revision: int,
        transition: ExperimentTransition,
    ) -> Experiment:
        ...

    def transition_node(...):
        ...

    def transition_run(...):
        ...

    def transition_attempt(...):
        ...

    def append_metric(...):
        ...

    def record_incident(...):
        ...
```

The repository emits events as part of transitions.

## 5. Event

```python
class Event(BaseModel):
    id: EventId
    sequence: int

    aggregate_type: str
    aggregate_id: str
    aggregate_revision: int

    event_type: str

    experiment_id: str

    occurred_at: datetime

    actor: Actor

    payload: dict[str, Any]

    schema_version: str
```

## 6. Event ordering

Guarantees:

- total ordering per SQLite database through `sequence`
- strict revision ordering per aggregate
- no guarantee of wall-clock ordering across remote runtime sources

Consumers should use `sequence`.

## 7. Outbox

Purpose:

- MLflow sink
- W&B sink
- external event bus
- webhook integration
- remote monitoring

Outbox row:

```python
class OutboxRecord:
    id: str
    event_id: str
    destination: str
    state: Literal["pending", "sending", "sent", "failed"]
    attempts: int
    next_attempt_at: datetime | None
```

Core state does not depend on outbox delivery success.

## 8. External event bus

Optional later:

- Kafka
- NATS
- Redis streams

External bus consumes from the outbox.

Do not make distributed messaging required for core execution.

## 8b. Intervention applications are events

An `InterventionApplication` is appended to the event stream, not stored as mutable
state (ADR-011). Three consequences for the schema:

- `event.sequence` is the canonical order. `TrainingPosition` is recorded alongside it
  but is not monotonic, because a restore rewinds it.
- `RunRealizationFingerprint` hashes the ordered applications, so a run that re-applied
  an intervention after a rollback is distinguishable from one that did not.
- `RunRealization` is a projection over these events. The event log is authoritative and
  the projection must be recomputable from it; a stored projection that disagrees with a
  rebuild is a provenance bug.

The tables list therefore gains `training_interventions` and `intervention_applications`.

## 8c. The operation journal is not the outbox

Both record work that happens outside the transaction, and they must not share a
mechanism (ADR-013). The outbox publishes events outward and its records are safe to
redeliver, because sinks are idempotent consumers. The operation journal records effects
the controller initiates on a runtime, and redelivering one of those may start a second
workload.

The tables list therefore gains `runtime_operations`, written in the same transaction as
the attempt it belongs to, so the *intent* is durable before the effect happens.

## 9. Snapshots

Do not implement full event sourcing in the MVP.

Materialized SQLite state is authoritative.

Events are the durable audit/provenance stream.

## 10. Controller leases

For LocalDaemon:

```text
controller_id
heartbeat_at
lease_expires_at
```

Prevent two local daemons controlling the same database.

Remote distributed controller locking is deferred.

## 11. Database migrations

Use a migration mechanism.

Requirements:

- monotonic schema version
- forward migration
- migration test from prior release fixture
- no destructive migration without explicit release note

## 12. Crash tests

Required integration cases:

1. crash after transition before outbox delivery
2. crash during runtime submit before runtime ID persist
3. runtime accepts idempotent operation but controller dies
4. crash after event insert but before transaction commit
5. concurrent transition with stale revision
6. controller restart during recovery
