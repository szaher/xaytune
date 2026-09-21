# ADR-005 — The persistence transaction contract

## Status
Proposed — expanded 2026-09-21. **This is the gate on band B, including PR-004.**

The original version of this ADR was fifteen lines describing one transaction:
validate revision, update aggregate, increment revision, insert event, insert
outbox, commit. That was adequate when band B owned `Experiment`, `Node`, `Run`,
`RunAttempt`, events and the outbox.

It no longer is. Band B now also owns `RuntimeOperation` (ADR-013), the minimal
`Action` substrate and its linkage to operations (PR-006a), operation and
cancellation intent, `RunRealization` projections (ADR-011), and telemetry
generation durability (ADR-014). Accepting the short version would have frozen
the schema while leaving the transaction boundaries that matter unstated — and
those boundaries are now the highest-risk decision in the project, because a
missing one is only observable after a crash.

## Context

Everything downstream assumes that a controller can die at any instant without
the record becoming a lie. The specific failures this ADR exists to prevent are
all of one shape: **two facts that must be true together, written separately.**

```text
state says RUNNING          event log says nothing
operation says SENT         no attempt owns it
action says "cancel"        no operation carries the request
operation exists            nothing records why
artifact exists             no event says it was produced
```

Each is unrecoverable by inspection after the fact, because there is no way to
tell a missing write from a write that was never intended.

## Decision

### 1. Authority model

**SQLite materialized state is authoritative. The event log is durable
provenance, not an event-sourced projection source.**

This is not event sourcing. Aggregates are read from their tables, not folded
from events. The event log exists so that history is answerable and so that
projections can be rebuilt and checked — not because state is derived from it.

The one exception is `RunRealization`, which *is* a projection (§7) and is
required to be recomputable from events.

Stating this matters because the two designs have different failure modes, and a
repository that is ambiguous about which it is will eventually be written as
both.

### 2. Optimistic concurrency

Every aggregate carries `revision`. Every write:

```text
UPDATE ... WHERE id = ? AND revision = ?
```

A zero-row update is a lost race and raises `ConcurrentModificationError`. The
caller re-reads and retries; it never re-issues a write with a stale revision,
and it never reads-then-writes without the guard.

`revision` increments by exactly one per committed transition, and the event
written in that transaction records the revision it produced. That pairing is
what makes §10's consistency check possible.

### 3. Atomic aggregate transition

The original decision, unchanged and still the base case:

```text
BEGIN IMMEDIATE
  validate expected revision
  update aggregate state
  increment revision
  insert DomainEvent
  insert OutboxRecord(s)
COMMIT
```

No transition writes state without its event. No event is written for a
transition that did not commit.

### 4. Atomic external-effect intent

**Intent is durable before the effect is attempted.**

```text
BEGIN IMMEDIATE
  create RunAttempt
  create RuntimeOperation(state=INTENDED, request_digest=...)
  insert DomainEvents
  insert OutboxRecords
COMMIT

runtime.submit_or_get(operation_id, plan)      <- the only step outside a transaction

BEGIN IMMEDIATE
  RuntimeOperation -> CONFIRMED
  persist RuntimeRef
  insert DomainEvent
COMMIT
```

A crash between the first commit and the runtime call leaves an operation in
`INTENDED` — which ADR-013 defines as "reconcile, do not re-issue blindly". A
crash after the call and before the second commit leaves it in `INTENDED` or
`SENT`, and `lookup_operation()` resolves what actually happened. Neither state
is ambiguous, and neither requires guessing.

### 5. Action–effect atomicity

An `Action` records desired intent; a `RuntimeOperation` records the external
effect. **They commit together.**

```text
BEGIN IMMEDIATE
  create/transition Action
  create the RuntimeOperation(s) that Action causes, in INTENDED
  link operation.caused_by_action_id
  insert DomainEvents
  insert OutboxRecords
COMMIT
```

Splitting them produces two states that ADR-013's model cannot represent:

```text
Action says cancellation was requested, but no operation exists
    -> the intent is durable and will never be acted on; nothing retries it,
       because nothing is unresolved

a cancel operation exists, but no Action records why
    -> an external effect with no recorded cause, which is precisely what the
       Action aggregate exists to prevent
```

The general form, which is the invariant to implement and test:

> **Any durable controller intent that implies an external side effect must
> commit the intent record and the corresponding `INTENDED` `RuntimeOperation`
> in the same transaction, before the external call.**

This covers cancellation today and every mutating action type from Phase 4
onwards without restating the rule per action.

### 6. Idempotency

Creation and transition are both idempotent, and they fail differently:

```text
same operation_id, same target/type/request_digest   -> return the existing record
same operation_id, anything else different           -> IdempotencyConflict
transition to the state already held                 -> no-op, revision unchanged
transition along an edge the machine forbids         -> InvalidTransition
```

An unresolved outcome stays `INTENDED`/`SENT`. **A lost response is never
recorded as `FAILED`**, because that would convert "we do not know" into "it did
not happen", and the reconciler would stop looking.

Event handlers are idempotent on `(target, stream_generation, sequence)` per
ADR-014, so a redelivered `CheckpointCommitted` does not produce two checkpoint
records.

### 7. Projection consistency

`RunRealization` is a projection over the event log (ADR-011) and must satisfy:

```python
assert repository.get_run_realization(run_id) == projector.rebuild_run_realization(
    repository.events_for_run(run_id)
)
```

A stored projection that disagrees with a rebuild is a **provenance bug, not a
cache bug**, and the test above is required rather than advisory. Materializing
`final_effective_state` as independently mutable state is forbidden: it
reintroduces exactly the divergence this ADR exists to prevent.

Both run fingerprints (`RunHistoryFingerprint`, `ArtifactLineageFingerprint`)
are computed from the projection and are provisional until the run is terminal.

### 8. Transaction isolation and writer serialization

```text
journal_mode = WAL
synchronous  = FULL
foreign_keys = ON
busy_timeout set explicitly
```

Every write transaction uses **`BEGIN IMMEDIATE`**, not the default deferred
begin. SQLite's deferred transactions take the write lock lazily, so a
read-then-write transaction can fail with `SQLITE_BUSY` at commit time after its
logic has already run — turning a clean serialization failure into a late,
partial-feeling one. Taking the lock up front makes contention a fast, obvious
failure that the retry path handles.

**Exactly one writer process per database.** The singleton lease of ADR-004
enforces it at the host level; the repository does not attempt multi-writer
coordination, and this assumption must be stated wherever a remote controller is
later proposed.

### 9. Failure semantics

Four crash points, each with a defined resolution:

| Crash | State | Resolution |
|---|---|---|
| Before commit | nothing written | nothing happened; no reconciliation needed |
| After commit, before external call | operation `INTENDED` | reconcile: `lookup_operation()`, then submit or adopt |
| After external call, before confirm | `INTENDED`/`SENT`, effect may exist | `lookup_operation()`; if the adapter cannot report completed operations, escalate rather than resubmit (ADR-013 §3) |
| After confirm | consistent | reattach to the running workload (PR-012a) |

The asymmetry is deliberate: it is always safe to have durable intent with no
effect, and never safe to have an effect with no durable intent. Every boundary
above is arranged so that the first is the only possible inconsistency.

### 10. Repository invariants

Checkable, and checked in tests rather than assumed:

1. A `RuntimeOperation`'s `target_id` exists in the table named by its
   `target_kind`. SQLite cannot enforce this across heterogeneous targets
   (ADR-013), so the repository does.
2. An aggregate's `revision` equals the revision recorded by its latest event.
3. Terminal records are immutable. A transition out of a terminal state is
   rejected, never silently ignored.
4. No external side effect exists without a durable `INTENDED` record that
   preceded it.
5. Every `RuntimeOperation` caused by an `Action` carries
   `caused_by_action_id`, and that action exists.
6. Outbox consumers publish events. **They never submit or cancel workloads** —
   an at-least-once outbox driving a runtime call would duplicate side effects,
   which is the failure ADR-013 exists to prevent.

### 11. Required crash and concurrency tests

Persistence is only as good as the tests that kill it at the wrong moment:

1. state, event and outbox commit together; failure between them rolls back all
2. stale revision raises `ConcurrentModificationError`
3. concurrent writers: one wins, one retries, no lost update
4. attempt + `INTENDED` operation are atomic; injected failure rolls back both
5. `Action` + caused operation are atomic; neither can exist alone
6. committed intent and `request_digest` survive a database reopen
7. duplicate operation creation is idempotent; a conflicting request is rejected
8. illegal transitions and stale revisions are rejected atomically
9. cancellation intent survives restart independently of observed attempt status
10. an operation targeting an evaluation attempt behaves identically to one
    targeting a training attempt
11. `RunRealization` rebuilt from events equals the stored projection
12. a lost runtime response leaves the operation unresolved, never `FAILED`

## Consequences

- **This ADR must be accepted before band B starts, including PR-004.** PR-004
  writes the tables and revision semantics; deferring acceptance to PR-005 puts
  the decision one PR after the point where it freezes.
- The repository is opinionated. Callers cannot write state without an event, or
  request an external effect without durable intent, because the API does not
  offer those operations separately.
- Band B is larger than it was: `runtime_operations` (migration 001) and
  `actions` (migration 002) are both prerequisites for Phase 2 cancellation.
- External sinks are eventually consistent through the outbox. That is a
  deliberate trade: the alternative is a distributed transaction with a system
  that may be down.
- Single-writer is assumed. A future remote controller must revisit §8 rather
  than inherit it silently.

## Rejected alternatives

**Independent StateStore and EventStore writes.** The original motivation for
this ADR: two stores that can diverge after a crash, with no way to tell which
one is right.

**Full event sourcing.** Rebuilding every aggregate from events on every read is
a larger change than the problem requires, and it makes the common query path
pay for the rare audit path. Projections are used where they earn their keep
(§7) and nowhere else.

**Recording a lost response as `FAILED`.** Convenient, and wrong: it discards
the distinction between "did not happen" and "unknown", and the reconciler stops
looking exactly when it should be looking hardest.

**Letting the outbox drive runtime submission.** Tempting, because the outbox
already guarantees at-least-once delivery. At-least-once is correct for
publishing an event and catastrophic for starting a GPU job.
