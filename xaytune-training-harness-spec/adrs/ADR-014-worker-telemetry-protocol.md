# ADR-014 — The worker telemetry protocol

## Status
Accepted — 2026-09-21.

Defines `xaytune.telemetry/v1alpha1`, which `TrainingExecutionSpec.telemetry`
already names and which nothing specified. Required by PR-005 (the event
schema) and PR-011 (`LocalRuntime`).

## Context

`TrainingExecutionSpec` carries a `TelemetryContract` naming
`xaytune.telemetry/v1alpha1`, and `RuntimeBackend.watch()` returns
`AsyncIterator[RuntimeEvent]`. Neither the event type nor the stream's
semantics were defined anywhere.

This is not a gap in documentation. It is a gap in the contract that everything
downstream depends on:

- checkpoint tracking — the controller learns a checkpoint exists only here
- progress and metrics — every number a planner decides on arrives this way
- incident detection — a divergence is an observation, not a process exit
- recovery — an attempt's resume position comes from its last committed event
- completion — the controller cannot mark a run `SUCCEEDED` on process exit
  alone, because a zero exit code does not mean the training converged

Without it, `watch()` is a type signature rather than an implementable
interface, and two runtimes would make incompatible assumptions about the same
stream.

The hard part is not the event shapes. It is that **the worker and the
controller fail independently**, and the stream between them is the only thing
that knows what happened.

## Decision

### 1. The event envelope

```python
class WorkerEvent(FrozenDomainModel):
    protocol_version: str        # "xaytune.telemetry/v1alpha1"
    event_id: str                # ULID-shaped, unique per (attempt, sequence)
    run_id: RunId
    attempt_id: AttemptId
    sequence: int                # monotonic per attempt, starts at 0, no gaps
    emitted_at: datetime         # worker clock, advisory only
    type: WorkerEventType
    payload: FrozenDict
```

`sequence` is the contract, not `emitted_at`. Worker clocks are not
trustworthy — they skew, they jump, and under a distributed launcher there are
several of them. Ordering, deduplication and gap detection all key on
`(attempt_id, sequence)`.

### 1a. Exactly one process assigns `sequence`

A gapless per-attempt sequence and a multi-rank workload are in tension: if
eight ranks each emit `sequence=15`, the key is not unique, and if they
coordinate before every event, telemetry has put a distributed sequencer on the
training hot path. Neither is acceptable, so the ownership is stated rather than
left to the implementer.

**One telemetry supervisor per attempt assigns the sequence.** In a distributed
launcher that is the driver — rank 0 or the equivalent coordinator:

```text
rank workers          raw local signals (not WorkerEvents)
      ↓
telemetry supervisor  assigns attempt-scoped sequence, emits WorkerEvent
      ↓
controller            dedups on (attempt_id, sequence)
```

Only the supervisor emits canonical `WorkerEvent`s. Ranks report to it in
whatever form the runtime finds convenient; that channel is **not** part of this
protocol and runtimes may implement it differently.

Consequences worth stating plainly:

- Per-rank information that matters must be carried in the payload — a rank
  identifier on a metric, for example — not inferred from who emitted the event.
- Losing the supervisor is losing telemetry for the attempt. That is the correct
  blast radius: the controller detects the gap and reconciles from
  `get_status()`, which is the same path a controller restart takes.
- `sequence` says nothing about wall-clock simultaneity across ranks, and no
  consumer may assume it does.

#### If the supervisor dies

The replacement supervisor for the same `RunAttempt` **must recover the next
sequence from durable or runtime-retained state.** Restarting at 0 would
re-issue keys the controller has already applied, and since deduplication is on
`(attempt_id, sequence)`, the controller would silently discard the new events
as duplicates — the worst available outcome, because it looks like silence
rather than an error.

Where sequence continuity cannot be guaranteed, the runtime **must create a new
`RunAttempt`** rather than resume the old attempt's stream. A new attempt is
cheap and honest; a reused attempt id with a restarted counter corrupts the
record.

This makes `(attempt_id, sequence)` a durable identity contract rather than an
in-process counter, which is what the controller's deduplication has been
assuming all along.

The rejected alternative was per-producer sequences
(`producer_id` + `producer_sequence`), which pushes ordering into every consumer
and gives the controller no total order for the attempt — which is the one thing
it actually needs.

### 2. Event families

```text
lifecycle     WorkerReady, TrainingStarted, TrainingCompleted, TrainingFailed
progress      StepCompleted, MetricObserved, Heartbeat
checkpoint    CheckpointStarted, CheckpointCommitted
incident      IncidentObserved
artifact      ArtifactProduced
```

`CheckpointCommitted` is the load-bearing one. It must carry the full
`DataCursor` and `ResumeGuarantee` of ADR-012, because it is the only point at
which the controller learns a resumable position exists. A checkpoint the
controller does not know about cannot be resumed from, however valid it is on
disk.

`IncidentObserved` is deliberately distinct from `TrainingFailed`. A loss
divergence, an OOM that was recovered internally, or a gradient overflow are
observations about a run that is still going. Collapsing them into failure is
how a recoverable condition becomes a lost run.

### 3. Delivery semantics

**At-least-once, with the controller deduplicating on
`(attempt_id, sequence)`.** Exactly-once delivery across a process boundary
that can fail on either side is not available, and pretending otherwise puts
the burden in the wrong place. The worker may re-emit after a reconnect; the
controller must be idempotent.

This means every event handler must be idempotent. `CheckpointCommitted`
arriving twice must not create two checkpoint records.

### 4. Cursor and reconnect

`watch(runtime_ref, cursor)` resumes after the given cursor. The cursor is the
last `sequence` the controller durably recorded — **not** the last it received.
The distinction is the entire point: an event that was received and then lost
in a crash must be redelivered.

A runtime that cannot replay from a cursor declares
`supports_event_replay: false` in its `CapabilityDocument`. The controller then
treats a reconnect as a **gap**, reconciles by polling `get_status()`, and
records that observability was degraded for that interval — rather than
silently assuming nothing happened while it was away.

### 5. Gap detection

A `sequence` jump means events were lost. The controller must not interpolate.
It records an `EventGapDetected` incident naming the missing range and
reconciles from `get_status()` and the checkpoint store, which are
authoritative in a way the stream is not.

Gaps are expected, not exceptional. They are what a controller restart looks
like from the stream's point of view.

### 6. Heartbeat and liveness

`Heartbeat` carries the worker's current `sequence` and the wall-clock interval
it expects between beats. Absence of heartbeats past that interval makes the
attempt **suspect**, not failed.

The controller must confirm through `get_status()` before acting, because the
common cause of missing heartbeats is a slow or partitioned network, and the
expensive mistake is killing a healthy run that was merely quiet. A worker
performing a long checkpoint write to remote storage can legitimately go
silent.

### 7. Ordering

Per attempt, `sequence` is total. **Across attempts there is no ordering**, and
none is needed: attempts are independent executions and the controller already
orders them through the RunAttempt state machine.

## Consequences

- `RuntimeBackend.watch()` becomes implementable, and two runtimes implementing
  it will agree.
- The controller's event handlers must all be idempotent. This is a real
  constraint on PR-005 and is easy to violate.
- Runtimes that cannot replay are supported but honest about it, through a
  declared capability rather than a silent difference in behaviour.
- The protocol is versioned in the envelope, so `v1alpha2` can change payloads
  without breaking a controller that pins the major version (ADR-008).

## Acceptance criteria

1. Every `WorkerEvent` carries `protocol_version`, `event_id`, `run_id`,
   `attempt_id`, `sequence` and `type`.
1a. Exactly one telemetry supervisor per attempt assigns `sequence`; individual
   ranks never emit `WorkerEvent`s directly.
1b. A replacement supervisor for the same attempt resumes the sequence from
   durable state; if it cannot, the runtime creates a new `RunAttempt` rather
   than restarting the counter.
2. `sequence` is monotonic and gapless per attempt, starting at 0.
3. Duplicate `(attempt_id, sequence)` is a no-op in every handler.
4. `CheckpointCommitted` carries a complete `DataCursor` and `ResumeGuarantee`
   per ADR-012.
5. `watch(cursor=N)` yields events with `sequence > N`, or the runtime declares
   `supports_event_replay: false`.
6. A `sequence` gap raises `EventGapDetected` and triggers reconciliation; it
   never silently continues.
7. Missing heartbeats mark an attempt suspect and trigger `get_status()`; they
   never directly transition it to `FAILED`.
8. `IncidentObserved` never transitions a run to a terminal state by itself.
9. A controller restart mid-stream loses no durably recorded event and
   double-applies none.
