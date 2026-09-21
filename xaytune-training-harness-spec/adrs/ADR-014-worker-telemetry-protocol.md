# ADR-014 — The worker telemetry protocol

## Status
Accepted — 2026-09-21.

Defines `xaytune.telemetry/v1alpha1`, which `TrainingExecutionSpec.telemetry`
already names and which nothing specified. Required by PR-005 (the event
schema) and PR-011 (`LocalRuntime`).

## Context

`TrainingExecutionSpec` carries a `TelemetryContract` naming
`xaytune.telemetry/v1alpha1`, and `RuntimeBackend.watch()` returned an
undefined `AsyncIterator[RuntimeEvent]`. Neither the event type nor the
stream's semantics were defined anywhere. This ADR defines them, and
`watch()` now returns `AsyncIterator[WorkerEvent]`.

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
    event_id: str                # ULID-shaped, unique per stream position
    run_id: RunId
    attempt_id: AttemptId
    stream_generation: int       # which telemetry stream of this attempt (see 1a)
    sequence: int                # monotonic within the generation, from 0, no gaps
    emitted_at: datetime         # worker clock, advisory only
    type: WorkerEventType
    payload: FrozenDict
```

`sequence` is the contract, not `emitted_at`. Worker clocks are not
trustworthy — they skew, they jump, and under a distributed launcher there are
several of them. Ordering, deduplication and gap detection all key on
`(attempt_id, stream_generation, sequence)`, ordered lexicographically: a
higher generation is always later than any sequence of a lower one.

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
controller            dedups on (attempt_id, stream_generation, sequence)
```

Only the supervisor emits canonical `WorkerEvent`s. Ranks report to it in
whatever form the runtime finds convenient; that channel is **not** part of this
protocol and runtimes may implement it differently.

Consequences worth stating plainly:

- Per-rank information that matters must be carried in the payload — a rank
  identifier on a metric, for example — not inferred from who emitted the event.
- Losing the supervisor is losing telemetry for the attempt, not losing the
  attempt. That is the correct blast radius: the controller detects the gap and
  reconciles from `get_status()`, which is the same path a controller restart
  takes.
- `sequence` says nothing about wall-clock simultaneity across ranks, and no
  consumer may assume it does.

#### If the supervisor dies

A replacement supervisor that continues the same stream **must recover the next
sequence from durable or runtime-retained state.** Restarting at 0 within the
same generation would re-issue keys the controller has already applied, and
since deduplication is on `(attempt_id, stream_generation, sequence)`, the
controller would silently discard the new events as duplicates — the worst
available outcome, because it looks like silence rather than an error.

Recovering the counter is also **not sufficient**. Knowing that the next
sequence is 42 says nothing about 41:

```text
supervisor emits seq 41
controller receives it
controller crashes before committing it
supervisor crashes
```

The controller's durable cursor is still 40, so on reconnect it asks for
everything after 40 — and 41 exists only in the memory of a process that is
gone. So continuing a generation requires history, not just a counter:

> Continuing a telemetry stream generation requires producer- or
> runtime-retained event history sufficient to replay **every event after the
> controller's durable cursor**, not merely recovery of the next counter value.

#### When continuity is lost: a new generation, not a new attempt

Where that history cannot be retained, the stream cannot be continued. **The
attempt is not what ended.** A `RunAttempt` is one infrastructure/runtime
execution attempt (`03-domain-model.md` §5), and the failure being handled here
is observability:

```text
training process still running normally
        │
telemetry supervisor dies
        │
event history unavailable
```

Minting `attempt_2` here would record an execution retry that never happened.
It would corrupt the record it is meant to protect — retry counts, per-attempt
resource usage, incident attribution and the recovery history all become
fiction — and it is a strictly worse version of the mechanism this ADR already
has for exactly this case: a declared gap, reconciliation from `get_status()`,
and a recorded interval of degraded observability.

So the two failures are separated, because they are different failures:

```text
telemetry stream ends, workload still executing
    same RunAttempt
    stream_generation += 1, sequence restarts at 0
    EventGapDetected naming the unrecoverable range
    the interval is marked observability-degraded in provenance
    reconcile runtime and checkpoint state from get_status()

runtime restarts or replaces the workload
    new RunAttempt (as it always was)
    stream_generation starts again at 0
```

The controller confirms which case it is through `get_status()` and
`lookup_operation()` before deciding — it does not infer a workload restart
from a dead stream.

**The generation counter is assigned by the controller-side adapter, not by the
worker.** This is deliberate: worker-side durable state is precisely what could
not be guaranteed, so requiring the replacement supervisor to remember its
generation would reintroduce the problem one level down. The controller knows
how many streams it has attached to this attempt because it is the thing that
attaches, so `RunAttempt.telemetry_generation` is durable controller state and
a fresh supervisor may safely start at sequence 0.

Before starting or attaching a replacement telemetry supervisor, the
controller/runtime adapter durably allocates the next generation if continuity
was lost, or reuses the current generation if complete replay is available. It
passes that assigned generation in the startup/attachment contract. The
supervisor includes it unchanged in every `WorkerEvent` and allocates only the
sequence within that generation.

```text
RunAttempt        execution history
telemetry stream  observability history
```

These are related but not the same history, and collapsing one into the other
loses both. That is why the key is three-part rather than two.

Two alternatives were rejected. **Per-producer sequences**
(`producer_id` + `producer_sequence`) push ordering into every consumer and give
the controller no total order for the attempt — which is the one thing it
actually needs. `stream_generation` is not that: generations are sequential, so
`(generation, sequence)` is still a total order.

**A new `RunAttempt` per stream discontinuity** — the earlier draft of this
section — was rejected for the reason above: it makes `RunAttempt` mean two
things at once and writes a false execution history to avoid a key collision
that a generation counter resolves for free.

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
`(attempt_id, stream_generation, sequence)`.** Exactly-once delivery across a process boundary
that can fail on either side is not available, and pretending otherwise puts
the burden in the wrong place. The worker may re-emit after a reconnect; the
controller must be idempotent.

This means every event handler must be idempotent. `CheckpointCommitted`
arriving twice must not create two checkpoint records.

### 4. Cursor and reconnect

`watch(runtime_ref, cursor)` resumes after the given cursor:

```python
class StreamCursor(FrozenDomainModel):
    generation: int
    sequence: int
```

It is the last position the controller durably recorded — **not** the last it
received. The distinction is the entire point: an event that was received and
then lost in a crash must be redelivered.

The cursor carries the generation because the sequence alone stopped being
unique within an attempt once generations existed. It remains a pair of
integers with defined meaning rather than an opaque provider token: a runtime
cannot smuggle its own pagination state through it, which is what a `str`
cursor invited.

A runtime that cannot replay from a cursor declares
`supports_event_replay: false` in its `CapabilityDocument`. The controller then
treats a reconnect as a **gap**, reconciles by polling `get_status()`, and
records that observability was degraded for that interval — rather than
silently assuming nothing happened while it was away.

### 5. Gap detection

Gap detection rests on an ordering guarantee, so state it first:

> `RuntimeBackend.watch()` MUST yield canonical `WorkerEvent`s in increasing
> `(stream_generation, sequence)` order. The runtime adapter buffers out-of-order transport delivery
> until the missing sequence arrives, or until the replay/gap policy determines
> it is unavailable.

Without it, a transport that delivers `11, 13, 12` makes the controller declare
`12` lost at the moment it sees `13`, and raise an incident for an event that
arrives a millisecond later. Ordering is the adapter's job precisely because it
is the only component that knows its transport's reordering behaviour.

Given ordering, a `sequence` jump within a generation means events were lost.
The controller must not interpolate. It records an `EventGapDetected` incident
naming the missing range and reconciles from `get_status()` and the checkpoint
store, which are authoritative in a way the stream is not.

A **generation advance** is the same signal at a larger scale: everything after
the durable cursor in the previous generation is unrecoverable. It records the
gap, reconciles identically, and marks the interval observability-degraded in
provenance — so a later reader can tell "nothing happened" from "we were not
watching".

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

Within a generation, `sequence` is total; across generations of one attempt,
`(generation, sequence)` is total. **Across attempts there is no ordering**, and
none is needed: attempts are independent executions and the controller already
orders them through the RunAttempt state machine.

## Consequences

- `RuntimeBackend.watch()` becomes implementable, and two runtimes implementing
  it will agree.
- The controller's event handlers must all be idempotent. This is a real
  constraint on PR-005 and is easy to violate.
- `RunAttempt` gains `telemetry_generation` — a column in migration 001, not a
  payload field, because the controller assigns it and reconciliation queries
  it — and the worker-event schema gains `stream_generation`. Both are frozen by
  the persistence band, which is why this belongs here rather than in the first
  runtime PR.
- A run's provenance can now say *we were not observing between here and here*,
  which it previously could only express by fabricating an attempt or by saying
  nothing at all.
- Runtimes that cannot replay are supported but honest about it, through a
  declared capability rather than a silent difference in behaviour.
- The protocol is versioned in the envelope, so `v1alpha2` can change payloads
  without breaking a controller that pins the major version (ADR-008).

## Acceptance criteria

1. Every `WorkerEvent` carries `protocol_version`, `event_id`, `run_id`,
   `attempt_id`, `stream_generation`, `sequence` and `type`.
1a. Exactly one telemetry supervisor per attempt assigns `sequence`; individual
   ranks never emit `WorkerEvent`s directly.
1b. A replacement supervisor continues the current generation only if it resumes
   the sequence from durable state **and** can replay every event after the
   controller's durable cursor. If it cannot do both, the controller advances
   `stream_generation` and the sequence restarts at 0.
1c. A lost telemetry stream **never** creates a `RunAttempt` on its own. Given a
   dead supervisor and a `get_status()` confirming the same workload is still
   executing, the attempt id is unchanged, the generation advances, and the
   interval is recorded as observability-degraded. A new `RunAttempt` is created
   only when the runtime restarts or replaces the workload.
1d. `stream_generation` is assigned from durable controller-side state
   (`RunAttempt.telemetry_generation`); a supervisor never has to remember it.
   The startup/attachment contract passes this assigned generation to the
   supervisor, which emits it unchanged on every event.
2. `sequence` is monotonic and gapless within a generation, starting at 0;
   `stream_generation` is monotonic within an attempt, starting at 0.
3. Duplicate `(attempt_id, stream_generation, sequence)` is a no-op in every
   handler.
4. `CheckpointCommitted` carries a complete `DataCursor` and `ResumeGuarantee`
   per ADR-012.
5. `watch(cursor=StreamCursor(g, n))` yields events ordered after `(g, n)`, or
   the runtime declares `supports_event_replay: false`.
5a. `watch()` yields in increasing `(generation, sequence)` order; a reordered
   transport is reassembled by the adapter and never surfaces as a gap.
6. A `sequence` gap or a generation advance raises `EventGapDetected` and
   triggers reconciliation; neither silently continues.
7. Missing heartbeats mark an attempt suspect and trigger `get_status()`; they
   never directly transition it to `FAILED`.
8. `IncidentObserved` never transitions a run to a terminal state by itself.
9. A controller restart mid-stream loses no durably recorded event and
   double-applies none.
