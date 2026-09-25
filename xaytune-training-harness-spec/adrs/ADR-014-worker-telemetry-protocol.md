# ADR-014 — The worker telemetry protocol

## Status
Accepted — 2026-09-21.

Defines `xaytune.telemetry/v1alpha2`, which `TrainingExecutionSpec.telemetry`
already names and which nothing specified. Required by PR-005 (the event
schema) and PR-011 (`LocalRuntime`).

## Context

`TrainingExecutionSpec` carries a `TelemetryContract` naming
`xaytune.telemetry/v1alpha2`, and `RuntimeBackend.watch()` returned an
undefined `AsyncIterator[RuntimeEvent]`. Neither the event type nor the
stream's semantics were defined anywhere. This ADR defines them, and
`watch()` now returns `AsyncIterator[RuntimeEventEnvelope]`.

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

The envelope is workload-neutral; the payload is not.

```python
class RuntimeEventEnvelope(FrozenDomainModel):
    protocol_version: str        # "xaytune.telemetry/v1alpha2"
    event_id: str                # ULID-shaped, unique per stream position

    target: RuntimeOperationTarget   # ADR-013: training-attempt | evaluation-attempt

    stream_generation: int       # which telemetry stream of this target (see 1a)
    sequence: int                # monotonic within the generation, from 0, no gaps
    emitted_at: datetime         # worker clock, advisory only

    payload: RuntimeEventPayload
```

```python
RuntimeEventPayload = TrainingEventPayload | EvaluationEventPayload
```

An earlier version of this envelope carried `run_id` and `attempt_id` directly
and a training-shaped `type`. That made the protocol training-only, while
ADR-015 claimed it applied to evaluation unchanged — which it could not, because
an `EvaluationAttempt` has no `run_id`, produces no checkpoints and never emits
`TrainingStarted`.

The split is the point: **sharing an execution transport does not require
sharing a domain aggregate.** Delivery, ordering, deduplication, cursors,
generations and gap detection are properties of a stream and are identical for
both. What differs is what the events *say*, and that lives in the payload. This
keeps ADR-015 §2's refusal to introduce `Execution`/`ExecutionAttempt` intact:
there is a shared envelope, not a shared aggregate.

`target` is the same typed reference ADR-013 uses for the operation journal, so
an event stream and the operation that started it name their subject the same
way.

**The target travels with the plan.** `RuntimeBackend.submit_or_get` receives an
operation id and a `ResolvedExecutionPlan`, and nothing else, so the plan is the
only place a backend can learn which attempt it is running for. It is carried as
a field on the plan rather than passed beside it, which also puts it inside
`request_digest` — correct, because the same spec submitted for a different
attempt is a different request, and get-or-create must not hand the second
attempt the first one's running workload.

A backend that did not know its target could not emit a single valid envelope:
`target` is required, and the payload family is pinned to `target.kind`.

`sequence` is the contract, not `emitted_at`. Worker clocks are not
trustworthy — they skew, they jump, and under a distributed launcher there are
several of them. Ordering, deduplication and gap detection all key on
`(target, stream_generation, sequence)`, ordered lexicographically: a higher
generation is always later than any sequence of a lower one.

### 1a. Exactly one process assigns `sequence`

A gapless per-attempt sequence and a multi-rank workload are in tension: if
eight ranks each emit `sequence=15`, the key is not unique, and if they
coordinate before every event, telemetry has put a distributed sequencer on the
training hot path. Neither is acceptable, so the ownership is stated rather than
left to the implementer.

**One telemetry supervisor per target assigns the sequence.** In a distributed
launcher that is the driver — rank 0 or the equivalent coordinator:

```text
rank workers          raw local signals (not envelopes)
      ↓
telemetry supervisor  assigns target-scoped sequence, emits the envelope
      ↓
controller            dedups on (target, stream_generation, sequence)
```

Only the supervisor emits canonical envelopes. Ranks report to it in
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
since deduplication is on `(target, stream_generation, sequence)`, the
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
attaches, so the generation is durable controller state and a fresh supervisor
may safely start at sequence 0.

**Every attempt type owns its own generation counter**, because the envelope's
target is typed and evaluation attempts stream through the same protocol:

```text
RunAttempt.telemetry_generation
EvaluationAttempt.telemetry_generation
```

An earlier version named only `RunAttempt.telemetry_generation`, which left an
evaluation telemetry supervisor with nowhere durable to allocate a new
generation after a restart — the same mismatch as `RuntimeOperation.attempt_id`,
one layer later.

The generation is a property of the **target**, so the conceptual field is
`target.telemetry_generation` and each attempt aggregate stores it. A separate
`TelemetryStreamState` table keyed by target would also work and is cleaner if
more workload types arrive soon, but it adds a table and a join to buy
generality that ADR-015 §2 says we do not yet need. Revisit it with the same
trigger: a third workload type.

Before starting or attaching a replacement telemetry supervisor, the
controller/runtime adapter durably allocates the next generation if continuity
was lost, or reuses the current generation if complete replay is available. It
passes that assigned generation in the startup/attachment contract. The
supervisor includes it unchanged in every envelope and allocates only the
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

Some families are properties of any workload, so they live on the envelope side
and are emitted by both:

```text
stream        WorkerReady, Heartbeat
incident      IncidentObserved
artifact      ArtifactProduced
```

`TrainingEventPayload`:

```text
lifecycle     TrainingStarted, TrainingCompleted, TrainingFailed
progress      StepCompleted, MetricObserved
checkpoint    CheckpointStarted, CheckpointCommitted
```

`EvaluationEventPayload`:

```text
lifecycle     EvaluationStarted, EvaluationCompleted, EvaluationFailed
progress      EvaluationProgress, MetricObserved
```

Evaluation has no checkpoint family because it produces no checkpoints — the
same reason its state machine has no `CHECKPOINTING` (ADR-015 §1). A payload
union rather than one flat enum is what makes that absence a type error instead
of a convention.

`CheckpointCommitted` is the load-bearing training payload. It must carry the full
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
`(target, stream_generation, sequence)`.** Exactly-once delivery across a process boundary
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
unique within a target once generations existed. It remains a pair of
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

> `RuntimeBackend.watch()` MUST yield canonical envelopes in increasing
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

`Heartbeat` carries the wall-clock interval it expects between beats. Its
stream position is the enclosing `RuntimeEventEnvelope.sequence`, and the
payload does not repeat it: there is one canonical sequence per event, assigned
once by the telemetry supervisor (§1a), and a second copy could disagree with
it. Absence of heartbeats past the declared interval makes the attempt
**suspect**, not failed.

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
- Every attempt aggregate gains `telemetry_generation` — a column, not a payload
  field, because the controller assigns it and reconciliation queries it. It is
  in migration 001 for `run_attempts` and must be present on
  `evaluation_attempts` when ADR-015's tables land. The worker-event schema gains
  `stream_generation`. Both are frozen by the persistence band, which is why this
  belongs here rather than in the first runtime PR.
- A run's provenance can now say *we were not observing between here and here*,
  which it previously could only express by fabricating an attempt or by saying
  nothing at all.
- Runtimes that cannot replay are supported but honest about it, through a
  declared capability rather than a silent difference in behaviour.
- The protocol is versioned in the envelope, so `v1alpha2` can change payloads
  without breaking a controller that pins the major version (ADR-008).

## Acceptance criteria

1. Every `RuntimeEventEnvelope` carries `protocol_version`, `event_id`,
   `target`, `stream_generation`, `sequence` and a typed `payload`.
1e. The envelope is workload-neutral: an evaluation attempt streams through it
   without a `run_id`, a checkpoint family or a training lifecycle event.
1a. Exactly one telemetry supervisor per target assigns `sequence`; individual
   ranks never emit envelopes directly.
1b. A replacement supervisor continues the current generation only if it resumes
   the sequence from durable state **and** can replay every event after the
   controller's durable cursor. If it cannot do both, the controller advances
   `stream_generation` and the sequence restarts at 0.
1c. A lost telemetry stream **never** creates an attempt of any kind on its own. Given a
   dead supervisor and a `get_status()` confirming the same workload is still
   executing, the attempt id is unchanged, the generation advances, and the
   interval is recorded as observability-degraded. A new `RunAttempt` is created
   only when the runtime restarts or replaces the workload.
1d. `stream_generation` is assigned from durable controller-side state — the
   `telemetry_generation` of the targeted attempt, whichever attempt type it is;
   a supervisor never has to remember it.
   The startup/attachment contract passes this assigned generation to the
   supervisor, which emits it unchanged on every event.
2. `sequence` is monotonic and gapless within a generation, starting at 0;
   `stream_generation` is monotonic within a target, starting at 0.
3. Duplicate `(target, stream_generation, sequence)` is a no-op in every
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

## PR-009a — typed observation bodies

The family boundary is now executable at both levels:

```python
TrainingEventPayload(workload="training", data=TrainingObservation)
EvaluationEventPayload(workload="evaluation", data=EvaluationObservation)
```

Each `data` union is discriminated by its literal `type`; that type lives once,
inside the body. Python callers can read `payload.type` as a derived property.
A checkpoint cannot be an evaluation observation, and an evaluation lifecycle
cannot be a training observation. Tests derive target coverage dynamically from
`OperationTargetKind`, and exercise every training union variant.

Training bodies include lifecycle, named scalar metrics, training/resource/data/
distributed/alignment metrics, numerical-health observations, typed checkpoints
and profiler lifecycle/artifact references. Evaluation retains its own lifecycle
and progress bodies plus shared worker-ready, heartbeat, incident, artifact,
resource and named metric observations. Common names do not relax the enforced
target-to-family pairing.

A heartbeat declares its expected interval; its canonical sequence lives in the
envelope, avoiding two conflicting sequence claims. Optional `TraceContext` and
`CorrelationContext` are advisory and never override the target, generation or
sequence. Contradictory target/generation context is rejected.

`CheckpointCommittedPayload` requires ADR-012's independent state/data/boundary
guarantees and validates them against a cursor and captured-state manifest.
Strong guarantees cannot be attached to a bare checkpoint URI. State is
referenced through artifacts, not embedded tensors/RNG bytes. See
[Observability and Provenance §9](../21-observability-and-provenance.md#9-pr-009a-shared-observation-contracts)
for units, state evidence, redaction, callback boundaries and deferred consumers.

PR-009a advances the protocol from `xaytune.telemetry/v1alpha1` to
`xaytune.telemetry/v1alpha2`: typed bodies replace the pre-NativeWorker dictionary
body. LocalRuntime refuses plans requesting the old protocol and raises on
incompatible complete retained records rather than silently skipping them.
Existing streams need an explicit migration; missing checkpoint evidence must
never be synthesized. A trailing incomplete JSONL write remains retryable. Existing
untyped checkpoint events must not be upgraded by inventing missing state.
LocalRuntime constructs the typed shared worker-ready/incident bodies; it still does not report training completion from
process exit. Stream ownership, replay, sequencing and reconciliation rules above
are unchanged.

### `xaytune.telemetry/v1alpha3` — evaluation results travel inline (PR-013)

`v1alpha3` makes an evaluation's result part of the stream. Its
`EvaluationCompleted` **must** carry `metrics: tuple[MetricResult, ...]` -- the
final, decision-grade result set -- with `result_ref` still pointing at the full
report. So the controller records a result without reading the worker's files,
and a remote runtime needs no storage shared with the controller. Streamed
`MetricObserved` stays observation: a final result is never reconstructed from
progress telemetry.

The change is a version, not a silent widening of `v1alpha2`: domain models
forbid unknown fields, so a `v1alpha2` reader would reject the new field, and
the same version with a different payload would not be the same protocol.

```text
v1alpha2   training telemetry                  yes
           EvaluationCompleted without metrics yes (read for recovery)
           EvaluationCompleted with metrics    refused
v1alpha3   EvaluationCompleted with metrics    required
```

- **Both are read.** A training workload started before an upgrade still
  writes `v1alpha2`, and a new controller still adopts it. The envelope accepts
  either version and enforces the pairing above; `TrainingExecutionSpec` stays
  on `v1alpha2` and `EvaluationExecutionSpec` requires `v1alpha3`.
- **The launcher writes what the plan declares**, from
  `plan.spec.telemetry.protocol_version`, not a constant; LocalRuntime refuses a
  plan asking for any other version at submission.
- **A completion is not a success.** The controller holds it until the runtime
  reports the workload `succeeded`, then records the result, the attempt's and
  run's `SUCCEEDED` and the cursor at the completion in one commit. Exit 0
  with no completion is a failed evaluation, not an empty success.
