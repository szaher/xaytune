# ADR-013 — External operation identity, replay, and cancellation intent

## Status
Accepted — 2026-09-21.

Required before the first runtime implementation, not after — an unreconciled
submission is an orphaned GPU job, and the journal must exist before anything
can submit.

Required before PR-005. The persistence structures it freezes must carry the operation
journal described here, and must not conflate it with the outbox.

## Context

The controller initiates side effects it cannot take back. A submission starts real
workloads on real hardware, and the record of having started them is written *after* the
call returns:

```text
BEGIN; persist attempt; COMMIT
runtime.submit(plan)          -> 64 GPUs start
                              <- controller crashes before the reply is stored
restart: attempt exists, RuntimeRef does not
```

A controller that resubmits here starts another 64 GPUs. One that does not resubmit may
strand a workload nobody is watching. Neither is acceptable, and no amount of retry logic
fixes it, because the ambiguity is in the record rather than in the call.

An `operation_id` passed to the runtime is necessary but not sufficient: the controller
also needs its own durable record of having *intended* the operation, written before the
call, so that a restart can ask what happened rather than guess.

The same problem applies to cancellation, plus a second one. Cancellation is not
instantaneous on a remote runtime, so between request and confirmation there is a period
where the controller wants the workload stopped and the workload is still running. That
is a difference between desired and observed state, and it needs somewhere to live.

## Decision

### 1. Every external effect has a durable operation record

```python
class RuntimeOperationTarget(FrozenDomainModel):
    kind: Literal["training-attempt", "evaluation-attempt"]
    id: str


class RuntimeOperation(BaseModel):
    id: OperationId
    target: RuntimeOperationTarget

    type: Literal["submit", "cancel"]
    request_digest: str

    state: Literal["intended", "sent", "confirmed", "failed"]
    runtime_ref: RuntimeRef | None

    created_at: datetime
    updated_at: datetime
```

#### The target is typed, not a `RunAttemptId`

An earlier version of this record held `attempt_id: RunAttemptId`. That made the
operation journal training-only, which contradicts ADR-015 §4 — evaluation
submission is supposed to go through `submit_or_get` "exactly as training does",
and an `EvaluationAttemptId` will not fit in a column that references
`run_attempts(id)`.

The fix is a typed target reference, **not** a generic `Execution` aggregate.
ADR-015 §2 deliberately declines to unify training and evaluation as domain
objects, and nothing here reverses that: what is shared is the *external side
effect* — one submission, one cancellation, one idempotency key — not the
domain meaning of the workload. A runtime does not care which aggregate asked.

The cost is that the database loses a foreign key, because the target rows live
in different tables. That is accepted, and the alternative was worse:

```sql
run_attempt_id        TEXT REFERENCES run_attempts(id),
evaluation_attempt_id TEXT REFERENCES evaluation_attempts(id),
CHECK ((run_attempt_id IS NULL) != (evaluation_attempt_id IS NULL))
```

which buys referential integrity and pays for it with a schema migration for
every new workload kind — and ADR-015 §2 already names data preparation and
reward-model scoring as the likely third and fourth. Referential integrity for
the target is enforced by the repository instead.

`target_kind` is deliberately **not** constrained by a `CHECK` either, for the
same reason: SQLite cannot alter one in place, so freezing the vocabulary in
the schema would mean rebuilding the table for each new workload kind — which
is the migration this design exists to avoid. The database enforces structural
shape; the domain type and the repository's target resolution validate the
vocabulary.

The record is written **before** the call and updated after it:

```text
BEGIN
  create attempt
  create operation op_123 in state INTENDED
  persist the execution-plan digest
COMMIT

runtime.submit_or_get(operation_id=op_123, plan=plan)

BEGIN
  operation op_123 -> CONFIRMED
  persist RuntimeRef
COMMIT
```

An operation found in `INTENDED` or `SENT` after a restart is the signal to reconcile
rather than to re-issue blindly.

### 2. The runtime contract is get-or-create, not create

```python
def submit_or_get(operation_id: OperationId, plan: ResolvedExecutionPlan) -> RuntimeRef
def lookup_operation(operation_id: OperationId) -> OperationOutcome | None
```

- **Same id, same digest** — return the original workload. Never start a second.
- **Same id, different digest** — fail with `IdempotencyConflict`. The controller is
  confused about what it asked for, and guessing would be worse than stopping.
- **After a restart** — `lookup_operation` answers what became of it.

#### `request_digest` hashes the external request, not the semantic fingerprint

```text
request_digest = hash(canonical(operation_type, ResolvedExecutionPlan))
```

covering everything that determines the side effect:

```text
entrypoint                  input refs
arguments                   output refs
candidate fingerprint       resource requirements
compiler identity           checkpoint contract
container digest            telemetry contract
dependency lock             runtime options
non-secret environment      secret *references* and version ids
```

Secrets are referenced, never hashed by value — a digest that changes when a
credential rotates would turn a routine rotation into an `IdempotencyConflict`,
and a digest computed *over* a secret leaks it into a durable record (ADR-016).

An earlier version defined this as "the `ExecutionFingerprint` plus the
operation's own parameters". That conflates two different questions:

```text
ExecutionFingerprint   are these executions scientifically/runtime equivalent?
request_digest         is this literally the same external side-effect request?
```

`ExecutionFingerprint` covers compiler, framework, code, runtime, GPU type,
world size and topology. Two submissions can agree on every one of those and
still differ in entrypoint, arguments, dataset or output location — so deriving
the idempotency key from it can return the *original* workload for a request
that was not the same request. That is the one failure mode get-or-create exists
to prevent, arriving through the key rather than through the call.

It must be produced by the canonical typed encoder from ADR-006, not by `hash()`
or a naive JSON dump, or the same request will appear to differ between
processes. Using that encoder is all `ExecutionFingerprint` and `request_digest`
share.

### 3. `lookup_operation` must be able to answer "it already finished"

This is the case a get-or-create contract alone does not cover. After a restart, an
operation in `SENT` has three possible truths:

```text
the runtime has the workload, running     -> adopt it
the runtime never received it             -> safe to re-issue
the runtime ran it and it already finished -> adopt the result, do not re-run
```

A lookup that can only say "running" or "not found" collapses the second and third into
one answer, and the controller will cheerfully re-run a completed workload. So
`OperationOutcome` must be able to report a terminal result, which means a runtime
adapter has to retain operation records for longer than the workloads they describe.

Where a runtime genuinely cannot distinguish the two — some batch systems forget
completed jobs — the adapter declares that capability as absent, and the controller
treats `SENT` with no record as requiring policy or human resolution rather than
resubmitting. Saying "I cannot tell" is a valid answer; guessing is not.

### 4. Cancellation intent is not workload state

**`CANCELLING` is not added to `RunAttemptStatus`.**

`RunAttempt.status` is *observed* state, reconciled from the runtime. A cancellation
request is *desired* state, owned by the controller. Putting intent into the lifecycle
enum mixes the two, and a reconciler reading runtime truth would then have to know not to
clobber it.

They are held separately instead:

```text
in flight                          confirmed

RunAttempt.status  RUNNING         RUNNING -> CANCELLED
CancelRun Action   EXECUTING       EXECUTING -> SUCCEEDED
RuntimeOperation   SENT            SENT -> CONFIRMED
```

A controller that restarts mid-cancellation knows the request exists, because the Action
and the operation are durable. No lifecycle state was needed to express it.

### 5. Observed terminal state wins over pending intent

If a workload reaches a terminal state before a cancellation is confirmed, the observed
outcome stands. An attempt that succeeded is `SUCCEEDED`, not `CANCELLED` — the cancel
simply arrived too late.

The Action records that it was superseded rather than failing: it did what it was asked,
and the answer was that there was nothing left to stop. Recording it as `FAILED` would
make routine races look like defects.

That is a first-class field, not a note in a payload:

```text
Action.status  = SUCCEEDED
Action.outcome = SUPERSEDED
```

`status` says whether the action completed; `ActionOutcome` says how it turned
out (`APPLIED` | `SUPERSEDED` | `NOOP`). A `SUPERSEDED` outcome under a
`FAILED` status would be contradictory, and burying it in `payload` would make a
lifecycle guarantee depend on an untyped dictionary. See `03-domain-model.md` §8.

### 6. Cancelling an experiment is a controller saga

```text
CancelExperiment
      ↓
create cancellation Actions for every non-terminal descendant
      ↓
issue cancel operations for active runtime attempts
      ↓
reconcile until descendants are terminal
      ↓
Experiment -> CANCELLED
```

> **Invariant.** `ExperimentStatus.CANCELLED` means Xaytune believes no owned workload is
> still executing.

That invariant is checkable at reconciliation time, which is what makes it worth having.

The experiment stays `ACTIVE` while cancellation propagates. Its own cancellation intent
lives in the Action, exactly as an attempt's does — the same rule as §4, one level up.

**When a descendant cannot be cancelled** — the runtime is unreachable, or an operation is
stuck in `SENT` with no resolvable outcome — the experiment **must not** reach
`CANCELLED`. It stays `ACTIVE` with an unresolved incident. The tempting implementation
is to time out and mark it cancelled anyway; that breaks the invariant precisely when it
matters, because the workload most likely still running is the one the controller lost
track of.

Abandoning such an experiment is a separate, explicit operation — a future
`AbandonExperiment` that records that Xaytune has stopped tracking workloads it may not
have stopped. It is not a timeout.

### 7. The operation journal is not the outbox

Both are durable records of work that happens outside a transaction, and PR-005 must not
treat them as the same mechanism:

| | Outbox | Operation journal |
|---|---|---|
| Direction | events published outward | effects we initiate on a runtime |
| Delivery | at-least-once | at-most-once effect, via get-or-create |
| Redelivery | free — sinks are idempotent consumers | **never blind** — a resend may start a workload |
| On restart | replay undelivered | reconcile against the runtime |

Giving submissions outbox redelivery semantics is the 64-GPU bug with extra steps.

### 8. LocalRuntime is bound by all of this

Restart safety is not a distributed-systems concern to defer; it is a property of the
first runtime implementation. A `LocalRuntime` that calls `subprocess.Popen` and forgets
the child cannot be reconciled, and a restarted daemon will relaunch work that is still
running.

It therefore records, durably:

```text
operation id
process identity   -- PID *and* process start time
plan digest
status / control path
```

**PID alone is not identity.** PIDs are recycled, so a daemon that adopts by PID after a
reboot can attach to — and later kill — an unrelated process. The start time is what
makes the pair unique.

## Acceptance criteria

- **AC-1.** Given an operation in `SENT` and a controller restart, when reconciliation
  runs, then the runtime is queried and no second workload is created.
- **AC-2.** Given the same `operation_id` and the same digest, when `submit_or_get` is
  called twice, then both calls return the same `RuntimeRef` and one workload exists.
- **AC-3.** Given the same `operation_id` and a different digest, then the call fails with
  `IdempotencyConflict` and no workload is created.
- **AC-4.** Given an operation whose workload completed while the controller was down,
  when reconciliation runs, then the result is adopted and the workload is not re-run.
- **AC-5.** Given a runtime that cannot report completed operations, when an operation in
  `SENT` has no record, then the controller escalates rather than resubmitting.
- **AC-6.** Given a cancellation in flight, when the controller restarts, then it knows
  cancellation was requested without any lifecycle state having encoded it.
- **AC-7.** Given a workload that succeeds before a cancellation is confirmed, then the
  attempt is `SUCCEEDED` and the Action is `status=SUCCEEDED, outcome=SUPERSEDED`.
- **AC-8.** Given an experiment cancellation where one descendant cannot be cancelled,
  then the experiment does **not** reach `CANCELLED`.
- **AC-9.** Given `ExperimentStatus.CANCELLED`, when reconciliation runs, then no owned
  workload is executing.
- **AC-10.** Given a `LocalRuntime` daemon restart, when a recorded process is still
  alive, then it is adopted rather than relaunched — matched on PID **and** start time.
- **AC-11.** Given a recorded PID that has been recycled by an unrelated process, when
  the daemon reconciles, then it does not adopt it.

### 6a. Cancelling an evaluation uses the same path

An `EvaluationAttempt` is cancelled exactly as a training attempt is: an
`Action` holds the intent, a `RuntimeOperation` carries the effect, and both
commit together (ADR-005 §5).

```python
Action(type="cancel-attempt",
       target=ActionTarget(kind="evaluation-attempt", id=...))
RuntimeOperation(target=RuntimeOperationTarget(kind="evaluation-attempt", id=...))
```

`CancelAttempt` is workload-neutral rather than split per workload type, and the
two target vocabularies use the same spellings on purpose — an Action's target
and the target of the operation it causes name the same subject.

This is what makes ADR-015's AC-7 — a cancelled evaluation leaves no executing
workload — actually reachable. Without an `evaluation-attempt` action target
there would be no legal Action to own the intent, and the ADR-005 invariant
would forbid issuing the cancel operation at all.

### 7. Sequencing: the Action substrate comes before cancellation

Cancellation as defined above needs a durable `Action` to hold the intent while
the `RuntimeOperation` carries the effect. `handle.cancel()` is public API from
the compile/execute phase onwards, so the Action aggregate, its state machine
and its repository are **band B work** (PR-006a), not part of the later policy
phase.

Only the substrate and the three cancellation action types are needed that
early. `PolicyEngine`, approvals, budget authorization and every mutating action
type stay where they were: those exist to answer *may this happen*, which
cancellation does not ask.

## Consequences

PR-005 implements `runtime_operations` in migration 002 and records its state
transitions in the existing domain event log, atomically with the outbox. No
separate operation-transition table is required. The repository writes the
INTENDED submit operation and request digest in the same transaction as the new
attempt, which makes intent durable before the effect happens. PR-009 depends
on these repository APIs; the journal is not deferred to a later migration.

Runtime adapters gain `submit_or_get` and `lookup_operation`, and must declare whether
they can report completed operations. That capability is not optional metadata — the
controller's behaviour on §3's third case depends on it.

`LocalRuntime` needs a supervisor that outlives the controller process. This is more work
than `Popen`, and it is what makes "kill the client, the controller continues" true rather
than aspirational.

## Rejected alternatives

**Pass `operation_id` and rely on the runtime alone.** Leaves no controller-side record of
intent, so a crash before the reply still cannot distinguish "never sent" from "sent and
lost".

**Add `CANCELLING` to `RunAttemptStatus`.** Mixes desired state into an enum that
reconciliation overwrites from observed runtime state.

**Time out a stuck cancellation and mark the experiment `CANCELLED`.** Breaks the one
invariant the state is worth having, in exactly the case where it matters.

**Reuse the outbox for submissions.** Outbox records are safe to redeliver. Submissions
are not.
