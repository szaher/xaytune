# ADR-015 — Evaluation has a durable execution lifecycle

## Status
Accepted — 2026-09-21.

Extends ADR-007, which established that evaluation is independent of trainers
but did not give it an execution model. Required before PR-005 freezes the
persistence schema.

## Context

ADR-007 is right that evaluation must not be a trainer callback. But it left
evaluation with only:

```text
EvaluationSpec  ──→  EvaluationResult
```

while training has `Run`, `RunAttempt`, `RuntimeRef` and a state machine.

That asymmetry is not a cosmetic inconsistency. Evaluation is a workload. It
queues, it starts, it fails, it gets preempted on spot capacity, it needs
retrying, it can be cancelled, it can run on a different runtime from the
training that produced the checkpoint, and it can be in flight when the
controller restarts.

With only a spec and a result, every one of those has no representation. The
practical failure is specific and silent: **a node sits in `EVALUATING`
forever** because the evaluation process died and nothing recorded that it had
ever started. There is no attempt to observe, so there is nothing to time out,
retry, or report.

Evaluation is also where the experiment's decisions come from. An evaluation
that is quietly lost does not just waste GPU hours — it stalls the search.

## Decision

### 1. Evaluation gets Run/Attempt, mirroring training

```python
class EvaluationRun(AggregateModel):
    id: EvaluationRunId
    node_id: NodeId
    spec: EvaluationSpec
    subject: ArtifactRef            # the checkpoint or model under evaluation
    fingerprint: EvaluationFingerprint

    seed: int | None                # NOT on EvaluationSpec -- see section 3
    replicate: int | None

    status: EvaluationRunStatus

class EvaluationAttempt(AggregateModel):
    id: EvaluationAttemptId
    evaluation_run_id: EvaluationRunId
    attempt_number: int
    runtime_ref: RuntimeRef | None
    telemetry_generation: int         # ADR-014, same contract as RunAttempt
    status: EvaluationAttemptStatus
```

The state machines follow the **same lifecycle principles** as `Run` and
`RunAttempt` — not the same tables. Evaluation produces no checkpoints, so it
has no `CHECKPOINTING` state and nothing to recover into, so no `RECOVERING`
either. A retry creates a new attempt; it never `RECOVER`s the old one.

```text
EvaluationRun

CREATED ──→ ACTIVE ──→ SUCCEEDED
  every non-terminal state also ──→ FAILED, CANCELLED
```

| From | To |
|---|---|
| CREATED | ACTIVE, CANCELLED, FAILED |
| ACTIVE | SUCCEEDED, FAILED, CANCELLED |
| SUCCEEDED, FAILED, CANCELLED | *(terminal)* |

```text
EvaluationAttempt

CREATED ──→ QUEUED ──→ STARTING ──→ RUNNING ──→ SUCCEEDED
  every non-terminal state also ──→ FAILED, CANCELLED
  QUEUED onwards also          ──→ PREEMPTED
```

| From | To |
|---|---|
| CREATED | QUEUED, CANCELLED, FAILED |
| QUEUED | STARTING, CANCELLED, FAILED, PREEMPTED |
| STARTING | RUNNING, CANCELLED, FAILED, PREEMPTED |
| RUNNING | SUCCEEDED, CANCELLED, FAILED, PREEMPTED |
| SUCCEEDED, FAILED, PREEMPTED, CANCELLED | *(terminal)* |

The ADR-002 rules that do carry over: every non-terminal state reaches `FAILED`
and `CANCELLED`, and `PREEMPTED` applies from `QUEUED` onwards because that is
when the workload becomes known to a scheduler.

Following the principles rather than copying the table means the controller's
reconciliation logic, the operation journal of ADR-013 and the telemetry
protocol of ADR-014 all still apply unchanged, without evaluation carrying two
states it can never enter.

### 2. Not a generic `Execution` abstraction — yet

The obvious move is to unify training and evaluation under
`Execution`/`ExecutionAttempt`. **This ADR deliberately does not.**

Two aggregates with the same shape are not yet evidence of a shared
abstraction, and the differences are real: training produces checkpoints and
accepts interventions mid-flight; evaluation consumes a subject and a spec
without mutating training state, and has neither a checkpoint nor an
intervention lifecycle. Unifying now would mean carrying
training-only concepts into evaluation and weakening the type of both.

Revisit when a third workload type appears — a data-preparation job or a reward
model scoring pass are the likely candidates. Recorded in
`22-open-questions.md`.

### 3. Reuse depends on whether the evaluator is deterministic

The tempting rule — same artifact plus same fingerprint means reuse the result —
is wrong, because not every evaluator is deterministic. LLM-judge evaluators
carry `temperature`, `top_p` and a provider; agent or environment evaluations
are stochastic by construction; and a seeded evaluator is reproducible only
given its seed. A hosted model behind an API does not guarantee reproducible
output even at temperature zero, and may change underneath a fixed model name.

So the evaluator declares its own reuse class as a capability:

```python
class EvaluatorDeterminism(StrEnum):
    DETERMINISTIC = "deterministic"   # same inputs, same output, always
    SEEDED = "seeded"                 # reproducible given the same seed, locally
    STOCHASTIC = "stochastic"         # each run is a sample
```

| Class | Reuse key |
|---|---|
| `DETERMINISTIC` | `(artifact digest, EvaluationFingerprint)` — may be reused freely |
| `SEEDED` | `(artifact digest, EvaluationFingerprint, EvaluationRun.seed)` — and only where execution is local; a hosted provider is never `SEEDED` |
| `STOCHASTIC` | No implicit cross-run reuse. A completed run is **a historical sample, not the answer**, and is reused only where `EvaluationReusePolicy` explicitly permits memoizing a sample |

Note where the seed enters: as a **component of the lookup key**, not as a
component of `EvaluationFingerprint`. That is what keeps replicate identity out
of the fingerprint, so two seeds of the same evaluation remain recognisably two
samples of one thing rather than two different evaluations.

The failure this prevents is quiet and statistical: asking for another sample of
a stochastic evaluation and silently receiving the previous one. That does not
error — it just makes a variance estimate wrong, and a planner comparing
candidates on noisy metrics will draw a confident conclusion from one sample it
believes is several.

Replicated evaluation follows from this: **`seed` and `replicate` live on
`EvaluationRun`, not on `EvaluationSpec`**, exactly as seed and replicate belong
to `Run` and not to `CandidateSpec`. A seed on the spec would make
`evaluation with seed 1` and `evaluation with seed 2` two different evaluation
specifications instead of two replicates of one — the same identity error that
moving `seed` off the candidate fixed for training.

The cache key additionally requires a **terminal** `EvaluationRun`. An in-flight
run must not be matched against, for the same reason a
a run's fingerprints are provisional until terminal (ADR-011).

### 4. Evaluation submission uses the operation journal

Evaluation submission goes through `submit_or_get(operation_id, plan)` exactly
as training does (ADR-013). A controller crash between submitting an evaluation
and persisting its `RuntimeRef` is the same problem with the same solution, and
it would be strange to solve it twice.

For that to be true rather than aspirational, the operation record must be able
to *name* an evaluation attempt. ADR-013's `RuntimeOperation` therefore carries
a typed target:

```python
RuntimeOperationTarget(kind="evaluation-attempt", id=evaluation_attempt_id)
```

This is not the generic `Execution` aggregate that §2 declines to build. The
journal records external side effects — one submission, one cancellation, one
idempotency key — and a runtime does not care which aggregate asked. Sharing
the effect is not sharing the domain object.

### 5. A node blocked on evaluation is visible as such

The obvious form of this invariant — `EVALUATING` implies at least one
non-terminal `EvaluationRun`, else raise an incident — is too strict, because
ordinary completion passes through a state that violates it:

```text
EvaluationRun -> SUCCEEDED
controller holds the result
node still EVALUATING, not yet transitioned to DECIDING
        ↑
  zero non-terminal runs, and nothing is wrong
```

Raising an incident there would make every successful evaluation look like
corruption for the width of one transaction. So it is a **reconciliation**
invariant, evaluated as three cases rather than one:

| At reconciliation, a node in `EVALUATING` | Meaning | Action |
|---|---|---|
| **A.** at least one required `EvaluationRun` is non-terminal | in progress | wait |
| **B.** all required runs terminal, required results present | finished, node lagging | reconcile the node to `DECIDING` |
| **C.** neither | stalled — runs missing, or terminal without results | raise `EvaluationStalled` |

Case B is the one that makes this work: the reconciler *repairs* the lag instead
of reporting it, so a controller that died between the run's terminal
transition and the node's cannot leave the node stuck.

Case C is still the point of the whole ADR. It is checkable, and it converts the
silent-stall failure into a detected one.

## Consequences

- The persistence schema gains `evaluation_runs` and `evaluation_attempts`, and
  PR-005 can define them now rather than discovering the need later.
  `evaluation_attempts` carries `telemetry_generation`, because ADR-014's
  envelope targets evaluation attempts and a generation counter that exists only
  on `run_attempts` cannot serve them.
- Evaluation retry, cancellation and preemption are expressible.
- Evaluation reuse is available and safe, and it is the cheapest optimisation
  in the system.
- Two near-identical state machines exist until a third workload justifies
  unifying them. This is accepted, and named, rather than papered over.

## Acceptance criteria

1. `EvaluationRun` and `EvaluationAttempt` exist with the transition tables
   above, including all `FAILED`/`CANCELLED` edges, and with **no**
   `CHECKPOINTING` or `RECOVERING` state.
2. `EvaluationAttemptStatus` includes `PREEMPTED`, reachable from `QUEUED`
   onwards.
3. Reconciling a node in `EVALUATING` resolves to exactly one of: wait (a
   required run is non-terminal), advance it to `DECIDING` (all required runs
   terminal with results), or raise `EvaluationStalled`. A successful evaluation
   awaiting its node transition never raises an incident.
4. Every evaluator declares an `EvaluatorDeterminism` class. A `DETERMINISTIC`
   result is reused on `(artifact digest, EvaluationFingerprint)`; a
   `STOCHASTIC` one is reused only where `EvaluationReusePolicy` permits it;
   in-flight runs are never matched.
4a. Requesting an additional replicate of a stochastic evaluation never returns
   a cached sample.
4b. `seed` and `replicate` are fields of `EvaluationRun`; `EvaluationSpec` has
   no `seed`. A `SEEDED` reuse lookup keys on the run's seed, which is not part
   of `EvaluationFingerprint`.
4c. Every `EvaluationResult` carries the `evaluation_run_id` of the run that
   produced it, so an individual sample of a stochastic evaluation is
   attributable to its execution.
5. Evaluation submission is idempotent through `submit_or_get` (ADR-013).
6. A controller restart mid-evaluation recovers the attempt rather than
   orphaning it or resubmitting it.
7. A cancelled evaluation leaves no executing workload, per the ADR-013
   cancellation invariant. The intent is owned by a `CancelAttempt` `Action`
   targeting the `evaluation-attempt` (ADR-013 §6a), committed with its cancel
   operation as ADR-005 §5 requires — evaluation cancellation uses the same path
   as training, not a parallel one.

## Implementation notes (PR-013)

- **"Required runs" are the runs of the node's current evaluation cycle.** A
  node can evaluate, decide, return to `ACTIVE` and evaluate again, so §5 would
  be unsound over all of a node's runs: the first round's successful results
  would satisfy the second. `ExperimentNode.evaluation_cycle` advances in the
  same write as the transition into `EVALUATING`, every `EvaluationRun` records
  its cycle, and the cycle's runs are written in that same commit
  (`begin_evaluation_cycle`). Reconciliation reads only the current cycle.
- **A completion is held durably.** An `EvaluationCompleted` is not yet a
  result -- the workload may still fail -- so it is written to the attempt
  (`pending_completion_json`) in the commit that advances the cursor past it,
  and becomes the result only when the runtime reports success. A stream that
  dies over the live workload moves the attempt to a new generation (ADR-014
  §1a); the completion is in the record, not that stream, so neither that nor
  a controller crash can lose it.
- **Preemption has no retry yet.** A preempted attempt is `PREEMPTED` and its
  run `FAILED`, so no run stays `ACTIVE` with nothing executing it; the node
  then stalls. A retry policy, when one exists, creates a new attempt.
- **A result agrees with its run in every provenance field**: node,
  fingerprint, subject (identity *and* digest -- two artifacts can share
  bytes), and each metric's evaluator, evaluator version and seed, a missing
  seed included. A report's producer is stamped by the controller with the
  result's id; a worker-supplied producer is refused. The repository
  enforces **all** of this (`result_provenance_problems`), and a worker
  reporting drift fails its evaluation, with the reason on the event. The
  database independently enforces the column-level subset -- the result's
  run, node, evaluation fingerprint, subject artifact id and subject digest,
  one result per run, and immutability -- so a writer that bypassed the
  repository still cannot misfile a result; the per-metric and report rules
  live in the payload and are the repository's.
- **A stall is recorded, not raised.** `EvaluationStalled` is an event on the
  node, once per cycle; the node stays `EVALUATING`. What to do about it is a
  decision.
- **Success needs the result.** `EvaluationRun` and `EvaluationAttempt` reach
  `SUCCEEDED` only through `record_evaluation_result`, together with the
  result; no other path can write it. Results are rows of their own
  (`evaluation_results`, migration 006), one per run, immutable, and a trigger
  refuses one whose node, fingerprint or subject disagrees with its run.
- **Implemented:** AC-1, 2, 3, 4b, 4c, 5, 6, 7 -- and AC-4's declaration half:
  every evaluator declares its `EvaluatorDeterminism`, recorded with the bound
  spec. **Not yet:** the reuse lookup itself (AC-4, 4a). The identity it keys
  on is indexed (`idx_evaluation_runs_reuse`); the lookup and
  `EvaluationReusePolicy` come with the planner, which is what would ask for a
  replicate.

