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
    status: EvaluationRunStatus

class EvaluationAttempt(AggregateModel):
    id: EvaluationAttemptId
    evaluation_run_id: EvaluationRunId
    attempt_number: int
    runtime_ref: RuntimeRef | None
    status: EvaluationAttemptStatus
```

The state machines follow the **same lifecycle principles** as `Run` and
`RunAttempt` — not the same tables. Evaluation produces no checkpoints, so it
has no `CHECKPOINTING` state and nothing to recover into, so no `RECOVERING`
either. A failed evaluation is retried as a new attempt.

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
is wrong, because not every evaluator is deterministic. `EvaluationSpec` carries
a `seed`; LLM-judge evaluators carry `temperature`, `top_p` and a provider; and
agent or environment evaluations are stochastic by construction. A hosted model
behind an API does not guarantee reproducible output even at temperature zero,
and may change underneath a fixed model name.

So the evaluator declares its own reuse class as a capability:

```python
class EvaluatorDeterminism(StrEnum):
    DETERMINISTIC = "deterministic"   # same inputs, same output, always
    SEEDED = "seeded"                 # reproducible given the same seed, locally
    STOCHASTIC = "stochastic"         # each run is a sample
```

| Class | Reuse rule |
|---|---|
| `DETERMINISTIC` | `(artifact digest, EvaluationFingerprint)` may be reused freely |
| `SEEDED` | Reusable only when the seed is part of the fingerprint **and** execution is local; a hosted provider is never `SEEDED` |
| `STOCHASTIC` | A completed run is **a historical sample, not the answer.** Reuse only when `EvaluationReusePolicy` explicitly permits memoizing a sample |

The failure this prevents is quiet and statistical: asking for another sample of
a stochastic evaluation and silently receiving the previous one. That does not
error — it just makes a variance estimate wrong, and a planner comparing
candidates on noisy metrics will draw a confident conclusion from one sample it
believes is several.

Replicated evaluation follows from this: replicate identity belongs to
`EvaluationRun`, exactly as seed and replicate belong to `Run`.

The cache key additionally requires a **terminal** `EvaluationRun`. An in-flight
run must not be matched against, for the same reason a
`RunRealizationFingerprint` is provisional until terminal (ADR-011).

### 4. Evaluation submission uses the operation journal

Evaluation submission goes through `submit_or_get(operation_id, plan)` exactly
as training does (ADR-013). A controller crash between submitting an evaluation
and persisting its `RuntimeRef` is the same problem with the same solution, and
it would be strange to solve it twice.

### 5. A node blocked on evaluation is visible as such

`ExperimentNode.EVALUATING` must correspond to at least one non-terminal
`EvaluationRun`. If none exists, the node is stuck and the controller must
raise an incident rather than wait.

This invariant is the point of the whole ADR. It is checkable, and it converts
the silent-stall failure into a detected one.

## Consequences

- The persistence schema gains `evaluation_runs` and `evaluation_attempts`, and
  PR-005 can define them now rather than discovering the need later.
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
3. `ExperimentNode.EVALUATING` implies a non-terminal `EvaluationRun`; the
   absence of one raises an incident.
4. Every evaluator declares an `EvaluatorDeterminism` class. A `DETERMINISTIC`
   result is reused on `(artifact digest, EvaluationFingerprint)`; a
   `STOCHASTIC` one is reused only where `EvaluationReusePolicy` permits it;
   in-flight runs are never matched.
4a. Requesting an additional replicate of a stochastic evaluation never returns
   a cached sample.
5. Evaluation submission is idempotent through `submit_or_get` (ADR-013).
6. A controller restart mid-evaluation recovers the attempt rather than
   orphaning it or resubmitting it.
7. A cancelled evaluation leaves no executing workload, per the ADR-013
   cancellation invariant.
