# Control-plane concepts

The ideas the control plane is built on, as implemented today. The
[architecture specification](https://github.com/szaher/xaytune/tree/main/xaytune-training-harness-spec)
has the full contracts and the ADRs behind them. This page is what you need
to use the control plane.

## Control the experiment, delegate execution

Xaytune decides and records *what* runs and *why*. It does not move tensors.

```text
CandidateSpec ──compile──> TrainingExecutionSpec ──resolve──> ResolvedExecutionPlan ──submit──> runtime
     what to test               how to run it                  what this runtime runs          runs it
```

A **trainer compiler** (`NativeCompiler`, `TRLCompiler`) turns a candidate
into a plan. A **runtime** (`LocalRuntime` today) executes the plan and streams
observations back. The **controller** (`EmbeddedControllerHost`) ties them
together and writes everything down. Each boundary carries data, never live
objects, so every step can be recorded, replayed and checked.

## The record

Everything is persisted to SQLite, and **the record is the truth**. A handle
answers from it. A restarted process reads it. Nothing important lives only in
memory.

```text
Experiment                   one question, with an objective
└── ExperimentNode           one candidate: its spec and fingerprint
    ├── Run                  one execution of that candidate, with a seed
    │   └── RunAttempt       one try at that run, on a runtime
    └── EvaluationRun        one evaluation of the trained model
        └── EvaluationAttempt
```

Each has its own state machine and changes only through legal transitions.
Every transition is committed **in one transaction with its event**, so the
event log is complete provenance: `handle.events()` replays it and then
follows new commits.

Two more records make external effects safe:

- **`RuntimeOperation`**: before Xaytune asks a runtime to do anything (submit,
  cancel), it records the intent, and afterwards the outcome. A crash in
  between leaves a recorded intent that reconciliation resolves. It never
  leaves an effect nobody knows about.
- **`Action`**: a request such as "cancel this experiment" is recorded as an
  Action, carried out through operations, and closed with its outcome.

## Candidates and identity

A `CandidateSpec` is one hypothesis: model, data, and how to train. It must
declare **every value that changes what the model learns**. The compilers
refuse a candidate that leaves one to a trainer default, and list every
reason.

`candidate_fingerprint()` is a versioned projection of the candidate, not a
hash of whatever the schema is today. Adding a field later does not change the
identity of candidates already recorded.

The **seed belongs to the run**, not the candidate. Two seeds are two samples
of the same hypothesis, under one fingerprint. That is why `ExperimentSpec`
takes `seed` separately.

## Submission is idempotent

A runtime is asked through `submit_or_get(operation_id, plan)`. The operation
id is minted once and recorded before the call, so asking again after a crash
returns the workload already running and never starts a second one.

The request is identified by `plan.request_digest("submit")`, a hash of the
whole request. When a submission has to be re-issued after a restart, the
controller rebuilds the plan from the record and checks that the digest
matches. The same request is re-sent, never a different one that happens to
look similar.

## Restart safety

The workload belongs to the runtime. `host.close()`, or the process dying,
stops observation but not training. `host.attach(experiment_id)` in any later
process then:

- **adopts** an attempt the runtime still holds, observing it from the durable
  telemetry cursor, so no observation is applied twice or lost;
- **looks up** a submission whose outcome was never recorded;
- **issues** only a submission the runtime never received, under its recorded
  identity;
- **escalates** with `ReconciliationEscalatedError` when the record cannot
  safely answer, for example a workload that ended with no recorded outcome.

## What `wait()` means

`handle.wait()` returns when the controller is **quiescent**: every piece of
work it can currently execute is settled, and its telemetry is drained. It does
not mean the experiment is finished. The returned `ExperimentResult` says so
explicitly:

- `status`: the experiment's own status. After a successful training run it is
  still `ACTIVE`, because ending an experiment is a decision, and nothing makes
  decisions yet.
- `quiescent`: always true for a result `wait()` returns.
- `next_stage`: the work that would move things on. `"evaluation"` means a
  trained candidate nothing has evaluated yet. `"decision"` means a candidate
  evaluated into `DECIDING`. `"failure-handling"` means a run or evaluation
  failed or was cancelled. `None` means the experiment is terminal.
- `nodes`: each candidate with its runs, attempts, artifacts and evaluations.

## Cancellation

`handle.cancel()` records an Action, then issues a cancel operation for each
live attempt. There is no `CANCELLING` status: the experiment stays `ACTIVE`
until no workload it owns is running, then becomes `CANCELLED`. Calling
`cancel()` again while one is in flight records nothing new.

## Evaluation

Set `ExperimentSpec.evaluation` to an `EvaluationSpec` naming **one**
evaluator (several evaluators are several evaluation runs), a dataset and
slices. After training succeeds, the host:

1. moves the candidate to `EVALUATING`, starting a new **evaluation cycle**,
   so results from an earlier cycle can never satisfy a later one;
2. asks the `Evaluator` to *prepare* an evaluation of the trained model, which
   produces a plan, the same way a compiler does, and starts nothing;
3. submits it through the same operation journal and runtime as training;
4. records the result (metrics tied to the run, evaluator version and seed)
   once the evaluation has both reported completion and exited successfully;
5. moves the candidate to `DECIDING`.

The evaluation spec is orchestration, not identity: it never enters the
candidate or its fingerprint, so changing how you evaluate never means
retraining.

**No evaluator is built in yet.** The next step wraps Xaytune's metrics and
lm-eval behind the `Evaluator` contract. Until then, you pass your own to
`EmbeddedControllerHost(..., evaluators={"name": factory})`.

## Not yet

These are designed in the specification and planned, but **not
implemented**: deciding what an evaluated candidate becomes, policy and
budgets, checkpoints and semantic recovery, planners and branching, daemon
hosting, and runtimes other than local.
