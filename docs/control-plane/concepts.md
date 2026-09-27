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

- `status`: the experiment's own status. `SUCCEEDED` or `FAILED` once its
  candidate has been evaluated and decided. `ACTIVE` while it has not: after
  training with no evaluation configured, or when the decision was deferred.
- `quiescent`: always true for a result `wait()` returns.
- `next_stage`: the work that would move things on. It is advice, not a
  status:

  ```text
  None                 the experiment is terminal
  "decision"           a candidate is DECIDING: its decision was deferred
  "evaluation"         a trained candidate is unevaluated, or evaluating
  "planning"           every candidate was rejected on its merits, and the
                       experiment is still ACTIVE: another candidate is needed
  "failure-handling"   training or evaluation failed or was cancelled
  ```

  `"planning"` comes only from a scientific outcome. A candidate that failed
  or was cancelled did not establish a result, so it leads to
  `"failure-handling"`, even beside a rejected one.
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

An evaluator declares how far its results can be trusted to repeat:
`DETERMINISTIC`, `SEEDED` or `STOCHASTIC`. The host records that declaration,
with the evaluator's version, when the experiment is submitted. It also asks
the evaluator whether it can run the spec exactly as declared (`supports()`),
so an impossible evaluation is refused then, not after training. If the
evaluator refuses the trained model itself when preparing, the evaluation run
fails with its reasons, and the cycle is reported stalled.

The built-in evaluator is `native`: next-token loss, perplexity and token
accuracy on a local held-out file pinned by its content digest. It is
`SEEDED`. Floating-point results depend on the device and library versions,
so it does not promise the same number everywhere. Other evaluators are
registered with `EmbeddedControllerHost(..., evaluators={"name": factory})`.

**Resolved once, at submission.** A spec can name something that changes:
the `lm-eval` evaluator takes a benchmark task name such as `arc_easy`, whose
definition ships inside lm-eval and whose dataset lives on the Hugging Face
Hub. The record must not hold anything that can change. So the host checks
the evaluation twice when it is submitted:

```text
supports(declared) → resolve() → supports(resolved) → recorded and fingerprinted
```

`resolve()` is optional: an evaluator that implements it
(`ResolvableEvaluator`) pins every mutable reference, and one that does not has
its spec recorded as declared. For `lm-eval` those references are the task's
definition (by digest, under lm-eval 0.4.13 exactly) and its dataset (by Hub
commit). It is the only step allowed to use the network. The second
`supports()` judges what the name turned out to be. `gsm8k` passes as a name
and is then refused as a generation task, before anything trains. The
resolved spec is what is recorded, so its fingerprint names the pinned task,
not the name. Nothing resolves it again: after a restart, the request is
rebuilt from the record, and the worker refuses to run a task whose installed
definition no longer matches the recorded digest.

`lm-eval` accepts one registered task at a time, defined in YAML, scored by
log-likelihood (`multiple_choice` or `loglikelihood`), and reporting `acc` and
`acc_norm`. It refuses groups, tags, tasks implemented in custom Python,
unsafe-code tasks, datasets it cannot pin to a commit, and generation tasks.
Generation brings decoding settings, stop sequences, filters and answer
extraction that decide the score, and no binding pins those yet. Each metric
records lm-eval's standard error and the number of documents actually scored,
after `limit`. It is `SEEDED`: the run's seed is lm-eval's Python, NumPy,
Torch and few-shot seed.

**No result stands in for a run.** Evaluating the same model the same way
twice runs twice. Reusing an earlier result (ADR-015's reuse lookup) is a
separate, later decision.

## Decisions

Once a candidate's evaluation cycle has its results, the node is `DECIDING`,
and a **decision engine** decides it.

- **From the record alone.** The engine sees a `DecisionContext`: the
  experiment's `Objective` and the results of that evaluation cycle, and
  nothing else (no clock, database or environment). Results from an earlier
  cycle never decide a later one.
- **Deterministic thresholds.** The built-in `ThresholdDecisionEngine`
  compares the recorded values with the objective:

  ```text
  a metric the objective or a constraint names is missing   → not decided
  any constraint violated          (<  <=  >  >=  ==  !=)    → REJECT
  no target                                                  → not decided
  target met   (maximize: value ≥ target, minimize: ≤)       → STOP_SUCCEEDED
  target not met                                             → STOP_FAILED
  ```

  It compares point estimates exactly and makes no statistical claim: it says
  whether 0.83 meets a stated threshold, not that 0.83 beats 0.81.
- **Durable, in one commit.** The `Decision` records the outcome, the
  evidence (each comparison, with the result it came from), the engine and
  its version, and a fingerprint of its inputs. It is written with what its
  outcome causes, in the same commit:

  ```text
  STOP_SUCCEEDED   candidate COMPLETED   experiment SUCCEEDED, best_node_id = it
  STOP_FAILED      candidate REJECTED    experiment FAILED
  REJECT           candidate REJECTED    experiment stays ACTIVE ("planning" next)
  ```

  `REJECT` judges the candidate, not the experiment: another candidate may
  still be proposed and succeed. Only a `STOP` ends the experiment.
- **Pure.** The engine returns a `DecisionProposal`: the outcome, reason,
  evidence and input fingerprint, with no id, time or actor. The same context
  always gives an identical proposal. The repository adds the id, time and
  actor when it records the decision. The input fingerprint is an explicit,
  versioned projection of the evidence (the objective, and each result's id,
  run, fingerprint, subject and metric values with their evaluator, seed,
  count and uncertainty). A field added to a result later does not change the
  identity of decisions already made.
- **Attributable.** The repository does not take a proposal's word for what
  it was decided on. It recomputes the input fingerprint from the stored
  objective and the cycle's results, and refuses a proposal whose fingerprint
  differs, names a result twice or not at all, or cites evidence from outside
  the cycle. It does not check the outcome: which outcome the evidence
  warrants is the engine's call, and a custom engine may use its own rules.
- **Nothing guessed.** An objective without a target means "optimize this",
  not "this is good enough". With one candidate there is nothing to compare,
  so the candidate stays `DECIDING`, as it does when a metric is missing. A
  `DecisionDeferred` event records why, once per cycle.
- **Once per cycle, across restarts.** A controller that died before deciding
  is replaced by one that decides when it attaches. Deciding the same cycle on
  the same inputs returns the decision already on record; a *different*
  decision for a cycle already decided is refused.

## Budgets

`ExperimentSpec.budget` limits what an experiment may spend. Nothing is
counted: every consequence of a run or an attempt is an entry in an
append-only **ledger**, written in the same commit as the change that causes
it, and every balance is derived from the entries.

```text
reserve   set aside before an effect      commit    the effect took place (subtracts nothing)
consume   what was actually spent         release   a reservation no longer needed

outstanding = max(reserved - consumed - released, 0)
remaining   = limit - consumed - outstanding
```

| Limit | Kind | How it is spent |
|---|---|---|
| `max_runs` | hard quota | a **run** is reserved when recorded, committed when the runtime accepts it, consumed when it ends; released if it never reached the runtime. Retrying a run spends nothing more. |
| `max_failures` | hard quota | 1 per `FAILED` attempt, training or evaluation (not preempted or cancelled) |
| `max_wall_time_seconds` | soft quota | each attempt's `ended_at - started_at`, when it ends; training and evaluation |
| `max_parallel_runs` | capacity | a slot per live training attempt, released when it ends; never spent |

- **Used up stops the next effect.** A quota with nothing left refuses the
  next run, attempt or evaluation before anything is written. What is already
  running is allowed to finish, and the experiment becomes `BUDGET_EXHAUSTED`
  only once nothing it owns is still executing, so `wait()` never sees it end
  while a worker lives. Reaching a limit exactly is exhaustion; passing it is
  a `BudgetOverrun` event, recorded, not acted on. What to do about an overrun
  is policy's, which is planned.
- **A full capacity waits.** When every parallel-run slot is held, the next
  attempt waits for one; it never exhausts the experiment.
- **Only what is measured is enforced.** `max_gpu_hours`, `max_tokens` and
  `max_cost` are refused at submission: GPUs requested times wall time is
  not GPU usage, and nothing meters tokens or prices yet.
- **Across restarts.** Each entry commits with its transition, so a crash
  cannot leave an ended attempt without its cost. An attach settles anything
  the ledger lacks, idempotently; normally that is nothing.

`ExperimentResult.budget` gives each limited dimension's limit, reserved,
committed, consumed and remaining.

## Not yet

These are designed in the specification and planned, but **not
implemented**: decisions that compare candidates (promotion, noise-aware
comparison across replicates), lm-eval generation tasks, reusing earlier
evaluation results, policy, budgets on GPU-hours, tokens and cost,
checkpoints and semantic recovery,
planners and branching, daemon hosting, and runtimes other than local.
