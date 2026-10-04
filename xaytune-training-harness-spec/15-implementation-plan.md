# Implementation Plan

## Ordering principle

The phases below are ordered by **what freezes what**, not by user-visible
value. The single largest architectural risk in this project is persistence
freezing before the contracts that determine its schema: once an event schema
ships and there are experiments on disk, changing what an event means costs a
migration and a compatibility story.

So every contract that determines a column or an event payload is settled
before PR-005 writes the schema. That is why ADR-011 through ADR-016 were
written ahead of the persistence PR rather than alongside it.

The second ordering constraint comes from ADR-011: **a `TrainingIntervention`
is the outcome of an approved `Action`.** Recovery is therefore not independent
of policy — an OOM that reduces micro-batch size is an operational action that
still needs deterministic authorization. Resilience cannot precede a minimal
Action/Policy substrate, which is a change from the original sequencing.

The third is ADR-013: runtime side effects need reconciliation before any
durable MVP claim, because an unreconciled submission is an orphaned GPU job.

### Three different things are called "reconciliation"

They land in three different bands, and conflating them is how a phase comes to
claim restart safety it has not built:

| | What it recovers | Band |
|---|---|---|
| **Repository recovery** | committed aggregate state and events, after the *process* restarts | B |
| **Runtime-operation reconciliation** | in-flight submissions and live attempts, through `lookup_operation()` and `get_status()` | C |
| **Daemon restart reconciliation** | whole-controller startup: lease acquisition, every owned experiment, budget settlement, control loops | H |

Only the first is available in band B, because there is no controller and no
runtime there yet. Only the third is a full controller restart. ADR-013 puts
the second in band C explicitly — **restart safety is a property of the first
runtime implementation, not something to defer** — so it is scheduled there
rather than left to the daemon phase.

```text
A  domain contract hardening     state machines, ADR-011 identity,
                                 deep immutability, data cursor (012),
                                 operation identity (013)
B  transactional persistence     events, outbox, operation journal, projections,
                                 repository recovery after process restart,
                                 minimal Action substrate (cancellation only)
C  compile boundary              NativeCompiler, restart-safe LocalRuntime,
                                 embedded controller, runtime-operation
                                 reconciliation and attempt reattachment
D  durable evaluation lifecycle  ADR-015
E  policy and budget             approvals over the band B Action substrate --
                                 prerequisite for F
F  checkpoint, recovery, interventions
G  planner and branching
H  daemon, kill/restart MVP      host leases, whole-controller startup
                                 reconciliation
I  LLM planner
J  Ray / TorchFT / Training Hub
```

The numbered phases below implement these bands. **There is one execution
order** — the phases were reordered to match, rather than left disagreeing with
a note about which wins:

| Band | Phase |
|---|---|
| A domain contract hardening | 0, 1 |
| B transactional persistence | 1 |
| C compile boundary, runtime, reconciliation | 2 |
| D durable evaluation | 3 |
| E Policy and Budget over the band B Actions | 4 |
| F checkpoint, recovery, interventions | 5 |
| G planner and branching | 6 |
| H daemon, kill/restart MVP | 7 |
| I LLM planner | 8 |
| J Ray / TorchFT / Training Hub | 9, 10 |

## Phase 0 — Architecture hardening

The gate is **per-ADR, not global**: an ADR must be settled before work that
depends on its unresolved semantics, not before all work.

A blanket "no implementation until every ADR is accepted" was the original
wording and it does not survive contact with the repository — `xaytune/core/`
already exists on `main`. An implementation contract that forbids what has
already shipped stops a coding agent for no reason.

### Ratified by merged implementation

These are settled by working, tested code on `main`. Their acceptance is a
matter of record rather than of review:

| ADR | Ratified by |
|---|---|
| ADR-002 — aggregate/state-machine separation | `xaytune/core/state/machines.py`, transition tables verified equal to `04-state-machines.md` |
| ADR-010 — core dependency boundary | `xaytune/core/` imports on a bare interpreter with only pydantic and pyyaml |

### Accepted by decision

| ADR | Gates |
|---|---|
| ADR-001 — experiment control plane | the premise every later ADR assumes |
| ADR-005 — the persistence transaction contract | band B, including PR-004; every transaction boundary and repository invariant the repository must hold |
| ADR-006 — fingerprints (identity model) | PR-007 fingerprint framework; reuse policy decided separately in ADR-017 |
| ADR-007 — evaluation independence | band D; extended by ADR-015 |
| ADR-008 — versioned plugin ABI | band C; accepted 2026-09-22 |
| ADR-009 — checkpoint layers | band F; accepted 2026-09-28, including PR-018's local-only and legacy read-surface boundaries |
| ADR-011 — candidates, interventions, overrides | PR-005 event schema; supersedes ADR-003's two-level lineage and extends ADR-006 |
| ADR-012 — data position and resume semantics | PR-005 checkpoint schema; all adaptive recovery |
| ADR-013 — operation identity and cancellation | the first runtime implementation |
| ADR-014 — worker telemetry protocol | `RuntimeBackend.watch()`; PR-005 event schema |
| ADR-015 — durable evaluation lifecycle | PR-005 evaluation tables |
| ADR-016 — specs versus implementations | PR-005 experiment record |
| ADR-017 — reuse policy | band G; accepted 2026-10-01 with v1 training-artifact reuse disabled |
| ADR-004 — durable controller hosting | band H; accepted 2026-10-03 with the SQLite request mailbox, atomic admission, flock singleton and the PR-027/PR-028 boundary; §8 (lease, fencing, sweep) settled 2026-10-04; §3 mutation kinds and §9 caller-side host settled 2026-10-04 (PR-029) |

### Still Proposed, and what each actually blocks

This remains `Proposed`. It names the work it gates, so nothing is blocked
that does not depend on it:

| ADR | Blocks |
|---|---|
| ADR-018 — agent harness candidates | future H02–H12 harness track; does not gate Phase 5/6 |

### Superseded

| ADR | By |
|---|---|
| ADR-003 — scientific vs execution lineage | ADR-011; retained for the reasoning that led there |

No ADR is half-accepted any more. Status is used as a gate, so an ADR that was
`Proposed — partially superseded` or `proposed but ratified in effect` could not
be acted on: ADR-001 is now `Accepted` because everything after it assumes it,
ADR-007 because ADR-015 depends on it, and ADR-006's genuinely open half became
ADR-017 rather than a second status on one document.

ADR-005 was accepted on 2026-09-21, which unblocked band B. (Current status is
kept in the spec README, not here, so this history does not go stale.) It was expanded before acceptance rather than accepted as written: the
original fifteen lines covered one transaction, which was adequate when band B
owned the experiment aggregates and the outbox. Band B also owns
`RuntimeOperation`, the Action substrate and its linkage, operation and
cancellation intent, `RunRealization` projections and telemetry generation
durability, so the short version would have frozen the schema while leaving the
transaction boundaries that matter unstated.

The gate was **before PR-004**, not PR-005, for this document's own "what
freezes what" reason: PR-004 is where the tables and revision-based persistence
are written, and a repository built against assumptions ADR-005 then contradicts
has to be rewritten — or, more likely, kept.

**No ADR now blocks work that is ready to start.** The remaining `Proposed`
one, ADR-018, gates the separate harness track. ADR-004 was accepted on
2026-10-03, before PR-027, which unblocked band H.
Each must be accepted before its own band, not before PR-004. ADR-008 was
accepted on 2026-09-22, unblocking band C.
ADR-009 was accepted on 2026-09-28, settling the checkpoint-layer gate for band F
before PR-018 merges. It freezes the local codec/store/manager architecture and
preserves legacy readability through the existing trainer loader.
ADR-017 was accepted on 2026-10-01, unblocking band G. Its v1 decision disables
training artifact reuse.

From here, changes to these contracts should come from an implementation
finding, a failing test or a demonstrated contradiction — not from another pass
over the design on paper.

Exit criteria, per band rather than globally:

- every ADR listed as gating a band is accepted before that band starts
- module ownership agreed
- core dependency boundary agreed *(done)*
- the PR list for a band is updated to match its ADRs before the band starts

---

## Phase 1 — Core domain and persistence

### PR-001 — core IDs and shared types

Implement:

- typed IDs
- Actor
- ArtifactRef
- DatasetRef
- ModelRef
- errors

Tests:

- serialization
- ID ordering/validation
- no heavy imports

### PR-002 — Experiment / Node / Run / Attempt models

Implement immutable/domain schemas.

No controller.

### PR-003 — separate state machines

Implement transition definitions and validation.

No persistence yet.

### PR-004 — SQLite repository

Prerequisite: ADR-005 is accepted before this PR starts. Its transaction,
revision and state/event/outbox consistency rules constrain the first persistent
repository schema, not only PR-005's event integration.

Implement tables and revision-based persistence.

### PR-005 — transactional events, outbox, and operation journal

Implement:

- atomic aggregate transition + event + outbox persistence
- `runtime_operations` in migration 002, with operation ID, typed target
  (`training-attempt` | `evaluation-attempt`), type, canonical `request_digest`,
  state, runtime reference and revision (ADR-013)
- `RuntimeOperation` repository APIs to create, get by operation ID, list by
  target, list unresolved operations, and transition with revision checks
- operation transitions: INTENDED → SENT / CONFIRMED / FAILED;
  SENT → CONFIRMED / FAILED; CONFIRMED and FAILED are terminal
- atomic creation of a `RunAttempt` and its INTENDED submit operation, including
  the request digest and their events/outbox records, before any runtime call
- durable cancellation intent for an existing attempt through the same journal
- primary-key operation lookup and indexes for target and unresolved-state queries
- every transaction boundary and repository invariant in ADR-005 §3-§10, with
  the crash and concurrency tests of §11

An unknown submission outcome stays unresolved, not FAILED; reconcile it using
ADR-013. Reusing an operation ID with the same request returns the existing
record; changing its attempt, type or digest raises `IdempotencyConflict`.
Operation transition history uses the domain event log, not a separate journal
transition table. The outbox publishes events; it never dispatches runtime calls.

Tests: crash rollback of attempt + intent + events + outbox as a unit; committed
intent survives reopen; duplicate/conflicting operation IDs; legal/illegal and
stale-revision transitions; unresolved-operation queries; cancellation intent
survives restart. No runtime is required for these persistence tests.

### PR-006a — minimal durable Action substrate

Ships **migration 003** (`actions`, plus `runtime_operations.caused_by_action_id`
added together with its `REFERENCES actions(id)` -- SQLite cannot attach a
foreign key to an existing column afterwards, so the column waits for its
table).
Both 001 and 002 must land before Phase 2, because `handle.cancel()` is public
API there and ADR-013 cancellation needs a durable Action to hold the intent
while the operation carries the effect. The split is sequencing, not
optionality: 001 belongs to PR-004, 002 to PR-005, and 003 to this PR, which is
the order they land in.

ADR-013 defines cancellation as `CancelExperiment → CancelRun → runtime cancel`,
with the Action holding the *intent* while the operation holds the effect. Phase
2 exposes `handle.cancel()`. So the Action aggregate is required two phases
before the `PolicyEngine` that was originally bundled with it, and splitting the
two is cheaper than dragging policy forward.

This PR is the substrate only:

- `Action`, `ActionId`, `ActionStatus` and the Action state machine
- `ActionRepository`, committed in the same transaction as its events
- linkage from an `Action` to the `RuntimeOperation`s it causes
- exactly three action types: `CancelAttempt`, `CancelRun`, `CancelExperiment`,
  registered in a domain **action registry** rather than a database `CHECK` —
  SQLite cannot alter a `CHECK` in place, and Phase 4 adds many types
- `CancelAttempt` is workload-neutral: it targets a `training-attempt` or an
  `evaluation-attempt`, using ADR-013's spellings so an Action's target and its
  operation's target are the same vocabulary
- `ActionOutcome` (`APPLIED` | `SUPERSEDED` | `NOOP`), so ADR-013 §5's
  cancellation race is representable without overloading `status`
- the Action state machine from `04-state-machines.md` §5, using the
  `VALIDATED → EXECUTING` path: a controller-owned cancellation is never marked
  `APPROVED` by nobody, and `APPROVAL_PENDING` stays reachable but unused until
  PR-023
- Action + caused `RuntimeOperation` committed in one transaction (ADR-005 §5)

Explicitly **not** here: `PolicyEngine`, approval rules, budget authorization,
or any mutating action type. Those stay in Phase 4, where the interesting
question is authorization rather than durability. An action in this PR is
proposed and executed by the controller itself.

The dependency chain this creates:

```text
minimal durable Action  →  runtime cancellation  →  evaluation
                        →  policy/budget enrichment
                        →  recovery and interventions
```

### PR-006 — experiment graph

Implement:

- parents
- children
- lineage
- roots
- descendants
- candidate comparison metadata
- cycle prevention

Phase exit:

- experiment can be persisted
- multiple nodes can exist
- events are durable
- operation intents, request digests and outbox records survive repository restart
- **repository restart does not lose committed state or events** — a new process
  against the same database reloads every aggregate and event exactly as
  committed

This phase makes no claim about active workloads. There is no controller and no
runtime yet, so there is nothing in flight to reconcile; that is band C.

---

## Phase 2 — Compile/execute boundary

### PR-007 — CandidateSpec / TrainingSpec / fingerprint contracts

The compiler consumes a `CandidateSpec`, not a `TrainingSpec` (ADR-011).
`TrainingSpec` is the training *program* only; model and data are its siblings:

```text
CandidateSpec
├── ModelSpec
├── DataSpec
├── TrainingSpec          # SFT / CONTINUED_PRETRAIN / DPO / GRPO
├── RewardSpec?
├── EnvironmentSpec?
└── TrainingSchedule?     # pre-registered interventions
```

Implement:

- CandidateSpec with the composition above
- TrainingSpec: SFT, CONTINUED_PRETRAIN, DPO, GRPO schema skeletons — the
  control-plane spelling from `05-training-spec-and-compilation.md` §2, not
  `PRETRAIN`; the legacy `xaytune.pretrain()` entry point keeps its name
- fingerprint framework: `CandidateFingerprint`, `RunHistoryFingerprint` and
  `ArtifactLineageFingerprint`
  (seed belongs to `Run`, so it feeds the realization and never the candidate)

### PR-008 — compiler/runtime protocols

Implement:

- TrainerCompiler
- TrainingExecutionSpec
- RuntimeBackend
- ResolvedExecutionPlan
- CapabilityDocument skeleton

### PR-009 — LocalRuntime

Prerequisite: PR-005's operation journal and atomic attempt/intent APIs have
landed. Persist intent before `submit_or_get`; runtime submission cannot use
outbox delivery as a substitute for the journal (ADR-013).

Subprocess + operation idempotency.

### PR-009a — Observability and training telemetry contracts

Contracts and tests only, after LocalRuntime and before NativeWorker:

- observation policy separate from the transport contract
- typed training/resource/data/distributed/alignment and numerical-health observations
- typed checkpoint cursor, captured-state evidence and three-dimensional resume guarantees
- profiler artifact lifecycle, structured logs, trace and correlation context
- declarative redaction and the EventSink plugin boundary
- immutable JSON validation and dependency-isolation tests

NativeWorker should emit one shared Xaytune telemetry contract rather than
establishing a Native-specific observability vocabulary that Ray, TRL and
Training Hub later have to translate or replace. Exporters, GPU collection,
profiler execution, callback integration and policy decisions are deferred.

### PR-010 — NativeCompiler / NativeWorker

Wrap existing training loop.

Goal:

```text
CandidateSpec → NativeCompiler → TrainingExecutionSpec → LocalRuntime
```

### PR-011 — TRLCompiler

Start with SFT only, and plain-text datasets only: prompt/completion and chat
datasets make TRL choose completion-only loss, which the candidate cannot yet
express. The second compiler is the proof that the boundary is
trainer-neutral, so the acceptance test runs one candidate through both
compilers and asserts the control-plane contract is the same -- not the
weights or the losses. Its runs share the candidate's fingerprint and differ in
execution identity (ADR-011 §5).

Introducing the second trainer surfaced semantics the first one had been
choosing silently, and both compilers now refuse them rather than disagree:
non-zero weight decay (the native loop decays every parameter, transformers
exempts biases and normalization weights, and the candidate cannot say which
is meant) until `OptimizerSpec` carries an explicit, identity-bearing
weight-decay policy in a new candidate projection; and a half precision the
device cannot run, which the native loop used to replace with fp32.

### PR-012 — public ExperimentHandle / EmbeddedControllerHost

Support:

```text
submit
status
wait
cancel
events
```

Not `pause`/`resume`. Decided before implementation:

- **Training success is not candidate success.** A successful run moves the
  `Run` and its final `RunAttempt` to `SUCCEEDED` and records the artifact. The
  node stays `ACTIVE`, because its next legal phase is `EVALUATING`, and the
  experiment stays `ACTIVE`, because terminalizing it is a controller/planner
  decision. No training-only state-machine edges are added.
- **`wait()` means controller quiescence**, not a terminal experiment: it
  returns when all work the embedded controller can currently execute is
  settled and telemetry is drained. The result says so explicitly
  (`quiescent`, and the stage that would run next), so returning is never read
  as experiment success. PR-013 makes evaluation executable, which extends what
  `wait()` waits through without changing what it means.
- **`events()` is the durable control-plane history**: `DomainEvent`s for the
  experiment, replayed and then followed using the database `sequence` as the
  cursor, so a handle from `attach()` sees the same stream. Runtime telemetry
  stays behind `RuntimeBackend.watch()`. Its controller-significant
  consequences are persisted; per-step metrics are not copied into the event
  table.
- **`cancel()` goes through the Action substrate** as the ADR-013 §6 saga, and
  `CANCELLED` still means no owned workload is executing.
- **Not restart-reconciling.** The host is backed by durable state but does not
  adopt an in-flight attempt after its process dies; that is PR-012a.

### PR-012a — runtime-operation reconciliation

ADR-013 requires this of the **first** runtime, not of the daemon phase: an
unreconciled submission is an orphaned GPU job, and a controller that cannot
tell a lost submission from a running one either duplicates work or abandons
it. PR-009 introduces the operation id; this is where the controller learns to
use it after a restart.

Implement:

- the restart path: for every non-terminal attempt, `lookup_operation()` and
  `get_status()` before any resubmission decision
- reattachment to a live attempt, including resuming `watch()` from the durable
  `StreamCursor`
- escalation, not resubmission, when an adapter cannot report completed
  operations (ADR-013)
- the ADR-014 telemetry rule: a dead stream over a live workload advances
  `telemetry_generation` and records a degraded interval, and never mints a
  `RunAttempt`
- cancellation leaves no executing workload

As built, reconciliation runs in `attach()`, for the experiment attached to:

- implementations, only where they are used: the runtime is resolved, and a
  version other than the recorded one fails closed, only where an external
  effect is looked up, adopted or cancelled. The compiler is needed only to
  rebuild a request that must be re-issued, so only that path resolves it and
  checks its version. A running workload is adopted even if its compiler is
  no longer installed, and a settled experiment -- or a refusal already
  recorded -- attaches with neither
- per unsettled attempt, its submit operation decides: CONFIRMED is adopted;
  INTENDED/SENT is looked up; only "never received" is issued, under the
  recorded operation id and after checking the rebuilt plan's digest -- and
  only if the runtime can report completed operations, otherwise it escalates
- each adopted attempt has its own observer, tracked per attempt: `wait()`
  waits for all of them, including any adopted while it waits, and `close()`
  stops all of them
- telemetry resumes from the attempt's durable cursor
  (`telemetry_generation`, `telemetry_sequence`), which advances only in the
  commit of the effect an event caused
- a dead stream over a live workload advances the generation, records
  `TelemetryDegraded`, and reads the outcome from the runtime; an ending
  nothing observed escalates rather than being guessed
- an in-flight cancellation is carried on

Not in scope, deliberately: **ownership**. Two hosts attached to one
experiment at the same time would both adopt it. Neither can create a second
workload, because adoption never issues one, but their writes would conflict.
Leases belong to the daemon host (PR-027/PR-028).

Phase exit:

- SFT runs through new compile/execute path
- existing trainer remains functional
- process boundaries are serializable
- an `EmbeddedControllerHost` restarted mid-attempt reattaches to the running
  workload instead of orphaning or duplicating it

---

## Phase 3 — Evaluation and decision substrate

### PR-013 — Evaluation domain and durable lifecycle

This is where ADR-015 is actually implemented; the phase claims durable
evaluation, so it has to schedule it.

Implement:

- EvaluationSpec (no `seed` — see below) and MetricResult
- EvaluationRun and EvaluationAttempt, with `seed`/`replicate` on the run
- both state machines, per the ADR-015 tables: no `CHECKPOINTING`, no
  `RECOVERING`, `PREEMPTED` from `QUEUED` onwards
- `EvaluationResult.evaluation_run_id`, so a sample is attributable to the
  execution that produced it
- `evaluation_runs` and `evaluation_attempts` persistence, added as a migration
  on top of PR-005's schema
- idempotent evaluation submission through `submit_or_get` (ADR-013)
- restart reconciliation for in-flight evaluations
- the **reconciliation** rule for a node in `EVALUATING` (ADR-015 §5), resolving
  to exactly one of:

  ```text
  a required EvaluationRun is non-terminal        -> wait
  all required runs terminal, results present     -> reconcile node to DECIDING
  neither                                         -> raise EvaluationStalled
  ```

  Not "implies a non-terminal `EvaluationRun`, otherwise an incident" — ADR-015
  rejected that form, because it fires on every successful evaluation during the
  instant between the run reaching `SUCCEEDED` and the node advancing. The
  middle case is what repairs the lag instead of reporting it.

As built:

- **The public trigger is `ExperimentSpec.evaluation`** (optional; its
  `EvaluationSpec.evaluator` is singular, one evaluator measuring any number
  of metrics, bound by the host to the evaluator's version and declared
  determinism -- several evaluators are several runs in a cycle). `None` leaves a trained node `ACTIVE` with
  `next_stage="evaluation"`, as before. Set, the host evaluates the trained
  model on the experiment's runtime once training succeeds, and the node
  reaches `DECIDING` with `next_stage="decision"` -- advice only; deciding is
  PR-015. The evaluation is orchestration, never candidate identity: it does
  not enter the `CandidateSpec`, its fingerprint or its compilation. There is
  no `handle.evaluate()`: coordinating that is the controller's job.
- **Evaluation's own wire type.** `EvaluationExecutionSpec` is a sibling of
  `TrainingExecutionSpec`, sharing only transport fields; `ExecutionSpec` is the
  wire-only union, discriminated on `api_version`, and a plan refuses a spec
  whose workload is not its target's. The training spec's field set is
  unchanged, so no recorded training `request_digest` moves (pinned by a test).
  LocalRuntime stays workload-blind.
- **Results travel inline, under `xaytune.telemetry/v1alpha3`** (ADR-014): the
  completion carries the final `MetricResult`s; success needs that completion
  *and* the runtime's `succeeded`, recorded in one commit. The completion is
  held **durably** on the attempt from the moment it arrives, so a stream that
  dies over a live workload, and then the controller, cannot lose it.
- **Provenance is exact.** A result agrees with its run in node, fingerprint,
  subject identity and digest, every metric's evaluator, version and seed, and
  each report's producer; the repository refuses one that does not, and the
  database independently refuses the column-level subset (run, node,
  fingerprint, subject id and digest, one result per run, no edits). Drift in
  what a worker reports fails the evaluation, with the reason on the event.
- **Preemption** fails the run (the attempt `PREEMPTED`) and stalls the node;
  retry, when it exists, is a new attempt.
- **Cycles** (ADR-015 implementation notes): required runs are the current
  cycle's, so an earlier round never satisfies a later one.
- **The first evaluation sample's seed is the training run's, replicate 1** --
  the embedded controller's default, not a coupling: an evaluation seed means
  nothing about training, stays outside `EvaluationFingerprint`, and a planner
  may schedule further replicates with seeds of its own.
- **One reconciliation, two workloads.** Issue, adoption and restart
  reconciliation are shared with training; the evaluator, like the compiler,
  is resolved and version-checked only where a request is rebuilt. `attach()`
  also carries forward what a crash between two commits leaves behind.
- **The `Evaluator` contract** (`xaytune.evaluation`: descriptor, determinism,
  `prepare()`) with no built-in evaluators. Tests use a scripted evaluator whose
  worker is a real process.
- **Not in PR-013:** evaluation reuse lookups (ADR-015 AC-4, 4a) and choosing
  a different runtime for evaluation.

### PR-014 — the native evaluator

Wrap current metrics/lm-eval. As built, the native half, with lm-eval split
into PR-014b:

- **`NativeEvaluator`** (`xaytune.evaluation.native`), registered by default as
  `native`, with its worker `xaytune.workers.eval_native` under telemetry
  v1alpha3. The narrow, exact surface: a local model artifact, a local JSONL
  file of plain text pinned by `DatasetRef.content_digest` (which the worker
  verifies before measuring), and next-token `loss`, `perplexity` and
  `token_accuracy`, aggregated over tokens. Format, truncation length, batch
  size, precision (`fp32` only) and metrics are required `EvaluatorSpec.config`,
  so they are in the `EvaluationFingerprint`. The tokenizer is the model's own,
  so it is part of the subject. Slices, dataset revisions and splits, and
  dataset fingerprints the worker cannot verify are refused.
- **It reuses the trainer's text pipeline, not `xaytune.eval.evaluate()`.** At
  the time, that function scored the logits at *i* against the token at *i*,
  not at *i + 1*, and averaged losses per batch. Its numbers described neither
  next-token prediction nor the data independent of batching. It was
  corrected separately (issue #36).
- **`SEEDED`, never `DETERMINISTIC`.** Floating-point results depend on the
  device and library versions. The run's seed is applied and recorded on
  every metric, and the report names the environment.
- **`Evaluator.supports(spec)`**, asked by the host at submission: an
  evaluation that cannot run exactly as declared is refused before anything
  trains. A refusal only `prepare()` can make, about the trained subject,
  fails the evaluation run with the reasons and stalls the cycle. It no
  longer escapes the controller task.
- **No reuse.** A test pins that a second evaluation of the same artifact, with
  the same fingerprint and seed, runs a second workload. The reuse lookup
  (ADR-015 AC-4, 4a) stays a separate decision.
- **Restart safety with the real evaluator**: the PR-013 crash points
  (`evaluating`, `eval-lost-response`, `eval-never-sent`, `eval-stream-lost`)
  with the native worker, each finishing with one attempt and one workload.

### PR-014b — lm-eval evaluator

Built after `1.0.0a1`. It was split from PR-014 because it needs a resolution
step the contract did not have: an lm-eval task names a mutable definition and
a hub dataset, and `prepare()` must never resolve anything. As built:

- **`ResolvableEvaluator.resolve(spec)`**, an optional capability the host
  detects, called once, at submission, between two checks:
  `supports(declared) → resolve() → supports(resolved) → recorded and
  fingerprinted`. The `Evaluator` contract is unchanged; an evaluator without
  `resolve()` has its spec recorded as declared, which a test pins with an
  evaluator written to the pre-PR-014b contract.
  - It is the only evaluator step allowed to use the network.
  - The second `supports()` exists because a name can pass the first check
    and resolve into something refused, such as `gsm8k`, a generation task.
  - Resolution may not rename the evaluator.
  - Nothing resolves again after submission: on restart, `prepare()` rebuilds
    the request from the record, which a test pins with an evaluator that
    fails if asked.
  - `NativeEvaluator.resolve()` returns its spec unchanged.
- **`LMEvalTaskBinding`**, recorded under `EvaluatorSpec.config["binding"]`, so
  the `EvaluationFingerprint` covers it and no migration is needed. It holds:
  - the task, and its `metadata.version`;
  - a digest of the task's YAML definition, with includes merged and function
    references made independent of the install location;
  - the lm-eval release, `0.4.13` exactly, which the `eval` extra pins;
  - the dataset path, name and Hub **commit**;
  - the `output_type`, `num_fewshot` and the task's metric list.

  A binding supplied by the caller is refused, and `limit`, `batch_size` and
  `precision` stay ordinary config.
- **What is accepted.** One registered task, defined in YAML, scored by
  log-likelihood (`multiple_choice` or `loglikelihood`), reporting only `acc`
  and `acc_norm`.
  - Refused: groups and tags, tasks implemented in custom Python, `unsafe_code`,
    `custom_dataset`, datasets needing `trust_remote_code`, a dataset the Hub
    cannot pin, and a task with no explicit `metric_list`.
  - Generation is refused because it adds unpinned surface (decoding settings,
    stop sequences, filters, answer extraction), not because it samples.
  - Binding reads the task's YAML only and imports none of its Python.
- **The worker** (`xaytune.workers.eval_lmeval`, telemetry v1alpha3).
  - It refuses another lm-eval release, and a task whose installed definition
    no longer digests to the binding.
  - It loads the dataset through the task's `dataset_kwargs` at the recorded
    commit, and scores in fp32.
  - The run's seed is lm-eval's Python, NumPy, Torch and few-shot seed.
  - Each metric records lm-eval's standard error and its **effective** sample
    count, which is the documents scored after `limit`, not the dataset's size.
- **`SEEDED`**, and no reuse, as for `native`.
- **`examples/control_plane/06_train_and_benchmark.py`** trains, runs `arc_easy`
  (`--limit 20`) and decides on `acc`.

### PR-015 — DecisionEngine

Implement deterministic objective/constraint decisions. As built:

- **`DecisionContext` → `DecisionProposal` → `Decision`**
  (`xaytune.core.domain.decision`).
  - The context is assembled from durable state: the experiment's objective
    and the node's **current-cycle** results.
  - The engine returns a proposal: the outcome, the reason, the evidence
    (each comparison, with its value, threshold, result and evaluator), the
    engine name and version, the result ids, and an **input fingerprint**.
  - The repository records it as an immutable `Decision`, adding an id, a
    time and an actor.
  - The fingerprint is `decision_input_identity_v1`, an explicit versioned
    projection of what is evidence: the objective, and each result's id, run,
    evaluation fingerprint, subject id and digest, and each metric's name,
    value, slice, evaluator and version, seed, sample count, confidence
    interval and standard error. It is canonically sorted, and excludes
    timestamps, reports and metadata.
- **`DecisionEngine`** (`xaytune.decision`), synchronous and pure: nothing but
  its context is read and nothing is minted, so the same context gives a
  byte-identical proposal. The spec sketch in 10-evaluation-and-decisioning §8
  is `async`; purity made that unnecessary.
- **`ThresholdDecisionEngine`, the v1 rules**:
  - a missing objective or constraint metric → undecidable;
  - any constraint violated, under all six operators applied exactly → `REJECT`;
  - no target → undecidable;
  - target met (maximize ≥, minimize ≤) → `STOP_SUCCEEDED`;
  - otherwise `STOP_FAILED`.

  It compares point estimates only, reads no `sample_count`, and considers
  only unsliced metrics; a metric reported twice in a cycle is ambiguous and
  undecidable. The outcome vocabulary is these three; the rest of §9 arrives
  with what can act on it.
- **Persistence, migration 007.** `record_decision` writes the decision,
  moves the node with its id, and applies the outcome to the experiment, with
  events, in **one commit**:
  - `STOP_SUCCEEDED`: node `COMPLETED`, and an `ACTIVE` experiment
    `SUCCEEDED` with `best_node_id` set;
  - `STOP_FAILED`: node `REJECTED`, and the experiment `FAILED`;
  - `REJECT`: node `REJECTED`, and the experiment **unchanged**. It judges the
    candidate, not the experiment, so the planner of band G can propose
    another without undoing a persistence rule. `next_stage` is then
    `"planning"`.

  It is idempotent per cycle: the same proposal returns the recorded
  decision, and a different one raises `DecisionConflictError`.

  The decision must name exactly the node's current-cycle results
  (`ProvenanceError`). The database enforces one decision per node per cycle,
  refuses a decision filed under another experiment, and refuses edits.
  `defer_decision` records `DecisionDeferred` once per cycle for an
  undecidable node, which stays `DECIDING`.
- **Host.** Deciding follows reconciliation into `DECIDING`, and `attach()`
  decides a node a crash left there. A crash point, `deciding`, proves a
  restart decides once, and a decided experiment is never decided again.

### PR-015b — adaptive decisions (`BRANCH`)

Lands before PR-024, because it defines what a candidate that finished valid
but short of the target looks like, and the planner builds on that. As built:

- **`DecisionOutcome.BRANCH`** (spec 10 §9): constraints held and the target is
  not met, so another candidate may be explored. The node becomes
  `COMPLETED`, not `REJECTED`. The experiment is **unchanged** and stays
  `ACTIVE`, and `best_node_id` stays unset. `BRANCH` creates no candidate:
  PR-024 proposes one and PR-025 materializes it.
- **`AdaptiveThresholdDecisionEngine`** (`adaptive-threshold` 1.0.0) uses
  `ThresholdDecisionEngine`'s comparison exactly. The only difference is that
  a missed target is `BRANCH` rather than `STOP_FAILED`. A violated constraint
  is still `REJECT`, a met target is still `STOP_SUCCEEDED`, and no target is
  still undecidable.
- **`ThresholdDecisionEngine` 1.0.0 is unchanged** and stays the host
  default. The single-candidate reading of a missed target does not silently
  change, and the two engines give identical input fingerprints for the same
  context.
- **Not budget-aware.** A decision answers what the evaluation established
  about the candidate. Whether the experiment may still spend on another
  candidate is asked when one is proposed. An exhausted budget ends the
  experiment there (`BUDGET_EXHAUSTED`) and never rewrites a decision.
- **`next_stage = "planning"`** now follows when every candidate of an
  `ACTIVE` experiment is decided on its merits: `REJECTED`, or `COMPLETED`
  after a `BRANCH`. A failed or cancelled candidate is still
  `"failure-handling"`.
- No migration: `decisions.outcome` is unconstrained text, and
  `DECIDING → COMPLETED` was already a node transition.

### PR-016 — BudgetLedger

Implement reserve/commit/consume/release. As built:

- **The ledger** (migration 008) is append-only; the database refuses updates
  and deletes. Each entry is idempotent under (subject kind, subject id,
  dimension, kind), and its amount is a positive decimal (no zero
  reservations).
  - Balances are derived, never stored:
    `outstanding = max(reserved − consumed − released, 0)` and
    `remaining = limit − consumed − outstanding`. A commit is provenance and
    subtracts nothing again.
  - **Atomic settlement is the invariant.** Every entry commits with the
    transition that causes it: an attempt's failure and its slot's
    release commit with it reaching `FAILED`. `settle_budget()` on
    attach is only a safety net, and normally writes nothing.
- **Dimensions.**
  - `max_runs` is a hard quota per `Run`, not per attempt. It is reserved when
    the run is recorded, committed on the first confirmed submission, consumed
    when the run ends, and released if it ends unsubmitted.
  - `max_failures` is a hard quota: one per `FAILED` training or evaluation
    attempt.
  - `max_parallel_runs` is a capacity (a semaphore on live training
    attempts). When it is full, the next attempt waits.
  - Refused at submission: `max_wall_time_seconds`, `max_gpu_hours`,
    `max_tokens`, `max_cost`. Wall time was built and withdrawn in review:
    attempt timestamps are stamped when the controller observes a change, so
    a restart that replays a worker's start after it finished records
    seconds for an hour's run. It returns once the runtime contract reports
    duration authoritatively (a runtime `started_at`/`finished_at`, or
    resource usage).
  - Evaluations spend failures, not runs or parallel runs.
- **Exhaustion.** A used-up quota refuses the next budgeted effect before any
  write. In-flight work drains, and the experiment becomes `BUDGET_EXHAUSTED`
  once nothing is live. Consuming past a limit emits `BudgetOverrun`;
  reaching it exactly does not. What an overrun should cause is left to
  PR-023.

Phase exit:

- experiment can train → evaluate → finish
- budget and evaluation metadata are durable
- an evaluation killed mid-flight is recovered on controller restart rather
  than leaving the node in `EVALUATING`

---

## Phase 4 — Policy, budget and mutating actions

> **Reordered.** Resilience was Phase 4 and the Action substrate Phase 5. ADR-011
> makes a `TrainingIntervention` the outcome of an **approved Action**, so an OOM
> that reduces micro-batch size is an operational action that still needs
> deterministic authorization. Recovery cannot precede the thing that authorizes
> it. The PR numbers below keep their original identities so cross-references
> elsewhere still resolve; only the phase order changed.
>
> **The Action aggregate itself is no longer here.** PR-006a builds the durable
> substrate and the cancellation actions in band B, because ADR-013 cancellation
> needs them by Phase 2. What remains in this phase is authorization and the
> mutating action types — the part that genuinely depends on policy and budget.


### PR-022 — mutating action types

The substrate exists from PR-006a; this adds the types that change training:
`ChangeLearningRate`, `ResizeMicrobatch`, `ChangeRewardCoefficient` and the rest,
each with its validation rules.

**As built** (`xaytune/core/domain/actions/`): typed, versioned *intent*
compiled into the existing `Action.type + target + payload`. There is no
migration, and nothing is executed.

- **Contract.**
  - `ActionSpec` carries `type`, `version`, a typed `target` and typed
    parameters. It refuses unknown fields, NaN and infinity, and holds data
    only.
  - `ActionDescriptor` records the type, version, `MutationClass`, target
    kinds, spec class, provider (`None` means built in, otherwise a
    `PluginDescriptor`) and an optional static `validator`.
  - `MutationClass` (`OPERATIONAL`, `SCIENTIFIC_INTERVENTION`, `EXPERIMENT`)
    is declared by each type, never inferred. `OPERATIONAL` means it does not
    alter scientific identity, not that it must produce an `ExecutionOverride`.
- **Vocabulary.**
  - Operational: `resize-microbatch`, `change-gradient-accumulation`,
    `change-worker-count`, `change-checkpoint-interval`, `cancel-attempt`,
    `cancel-run`.
  - Scientific: `change-learning-rate`, `change-scheduler`, `change-warmup`.
  - Experiment: `cancel-experiment`, `reject-candidate`, `promote-candidate`.
  - Deferred: `change-reward-coefficient` (`RewardSpec` has no weights),
    `stop-experiment` (no draining lifecycle), and changes to the dataset,
    base model, algorithm, optimizer, LoRA rank or adapter.
- **Durable envelope.** The payload is `{"schema_version", "parameters"}`,
  canonical at every depth, plus `provider` (provider, plugin name, API
  version) for a plugin's type. `plugin_version` is excluded on purpose: the
  payload is part of an action's request identity, and a compatible upgrade
  is the same contract (ADR-008). `type` and `target` are not repeated.
  Cancellations keep `{}` as implicit version 1, so the rows `1.0.0a1` wrote
  are unchanged.
- **Reading and writing.**
  - `ActionStore._insert` runs `validate_intent` (schema, then the plugin
    validator), so a malformed action is never written.
  - Reads never need the schema, so history outlives a plugin. `spec_of`
    fails closed on an unknown type, version or provider.
- **Registration.** Only explicit `register_action`, with no entry-point
  discovery; one shared discovery for every plugin kind is planned instead.
  `register_action_type(str)` stays importable but raises.
- **Runtime-neutral.** `workers` counts logical training workers, as
  `ResourceRequirements.workers` does. Mapping them onto pods, actors or GPUs
  is the runtime's job (PR-033).
- **Context goes to PR-023.** Whether the target exists and is live, runtime
  support, elastic ranges, budget, approval and policy are PR-023's to check.
- **Later extension points.**
  - Custom budget dimensions need a meter that stays authoritative across a
    controller restart before they can be hard-enforced; otherwise they are
    refused or observe-only (the lesson of PR-016's wall time).
  - Custom objectives extend through `DecisionEngine`, not by subclassing
    `Objective`.

### PR-023 — PolicyEngine

Budget authorization pieces land here too: recovery in Phase 5 proposes Actions,
and an Action that cannot be authorized cannot be applied.

**As built.** Validation, then authorization, then optional human approval,
recorded as a durable governed Action. Nothing is executed.

- **Flow.** `ControlPlaneRepository.propose_action` (host:
  `ExperimentHandle.propose`):
  - `PROPOSED → VALIDATING`, then `REJECTED` if the action does not apply,
    with its problems recorded and no policy consulted;
  - otherwise `VALIDATED`, and then policy decides:
    - `ALLOW` leaves it `VALIDATED`; the decision is the authorization, and
      there is no `AUTHORIZED` state;
    - `DENY` moves it to `REJECTED`;
    - `REQUIRE_APPROVAL` moves it to `APPROVAL_PENDING`.
  - The Action, its decision and every step's event commit together.
- **Validation** (`applicability_problems`) is built in and pure.
  - For every action: the experiment is `ACTIVE`, and the target exists in it.
  - Operational run changes need a run that has not ended; interventions need
    an `ACTIVE` run; `reject-candidate` needs a node in no terminal state of
    `NODE_MACHINE` (`FAILED` included), and `promote-candidate` a `COMPLETED`
    one.
  - `change-worker-count` needs `elasticity.supported is True` and a count
    within the elastic range and, when declared, the distributed range. If
    either section is missing or says unsupported, the proposal is refused.
- **Authorization.**
  - `PolicyEngine.evaluate(spec, context)` is pure, like `DecisionEngine`.
  - `PolicyContext` is the snapshot, and `policy_input_identity_v1` gives the
    fingerprint: experiment status and revision, the action's type, version,
    provider, mutation class and parameters, the target's status and
    revision, the proposer's type and id, `BudgetStatus` and the runtime's
    `CapabilityDocument`. No time.
  - Everything policy can read is identified. The proposer is a
    `PolicyProposer` (type, id), not an `Actor`: metadata is provenance, kept
    on the Action. A test pins the field set of every model in the snapshot,
    so a new field is a deliberate v1-or-v2 choice.
  - The repository reads the snapshot again inside the transaction, refuses a
    changed one (`StalePolicyContextError`), and refuses a proposal whose
    fingerprint is not the snapshot's or whose engine name and version are
    not those of the engine asked (`ProvenanceError`), before writing
    anything.
  - `DenyAllPolicy` is the host's default, so with no policy configured every
    proposal is denied, with a durable decision. `RulePolicyEngine` is first
    match wins, and its version names the rule set.
- **Record.** Migration 009, `policy_decisions`:
  - columns: id, `action_id` (unique), experiment, engine name and version,
    verdict, `input_fingerprint`, `payload_json` holding the decision and its
    whole snapshot, and `created_at`;
  - append-only triggers, and a trigger checking that a decision belongs to
    its action's experiment.
  - `Action.policy_decision_id` is set in the same transaction.
- **Approval.**
  - Only an actor of type `human` may approve or reject, and only from
    `APPROVAL_PENDING`. It approves the recorded proposal, never re-running
    policy.
  - The same human (type and id) giving the same answer and reason again is
    recognised and writes nothing, whatever the actor's metadata, which stays
    provenance on the original event. Any other answer is refused (`ApprovalConflictError`).
  - There is no role or group model yet.
- **Cancellation** is outside policy. `propose_action` refuses `cancel-*`
  specs (`CancellationNotGovernedError`), so there is no second cancellation
  path, and cancellations keep `policy_decision_id = NULL`.
- **Quiescence and next stage.**
  - `unsettled_work` means control work actively in progress whose outcome
    must still be driven or reconciled: operations `INTENDED` or `SENT`, and
    actions `PROPOSED`, `VALIDATING` or `EXECUTING`.
  - A governed action resting in `VALIDATED`, `APPROVAL_PENDING` or
    `APPROVED` waits for a person or an executor, not a controller, so it
    does not keep `wait()` from returning. Cancellations never rest in those
    states.
  - Instead, `ExperimentResult.next_stage` shows it, ahead of the stages
    derived from the candidates: `"action-approval"` for `APPROVAL_PENDING`,
    and `"action-execution"` for an action that `awaits_execution`
    (`VALIDATED` with `ALLOW`, or `APPROVED` with `REQUIRE_APPROVAL`).
- **Identity versions are frozen.** `policy_input_identity_v1` is an explicit
  projection of named fields, including those of `BudgetStatus` and every
  `CapabilityDocument` section. A field added to those models later does not
  change v1. A field policy must see becomes `policy_input_identity_v2`,
  recorded as such. `PolicyContext` rebuilds every nested model as its exact
  v1 type from the projected fields, and `parameters`, `provider` and
  capability `extensions` as base `FrozenDict`s of base values at every
  depth, so a subclass carrying a later field or attribute reaches no engine, and the stored snapshot is exactly what policy saw. A
  runtime input policy must see goes in `CapabilityDocument.extensions`
  (identified by v1) or in a v2.
- **Budget** is policy input only. None of the built-in actions has an
  authoritative charge, so nothing is reserved.
- **For the executor (Phase 5).**
  - It may carry out a non-cancellation action only if it is either
    `VALIDATED` with an `ALLOW` decision, or `APPROVED` with a
    `REQUIRE_APPROVAL` decision. A `VALIDATED` action with no decision is
    never executable.
  - It must check again, immediately before any effect, that the action still
    applies, and refuse or supersede a stale one.
  - A policy decision authorizes the snapshot it recorded, and approval
    approves that proposal; neither speaks for later state.

## Phase 5 — Resilience, recovery and interventions

### PR-017 — incident model and detectors

**As built.** Structured observation → detector candidates → deterministic
classification → immutable incident and its event → stop.

- The domain model uses the full category vocabulary in the resilience spec;
  the initial detectors recognize process failure, explicit CUDA OOM, symbolic
  NaN/Inf, and checkpoint write/corruption/compatibility failures. Exact reason
  codes are classified; free-form detail and ambiguous exception names are
  evidence only. Unrecognized or contradictory diagnoses become `UNKNOWN`.
- Training and evaluation reuse their existing telemetry observations and
  envelopes. Evidence preserves the full envelope, detector names/versions,
  candidates, and classifier name/version. Intentional process cancellation is
  distinguished by the runtime's structured cancellation flag.
- Migration 010 stores incidents once per `(target, stream generation,
  sequence)` (ADR-014). This is observation identity, not a recovery-loop
  signature. Identical replay returns the original incident, including after
  restart or a classifier upgrade; conflicting evidence is refused.
- `record_incident` validates ownership from the recorded attempt/run, and
  commits the incident, observational event/outbox, and replay cursor together.
  The incident table is authoritative; `incidents.for_attempt(target)` reads its
  members without maintaining a second ID list on the attempt or advancing its
  aggregate revision. SQL also enforces ownership and append-only records.
- Both controller observation loops record incidents. Polling still determines
  the workload's terminal outcome; it is not used to invent a diagnostic signal
  or infer a lost event. No Action, RuntimeOperation, attempt, or node is created
  by incident detection. No recovery decision, retry, or checkpoint handling.

### PR-018 — checkpoint codec/store/manager

Start local-only.

**As built:**

- `xaytune.checkpoints` separates codec, store and manager (ADR-009).
  `SerializedStateCodec` packages already serialized components using the
  ADR-012 state manifest. Trainer adapters remain responsible for capturing
  and applying actual state. The codec verifies every state/sampler/worker RNG
  reference against bundled bytes, digest and producer, without loading ML
  libraries or pickle.
- Legacy directories remain readable unchanged through
  `xaytune.trainer.checkpointing.load_checkpoint`. They are not automatically
  eligible for the new manager/recovery path; absent provenance, cursor, RNG and
  intervention evidence cannot support `FULL + EXACT`. Explicit import/migration
  remains future work, as clarified in accepted ADR-009 and ADR-012.
- The versioned manifest carries producer/candidate/execution provenance,
  training position, cursor, intervention capture, guarantee, codec/version,
  exact-match compatibility and mandatory manifest/file digests. The existing
  telemetry validator enforces evidence for resume claims; missing legacy
  state is never upgraded into `FULL` or `EXACT`.
- `LocalCheckpointStore` validates and fsyncs same-filesystem staging, then
  atomically renames into the committed namespace. Only published valid
  bundles can be listed/localized. Stable checkpoint IDs and manifest identity
  make publication replay-safe. Changed content under one ID conflicts,
  including across processes. Identical retries discard their redundant staging.
- The manager checks candidate, codec/layout/framework/topology compatibility,
  dataset/ordering identity and requested guarantee before decode.
  `restore_recorded` additionally binds bytes to the durable producer,
  candidate, execution fingerprint and reported capture. Decode returns local
  encoded files for an adapter to apply; it records no achieved trainer restore.
- Migration 011 and the training observation loop record typed
  `CheckpointCommitted` reports. Report, `CheckpointRecorded` event/outbox,
  delivery receipt and cursor share one transaction. Replay or re-emission
  records one semantic checkpoint/event; changed evidence or producer/payload
  is refused. Reports and receipts are append-only. Reports do not prove byte
  integrity: a coordinator must use the manager before selecting a checkpoint.
  `repository.checkpoints.for_attempt(id)` is authoritative; recording does
  not maintain a second output projection or advance the attempt revision.
- No recovery plan/coordinator, automatic resume, action execution, new
  attempt/node, live trainer integration, remote/distributed store, resharding,
  retention/deletion or legacy conversion. Existing compilers still refuse
  checkpoint intent until their adapters implement capture. Local storage
  requires a trusted filesystem supporting atomic directory rename and fsync.
  File I/O is synchronous within the async local APIs.

### PR-019 — recovery episodes + decision-only coordinator

**As built:**

- Immutable attempt-scoped `RecoveryEpisode`, `RecoveryEpisodeId`, original explicit
  `RecoveryRequest`, append-only accepted/late incident membership and `RecoveryPlan`
  decision revisions. Closure derives from durable successor existence; no mutable
  episode state/effective pointer. Every accepted evidence extension appends a plan.
- Canonical versioned `RecoveryInputsV1` replaces aggregate dictionary snapshots.
  Pure authority-based arbitration considers all retained diagnoses. Fatal evidence
  fails, specialised/unknown/evaluation paths pause, compatible generic evidence
  selects validated resume or explicitly permitted retry. No category priority table.
- Same-run and prior-attempt checkpoint selection preserves FULL + EXACT optimizer
  boundaries, consumer RestoreContext, deterministic newest-valid fallback and
  validation-only byte/provenance inspection outside the write lock. The published
  optional codec validation declaration preserves the historical v1alpha1 ABI.
  Compatibility/corruption failures become ineligible; programmer errors propagate.
  Cheap blocking paths do not assess checkpoints or fabricate eligibility entries.
- Actual-attempt and experiment recovery limits count episode reservations, not
  observations/revisions. A replacement effective decision releases/replaces its
  own unit. Repeat limits count distinct prior accepted-signature episodes: limit 2
  permits prior 0/1/2, refuses 3. Execution identity equality is not a generic loop.
- Migration 012 contains three append-only tables, sequence/ownership/closure
  guards and derived effective/closure views. First episode/memberships/decision
  and each later revision commit atomically with RecoveryPlanned/event outbox.
  Separate incident membership admission emits RecoveryEvidenceAttached and may
  intentionally leave a fail-closed uncovered-evidence gap until replanning.
- Recording rechecks typed database projections/report bindings under the write
  lock and derives the decision from trusted coordinator eligibility evidence.
  It cannot prove current checkpoint bytes from a DB row. Future execution MUST
  revalidate selected checkpoint bytes and atomically guard openness/current
  revision/coverage before creating a successor and applying governance.
- Reconciliation processes typed attempt groups, replays without current config,
  repairs evidence gaps with stored requests, and requires explicit reconstruction
  only for first episode creation. It stops at the first unresolved request gap
  because later limits/repeats depend on preceding history; reconstruct and rerun.
  Superseded targets with no episode remain historical Incident audit only.
- Future append-only execution receipts can reference episode/plan/outcome/successor
  without mutating episodes/plans. No receipt behavior, recovery execution, Actions,
  RuntimeOperations, new attempts/nodes, overrides, scientific mutations, budget
  ledger reservations, controller auto-execution or harness implementation here.

### PR-020 — adaptive OOM recovery

Implement in reviewable layers:

- pure deterministic OOM planner with a versioned failed-execution input and one
  `ResizeMicrobatch` action spec carrying both micro-batch and accumulation;
  autonomous proposals preserve effective batch exactly;
- append-only `RecoveryExecutionReceipt` linked to episode, decision revision,
  governed action and successor/operation where executed;
- governed execution with checkpoint and effective-plan revalidation, then atomic
  successor attempt/override/runtime-operation intent recording;
- compilation/runtime resolution of durable successor overrides and injected
  OOM → resumed continuation validation.

### PR-021 — numerical recovery

Recovery that changes LR to stabilise a continuing run records a `TrainingIntervention`
on that run through the Action path (ADR-011). It does not create a new node. Forking
is for alternatives you want to compare.

Implement in reviewable layers:

- pre-executor foundation: immutable `TrainingIntervention` / `InterventionApplication`
  with typed triggers, explicit replay policy and `LearningRateMutation`; append-only
  persistence (migration 014); the `RunRealization` projection with history and
  artifact-lineage fingerprints; a pure `NumericalRecoveryPlanner` driven by an
  explicit `NumericalRecoveryPolicyV1`; a separate numerical Action binding; and
  governed `ChangeLearningRate` proposals through the existing policy path;
- execution (PR-021b), after its own architecture review, using the checkpoint-backed
  successor model pinned in 08 §9a (no live-worker mutation in v1): validated
  FULL+EXACT checkpoint → atomic successor creation → restore and apply the
  intervention → confirmed `InterventionApplication` → Action `SUCCEEDED`, under
  ADR-013 intent-first rules. The successor closes the numerical episode.

Phase exit:

- injected OOM recovers automatically
- scientific vs operational lineage is correct

---

## Phase 6 — Planner, branching and local MVP

> Moved after resilience. The end-to-end MVP scenario is *adaptive* training —
> it OOMs, recovers with an execution override and continues — so it cannot be
> demonstrated before recovery exists. Previously PR-026 sat two phases ahead of
> the machinery it exercises.

### PR-024 — RuleBasedPlanner

The pure planning contract and the first useful deterministic planner. As
built:

- **Ownership.** The original rule list predated PR-015 and PR-019–021 and
  is retired. Each item is owned elsewhere:
  - objective and constraint verdicts belong to the DecisionEngine
    (`STOP_SUCCEEDED`, `STOP_FAILED`, `REJECT`, and `BRANCH` from PR-015b);
  - OOM, numerical and other execution failures belong to recovery
    (PR-019–021);
  - plateau detection is **deferred**: training metrics are not durable, and
    a plateau over candidates needs the multi-candidate history PR-025/026
    create.

  The planner answers one question: *which candidate should be tried next?*
- **`PlannerSpec`** (ADR-016) sits beside `CompilerSpec` and `RuntimeSpec`.
  `ExperimentSpec.planner` is optional, and the host binds it at submission.
  Binding resolves the kind and checks the plugin API. It also validates the
  config into the planner's typed configuration and records the version, so
  `Experiment.planner` holds the bound spec. A spec that cannot be bound is
  refused with nothing recorded, and a restarted host rebuilds the planner
  from the record (`_recorded_planner`). Records written before PR-024 have
  `planner = None` and load unchanged. Nothing invokes the planner yet.
- **`Planner` protocol** (`xaytune.planning`): `descriptor`, the bound `spec`,
  and `async propose(PlanningContext) -> tuple[CandidateProposal |
  ActionProposal, ...]`. It is pure: no clock, id, repository, environment or
  runtime. The same bound planner and context give identical proposals.
  `NoOpPlanner` always returns `()`.
- **`PlanningContext`** (`xaytune.core.domain.planning`) is a serializable
  projection that `ControlPlaneRepository.planning_context()` assembles
  read-only from the relationships, never from child-id lists. It holds:
  - the experiment's status and objective;
  - each node's status, parents and candidate, with its **current** (v2)
    fingerprint recomputed from the stored snapshot;
  - each node's decisions, and its evaluation results with their cycles;
  - the budget status.

  Its identity is `planning_context_identity_v1`, an explicit versioned
  projection hashed by `input_fingerprint()`. It covers **everything a planner
  can read**:
  - each candidate through `planning_candidate_projection_v1`, which is its v2
    identity plus each field identity leaves out (`metadata` at every level,
    `TrainingSpec.api_version`, scheduled-intervention rationales), named one
    by one;
  - every budget balance (limit, reserved, committed, consumed, released,
    outstanding, remaining).

  A tripwire test pins the field sets of every model the context exposes.
- **Proposals** carry a `ProposalProvenance`:
  - the planner's provider, name, version and plugin API version;
  - the bound spec's kind and version;
  - `planner_spec_fingerprint` (`planner_spec_identity_v1`), which hashes the
    bound spec with its canonical **config** and the descriptor contract, so
    two configurations of one planner are told apart;
  - the context identity version and fingerprint.
  - `CandidateProposal` holds the full candidate, its fingerprint (checked),
    parents, hypothesis, reason, the mutation as data and evidence refs. It
    mints no node id or timestamp.
  - `ActionProposal` wraps a typed `ActionSpec` instance that must be
    **registered**, with the registered descriptor's schema class exactly
    `type(action)`. It refuses mappings, the base class, unregistered
    subclasses, and subclasses posing as a registered type.
  - Nothing is persisted. There is no `planning_rounds` table; durable
    planner audit arrives where proposals are consumed.
- **`RuleBasedPlanner`** works through an ordered list of typed mutation
  rules (no JSON patches or field paths). The first rule is
  `increase-lora-rank` (`factor` ≥ 2, `max_rank` ≥ 1). It applies only to a
  `lora` adapter with an explicit positive rank, uses `min(rank × factor,
  max_rank)`, and never shrinks the rank. It acts only at the planning stage,
  which is judged by status **and** decision (`settled_for_planning`): the
  experiment is `ACTIVE`, every node is `COMPLETED` by a latest `BRANCH` or
  `REJECTED` by a latest `REJECT`, and no quota is exhausted.
  - A `STOP_*` decision anywhere, which a paused experiment does not apply, or
    a settled node with no decision, blocks planning.
  - The host's `next_stage` uses the same predicate and reports `"decision"`
    in those cases instead of `"planning"`.
  - **Parent.** The best `COMPLETED` node whose latest decision is `BRANCH`,
    by the unsliced primary metric measured exactly once in that decision's
    results. Ties go to node id. `REJECTED` nodes are never parents, and a
    missing or repeated measurement makes a node ineligible rather than
    guessed at.
  - **Rules.** Tried in declared order. A rule is skipped if it doesn't apply
    or if its candidate's current fingerprint is already in the experiment
    (lineage deduplication). The first novel candidate is proposed; otherwise
    nothing is.
- **Budget** is read, never reserved: the planner writes no ledger entry and
  marks nothing exhausted. Authoritative reservation stays with the
  branching/control path.
- **No artifact reuse** (ADR-017 v1): every run a proposal leads to executes
  as new work.
- **Exit criterion.** Node A is `COMPLETED` after `BRANCH` at `task_success
  = 0.79` with LoRA rank 16. The planner proposes a novel LoRA-32 candidate
  with parent A, provenance and the context fingerprint, and writes no row.
  PR-025 materializes node B.

### PR-025 — experiment branching

An alternative candidate creates a new node; an in-run scientific change records a
`TrainingIntervention` (ADR-011). As built, branching consumes a
`CandidateProposal` and creates a node, but **no run**:

- **`ControlPlaneRepository.materialize_candidate_proposal(experiment_id,
  proposal, actor=...)`** does everything in one write transaction. Either it
  creates exactly one child node, moves it `CREATED → PLANNED`, writes its
  edge, `NodeCreated` and `ExperimentNodeStatusChanged`, and returns the node,
  or it writes nothing. Checks, in order:
  - the candidate's current fingerprint is the proposal's, else
    `ProvenanceError`;
  - the experiment already has this candidate (current v2 identity,
    recomputed from every stored snapshot): if it came from the **same**
    proposal, the existing node is returned and nothing is written
    (idempotent retry); otherwise `CandidateConflictError`;
  - the experiment is `ACTIVE`, else `BranchRefusedError`;
  - the proposal was made under the experiment's **recorded** `PlannerSpec`:
    kind, version and spec identity version must match, and the
    planner-spec fingerprint is recomputed from the recorded spec, so a
    factor-4 proposal is refused for a factor-2 experiment. Else
    `ProvenanceError`;
  - the current `PlanningContext` fingerprint equals the proposal's, else
    `StaleProposalError`. It is checked in the same transaction, and the
    repository never re-plans;
  - no budget quota is exhausted, else `BranchRefusedError`;
  - every evidence ref is a decision or evaluation result of one of the
    proposal's parents, else `ProvenanceError`;
  - the parents are sound, using the existing graph authority: they exist,
    are in the same experiment, are not named twice, and form no cycle.
    Else `LineageError`.
- **The controller half.** `EmbeddedControllerHost._materialize_candidate_proposal`
  rebuilds the recorded planner and calls `require_proposed_by`, which
  compares the full provenance, including the real descriptor provider and
  plugin API version that storage cannot know. Only then does it call the
  repository. Storage never imports a planner.
- **Durable origin.** `ExperimentNode.branch_origin: CandidateBranchOrigin |
  None` holds the proposal fingerprint (`candidate_proposal_identity_v1`,
  versioned), the full `ProposalProvenance` (planner, spec fingerprint,
  context fingerprint), the mutation and **typed** `EvidenceRef`s. Candidate,
  parents, hypothesis, reason and creator stay on the node itself. Existing
  nodes load with `None`, and there is no `planning_rounds` table.
- **Typed evidence.** `EvidenceRef(kind="decision" | "evaluation-result",
  id)` replaces PR-024's opaque strings, so evidence is validated rather than
  decorative.
- **Boundaries.**
  - The node stops at `PLANNED`. `PLANNED → READY → ACTIVE`, run creation,
    compilation and submission are PR-026's.
  - `ExperimentResult.next_stage` reports the new `"training"` stage for an
    open experiment with a `PLANNED` node -- an accepted candidate that needs
    its first run -- after action, decision and evaluation work and ahead of
    `"planning"` and `"failure-handling"`.
  - No `max_runs` reservation or ledger entry: a run's quota is reserved only
    by `create_run()`, and a budget spent between branching and the first run
    is that run's `BudgetExhaustedError` to report.
  - A `CandidateProposal` is not an Action and does not pass through
    PolicyEngine; it cannot grant capabilities.
  - Candidate governance is the recorded planner, exact provenance, a fresh
    context, candidate and lineage validation, deduplication, the budget
    precheck and an explicit actor. Organization-specific candidate approval
    is a separate future extension point, and `BranchExperiment` is not
    invented.
  - Same candidate means no duplicate node. It never means reusing an
    artifact (ADR-017).
- **Lineage representation.** `ExperimentNode.parent_ids` (the payload) and
  `experiment_edges` (the query authority) are written in the same insert,
  and `validate_parents` refuses any lineage on which they could disagree.

### PR-026 — end-to-end adaptive MVP

Reference scenario in `18-mvp-reference-scenario.md`. As built:

- **One submission, no further calls.** `EmbeddedControllerHost` carries an
  open experiment from `"planning"` to `"training"` itself, through
  `_continue_adaptive_experiment`: each round reads `next_stage` from the
  record, does one durable step, and reads again -- never a callback chain
  carrying objects from stage to stage. It is invoked after an evaluation
  settles and its decision is applied, at the end of `attach()`'s
  reconciliation, and by `wait()` before it calls the experiment settled.
  `wait()` is a safety net, not the driver: an experiment submitted and never
  waited on still progresses while the host lives.
- **Only planning and training are driven.** Evaluation, decision and
  recovery stay with the paths that own them:

  ```text
  DecisionEngine decides → planner proposes → branching admits lineage
    → adaptive controller realizes admitted work → existing compiler,
      runtime, recovery and evaluation machinery executes it
  ```

- **Planning.** The experiment's *recorded* `PlannerSpec`, bound again on
  this host; `planning_context()`; `propose()`. No proposal: the experiment
  stays `ACTIVE` at `"planning"`. One `CandidateProposal`: branched through
  PR-025. Several proposals, or an `ActionProposal`: escalated
  (`wait()` raises `ReconciliationEscalatedError`), nothing chosen, no Action
  created -- there is no selection or scheduling policy yet. A used-up quota
  ends the experiment `BUDGET_EXHAUSTED` instead of planning; the decision
  that asked for another candidate stands.
- **Only planner-created nodes run automatically.** A `PLANNED` node is
  realized only if its `branch_origin` names the experiment's recorded
  planner exactly (`require_provenance_of` on the host; the recorded spec
  fingerprint again in the repository). A `PLANNED` node with no origin is
  left for whoever planned it; one whose origin names another planner or
  configuration is escalated. `created_by`, status or a planner name alone
  never authorize a run. Because `branch_origin` now authorizes execution,
  it is repository-issued: `create_node()` refuses one, and only
  `materialize_candidate_proposal()` mints it.
- **Seed.** A branched node's first run inherits the seed of its single
  parent's single successful training run, and records
  `Run.seed_origin = RunSeedOrigin(kind="parent-run", source_run_id)`.
  Provenance only: it enters no fingerprint, and existing runs load with
  `None`. Several parents, or a parent with zero or several runs, is
  escalated -- v1 has no replicate-selection policy and does not guess.
  Same seed is never reuse: Run B is a new run, attempt, execution spec and
  submission (ADR-017).
- **Revalidation.** Before any write, the recorded compiler (at its recorded
  version) must `supports()` the branched candidate; otherwise it is
  escalated with no run, attempt, reservation or effect.
- **Atomic realization.** `realize_planned_node` re-checks the whole v1
  seed contract inside its transaction -- one parent with exactly one run,
  `SUCCEEDED` and seeded, named by the required `seed_origin`, same seed,
  `replicate == 1` -- so a run added after the host's read is refused, not
  silently chosen from. It writes, in one transaction:
  node `PLANNED → READY → ACTIVE`, `RunCreated` with the `max_runs`
  reservation, run `CREATED → ACTIVE`, the attempt with its `INTENDED`
  submit and its parallel-run slot -- each through its state machine with
  its event. The `PLANNED` node is the durable gate: a second caller finds it
  no longer `PLANNED` and writes nothing. The runtime is called after the
  commit, through the existing `_issue → submit_or_get → confirm → _adopt →
  _observe` path. The transactional repository methods were factored into
  shared in-transaction helpers (`_apply_transition`,
  `_insert_run_with_reservation`, `_insert_training_attempt_with_intent`);
  no write transaction nests another.
- **Budget.** Run B's `max_runs` reservation is the first budget effect of
  branching. No run left: nothing is written and the experiment ends
  `BUDGET_EXHAUSTED`. A full `max_parallel_runs` is capacity: nothing is
  written and realization waits.
- **Restart.** Crash after `BRANCH` with no child, or with a `PLANNED` child
  and no run: `attach()` continues from the record. Crash after realization
  with the submission `INTENDED`: the existing ADR-013 reconciliation adopts
  or re-issues it; no node or run is created again.
- **Decision engine.** The acceptance test configures
  `AdaptiveThresholdDecisionEngine` explicitly; the host's default is
  unchanged.
- **Acceptance.** `tests/test_experiment/test_adaptive_mvp.py`: A1 CUDA OOM
  → A2 restored and resized under the same Run A → 0.79 → `BRANCH` →
  LoRA 16 → 32 → node_B → Run B (seed inherited) → 0.83 →
  `STOP_SUCCEEDED` → `SUCCEEDED`, `best_node_id = node_B`, from
  `handle.wait()` and across both restart gaps.

Out of scope: `DecisionEngineSpec` persistence, GPU-hour metering, LLM
planner, search providers, multiple proposals or parallel branches,
replicate scheduling, multi-parent seed selection, candidate approval,
generic `ActionProposal` execution, daemon hosting and leases (PR-027/028),
CLI, the rich result/export projection, plateau detection, artifact reuse.

Follow-ups recorded by this PR:

- **`DecisionEngineSpec` persistence under ADR-016.** The host's decision
  engine is not recorded with the experiment, so a host attaching to an
  adaptive experiment must be configured with the same engine.
- **Future budget work -- authoritative GPU-hour metering.** Runtime-reported
  measured GPU usage; durable resource-usage attribution; conversion into
  `BudgetLedger` consumption; restart-safe accounting; distributed and
  multi-worker semantics. Requested resources must never be presented as
  actual usage. Until then `max_gpu_hours` stays refused at submission and
  spec 18 enforces `maxRuns` only.
- **Rich result / provenance export projection.** Spec 18's
  `best_artifact`, `experiment_graph`, `incidents`, `recovery_history`,
  `resource_usage`, `provenance_bundle` -- a projection over the durable
  record, not fields of today's `ExperimentResult`.

Phase exit:

- full adaptive experiment works locally without LLM

---

## Phase 7 — Durable local controller

Runtime-operation reconciliation already exists from PR-012a. What this phase
adds is *host* reconciliation: ownership, and recovering the whole controller
rather than one attempt.

### PR-027 — LocalDaemonControllerHost

Persistent process, singleton locking per state database, controlled shutdown.
Contract: ADR-004 (accepted before this PR). Delivered as the server half,
`LocalDaemonControllerServer`, and the mailbox client `DaemonClient`; the
`ControllerHost`-compatible `LocalDaemonControllerHost` is PR-029's (ADR-004
§1).

- **SQLite mailbox.** `controller_requests` (`submit`, `attach`), idempotent
  by client-generated id; `PENDING → ACCEPTED → COMPLETED`, `PENDING →
  FAILED`; no `PROCESSING` state. No socket, HTTP or gRPC.
- **Atomic admission.** The submission's experiment, root node, run 1,
  attempt 1, reservations and `INTENDED` submit -- and the request's
  `ACCEPTED` -- commit in one transaction before the runtime is called. The
  embedded host's `submit()` uses the same admission.
- **Unfinished requests only** are recovered at daemon start: `PENDING`
  processed, `ACCEPTED` resumed through `attach()`/ADR-013 reconciliation.
  A `COMPLETED` request's experiment is not adopted again without an explicit
  `attach` request.
- **Singleton.** `fcntl.flock(LOCK_EX | LOCK_NB)` on `<resolved db>.lock`,
  held for the process lifetime; diagnostic content only; POSIX only.
- **Process.** Foreground `python -m xaytune.daemon --state … --config
  module:factory`; every implementation from the explicit configuration.
- **Provenance.** `ControllerHostRef(kind="local_daemon", id=<instance id>)`,
  injected into the delegated embedded controller.
- **Shutdown.** SIGTERM/SIGINT: stop dequeuing, cancel observers, close
  runtimes and the database, release the lock last; workloads keep running.

Out of scope: leases and heartbeats, the whole-database startup sweep,
budget settlement after downtime (PR-028); the user CLI (PR-029);
HTTP/Unix-socket/gRPC; `DecisionEngineSpec` persistence; Windows locking.

### PR-028 — host leases and whole-controller startup reconciliation

Durable controller ownership, and a restarted daemon recovering everything it
owns, through the existing ADR-013 effect identities and record-driven loops.
Contract: ADR-004 §8 (settled 2026-10-04). As built:

- **Two layers.** The PR-027 flock stays; the durable lease is added on top.
  Startup: flock → migrate → acquire or wait for the lease → fenced controller
  → unfinished mailbox requests → sweep → mailbox and heartbeat loop. The flock
  is released last. Local SQLite only.
- **Lease.** Migration 017, a singleton `controller_leases` row
  (`controller_id`, `epoch`, `heartbeat_at`, `lease_expires_at`), never
  deleted. The epoch starts at 1 and increments on every new acquisition,
  clean or crash; renewal keeps it. A live lease is waited out, never taken --
  no host/PID/process shortcut. Renewal (every TTL/3) and release are
  compare-and-sets on `(controller_id, epoch)`; an expired lease is not
  revived. `DaemonConfig.lease_ttl_seconds`, default 30, is the one timing
  setting; no CLI flag.
- **Fencing.** Every `ControlPlaneRepository` write goes through one
  transaction helper that checks, after the write lock and before the first
  mutation: owner + epoch + unexpired for the daemon's controller
  (`LeaseLostError`), no live lease for an embedded host
  (`ControllerLeaseHeldError`) -- on every write, so a host opened before the
  daemon is refused afterwards; reads stay open. `DaemonClient` mailbox writes
  and runtime calls are not fenced (ADR-013 idempotency covers a stale
  re-issue). An AST test keeps the helper the only transaction opener.
- **Lease loss is process-fatal** wherever it surfaces: the daemon stops,
  records no workload outcome, leaves the lease alone and exits 5.
- **Sweep.** Nonterminal experiments with `controller_host.kind ==
  "local_daemon"` or a `COMPLETED` attach request, each reconciled by
  `attach()`; embedded experiments never attached are not swept. Running it
  twice changes nothing the first run did not.
- **Pause-safe reconciliation.** `PAUSED` experiments are swept: existing work
  is adopted and settled, nothing new starts. `_continue_to_evaluation()` now
  requires `ACTIVE`, and an evaluation run with no attempt gets none while
  paused.
- **Settle durable budget accounting after downtime and apply existing
  budget-exhaustion rules before starting further work.** The sweep runs
  `settle_budget()` and moves no experiment; the existing rules apply at the
  existing effect boundaries (new run, evaluation cycle, evaluation attempt,
  recovery successor) through `attach()`, unchanged by a restart. A resting
  experiment with a used-up quota stays `ACTIVE`; a `PAUSED` one is settled
  and stays paused. No deadline, wall-time, GPU-hour, token or cost
  accounting: those dimensions are refused at submission, and downtime costs
  nothing.

Out of scope: `LocalDaemonControllerHost` and the CLI (PR-029); cancel,
propose and approve request kinds; pause/resume; deadlines and time-based
budgets; remote or distributed controllers, network filesystems and runtime
fencing tokens; the event actor rename; mailbox retry backoff.

### PR-029 — LocalDaemonControllerHost + CLI submit/attach/watch

`LocalDaemonControllerHost`, the caller-side `ControllerHost` over the PR-027
mailbox (ADR-004 §1, §9), and the CLI built on it rather than on a second way
of talking to the daemon. As built:

```text
LocalDaemonControllerHost
  submit(spec) -> ExperimentHandle     submit request, until COMPLETED
  attach(id)   -> ExperimentHandle     attach request, until COMPLETED
  handle(id)   -> ExperimentHandle     no request: an experiment it owns
  approve_action, reject_action        approve-action, reject-action requests
  cancel(id, request_id=...)           what handle.cancel() sends, retryable

daemon-backed ExperimentHandle         the embedded host's handle, over a
  status(), actions(), events()        host protocol: record reads
  wait()                               record polling, until the controller's
                                       recorded rest is the latest event
  cancel(), propose()                  cancel, propose-action requests

CLI   submit, attach, status, watch, events, results, actions, cancel,
      approve, reject -- over that host, nothing else
```

- **Mutation kinds.** `cancel`, `propose-action`, `approve-action`,
  `reject-action`, each `PENDING → COMPLETED | FAILED` with exactly its
  payload (migration 018 rebuilds `controller_requests` for the kind
  `CHECK`). Each carries the identity of what it records or resolves -- a
  client-minted `cancel-experiment` or proposed action id, or the action it
  resolves -- so a crash between its intent and its effects, or a resend, is
  finished rather than repeated. `FAILED` only for a refusal before anything
  was recorded; the host's cancel, approve and reject are split into a
  recording step and a carry-on step so the daemon can tell the two apart.
- **Record, attach, carry on.** The daemon records the intent (the only
  refusable step, calling no runtime), then attaches the experiment so it
  observes the effects, then carries the intent on; after the intent nothing
  fails the request. A `COMPLETED` request of any kind but `submit` counts as
  adoption for the sweep, and a `FAILED` one adopted nothing.
- **One Action per request.** A cancel request is carried out through the
  Action it names or refused (`CancellationInFlightError`) -- never answered
  with another cancellation in flight. `propose()` takes a `request_id` too,
  reusing the recorded Action id: a lost answer is retried as the same
  request, judged once. A daemon-backed `cancel()` beside another
  cancellation joins it by a separate `attach` request -- adopting the
  experiment and carrying that cancellation on -- and returns only if the
  experiment is then cancelled or cancelling; otherwise it raises the
  refusal.
- **Rest.** A client cannot run the controller's `wait()`, and the record
  cannot tell rest from the instant between two steps (a trained run whose
  evaluation cycle is about to begin). After each request and swept
  experiment the daemon awaits its controller's `wait()` and records a
  `controller_rests` row at the experiment's latest event sequence, with any
  escalation; a client's `wait()` returns once, in one snapshot, that is still
  the latest event, no request for it is unfinished and the result is
  quiescent -- or the experiment is terminal and settled.
- **Retry.** Every mutating CLI command prints its request id before sending
  it and takes `--request-id`; the host reuses the recorded experiment or
  action id for a request id it has seen. A request waits in the mailbox when
  no daemon runs; `--timeout` bounds the wait for its handoff.

Every mutation is a request kind the daemon carries out; nothing runs a
controller or calls a runtime in the caller. `pause`/`resume` are **not** in
PR-029: the current `ExperimentHandle` does not implement them, and they need
their own durable design (pause semantics, runtime behaviour, transitions,
reconciliation). No HTTP, socket or remote daemon; no LLM planner work.

Exit (`tests/test_daemon/test_daemon_cli.py`, against a daemon process):

- submit experiment from the CLI, which exits
- kill a client mid-watch, and one mid-submission with no daemon running
- controller continues; the killed submission is carried out once a daemon
  starts, and retried by its request id without a second experiment
- reconnect and inspect: watch, results, actions; approve a pending action;
  cancel, with the runtime called by the daemon

---

## Phase 8 — LLM planner

### PR-030 — AgentModel protocol

As built (`xaytune.agent`): the model-invocation boundary a future
`LLMPlanner` sits on. It asks a model for structured output and knows nothing
of experiments, candidates, actions, policy, storage, runtimes or controllers;
it imports only `xaytune.core`, and an architecture test holds that (and keeps
provider SDKs and HTTP clients out).

- **Contract.** `AgentModel` has a `descriptor: AgentModelDescriptor` (the
  adapter's `PluginDescriptor` and the model's `AgentModelIdentity` --
  provider, name, pinned revision or `None` -- kept separate) and `async
  generate(AgentModelRequest) -> AgentModelResponse`. Adapter configuration,
  credentials included, is in neither; only requests, responses and
  descriptors are serialized.
- **Request.** System prompt, messages (user/assistant, ending with the
  user's turn), a mandatory `response_schema`, `temperature` and
  `max_output_tokens`. Frozen, closed and canonical.
- **Schema subset.** Xaytune's closed response-schema subset in JSON Schema
  vocabulary, which `xaytune.agent.schema` fully implements (it is not a
  standards-complete JSON Schema validator): `type` or `anyOf`, `enum`,
  `const`; closed objects (`additionalProperties` absent or false); arrays
  with `items`; string lengths; numeric bounds; string `title`/`description`. A keyword outside it is refused when the request is
  built, never ignored when an answer is checked. Checking is typed: `true` is
  not an integer, `1.0` is not an integer, and `enum`/`const` compare by
  canonical encoding.
- **Fail closed.** Callers go through `invoke_agent_model()`. It refuses an
  adapter on an unsupported plugin API before asking. It wraps an adapter's
  own exception in `AgentModelInvocationError`: provider/adapter exception
  text and chaining are discarded at the boundary (no `__cause__`, no
  `__context__`), and only the exception type and the logical request
  fingerprint cross it, since SDK text can carry URLs, headers or keys
  (PR-030a). An adapter with a deliberately sanitized diagnostic raises
  `AgentModelInvocationError` itself. PR-032 never persists or serializes
  provider SDK exception objects, `__cause__`, `__context__`, traceback
  locals or raw provider error payloads. It refuses
  the whole answer (`AgentModelOutputError`) if it is not an
  `AgentModelResponse`, if it reports a revision other than a pinned one (a
  missing reported revision is accepted; a more specific model name, an
  alias resolved, is too), or if its content breaks the schema (every
  violation by path).
- **Identity.** `agent_model_request_identity_v1(model, request)` covers the
  model identity, system prompt, messages, schema and generation parameters.
  Not the adapter or its version (how a request is carried, not what it asks),
  and not usage, latency, provider request ids or the answer. A test pins the
  digest, and one pins both models' fields against the projection.
- **Response.** `content` (the structured answer), the model and revision the
  provider reports, `finish_reason`, `usage` (input/output tokens; unreported
  counts stay `None`), `provider_request_id`, `latency_seconds`. A rationale
  is a field of the caller's schema; nothing requests or records hidden
  reasoning.
- **One call, one invocation.** An adapter may retry transport failures
  inside `generate`; the caller sees one call for one request identity.
  Recording attempts is PR-032, which keeps the full `AgentModelDescriptor`
  in the invocation record beside the request fingerprint rather than
  changing `agent_model_request_identity_v1`.
- **`ScriptedAgentModel`** answers from a script (content, a whole response,
  or an exception), records every request, and checks nothing itself, so a
  malformed scripted answer is refused at the boundary like a real one. It is
  PR-031's backend in CI: no network, no keys.

Not here: `LLMPlanner`, prompts, `PlanningContext` → request conversion,
proposal parsing, policy, persistence or audit tables, tool calling, and any
provider adapter.

### PR-031 — LLMPlanner

Structured actions only.

### PR-032 — agent audit/provenance

Exit:

- malformed model output cannot execute
- policy cannot be bypassed
- decisions reproducible/auditable

---

## Phase 9 — Ray and TorchFT

### PR-033 — RayTrainRuntime

### PR-034 — RayTuneSearchProvider

Keep separate from runtime.

### PR-035 — TorchFTResilienceProvider

### PR-036 — distributed failure tests

Exit:

- worker failure
- node failure
- checkpoint recovery
- experiment lineage preserved

---

## Phase 10 — Training Hub

### PR-037 — TrainingHubRuntime

### PR-038 — runtime capability discovery

### PR-039 — remote artifact/checkpoint flow

### PR-040 — OpenShift AI integration test

Preferred stack:

```text
Xaytune
→ Training Hub
→ Kubeflow Trainer / KubeRay
→ Kueue
```

---

## Phase 11 — Search/memory and Agent Training (model-weight optimization)

After core stabilizes:

- OptunaSearchProvider
- KatibSearchProvider
- structured experiment memory
- semantic memory plugin
- TRL/OpenEnv Agent Training compiler (weight updates through agent environments)
- verl compiler (model-weight optimization)
- torchtune compiler
- Studio migration

Agent Training changes model weights. Agent Harness Optimization changes how a
model is operated and can run without training any model. The track below is
separate from Phase 11's training integrations and experiment-memory plugins.

---

## Harness Optimization track (planned)

[Chapter 23](23-agent-harness-optimization.md) defines the proposed architecture.
[ADR-018](adrs/ADR-018-agent-harness-candidates.md) remains `Proposed`: H01 is
specification/review only; H02 onward requires acceptance and the generic
experiment/planner foundation. Existing historical PR numbers remain intact.

**Current implementation priority is unchanged:** PR-019 RecoveryPlan/coordinator,
PR-020 adaptive OOM recovery, PR-021 numerical recovery, PR-024 RuleBasedPlanner,
PR-025 branching, PR-026 adaptive MVP. Harness implementation depends on those
generic experiment/planning foundations and must not delay recovery. H01 lands
now so later planner contracts avoid unnecessary training-only assumptions.

| ID | Planned deliverable | Prerequisites / gate |
|---|---|---|
| H01 | Candidate-generalization ADR/spec | Architecture review; this spec-only PR |
| H02 | AgentHarnessSpec + HarnessFingerprint and compatible candidate envelope | ADR-018 accepted; PR-024–026 foundation; wire/projection review |
| H03 | Harness compiler/execution/runner protocol | H02; workload target/telemetry/version compatibility review |
| H04 | Benchmark/task/environment protocol and input resolution | H02; repeatability/pinning review; required before comparable execution |
| H05 | Trajectory artifact model | H03–H04; provenance/security/retention review |
| H06 | Harness evaluator + multi-objective metrics | H04–H05; replicate and objective contract review |
| H07 | Mutation/search provider protocols | H02, H06; PR-025 candidate-proposal governance settled before mutation execution; distinct CandidateProposal/ActionProposal paths |
| H08 | Pi adapter | H03–H05; declared supported subset and capability checks |
| H09 | Codex adapter | H03–H05; declared supported subset and capability checks |
| H10 | Claude Code adapter | H03–H05; declared supported subset and capability checks |
| H11 | Additional harness adapters: OpenCode, Hermes, custom production agents | H03–H05; each independently scoped |
| H12 | End-to-end harness optimization MVP | H02–H07 and at least one validated adapter; safe isolated pinned suite |

The first MVP optimizes **prompt, context policy and tool descriptions/configuration**
within a fixed, already authorized tool set. It retains success, quality, cost,
input/output tokens and latency separately, records trajectory and mutation
provenance, enforces policy/budgets, and returns an immutable harness artifact.
One validated adapter is sufficient; H09–H11 need not delay H12. Expand later
into middleware, delegation, persistent memory and joint model+harness work.
Search providers remain pluggable; no particular optimizer is required.
