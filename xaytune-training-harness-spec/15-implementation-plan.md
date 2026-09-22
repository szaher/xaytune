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
| **Daemon restart reconciliation** | whole-controller startup: lease acquisition, every active experiment, deadlines, control loops | H |

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
| ADR-006 — fingerprints (identity model) | PR-007 fingerprint framework; reuse policy split out to ADR-017 |
| ADR-007 — evaluation independence | band D; extended by ADR-015 |
| ADR-011 — candidates, interventions, overrides | PR-005 event schema; supersedes ADR-003's two-level lineage and extends ADR-006 |
| ADR-012 — data position and resume semantics | PR-005 checkpoint schema; all adaptive recovery |
| ADR-013 — operation identity and cancellation | the first runtime implementation |
| ADR-014 — worker telemetry protocol | `RuntimeBackend.watch()`; PR-005 event schema |
| ADR-015 — durable evaluation lifecycle | PR-005 evaluation tables |
| ADR-016 — specs versus implementations | PR-005 experiment record |

### Still Proposed, and what each actually blocks

These remain `Proposed`. Each names the work it gates, so nothing is blocked
that does not depend on it:

| ADR | Blocks |
|---|---|
| ADR-004 — durable controller hosting | band H (daemon, kill/restart) |
| ADR-008 — versioned plugin ABI | band C (compiler/runtime plugin loading) |
| ADR-009 — checkpoint layers | band F |
| ADR-017 — reuse policy | band G (planner reuse decisions); split out of ADR-006 |

### Superseded

| ADR | By |
|---|---|
| ADR-003 — scientific vs execution lineage | ADR-011; retained for the reasoning that led there |

No ADR is half-accepted any more. Status is used as a gate, so an ADR that was
`Proposed — partially superseded` or `proposed but ratified in effect` could not
be acted on: ADR-001 is now `Accepted` because everything after it assumes it,
ADR-007 because ADR-015 depends on it, and ADR-006's genuinely open half became
ADR-017 rather than a second status on one document.

**ADR-005 was accepted on 2026-09-21, so band B is unblocked and PR-004 may
start.** It was expanded before acceptance rather than accepted as written: the
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
ones gate later bands: ADR-008 blocks band C, ADR-009 band F, ADR-017 band G and
ADR-004 band H. Each must be accepted before its own band, not before PR-004.

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

Start with SFT only.

### PR-012 — public ExperimentHandle / EmbeddedControllerHost

Support:

```text
submit
status
wait
cancel
events
```

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

### PR-014 — existing eval adapter

Wrap current metrics/lm-eval.

### PR-015 — DecisionEngine

Implement deterministic objective/constraint decisions.

### PR-016 — BudgetLedger

Implement reserve/commit/consume/release.

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

### PR-023 — PolicyEngine

Budget authorization pieces land here too: recovery in Phase 5 proposes Actions,
and an Action that cannot be authorized cannot be applied.

## Phase 5 — Resilience, recovery and interventions

### PR-017 — incident model and detectors

Initial:

- process failure
- CUDA OOM
- NaN/Inf
- checkpoint failure

### PR-018 — checkpoint codec/store/manager

Start local-only.

### PR-019 — recovery plan + coordinator

### PR-020 — adaptive OOM recovery

Implement execution override:

- lower microbatch
- optionally preserve effective batch
- resume checkpoint

### PR-021 — numerical recovery

Recovery that changes LR to stabilise a continuing run records a `TrainingIntervention`
on that run through the Action path (ADR-011). It does not create a new node. Forking
is for alternatives you want to compare.

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

Rules:

- objective reached → stop
- plateau → propose evaluation/stop
- OOM → recovery path
- failed constraint → reject candidate

### PR-025 — experiment branching

An alternative candidate creates a new node; an in-run scientific change records a
`TrainingIntervention` (ADR-011).

### PR-026 — end-to-end MVP test

Reference scenario in `18-mvp-reference-scenario.md`.

Phase exit:

- full adaptive experiment works locally without LLM

---

## Phase 7 — Durable local controller

Runtime-operation reconciliation already exists from PR-012a. What this phase
adds is *host* reconciliation: ownership, and recovering the whole controller
rather than one attempt.

### PR-027 — LocalDaemonControllerHost

Persistent process, singleton locking per state database, controlled shutdown.

### PR-028 — host leases and whole-controller startup reconciliation

Implement:

- controller identity and lease acquisition, so two daemons cannot drive one
  database
- startup sweep: load every active experiment, node, run and attempt, then
  delegate per-attempt recovery to the PR-012a path rather than reimplementing
  it
- deadline and budget re-evaluation after downtime
- resuming control loops

Idempotent: running it twice changes nothing the first run did not.

### PR-029 — CLI submit/attach/watch

Exit:

- submit experiment
- kill client
- controller continues
- reconnect and inspect

---

## Phase 8 — LLM planner

### PR-030 — AgentModel protocol

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

## Phase 11 — Search/memory/agent training

After core stabilizes:

- OptunaSearchProvider
- KatibSearchProvider
- structured experiment memory
- semantic memory plugin
- TRL/OpenEnv agent-training compiler
- verl compiler
- torchtune compiler
- Studio migration
