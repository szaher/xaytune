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

```text
A  domain contract hardening     state machines, ADR-011 identity,
                                 deep immutability, data cursor (012),
                                 operation identity (013)
B  transactional persistence     events, outbox, operation journal, projections
C  compile boundary              NativeCompiler, restart-safe LocalRuntime,
                                 embedded controller + reconciliation
D  durable evaluation lifecycle  ADR-015
E  minimal Action/Policy/Budget  approvals -- prerequisite for F
F  checkpoint, recovery, interventions
G  planner and branching
H  daemon, kill/restart MVP
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
| E Action / Policy / Budget | 4 |
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
| ADR-001 — experiment control plane | nothing yet; it is the premise the rest assumes and is ratified in effect by ADR-011's acceptance |
| ADR-003 — scientific vs execution lineage | superseded in substance by ADR-011; retained for its history |
| ADR-004 — durable controller hosting | band H (daemon, kill/restart) |
| ADR-005 — transactional persistence | band B — **must be accepted before PR-005** |
| ADR-006 — fingerprints and reuse | extended by ADR-011; the reuse-policy half is still open and blocks band G (planner reuse decisions) |
| ADR-007 — evaluation independence | extended by ADR-015; blocks band D |
| ADR-008 — versioned plugin ABI | blocks band C (compiler/runtime plugin loading) |
| ADR-009 — checkpoint layers | blocks band F |

**ADR-005 is the live one.** It is the next ADR that must be accepted, because
band B cannot start without it.

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

Implement tables and revision-based persistence.

### PR-005 — event + outbox transaction

Integrate events/outbox atomically with transitions.

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
- controller restart does not lose state

---

## Phase 2 — Compile/execute boundary

### PR-007 — TrainingSpec

Implement:

- SFT
- pretrain
- DPO
- GRPO schema skeletons
- fingerprint framework

### PR-008 — compiler/runtime protocols

Implement:

- TrainerCompiler
- TrainingExecutionSpec
- RuntimeBackend
- ResolvedExecutionPlan
- CapabilityDocument skeleton

### PR-009 — LocalRuntime

Subprocess + operation idempotency.

### PR-010 — NativeCompiler

Wrap existing training loop.

Goal:

```text
TrainingSpec → NativeCompiler → TrainingExecutionSpec → LocalRuntime
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

Phase exit:

- SFT runs through new compile/execute path
- existing trainer remains functional
- process boundaries are serializable

---

## Phase 3 — Evaluation and decision substrate

### PR-013 — EvaluationSpec / MetricResult

### PR-014 — existing eval adapter

Wrap current metrics/lm-eval.

### PR-015 — DecisionEngine

Implement deterministic objective/constraint decisions.

### PR-016 — BudgetLedger

Implement reserve/commit/consume/release.

Phase exit:

- experiment can train → evaluate → finish
- budget and evaluation metadata are durable

---

## Phase 4 — Actions, policy, budget

> **Reordered.** Resilience was Phase 4 and the Action substrate Phase 5. ADR-011
> makes a `TrainingIntervention` the outcome of an **approved Action**, so an OOM
> that reduces micro-batch size is an operational action that still needs
> deterministic authorization. Recovery cannot precede the thing that authorizes
> it. The PR numbers below keep their original identities so cross-references
> elsewhere still resolve; only the phase order changed.


### PR-022 — typed actions

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

### PR-027 — LocalDaemonControllerHost

### PR-028 — reconciliation

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
