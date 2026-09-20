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

The numbered phases below map onto these bands. Where they disagree, the bands
are authoritative: the numbering predates ADR-011 and ADR-013.

## Phase 0 — Architecture hardening

No feature implementation before these ADRs are accepted.

### ADR set

1. experiment control plane / compile-execute boundary
2. aggregate/state-machine separation
3. scientific vs execution lineage
4. durable controller hosting
5. transactional state/event/outbox persistence
6. identity/fingerprint/reuse semantics

Plus the six written since, all of which gate PR-005:

7. ADR-011 — candidates, interventions, overrides *(Accepted)*
8. ADR-012 — data position and resume semantics *(Accepted)*
9. ADR-013 — operation identity and cancellation *(Accepted)*
10. ADR-014 — worker telemetry protocol *(Accepted)*
11. ADR-015 — durable evaluation lifecycle *(Accepted)*
12. ADR-016 — specs versus implementations *(Accepted)*

Exit criteria:

- all ADRs accepted
- module ownership agreed
- core dependency boundary agreed
- first 12 PRs updated to match ADRs

> **This gate is currently inconsistent with the repository and needs a
> decision.** ADR-001…010 are still `Status: Proposed`, but the implementations
> of ADR-002 (state machines) and ADR-010 (core dependency boundary) are merged
> on `main` and under test. An agent reading this section literally will stop
> before Phase 1, having been told a gate is closed that the repository has
> already walked through. Either accept the ADRs whose implementations have
> landed, or restate what Phase 0 actually requires. Tracked in
> `22-open-questions.md`.

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

## Phase 4 — Resilience

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

## Phase 5 — Planner, actions, policy

### PR-022 — typed actions

### PR-023 — PolicyEngine

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

## Phase 6 — Durable local controller

### PR-027 — LocalDaemonControllerHost

### PR-028 — reconciliation

### PR-029 — CLI submit/attach/watch

Exit:

- submit experiment
- kill client
- controller continues
- reconnect and inspect

---

## Phase 7 — LLM planner

### PR-030 — AgentModel protocol

### PR-031 — LLMPlanner

Structured actions only.

### PR-032 — agent audit/provenance

Exit:

- malformed model output cannot execute
- policy cannot be bypassed
- decisions reproducible/auditable

---

## Phase 8 — Ray and TorchFT

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

## Phase 9 — Training Hub

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

## Phase 10 — Search/memory/agent training

After core stabilizes:

- OptunaSearchProvider
- KatibSearchProvider
- structured experiment memory
- semantic memory plugin
- TRL/OpenEnv agent-training compiler
- verl compiler
- torchtune compiler
- Studio migration
