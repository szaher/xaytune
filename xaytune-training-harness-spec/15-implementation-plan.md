# Implementation Plan

## Phase 0 — Architecture hardening

No feature implementation before these ADRs are accepted.

### ADR set

1. experiment control plane / compile-execute boundary
2. aggregate/state-machine separation
3. scientific vs execution lineage
4. durable controller hosting
5. transactional state/event/outbox persistence
6. identity/fingerprint/reuse semantics

Exit criteria:

- all ADRs accepted
- module ownership agreed
- core dependency boundary agreed
- first 12 PRs updated to match ADRs

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

Recovery that changes LR must create a new node.

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

Scientific mutation creates new node.

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
