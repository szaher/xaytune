# Xaytune 2.0 — Agent-Native Model Training Harness

This package is the implementation specification for evolving Xaytune from an opinionated model training/fine-tuning library into an **agent-native experiment control plane and model training harness**.

The design assumes the current Xaytune repository continues to exist and is evolved incrementally. Existing recipes, trainer loop, callbacks, checkpointing, evaluation, pipelines, logging backends, CLI, Studio, and plugin discovery are treated as assets to be refactored behind new control-plane boundaries.

## Product definition

**User-facing:** Xaytune is an agent-native model training harness for adaptive, resilient, reproducible model training and post-training.

**Internal architectural definition:** Xaytune is an experiment control plane that compiles training intent into serializable execution plans, delegates execution to runtimes, observes the result, evaluates it, applies policy and budget constraints, and decides what should happen next.

The governing rule is:

> Xaytune controls the experiment. Trainer integrations compile training intent. Runtime integrations execute training.

## Why this exists

Modern model training uses multiple systems:

- PyTorch
- Transformers / PEFT
- TRL
- torchtune
- verl
- TorchFT
- Ray Train
- Ray Tune
- Kubeflow Trainer
- Kueue
- Training Hub
- MLflow / W&B

Each solves part of the problem. Xaytune owns the missing cross-cutting control layer:

- experiment semantics
- scientific lineage
- execution lineage
- agentic experiment planning
- adaptive training decisions
- resilience and recovery policy
- evaluation-driven branching
- budgets and policy enforcement
- provenance
- durable controller state
- runtime-neutral training intent
- capability negotiation

## Architecture at a glance

```text
                    Xaytune
             Experiment Control Plane
                      │
      ┌───────────────┼────────────────┐
      │               │                │
 Scientific       Adaptive          Agentic
 lineage          resilience        decisions
      │               │                │
      └───────────────┼────────────────┘
                      │
               Policies / Budget
                      │
                  Planner
        ┌─────────────┼─────────────┐
        │             │             │
      Rules         LLM        SearchProvider
                                  │
                         Ray Tune / Katib /
                              Optuna
                      │
                TrainingSpec
                      │
              TrainerCompiler
        ┌─────────────┼──────────────┐
        │             │              │
       TRL        torchtune         verl
        │             │              │
        └─────────────┼──────────────┘
                      │
            TrainingExecutionSpec
                      │
             Capability Resolver
                      │
             RuntimeBackend.submit
        ┌─────────────┼──────────────┐
        │             │              │
      Local       Ray Train     Training Hub
                                    │
                           Kubeflow / KubeRay
                                    │
                                  Kueue
```

## Package contents

- `01-product-and-scope.md` — product definition, scope, non-goals, invariants
- `02-architecture.md` — target architecture and boundaries
- `03-domain-model.md` — Experiment, Node, Run, Attempt, Action, Incident, artifacts
- `04-state-machines.md` — separate lifecycle state machines and transitions
- `05-training-spec-and-compilation.md` — TrainingSpec, TrainerCompiler, TrainingExecutionSpec
- `06-runtime-and-controller-hosting.md` — RuntimeBackend, durable controller, submit/attach/wait
- `07-persistence-and-events.md` — SQLite transaction model, outbox, revisions, reconciliation
- `08-resilience-and-recovery.md` — incident detection, adaptive recovery, TorchFT/Ray integration
- `09-agent-planner-policy-budget.md` — typed actions, planners, LLM agents, policy, budget ledger
- `10-evaluation-and-decisioning.md` — versioned metrics, judges, statistical safety, promotion
- `11-checkpointing.md` — codec/store/manager split, compatibility and commit semantics
- `12-capabilities-and-plugins.md` — parameterized capability schema and versioned plugin ABI
- `13-public-api-and-cli.md` — Python API, YAML, CLI, handles, remote-friendly semantics
- `14-repo-refactor-map.md` — how current Xaytune modules move behind the new architecture
- `15-implementation-plan.md` — phased PR plan and dependency order
- `16-testing-and-fault-injection.md` — test strategy, failure injection, architecture tests
- `17-coding-agent-contract.md` — rules coding agents must follow
- `18-mvp-reference-scenario.md` — acceptance scenario for adaptive resilient training
- `19-backward-compatibility.md` — compatibility plan for existing APIs and pipelines
- `20-security-and-governance.md` — agent authority, secrets, audit, supply chain, execution safety
- `21-observability-and-provenance.md` — event model, MLflow/W&B mapping, lineage
- `22-open-questions.md` — decisions intentionally deferred
- `adrs/` — architecture decision records required before implementation
- `schemas/` — proposed YAML and JSON/Python schema examples
- `agent-prompts/` — coding-agent execution prompts for the first implementation phases

## Implementation order

Do not begin feature implementation before the six foundation ADRs are accepted:

1. experiment control plane and `TrainingExecutionSpec` boundary
2. separate aggregate state machines
3. scientific lineage vs execution lineage
4. durable controller hosting and reconciliation
5. transactional state/event persistence with outbox
6. identity, fingerprints, seed/replicate/reuse semantics

Then implement in this order:

```text
domain + IDs
→ state machines
→ SQLite repository / outbox
→ event model
→ experiment graph
→ TrainingSpec / compiler contracts
→ LocalRuntime
→ current trainer as NativeCompiler
→ TRLCompiler
→ evaluation
→ action / policy / budget
→ incidents and recovery
→ rule-based planner
→ end-to-end MVP
→ LLM planner
→ Ray / TorchFT
→ Training Hub
```

## Definition of success

A successful Xaytune experiment can:

1. accept a training objective and immutable training specification
2. compile it into a serializable execution plan
3. execute it locally or through a remote runtime
4. survive controller restart
5. observe training through structured events
6. detect a training or infrastructure incident
7. classify whether it is operational or scientific
8. recover from a checkpoint
9. apply approved execution overrides or create a new scientific candidate
10. evaluate the result with versioned metrics
11. use policy, budget, prior experiments, rules, search, or an LLM to propose the next action
12. create a new experiment branch when training semantics change
13. stop when the objective is met or budget is exhausted
14. return artifact, full lineage, metrics, incidents, recovery history, cost/resource accounting, and decision provenance
