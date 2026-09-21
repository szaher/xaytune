# Xaytune

**An agent-native experiment control plane for model post-training and adaptation.**

Xaytune is being built to coordinate training, evaluation, branching, recovery,
interventions, experiment lineage, and agent-driven decisions across trainer and
runtime backends. It answers: **what should this training experiment do next?**

**Xaytune controls the experiment.** Trainer integrations compile training intent.
Runtime integrations execute training. Infrastructure schedules and runs workloads.

!!! info "Xaytune 2.0 is under active development"
    As of **2026-09-22**, the domain and Band B persistence foundations are merged
    on `main`. Band C, the compile/execute boundary, is current work. The
    end-to-end experiment controller and runtime integrations are not available
    yet. The existing v0.6.0 trainer API remains usable.

## What Xaytune owns

The target control plane owns experiment topology, candidate identity, scientific
lineage, evaluation-driven decisions, interventions, semantic recovery, agent
proposals, policy gates, budgets, and provenance. It delegates tensor execution,
distributed training, worker management, scheduling, infrastructure admission,
and low-level fault tolerance to trainer, runtime, and platform integrations.

Planned integrations include Xaytune Native / TRL / torchtune / verl compilers;
Local / Ray Train / Training Hub runtimes; and infrastructure or resilience
providers such as TorchFT, Kubeflow Trainer, KubeRay, Kueue, and OpenShift AI.
This describes the intended ecosystem, not currently available backend support.

## Development status

### Completed foundation

- Immutable core domain models and lifecycle state machines.
- SQLite persistence, versioned migrations, and revision-based optimistic concurrency.
- Atomic state + event + outbox transactions.
- Durable `RuntimeOperation` journal and atomic attempt + submit-intent persistence.
- Idempotent repository writes, unresolved-operation queries, and crash/concurrency tests.
- Action substrate and persisted cancellation intent, lifecycle, and outcomes.
- Experiment DAG, cycle prevention, lineage traversal, and candidate comparison.

These are repository guarantees. Runtime submission, workload reattachment,
external cancellation delivery, policy/budget enforcement, and daemon restart
reconciliation are not implemented by this foundation alone.

### Current work: Band C — compile/execute boundary

Next are `CandidateSpec`, `TrainingSpec`, `CandidateFingerprint`,
`RunHistoryFingerprint`, and `ArtifactLineageFingerprint`, then `TrainerCompiler`,
`TrainingExecutionSpec`, and capability resolution. Restart-safe LocalRuntime,
Native/TRL adapters, an embedded controller, and runtime-operation reconciliation
follow within Band C.

See the [core API](api/core.md) for the foundation and the
[implementation plan](https://github.com/szaher/xaytune/blob/main/xaytune-training-harness-spec/15-implementation-plan.md)
for the remaining work.

## Target architecture

Solid green boxes show the implemented foundation; dashed boxes show planned
components. The runtime feedback path is separate from the path that submits work.

![Xaytune target architecture: experiment control plane, trainer compilation, capability and resilience resolution, runtime backends, and infrastructure.](assets/architecture-overview.svg)

The boundary is **experiment topology, not GPU count**. Xaytune may control a
large distributed training job while its runtime handles second-scale worker
recovery and distributed tensor execution.

The [architecture specification](https://github.com/szaher/xaytune/blob/main/xaytune-training-harness-spec/02-architecture.md)
defines the contracts and dependency boundaries.

## Roadmap

| Architectural band | Status |
|---|---|
| Domain foundation | Complete; typed candidate/identity contracts continue in Band C |
| B — persistence and control records | Complete |
| C — compile/execute, local runtime, runtime reconciliation | **Current** |
| D — durable evaluation and decisioning | Planned |
| E — policy and budget over the Action substrate | Planned |
| F — checkpoints, semantic recovery, interventions | Planned |
| G — rule-based planner and experiment branching | Planned |
| H — daemon hosting and whole-controller restart | Planned |
| I — LLM planner | Planned |
| J — Ray / TorchFT / Training Hub integrations | Planned |

## Existing training capabilities

The existing PyTorch-based trainer stack remains available and will become the
Native trainer implementation under the new architecture. Its capabilities
include SFT, LoRA/QLoRA, DPO/GRPO alignment, evaluation, export, deterministic
multi-stage pipelines, YAML configuration, callbacks, and logging integrations.
Core SFT and DPO paths are tested and functional.

Additional alignment methods, FSDP, Studio, and data preparation remain
experimental. **DeepSpeed is experimental and partial:** optimizer/scheduler
and checkpoint ownership are still tracked by
[TASK-029](https://github.com/szaher/xaytune/blob/main/implementation-plan/backlog.md#task-029-is-still-open).
The existing trainer's evaluation and pipeline features are separate from the
planned durable evaluation coordinator and adaptive experiment controller.

## Quick example — existing trainer API

```python
import xaytune

state = xaytune.finetune(
    model="meta-llama/Llama-3.1-8B",
    dataset="data/train.jsonl",
    method="lora",
    format="alpaca",
    num_epochs=3,
)
```

Or use the CLI:

```bash
xaytune train --config configs/examples/lora_finetune.yaml
```

Installation and these examples use the existing training package. They do not
provide the planned end-to-end 2.0 experiment API.

## Next steps

- [Getting Started](getting-started.md) — install and run the existing trainer.
- [Recipes](recipes/index.md) — fine-tuning, pre-training, and alignment.
- [API Reference](api/index.md) — current API documentation.
- [CLI Reference](cli.md) — current commands.
- [Examples](examples.md) — notebooks and sample configs.
- [Control-plane specification](https://github.com/szaher/xaytune/tree/main/xaytune-training-harness-spec) — target architecture and implementation contracts.
