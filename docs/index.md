# Xaytune

**Xaytune is an agent-native experiment control plane for model post-training and adaptation.**

It records what an experiment is testing, compiles each candidate into a plan a
trainer can run, submits that plan to a runtime, and keeps a durable record of
everything that happened. That record is what lets a restarted controller pick
up running work instead of losing it or starting it twice. It is built to
answer: **what should this training experiment do next?**

**Xaytune controls the experiment.** Trainer integrations compile training intent.
Runtime integrations execute it. Infrastructure schedules and runs the workloads.

!!! info "Pre-release"
    The control plane is on `main` and is not in a release yet. The package on
    PyPI, `0.6.0`, is the [legacy trainer API](getting-started.md), which remains
    available. The first control-plane release is planned as `1.0.0a1`, once it
    can train, evaluate *and* decide.

## Start here

- [Control-plane getting started](control-plane/getting-started.md): install
  from `main`, compile a candidate, train it, cancel it, and attach to it from
  another process.
- [Control-plane concepts](control-plane/concepts.md): candidates, compilers,
  runtimes, the durable record, and what `wait()` means.
- [Legacy trainer API](getting-started.md): `finetune`, `align`, the CLI and
  pipelines from the `0.6.0` package.

## What Xaytune owns, and what it delegates

| Xaytune owns | Today |
|---|---|
| Experiment lifecycle and durable controller state | Available |
| Candidate identity and scientific lineage | Available |
| Training orchestration across trainer backends | Available: Native and TRL, on a local runtime |
| Evaluation orchestration | Lifecycle available; built-in evaluators are next |
| Decisions and branching | Planned |
| Resilience policy and semantic recovery | Planned |
| Policy gates, budgets and agent-driven control | Planned |

Xaytune delegates tensor execution, distributed training, worker management
and second-scale worker recovery, scheduling, and infrastructure admission to
trainer, runtime and platform integrations. The boundary is **experiment
topology, not GPU count**.

## Available today

Status as of **2026-09-25**, after PR-013:

- **A durable record**: SQLite persistence, versioned migrations, atomic
  state + event + outbox transactions, a `RuntimeOperation` journal that
  records every external effect before it is attempted, and cancellation as a
  recorded Action.
- **Candidates with identities**: `CandidateSpec` and a versioned
  `CandidateFingerprint`.
- **Compilation**: `NativeCompiler` and `TRLCompiler` compile full-parameter
  SFT into a serializable `TrainingExecutionSpec`, and refuse any candidate that
  leaves a training-relevant value to a trainer default.
- **A restart-safe local runtime**: `LocalRuntime`, with idempotent
  `submit_or_get` and versioned telemetry.
- **An embedded controller**: `EmbeddedControllerHost` and `ExperimentHandle`
  (`submit`, `status`, `wait`, `cancel`, `events`, `attach`). A restarted
  process adopts running work and never submits it twice.
- **The durable evaluation lifecycle**: evaluation runs, attempts, results and
  cycles, recorded and restart-safe. No evaluator is built in yet; you register
  your own `Evaluator` until the next PR.

## Planned

Production evaluators (next), then the DecisionEngine, policy and budgets,
checkpoints and semantic recovery, a rule-based planner and branching, daemon
hosting, an LLM planner, and Ray / TorchFT / Training Hub integrations. None of
these exist yet.

## Architecture

Solid green boxes are implemented; dashed boxes are planned. The runtime
feedback path is separate from the path that submits work.

![Xaytune target architecture: experiment control plane, trainer compilation, capability resolution, runtime backends, and infrastructure, with implementation status after PR-013.](assets/architecture-overview.svg)

The [architecture specification](https://github.com/szaher/xaytune/blob/main/xaytune-training-harness-spec/02-architecture.md)
defines the contracts and dependency boundaries, and the
[implementation plan](https://github.com/szaher/xaytune/blob/main/xaytune-training-harness-spec/15-implementation-plan.md)
the order the remaining work lands in.

## Roadmap

| Architectural band | Status |
|---|---|
| A — domain foundation | Complete |
| B — persistence and control records | Complete |
| C — compile/execute, local runtime, runtime reconciliation | Complete |
| D — durable evaluation and decisioning | **Current**: lifecycle complete; evaluators and DecisionEngine next |
| E — policy and budget over the Action substrate | Planned |
| F — checkpoints, semantic recovery, interventions | Planned |
| G — rule-based planner and experiment branching | Planned |
| H — daemon hosting and whole-controller restart | Planned |
| I — LLM planner | Planned |
| J — Ray / TorchFT / Training Hub integrations | Planned |

## Legacy trainer API

Xaytune began as an opinionated PyTorch training library. That library is what
PyPI's `0.6.0` contains, and it stays available: SFT with LoRA/QLoRA, DPO and
GRPO alignment, evaluation, export, multi-stage pipelines, YAML configuration,
callbacks and logging integrations. Its trainer is what `NativeCompiler` runs,
behind the compile/execute boundary.

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

Additional alignment methods, FSDP, Studio and data preparation are
experimental. **DeepSpeed is experimental and partial:** optimizer/scheduler
and checkpoint ownership are still tracked by
[TASK-029](https://github.com/szaher/xaytune/blob/main/implementation-plan/backlog.md#task-029-is-still-open).
The trainer's own `evaluate()` and pipeline evaluation stages are separate from
the control plane's durable evaluation lifecycle.

See [Getting Started](getting-started.md), [Recipes](recipes/index.md),
[CLI Reference](cli.md) and [Examples](examples.md).
