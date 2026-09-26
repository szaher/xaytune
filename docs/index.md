# Xaytune

**Xaytune is an agent-native experiment control plane for model post-training and adaptation.**

It records what an experiment is testing, compiles each candidate into a plan a
trainer can run, submits that plan to a runtime, and keeps a durable record of
everything that happened. That record is what lets a restarted controller pick
up running work instead of losing it or starting it twice. It is built to
answer: **what should this training experiment do next?**

**Xaytune controls the experiment.** Trainer integrations compile training intent.
Runtime integrations execute it. Infrastructure schedules and runs the workloads.

!!! info "Alpha"
    `1.0.0a1` is the first release of the control plane: declare an experiment,
    train, evaluate, and make a durable, deterministic decision. It is a
    pre-release: install it with `pip install "xaytune==1.0.0a1"` (a plain
    `pip install xaytune` still installs `0.6.0`), and expect the API to change
    before `1.0.0`. The [legacy trainer API](getting-started.md) is still
    included.

## Start here

- [Control-plane getting started](control-plane/getting-started.md): install
  it, compile a candidate, train it, cancel it, and attach to it from
  another process.
- [Control-plane concepts](control-plane/concepts.md): candidates, compilers,
  runtimes, the durable record, and what `wait()` means.
- [Legacy trainer API](getting-started.md): `finetune`, `align`, the CLI and
  pipelines, from `0.6.0` and still included.

## What Xaytune owns, and what it delegates

| Xaytune owns | Today |
|---|---|
| Experiment lifecycle and durable controller state | Available |
| Candidate identity and scientific lineage | Available |
| Training orchestration across trainer backends | Available: Native and TRL, on a local runtime |
| Evaluation orchestration | Available: the built-in native evaluator; lm-eval planned |
| Decisions and branching | Decisions available: deterministic thresholds on the objective; branching planned |
| Resilience policy and semantic recovery | Planned |
| Policy gates, budgets and agent-driven control | Planned |

Xaytune delegates tensor execution, distributed training, worker management
and second-scale worker recovery, scheduling, and infrastructure admission to
trainer, runtime and platform integrations. The boundary is **experiment
topology, not GPU count**.

## Available today

Status as of **2026-09-25**, after PR-015:

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
- **Durable evaluation, with a built-in evaluator**: evaluation runs, attempts,
  results and cycles, recorded and restart-safe. The `native` evaluator
  measures next-token loss, perplexity and token accuracy on a local held-out
  file pinned by its content digest.
- **Durable decisions**: an evaluated candidate is decided from the record --
  the objective's target and constraints against the recorded results -- and
  the decision is written with the transitions it causes, in one commit. A
  candidate that cannot be decided (no target, a missing metric) stays
  `DECIDING`, with the reason recorded.

## Planned

Policy and budgets, checkpoints and semantic recovery, a rule-based planner
and branching (with decisions that compare candidates), daemon hosting, an LLM
planner, and Ray / TorchFT / Training Hub integrations. An lm-eval evaluator, with
pinned task and dataset versions, is a planned integration alongside them.
None of these exist yet.

## Architecture

Solid green boxes are implemented; dashed boxes are planned. The runtime
feedback path is separate from the path that submits work.

![Xaytune target architecture: experiment control plane, trainer compilation, capability resolution, runtime backends, and infrastructure, with implementation status after PR-015.](assets/architecture-overview.svg)

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
| D — durable evaluation and decisioning | Complete: lifecycle, native evaluator, threshold decisions; lm-eval planned |
| E — policy and budget over the Action substrate | Planned |
| F — checkpoints, semantic recovery, interventions | Planned |
| G — rule-based planner and experiment branching | Planned |
| H — daemon hosting and whole-controller restart | Planned |
| I — LLM planner | Planned |
| J — Ray / TorchFT / Training Hub integrations | Planned |

## Legacy trainer API

Xaytune began as an opinionated PyTorch training library. That library is what
PyPI's `0.6.0` contains, and `1.0.0a1` still includes it: SFT with LoRA/QLoRA, DPO and
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
