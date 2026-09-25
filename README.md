<p align="center">
  <img src="https://szaher.github.io/xaytune/assets/logo.png" alt="xaytune" width="400">
</p>

# Xaytune

**Xaytune is an agent-native experiment control plane for model post-training and adaptation.**

It records what an experiment is testing, compiles each candidate into a plan a
trainer can run, submits that plan to a runtime, and keeps a durable record of
everything that happened. That record is what lets a restarted controller pick
up running work instead of losing it or starting it twice. The question it is
built to answer is:

> What should this training experiment do next?

**Xaytune controls the experiment.** Trainer integrations compile training intent.
Runtime integrations execute it. Infrastructure schedules and runs the workloads.

**[Documentation](https://szaher.github.io/xaytune/)** | **[Control-plane getting started](https://szaher.github.io/xaytune/control-plane/getting-started/)** | **[Examples](https://github.com/szaher/xaytune/tree/main/examples/control_plane/)** | **[Architecture specification](https://github.com/szaher/xaytune/blob/main/xaytune-training-harness-spec/README.md)**

> **Alpha.** `1.0.0a1` is the first release of the control plane: declare an
> experiment, train, evaluate, and make a durable, deterministic decision. It
> is a pre-release, so `pip install xaytune` still installs `0.6.0`; see
> [Install](#install). Expect the control-plane API to change before `1.0.0`.
> The trainer library from `0.6.0` is still included, as the
> [Legacy trainer API](#legacy-trainer-api).

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

| Xaytune delegates |
|---|
| Tensor execution and distributed training |
| Worker management and second-scale worker recovery |
| GPU scheduling and infrastructure admission |
| Cluster and Kubernetes workload execution |

The boundary is **experiment topology, not GPU count**. Xaytune may control a
large distributed training job while the runtime stays responsible for
distributed tensor execution and for recovering individual workers.

## Available today

Status as of **2026-09-25**, after PR-015.

- **A durable record.** SQLite persistence with versioned migrations and
  optimistic concurrency. Every state change commits in one transaction with
  its event and outbox entry. A `RuntimeOperation` journal records every
  external effect *before* it is attempted. Cancellation is a recorded Action
  whose outcome is also recorded.
- **Candidates with identities.** `CandidateSpec` declares everything that
  changes what a model learns. `CandidateFingerprint` is a versioned
  projection of it, so later schema additions cannot silently change the
  identity of candidates already recorded.
- **Compilation.** `TrainerCompiler` turns a candidate into a serializable
  `TrainingExecutionSpec` and never executes anything. `NativeCompiler` and
  `TRLCompiler` cover full-parameter SFT. A candidate that leaves a
  training-relevant value to a trainer default is refused, with every reason
  listed.
- **A restart-safe local runtime.** `LocalRuntime` runs workers as separate
  processes and streams versioned telemetry. It takes submissions through
  `submit_or_get`, so re-submitting under the same operation id returns the
  existing workload instead of starting a new one.
- **An embedded controller.** `EmbeddedControllerHost.submit()` returns an
  `ExperimentHandle` with `status`, `wait`, `cancel` and `events`. A new
  process can `attach()` to the experiment and adopt its running workload,
  without orphaning it or submitting it twice.
- **Durable evaluation, with a built-in evaluator.** An `ExperimentSpec` can
  name an `EvaluationSpec`. After training, the host evaluates the trained
  model through the same journal, runtime and telemetry, records the result,
  and moves the candidate to `DECIDING`. Evaluation runs, attempts, results
  and cycles survive a restart. The built-in `native` evaluator measures
  next-token loss, perplexity and token accuracy on a local held-out file
  pinned by its content digest. An evaluation it cannot run exactly as
  declared is refused at submission, before anything trains.
- **Durable decisions.** Once a candidate is evaluated, a decision engine
  decides it from the record alone: the experiment's objective and that
  evaluation's results. The built-in engine compares the recorded values with
  the objective's target and constraints. A target met gives a `COMPLETED`
  candidate and a `SUCCEEDED` experiment, which names it as its best
  candidate. A target missed gives `REJECTED` and `FAILED`. A violated
  constraint rejects the candidate but leaves the experiment open for
  another. The decision is recorded, with the evidence and a fingerprint of
  its inputs, in the same commit that applies it. A candidate it cannot decide (no target, a missing metric) stays
  `DECIDING`, with the reason recorded; nothing is guessed.
- **A reproducible environment.** `uv.lock` pins every dependency, and a CI job
  tests the locked environment. The TRL worker refuses any TRL or Transformers
  release it has not been classified against.

Seen from a caller:

```python
from xaytune.experiment import EmbeddedControllerHost

host = EmbeddedControllerHost("xaytune-workdir/state.db")
handle = await host.submit(spec)        # spec: an ExperimentSpec
result = await handle.wait()            # quiescent: every piece of work is settled
print(result.status, result.next_stage) # SUCCEEDED, None once evaluated and decided
```

`wait()` returns when the controller has nothing left it can run, which is not
the same as the experiment being finished. The result says which stage would
run next, if any. See
[the control-plane examples](https://github.com/szaher/xaytune/tree/main/examples/control_plane/) for runnable code.

## Planned

In the order the [implementation plan](https://github.com/szaher/xaytune/blob/main/xaytune-training-harness-spec/15-implementation-plan.md)
builds them:

1. **Policy and budgets** over the Action substrate.
2. **Checkpoints, semantic recovery and interventions.**
3. **A rule-based planner and experiment branching**, and with it decisions
   that compare candidates -- promotion, and noise-aware comparison across
   replicates -- which one candidate cannot support.
4. **Daemon hosting** and whole-controller restart.
5. **An LLM planner** proposing candidates under policy.
6. **Ray, TorchFT and Training Hub** integrations.

Alongside them, an **lm-eval evaluator** is a planned integration, with each
task's definition and dataset pinned to immutable versions when the
experiment is submitted.

None of these exist yet, and nothing in this repository should be read as
claiming they do.

## Architecture

The target architecture keeps scientific intent, compilation, capability
resolution and execution separate. Solid green boxes are implemented; dashed
boxes are planned. The runtime feedback path is separate from the path that
submits work.

![Xaytune target architecture: experiment control plane, CandidateSpec, trainer compilers, TrainingExecutionSpec, capability resolution, ResolvedExecutionPlan, runtime backends, and infrastructure, with implementation status after PR-015.](https://szaher.github.io/xaytune/assets/architecture-overview.svg)

See the [architecture specification](https://github.com/szaher/xaytune/blob/main/xaytune-training-harness-spec/02-architecture.md)
for protocol and dependency boundaries.

<details>
<summary>Implementation history</summary>

Persistence (band B): [SQLite repository (#18)](https://github.com/szaher/xaytune/pull/18),
[events, outbox, and journal (#19)](https://github.com/szaher/xaytune/pull/19),
[Action substrate (#20)](https://github.com/szaher/xaytune/pull/20),
[experiment graph (#21)](https://github.com/szaher/xaytune/pull/21),
[persistence hardening (#22)](https://github.com/szaher/xaytune/pull/22).

Compile/execute (band C): [CandidateSpec and fingerprints (#24)](https://github.com/szaher/xaytune/pull/24),
[compile/execute contracts (#25)](https://github.com/szaher/xaytune/pull/25),
[LocalRuntime (#26)](https://github.com/szaher/xaytune/pull/26),
[telemetry contracts (#27)](https://github.com/szaher/xaytune/pull/27),
[NativeCompiler and NativeWorker (#28)](https://github.com/szaher/xaytune/pull/28),
[TRLCompiler and TRLWorker (#29)](https://github.com/szaher/xaytune/pull/29),
[EmbeddedControllerHost and ExperimentHandle (#30)](https://github.com/szaher/xaytune/pull/30),
[runtime-operation reconciliation (#31)](https://github.com/szaher/xaytune/pull/31),
[dependency reproducibility (#32)](https://github.com/szaher/xaytune/pull/32).

Evaluation (band D): [durable evaluation lifecycle (#33)](https://github.com/szaher/xaytune/pull/33).

</details>

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

## Install

```bash
pip install "xaytune==1.0.0a1"          # the control plane, and the legacy trainer API
pip install "xaytune[trl]==1.0.0a1"     # adds the TRL trainer
```

`1.0.0a1` is a pre-release, so pip installs it only when asked: by version, as
above, or unpinned with `pip install --upgrade --pre xaytune` (`--upgrade`
matters where `0.6.0` is already installed, which pip would otherwise keep). A
plain `pip install xaytune` still installs **0.6.0, the legacy trainer API**
alone.

To install exactly what CI tests, or to work on Xaytune, install from a clone:

```bash
git clone https://github.com/szaher/xaytune && cd xaytune
uv sync --locked                # or: pip install -e .
uv sync --locked --extra trl    # adds the TRL trainer; or: pip install -e ".[trl]"
```

`import xaytune` does not import torch, transformers or TRL. The control plane
imports without them, and training runs in separate worker processes that do.

## Legacy trainer API

Xaytune began as an opinionated PyTorch training library, and that library is
still here: `finetune`, `pretrain`, `align`, `evaluate`, the `xaytune` CLI, and
multi-stage pipelines. It is what PyPI's `0.6.0` contains, and `1.0.0a1` still
includes it. It stays available
through the transition, and its trainer is what `NativeCompiler` runs, behind
the new compile/execute boundary.

The maturity labels in this section refer to **this trainer library**, not to
the control plane.

**Core (tested, recommended for use):**

- **SFT fine-tuning** — full, LoRA, QLoRA with proper prompt masking (only response tokens contribute to loss)
- **Multi-turn conversation masking** — per-turn label masking for chat and ShareGPT formats
- **DPO alignment** — Direct Preference Optimization with response-only log-probability scoring
- **GRPO alignment** — Group Relative Policy Optimization, reference-model-free by default
- **5 data formats** — Alpaca, ShareGPT, OpenAI chat, raw text, preference pairs
- **Multi-stage pipelines** — chain SFT → merge → DPO → eval → export in a single command
- **Evaluation** — built-in metrics (loss, perplexity, token accuracy) + lm-eval-harness benchmarks
- **Export** — merge LoRA adapters, push to HuggingFace Hub
- **Config system** — YAML configs with inheritance, CLI overrides, Pydantic validation
- **Callbacks** — event-driven hooks for checkpointing, early stopping, progress, custom logic
- **4 logging backends** — console, TensorBoard, W&B, MLflow

**Experimental (limitations vary; see [Maturity](#maturity)):**

- **ORPO / SimPO alignment** — implemented with numerically stable loss, needs real-world validation
- **REINFORCE alignment** — vanilla policy gradient, functional but minimal
- **PPO** — value head, rollout buffer, and multi-epoch training; requires online RL
- **Online RL** — generate→score→train pipeline for RL methods
- **DeepSpeed** — partial ZeRO integration; optimizer/scheduler and checkpoint ownership remain open (TASK-029)
- **FSDP** — wrapping with sharding strategy, CPU offload, mixed precision
- **GGUF conversion** — delegates to llama.cpp tools (requires separate installation)
- **Model merging** — Linear, SLERP, TIES, DARE weight interpolation
- **Agent fine-tuning** — tool-use data formats with per-message loss masking
- **Training Studio** — Gradio web UI for configuring and launching runs
- **Data preparation** — generate, filter, deduplicate, convert pipeline

### Legacy quickstart

`pip install xaytune` installs it (`0.6.0`), as does `1.0.0a1`; optional extras include
`[wandb]`, `[mlflow]`, `[tensorboard]`, `[deepspeed]`, `[eval]` (lm-eval-harness),
`[studio]` and `[all]`. The notebooks in [`examples/`](https://github.com/szaher/xaytune/tree/main/examples/) use this API.

#### Python API

```python
import xaytune

# LoRA fine-tuning (prompt tokens masked, trains on response only)
state = xaytune.finetune(
    model="meta-llama/Llama-3.1-8B",
    dataset="data/train.jsonl",
    method="lora",
    format="alpaca",
    num_epochs=3,
)

# DPO alignment (response-only log-prob scoring)
state = xaytune.align(
    model="output/sft-model",
    dataset="data/preferences.jsonl",
    method="dpo",
    format="preference",
)

# Evaluation
results = xaytune.evaluate(
    model="output/my-model",
    dataset=[{"input_ids": [1, 2], "labels": [1, 2]}],
    metrics=["loss", "perplexity", "token_accuracy"],
)
```

#### Multi-Stage Pipeline

Chain training stages in a single command:

```yaml
# pipeline.yaml
name: sft-to-aligned
output_dir: output/pipeline
stages:
  - name: sft
    recipe: finetune
    method: lora
    model_name: "meta-llama/Llama-3.1-8B"
    data: { path: "data/train.jsonl", format: alpaca }
    trainer: { num_epochs: 3, learning_rate: 2e-4 }

  - name: merge
    export: merge

  - name: dpo
    recipe: align
    method: dpo
    data: { path: "data/prefs.jsonl", format: preference }
    trainer: { num_epochs: 1, learning_rate: 5e-6 }

  - name: eval
    eval: { metrics: [loss, perplexity], benchmarks: [mmlu] }
```

```bash
xaytune pipeline --config pipeline.yaml
xaytune pipeline --config pipeline.yaml --dry-run
xaytune pipeline --config pipeline.yaml --resume-from dpo
```

#### CLI

```bash
# Train
xaytune train --config configs/examples/lora_finetune.yaml
xaytune train --config configs/examples/lora_finetune.yaml --override model.name=mistralai/Mistral-7B-v0.3

# Evaluate
xaytune eval --model output/my-model --benchmarks mmlu,gsm8k
xaytune eval --model output/my-model --dataset data/eval.jsonl --metrics loss,perplexity

# Export
xaytune export merge --checkpoint output/lora-ckpt --output output/merged
xaytune export push --model output/merged --repo username/my-model

# Distributed training
xaytune launch --config configs/examples/lora_finetune.yaml --nproc-per-node 4

# Training Studio (experimental)
xaytune studio --port 7860
```

#### Config file

```yaml
recipe: finetune
method: lora

model:
  name: meta-llama/Llama-3.1-8B

data:
  path: data/train.jsonl
  format: alpaca
  eval_split: 0.05
  packing: true
  max_seq_length: 2048

lora:
  rank: 16
  alpha: 32

trainer:
  batch_size: 4
  learning_rate: 2e-4
  num_epochs: 3
  mixed_precision: bf16
  checkpoint_every_n_steps: 500

eval:
  every_n_steps: 500
  metrics: [loss, perplexity]

logging:
  backends: [console, tensorboard]
```

### Recipes

| Recipe | Methods | Use case |
|--------|---------|----------|
| `finetune` | `full`, `lora`, `qlora` | Supervised fine-tuning on instruction data |
| `pretrain` | `full` | Pre-training or continued pre-training on raw text |
| `align` | `dpo`, `grpo` | Alignment with human preferences (recommended) |
| `align` | `orpo`, `simpo`, `reinforce`, `ppo` | Alignment (experimental — see [Maturity](#maturity)) |

### Maturity

| Feature | Status | Notes |
|---------|--------|-------|
| SFT (full/LoRA) | **Stable** | Prompt masking, multi-turn, sequence packing |
| QLoRA | **Stable** | Uses `prepare_model_for_kbit_training` |
| DPO | **Stable** | Response-only log-probs, frozen reference model |
| GRPO | **Stable** | Reference-model-free, optional KL via `kl_coeff` |
| Multi-stage pipeline | **Stable** | Sequential chaining with auto-inheritance |
| Evaluation | **Stable** | Metrics + lm-eval benchmarks |
| Export (merge, Hub push) | **Stable** | LoRA merge, HF Hub push |
| Config system | **Stable** | YAML, inheritance, Pydantic validation |
| Callbacks + logging | **Stable** | 4 backends, exception isolation |
| ORPO | **Experimental** | Numerically stable, needs real-world validation |
| SimPO | **Experimental** | Length-normalized, reference-free |
| REINFORCE | **Experimental** | Vanilla policy gradient |
| PPO | **Experimental** | Full PPO trainer with value head, rollout buffer, multi-epoch training. Requires `online_rl.enabled=True` |
| DeepSpeed | **Experimental / partial** | Optimizer/scheduler and checkpoint ownership remain open; see [TASK-029](https://github.com/szaher/xaytune/blob/main/implementation-plan/backlog.md#task-029-is-still-open) |
| FSDP | **Experimental** | Sharding, offload, mixed precision wrapping |
| GGUF export | **Experimental** | Requires llama.cpp tools installed separately |
| Model merging | **Experimental** | TIES, DARE, SLERP, linear interpolation |
| Agent fine-tuning | **Experimental** | Tool-use formats with per-message masking |
| Training Studio | **Experimental** | Gradio UI for job configuration and launch |
| Data preparation | **Experimental** | Generate, filter, deduplicate, convert |
| Online RL | **Experimental** | Generate→score→train for RL methods |

### Extensibility

Register custom components with decorators:

```python
from xaytune.data import register_format
from xaytune.eval import register_metric
from xaytune.recipes.align.rewards import register_reward
from xaytune.trainer import on

@register_format("my-format")
def parse_my_data(sample):
    return {"text": f"Q: {sample['q']}\nA: {sample['a']}"}

@register_metric("domain-accuracy")
def domain_accuracy(predictions, references, **kwargs):
    return sum(p == r for p, r in zip(predictions, references)) / len(predictions)

@register_reward("brevity")
def brevity_reward(prompt, response, *, max_len=100):
    return 1.0 if len(response) <= max_len else 0.0

@on("step_end")
def log_memory(state):
    print(f"Step {state.global_step}: loss={state.metrics.get('loss', 'N/A')}")
```

## License

Apache 2.0
