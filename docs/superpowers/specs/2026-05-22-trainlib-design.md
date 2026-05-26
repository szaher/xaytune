# xaytune — LLM Training & Fine-Tuning Library

## Overview

xaytune is an open-source Python library for LLM training and fine-tuning, built on PyTorch. It provides a recipe-based architecture with a layered API: simple one-liners for beginners, full control for experts. Config files and Python API are equal citizens.

**Scope:** Pre-training, continued pre-training, full fine-tuning, LoRA/QLoRA, and alignment (DPO, GRPO, PPO, ORPO, SimPO). Multi-node distributed training via DDP, FSDP, and DeepSpeed.

**Non-scope (v1):** Infrastructure management (handled by Ray, Slurm, K8s), Training Studio UI (future — see Section 11).

## 1. Architecture

Recipe-based, three layers:

```
┌─────────────────────────────────────────┐
│           CLI / Config Engine           │  Layer 3 — Interface
├─────────────────────────────────────────┤
│   pretrain │ finetune │ align (recipes) │  Layer 2 — Recipes
├────────┬────────┬─────────┬────────┬────┤
│ models │  data  │ trainer │  eval  │ exp│  Layer 1 — Building Blocks
└────────┴────────┴─────────┴────────┴────┘
         PyTorch / HuggingFace / DeepSpeed
```

**Layer 1 — Building Blocks:** Reusable primitives (model loading, data pipelines, training loop, evaluation, export). Each module has a clear responsibility and a public API.

**Layer 2 — Recipes:** Opinionated workflows that wire building blocks together with sensible defaults. Pre-training, fine-tuning (full/LoRA/QLoRA), and alignment (DPO/GRPO/PPO/ORPO/SimPO).

**Layer 3 — Interface:** CLI (`xaytune train`), config engine (YAML with inheritance and CLI overrides), and Python API (`xaytune.finetune(...)`). All three produce the same result.

## 2. Models & PEFT

### Model Loading

Unified interface for HuggingFace Hub and custom models:

```python
from xaytune.models import load_model

model = load_model("meta-llama/Llama-3.1-8B", quantization="4bit")
model = load_model(MyCustomModel, config=my_config)
```

### PEFT Methods

- **LoRA** — low-rank adapters on attention/MLP layers, configurable rank/alpha/target modules
- **QLoRA** — LoRA on a 4-bit quantized base model (bitsandbytes NF4)
- **Full fine-tuning** — all parameters trainable

### Design Decisions

- Model wrapping is lazy — quantization and PEFT adapters are applied at recipe setup, not at load time. Users can inspect/modify the base model first.
- Target modules are auto-detected per architecture but overridable.
- Checkpointing saves only adapter weights for PEFT methods. `merge_and_export()` produces a standalone model.

### Custom Model Registration

```python
from xaytune.models import register_model

@register_model("my-architecture")
class MyModel(TrainlibModel):
    def __init__(self, config):
        ...
```

## 3. Data Pipeline

### Dataset Loading

```python
from xaytune.data import load_dataset

dataset = load_dataset("data.jsonl", format="alpaca")
dataset = load_dataset("data.jsonl", format="sharegpt")
dataset = load_dataset("data.jsonl", format="chat")
dataset = load_dataset("corpus/", format="text", streaming=True)
dataset = load_dataset("tatsu-lab/alpaca", source="huggingface")
dataset = load_dataset("data.jsonl", format="custom", map_fn=my_formatter)
```

### Built-in Formats

- **alpaca** — instruction/input/output fields
- **sharegpt** — multi-turn conversation format
- **chat** — OpenAI chat message format
- **text** — raw text for pre-training
- **preference** — (prompt, chosen, rejected) triples for alignment

### Design Decisions

- Streaming by default for large datasets. Uses HF `datasets` with memory-mapped files.
- Chat templates are model-aware — tokenization applies the correct template automatically. Users can override.
- Data packing — multiple short examples packed into a single sequence. On by default for fine-tuning.
- Preference data — dedicated `PreferenceDataset` for DPO/ORPO, supporting UltraFeedback and similar formats.
- Preprocessing is lazy and cacheable — tokenization on-the-fly with optional disk caching.
- Data validation — validates a sample batch before training starts (tokenization, sequence lengths, special tokens).

### Custom Format Registration

```python
from xaytune.data import register_format

@register_format("my-format")
def parse_my_data(sample: dict) -> dict:
    return {"instruction": sample["q"], "response": sample["a"]}
```

## 4. Trainer & Training Loop

### Core Trainer

The trainer wraps PyTorch's training loop with distributed support, mixed precision, gradient accumulation, checkpointing, and logging.

```python
from xaytune.trainer import TrainerConfig

config = TrainerConfig(
    strategy="fsdp",
    mixed_precision="bf16",
    batch_size=4,
    gradient_accumulation=8,
    learning_rate=2e-4,
    num_epochs=3,
)
```

### Callbacks via Decorators

```python
from xaytune.trainer import on

@on("step_end")
def log_gpu_memory(state):
    print(f"Step {state.step}: {state.gpu_memory_used_mb}MB")

@on("eval_end")
def early_stop(state):
    if state.metrics["loss"] < 0.1:
        state.stop_training()
```

### Callback Events

- `train_start`, `train_end`
- `epoch_start`, `epoch_end`
- `step_start`, `step_end`
- `eval_start`, `eval_end`
- `checkpoint_saved`
- `error`

## 5. Distributed Training

### Launcher

Default launcher is `torchrun`:

```bash
# Single GPU
xaytune train --config finetune.yaml

# Multi-GPU (single node)
torchrun --nproc_per_node=4 -m xaytune train --config finetune.yaml

# Multi-node
torchrun --nnodes=2 --node_rank=0 --master_addr=host0 \
    -m xaytune train --config finetune.yaml
```

### Strategies

- **DDP** — each GPU holds a full model copy. Default for LoRA fine-tuning.
- **FSDP** — shards parameters, gradients, and optimizer states. Default for pre-training and full fine-tuning.
- **DeepSpeed** — ZeRO Stage 1/2/3. Optional dependency.

### Design Decisions

- Strategy is a recipe concern — each recipe has a sensible default. Users override only when needed.
- Checkpointing is distributed-aware — async saves, rank 0 metadata, sharded for FSDP.
- Resumption is automatic: `xaytune train --config finetune.yaml --resume`.
- No infrastructure coupling — xaytune uses whatever `torchrun` or the infrastructure layer provides.

## 6. Alignment & RL Methods

### Preference-Based (no reward model)

- **DPO** — Direct Preference Optimization. Chosen/rejected pairs.
- **ORPO** — Odds Ratio Preference Optimization. Combined SFT + preference.
- **SimPO** — Reference-model-free DPO variant.

### RL-Based (reward-driven)

- **GRPO** — Group Relative Policy Optimization. Generates multiple responses, ranks as group. No critic model needed. First-class citizen.
- **PPO** — Proximal Policy Optimization. Classic RLHF with reward model + value head.
- **REINFORCE** — Vanilla REINFORCE with baseline.

### API

```python
from xaytune import align

align(model="my-sft-model", dataset="prefs.jsonl", method="grpo")

align(
    model="my-sft-model",
    dataset="prefs.jsonl",
    method="grpo",
    group_size=8,
    kl_coeff=0.04,
    reward_fn=my_custom_reward,
)
```

### Custom Rewards

```python
from xaytune.align import register_reward

@register_reward("code-correctness")
def code_reward(prompt: str, response: str) -> float:
    passed = run_test_cases(prompt, response)
    return 1.0 if passed else 0.0
```

### Design Decisions

- GRPO is a first-class citizen — no critic model means less VRAM and simpler distributed setup.
- Modular reward — pluggable via decorator. Reward model, function, or API call (LLM-as-judge).
- New methods are easy to add — alignment recipe base class handles generation, scoring, policy updates. A new algorithm only implements the policy loss.
- Optional dependencies — PPO pulls in value head/reward model machinery. DPO/GRPO don't require it.

## 7. Evaluation & Export

### Evaluation

```python
from xaytune import evaluate

results = evaluate(model="output/my-model", dataset="eval.jsonl")
results = evaluate(model="output/my-model", benchmarks=["mmlu", "hellaswag", "gsm8k"])
```

- Built-in: perplexity, loss, token accuracy
- Benchmark integration via `lm-eval-harness` (optional dependency)
- Side-by-side: `xaytune compare model-a/ model-b/ --benchmarks mmlu,gsm8k`
- Results saved alongside checkpoints

### Custom Metrics

```python
from xaytune.eval import register_metric

@register_metric("domain-accuracy")
def domain_accuracy(predictions, references) -> float:
    return sum(p == r for p, r in zip(predictions, references)) / len(predictions)
```

### Export

```python
from xaytune import export

export.merge("output/lora-checkpoint", save_to="output/merged-model")
export.to_gguf("output/merged-model", quantization="Q4_K_M")
export.push_to_hub("output/merged-model", repo="username/my-model")
export.save("output/merged-model", include_metadata=True)
```

### CLI

```bash
xaytune eval --model output/my-model --benchmarks mmlu,gsm8k
xaytune export merge --checkpoint output/lora --output output/merged
xaytune export gguf --model output/merged --quant Q4_K_M
xaytune export push --model output/merged --repo username/my-model
```

## 8. Config System

### YAML Config

```yaml
recipe: finetune
method: lora

model:
  name: meta-llama/Llama-3.1-8B
  quantization: 4bit

lora:
  rank: 16
  alpha: 32
  target_modules: auto

data:
  path: data/train.jsonl
  format: alpaca
  eval_split: 0.05
  packing: true

trainer:
  strategy: ddp
  mixed_precision: bf16
  batch_size: 4
  gradient_accumulation: 8
  learning_rate: 2e-4
  num_epochs: 3
  checkpointing:
    every_n_steps: 500
    save_last: true

eval:
  every_n_steps: 500
  metrics: [loss, perplexity]

logging:
  backends: [tensorboard, wandb]
  project: my-finetune-run

output:
  dir: output/my-model
  merge_on_complete: true
```

### Config Features

- **CLI overrides** — `xaytune train --config finetune.yaml --model.name=different-model`
- **Inheritance** — `base: defaults/lora.yaml` with only overrides
- **Validation** — schema-validated before training. Clear errors with suggestions.
- **Export** — every run saves its resolved config. Any run is reproducible.

### Logging Integrations

- TensorBoard (built-in)
- Weights & Biases (optional)
- MLflow (optional)
- Console with rich progress bars (always on)

## 9. Developer Experience

### UX Principles

1. **Fail early, fail clearly** — validate configs, check GPU availability, verify compatibility before training. Error messages suggest fixes.
2. **Progress by default** — rich progress bars, ETA, loss curves in terminal, GPU utilization.
3. **Sensible defaults** — a recipe works with just a model name and dataset path.
4. **Discoverability** — `xaytune list recipes`, `xaytune list formats`, `xaytune list metrics` show all registered components.

### Decorator Summary

| Decorator | Purpose |
|-----------|---------|
| `@register_model` | Register custom model architectures |
| `@register_format` | Register custom data formats |
| `@register_metric` | Register custom evaluation metrics |
| `@register_recipe` | Register custom training recipes |
| `@register_reward` | Register custom reward functions |
| `@on(event)` | Hook into training loop events |

## 10. Project Structure

```
xaytune/
├── pyproject.toml
├── README.md
├── xaytune/
│   ├── __init__.py              # High-level API
│   ├── cli.py                   # CLI entry point
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schema.py            # Config validation & schemas
│   │   ├── parser.py            # YAML loading, inheritance, CLI overrides
│   │   └── defaults/            # Built-in default configs
│   ├── models/
│   │   ├── __init__.py          # load_model, register_model
│   │   ├── loader.py            # HF model loading, quantization
│   │   ├── peft.py              # LoRA/QLoRA wrapping
│   │   └── registry.py          # Custom model registry
│   ├── data/
│   │   ├── __init__.py          # load_dataset, register_format
│   │   ├── formats.py           # Built-in formats
│   │   ├── packing.py           # Sequence packing
│   │   ├── preferences.py       # Preference dataset for alignment
│   │   └── registry.py          # Custom format registry
│   ├── trainer/
│   │   ├── __init__.py          # Trainer, TrainerConfig, on()
│   │   ├── loop.py              # Core training loop
│   │   ├── distributed.py       # DDP, FSDP, DeepSpeed wrappers
│   │   ├── callbacks.py         # Callback system
│   │   └── checkpointing.py     # Save/resume
│   ├── recipes/
│   │   ├── __init__.py          # register_recipe, Recipe base class
│   │   ├── base.py              # Recipe base class, shared logic
│   │   ├── pretrain.py          # Pre-training recipe
│   │   ├── finetune.py          # Fine-tuning recipe (full, LoRA, QLoRA)
│   │   └── align/
│   │       ├── __init__.py
│   │       ├── base.py          # Alignment recipe base class
│   │       ├── dpo.py           # DPO, SimPO, ORPO
│   │       ├── grpo.py          # GRPO
│   │       └── ppo.py           # PPO, REINFORCE
│   ├── eval/
│   │   ├── __init__.py          # evaluate, register_metric
│   │   ├── metrics.py           # Built-in metrics
│   │   └── benchmarks.py        # lm-eval-harness integration
│   ├── export/
│   │   ├── __init__.py          # merge, to_gguf, push_to_hub
│   │   ├── merge.py             # LoRA adapter merging
│   │   ├── gguf.py              # GGUF conversion
│   │   └── hub.py               # HuggingFace Hub push
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── console.py           # Rich progress bars
│   │   └── integrations.py      # TensorBoard, W&B, MLflow
│   └── utils/
│       ├── __init__.py
│       ├── validation.py        # Pre-flight checks
│       └── registry.py          # Base registry pattern
├── configs/
│   └── examples/                # Example configs
└── tests/
    ├── test_models/
    ├── test_data/
    ├── test_trainer/
    ├── test_recipes/
    ├── test_eval/
    └── test_config/
```

### Packaging

- `pip install xaytune` — core (PyTorch, HF Transformers, peft, bitsandbytes)
- `pip install xaytune[deepspeed]` — adds DeepSpeed
- `pip install xaytune[eval]` — adds lm-eval-harness
- `pip install xaytune[wandb]` — adds W&B
- `pip install xaytune[all]` — everything

## 11. Future: Training Studio

Not in v1 scope, but the architecture is designed to support a future UI-driven training experience:

### Architectural Constraints (enforced now)

- **Config is fully serializable** — everything round-trips through JSON/YAML. A web UI can generate configs.
- **Training state is observable** — the callback/event system supports external subscribers (WebSocket, SSE). Metrics, progress, and status are queryable, not just printed.
- **API is embeddable** — recipes are launchable and controllable programmatically (start, pause, resume, cancel) from a web backend (FastAPI/Flask).
- **Job metadata** — every run gets a unique ID with stored config, status, and metrics. The studio will list, compare, filter, and resume runs.
- **No CLI assumptions** — core logic never assumes it's running in a terminal. Console output is a logging backend, not the only interface.

### Envisioned Studio Features (future)

- Visual config builder — drag-and-drop recipe configuration
- Real-time training dashboard — loss curves, GPU utilization, ETA
- Experiment comparison — side-by-side metrics across runs
- Dataset browser — preview and validate data before training
- Model registry — track trained models, their lineage, and benchmarks

## 12. Testing Strategy

- **Unit tests** — per-module tests for data formats, config parsing, model loading, metric computation
- **Integration tests** — end-to-end recipe runs on small models (tiny-llama scale) with a few steps
- **Distributed tests** — multi-GPU tests marked for CI environments with GPU access
- **Config validation tests** — ensure schema catches invalid configs and produces helpful errors
