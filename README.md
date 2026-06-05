<p align="center">
  <img src="docs/assets/logo.png" alt="xaytune" width="400">
</p>

<p align="center">
  An opinionated LLM training and fine-tuning library built on PyTorch.<br>
  Recipe-based architecture with a layered API: simple one-liners for beginners, full control for experts.
</p>

**[Documentation](https://szaher.github.io/xaytune/)** | **[Examples](https://szaher.github.io/xaytune/examples/)** | **[API Reference](https://szaher.github.io/xaytune/api/)**

> **Status: Alpha (v0.6.0)** — Core SFT and DPO paths are tested and functional. Some features are experimental. See [Maturity](#maturity) below.

## Features

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

**Experimental (functional but not battle-tested):**
- **ORPO / SimPO alignment** — implemented with numerically stable loss, needs real-world validation
- **REINFORCE alignment** — vanilla policy gradient, functional but minimal
- **PPO** — simplified clipped policy gradient (not a full PPO trainer — no rollout buffer, GAE, or value model)
- **Online RL** — generate→score→train pipeline for RL methods
- **DeepSpeed** — ZeRO integration via `ds.initialize()`, engine-aware training loop
- **FSDP** — wrapping with sharding strategy, CPU offload, mixed precision
- **GGUF conversion** — delegates to llama.cpp tools (requires separate installation)
- **Model merging** — Linear, SLERP, TIES, DARE weight interpolation
- **Agent fine-tuning** — tool-use data formats with per-message loss masking
- **Training Studio** — Gradio web UI for configuring and launching runs
- **Data preparation** — generate, filter, deduplicate, convert pipeline

## Install

```bash
pip install xaytune
```

Optional extras:

```bash
pip install xaytune[wandb]       # Weights & Biases logging
pip install xaytune[mlflow]      # MLflow logging
pip install xaytune[deepspeed]   # DeepSpeed distributed training
pip install xaytune[eval]        # lm-eval-harness benchmarks
pip install xaytune[studio]      # Training Studio web UI
pip install xaytune[all]         # Everything
```

## Quickstart

### Python API

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

### Multi-Stage Pipeline

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

### CLI

```bash
# Train
xaytune train --config configs/lora_finetune.yaml
xaytune train --config configs/lora_finetune.yaml --override model.name=mistralai/Mistral-7B-v0.3

# Evaluate
xaytune eval --model output/my-model --benchmarks mmlu,gsm8k
xaytune eval --model output/my-model --dataset data/eval.jsonl --metrics loss,perplexity

# Export
xaytune export merge --checkpoint output/lora-ckpt --output output/merged
xaytune export push --model output/merged --repo username/my-model

# Distributed training
xaytune launch --config configs/lora_finetune.yaml --nproc-per-node 4

# Training Studio (experimental)
xaytune studio --port 7860
```

### Config file

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

## Recipes

| Recipe | Methods | Use case |
|--------|---------|----------|
| `finetune` | `full`, `lora`, `qlora` | Supervised fine-tuning on instruction data |
| `pretrain` | `full` | Pre-training or continued pre-training on raw text |
| `align` | `dpo`, `grpo` | Alignment with human preferences (recommended) |
| `align` | `orpo`, `simpo`, `reinforce`, `ppo` | Alignment (experimental — see [Maturity](#maturity)) |

## Maturity

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
| PPO | **Experimental** | Simplified clipped PG — not full PPO (no rollout buffer, GAE, value model) |
| DeepSpeed | **Experimental** | ZeRO via `ds.initialize`, engine-aware loop |
| FSDP | **Experimental** | Sharding, offload, mixed precision wrapping |
| GGUF export | **Experimental** | Requires llama.cpp tools installed separately |
| Model merging | **Experimental** | TIES, DARE, SLERP, linear interpolation |
| Agent fine-tuning | **Experimental** | Tool-use formats with per-message masking |
| Training Studio | **Experimental** | Gradio UI for job configuration and launch |
| Data preparation | **Experimental** | Generate, filter, deduplicate, convert |
| Online RL | **Experimental** | Generate→score→train for RL methods |

## Extensibility

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

## Architecture

```
+-----------------------------------------+
|        CLI / Pipeline / Config          |  Layer 3 - Interface
+-----------------------------------------+
|   pretrain | finetune | align (recipes) |  Layer 2 - Recipes
+--------+--------+---------+--------+----+
| models |  data  | trainer |  eval  | exp|  Layer 1 - Building Blocks
+--------+--------+---------+--------+----+
         PyTorch / HuggingFace / DeepSpeed
```

## License

Apache 2.0
