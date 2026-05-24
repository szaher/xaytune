# trainlib

An opinionated LLM training and fine-tuning library built on PyTorch. Recipe-based architecture with a layered API: simple one-liners for beginners, full control for experts.

**[Documentation](https://szaher.github.io/trainlib/)** | **[Examples](https://szaher.github.io/trainlib/examples/)** | **[API Reference](https://szaher.github.io/trainlib/api/)**

## Features

- **3 recipes** — fine-tune (full/LoRA/QLoRA), pre-train, align (DPO, GRPO, PPO, ORPO, SimPO, REINFORCE)
- **4 data formats** — Alpaca, ShareGPT, chat template, raw text, plus preference pairs
- **Automatic tokenization** — text data is tokenized and collated automatically; pre-tokenized data passes through unchanged
- **Sequence packing** — pack short sequences to maximize GPU utilization
- **Distributed training** — DDP, FSDP, DeepSpeed via `trainlib launch`
- **4 logging backends** — console, TensorBoard, W&B, MLflow
- **LR finder** — automatic learning rate range test
- **Callbacks** — event-driven hooks for early stopping, checkpointing, progress, custom logic
- **Evaluation** — built-in metrics + lm-eval-harness benchmarks
- **Export** — merge LoRA adapters, GGUF conversion, push to HuggingFace Hub
- **Training Studio** — Gradio web UI for configuring and launching runs
- **8 CLI commands** — train, eval, export, compare, lr-find, list, studio, launch
- **Fully typed** — Pydantic configs, py.typed, mypy-clean

## Install

```bash
pip install trainlib
```

Optional extras:

```bash
pip install trainlib[wandb]       # Weights & Biases logging
pip install trainlib[mlflow]      # MLflow logging
pip install trainlib[deepspeed]   # DeepSpeed distributed training
pip install trainlib[eval]        # lm-eval-harness benchmarks
pip install trainlib[studio]      # Training Studio web UI
pip install trainlib[docs]        # MkDocs documentation site
pip install trainlib[all]         # Everything
```

## Quickstart

### Python API

```python
import trainlib

# LoRA fine-tuning
trainlib.finetune(
    model="meta-llama/Llama-3.1-8B",
    dataset="data/train.jsonl",
    method="lora",
    format="alpaca",
    num_epochs=3,
)

# Pre-training
trainlib.pretrain(
    model="meta-llama/Llama-3.1-8B",
    dataset="data/corpus.jsonl",
    format="text",
)

# DPO alignment
trainlib.align(
    model="output/sft-model",
    dataset="data/preferences.jsonl",
    method="dpo",
    format="preference",
)

# Evaluation
results = trainlib.evaluate(
    model="output/my-model",
    dataset=[{"input_ids": [1, 2], "labels": [1, 2]}],
    metrics=["loss", "perplexity"],
)
```

### CLI

```bash
# Train
trainlib train --config configs/lora_finetune.yaml
trainlib train --config configs/lora_finetune.yaml --override model.name=mistralai/Mistral-7B-v0.3
trainlib train --config configs/lora_finetune.yaml --dry-run

# Evaluate
trainlib eval --model output/my-model --benchmarks mmlu,gsm8k --num-fewshot 5
trainlib eval --model output/my-model --dataset data/eval.jsonl --metrics loss,perplexity

# Compare models
trainlib compare model-a/ model-b/ --benchmarks mmlu,gsm8k

# Export
trainlib export merge --checkpoint output/lora-ckpt --output output/merged
trainlib export gguf --model output/merged --output model.gguf --quant Q4_K_M
trainlib export push --model output/merged --repo username/my-model

# LR finder
trainlib lr-find --config configs/lora_finetune.yaml

# Distributed training
trainlib launch --config configs/lora_finetune.yaml --nproc-per-node 4

# Training Studio
trainlib studio --port 7860

# List components
trainlib list recipes
trainlib list formats
trainlib list metrics
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
| `align` | `dpo`, `grpo`, `ppo`, `orpo`, `simpo`, `reinforce` | Alignment with human preferences |

## Extensibility

Register custom components with decorators:

```python
from trainlib.data import register_format
from trainlib.eval import register_metric
from trainlib.recipes.align import register_reward
from trainlib.trainer import on

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

## Export

```python
from trainlib import export

# Merge LoRA adapters into base model
export.merge("output/lora-checkpoint", save_to="output/merged")

# Save with metadata
export.save(model, tokenizer, output_dir="output/final", metadata={"recipe": "finetune"})

# Push to Hugging Face Hub
export.push_to_hub("output/merged", repo="username/my-model")

# Convert to GGUF for local inference
from trainlib.export import to_gguf
to_gguf("output/merged", output="model.gguf", quantization="Q4_K_M")
```

## Architecture

```
+-----------------------------------------+
|           CLI / Config Engine           |  Layer 3 - Interface
+-----------------------------------------+
|   pretrain | finetune | align (recipes) |  Layer 2 - Recipes
+--------+--------+---------+--------+----+
| models |  data  | trainer |  eval  | exp|  Layer 1 - Building Blocks
+--------+--------+---------+--------+----+
         PyTorch / HuggingFace / DeepSpeed
```

## License

Apache 2.0
