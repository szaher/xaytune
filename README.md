# trainlib

An opinionated LLM training and fine-tuning library built on PyTorch.

trainlib provides a recipe-based architecture with a layered API: simple one-liners for beginners, full control for experts. Config files and Python API are equal citizens.

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
    dataset="data/corpus/",
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
# Train with config
trainlib train --config configs/examples/lora_finetune.yaml

# Train with overrides
trainlib train --config configs/examples/lora_finetune.yaml \
    --override model.name=mistralai/Mistral-7B-v0.3

# Dry run (validate and print config)
trainlib train --config configs/examples/lora_finetune.yaml --dry-run

# List registered components
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

lora:
  rank: 16
  alpha: 32

trainer:
  batch_size: 4
  learning_rate: 2e-4
  num_epochs: 3

logging:
  backends: [console, tensorboard]
```

See `configs/examples/` for more examples.

## Recipes

| Recipe | Methods | Use case |
|--------|---------|----------|
| `finetune` | `full`, `lora`, `qlora` | Supervised fine-tuning on instruction data |
| `pretrain` | `full` | Pre-training or continued pre-training on raw text |
| `align` | `dpo`, `grpo`, `ppo`, `orpo`, `simpo` | Alignment with human preferences |

## Extensibility

Register custom components with decorators:

```python
from trainlib.models import register_model
from trainlib.data import register_format
from trainlib.eval import register_metric
from trainlib.recipes.align import register_reward
from trainlib.trainer import on

@register_format("my-format")
def parse_my_data(sample):
    return {"instruction": sample["q"], "response": sample["a"]}

@register_metric("domain-accuracy")
def domain_accuracy(predictions, references):
    return sum(p == r for p, r in zip(predictions, references)) / len(predictions)

@on("step_end")
def log_memory(state):
    print(f"Step {state.global_step}: loss={state.metrics.get('loss', 'N/A')}")
```

## Export

```python
from trainlib import export

# Merge LoRA adapters into base model
export.merge("output/lora-checkpoint", save_to="output/merged-model")

# Save with metadata
export.save(model, tokenizer, output_dir="output/final", metadata={"recipe": "finetune"})

# Push to Hugging Face Hub
export.push_to_hub("output/merged-model", repo="username/my-model")
```

## Architecture

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

## License

Apache 2.0
