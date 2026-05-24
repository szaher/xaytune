# Getting Started

## Installation

Install xaytune from source:

```bash
pip install -e .
```

### Optional Dependencies

xaytune ships optional extras for evaluation, logging, and distributed training:

```bash
# Benchmark evaluation (lm-eval)
pip install -e ".[eval]"

# Logging backends
pip install -e ".[wandb]"
pip install -e ".[mlflow]"
pip install -e ".[tensorboard]"

# DeepSpeed distributed training
pip install -e ".[deepspeed]"

# Everything
pip install -e ".[all]"

# Development (pytest, ruff, mypy)
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.40+

---

## Quickstart: Fine-tune with LoRA

### Python API

The fastest way to start training is with the top-level `xaytune.finetune()` function:

```python
import xaytune

state = xaytune.finetune(
    model="meta-llama/Llama-3.1-8B",
    dataset="data/train.jsonl",
    method="lora",
    format="alpaca",
    num_epochs=3,
    learning_rate=2e-4,
    batch_size=4,
)
```

This loads the model, applies LoRA adapters, formats the data, runs training, and returns a `TrainState` object with the final metrics.

### YAML Config + CLI

For reproducible experiments, write a YAML config file:

```yaml
# my_config.yaml
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
  dropout: 0.05

trainer:
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2e-4
  num_epochs: 3

logging:
  backends: [console, tensorboard]
  project: my-finetune

output:
  dir: output/my-finetune
```

Then run:

```bash
xaytune train --config my_config.yaml
```

!!! tip "Dry run"
    Use `--dry-run` to validate your config without starting training:
    ```bash
    xaytune train --config my_config.yaml --dry-run
    ```

### Config Overrides

Override any config value from the command line using dot notation:

```bash
xaytune train --config my_config.yaml \
    --override model.name=mistralai/Mistral-7B-v0.3 \
    --override trainer.learning_rate=1e-4
```

---

## Quickstart: Evaluate

Evaluate a model against standard benchmarks:

```bash
xaytune eval --model output/my-finetune --benchmarks mmlu,gsm8k
```

Or evaluate on a custom dataset:

```bash
xaytune eval --model output/my-finetune --dataset data/eval.jsonl --metrics loss,perplexity
```

---

## Quickstart: Export

Merge LoRA adapters back into the base model:

```bash
xaytune export merge --checkpoint output/my-finetune --output output/merged
```

Push to Hugging Face Hub:

```bash
xaytune export push --model output/merged --repo username/my-model
```

Convert to GGUF for local inference with llama.cpp:

```bash
xaytune export gguf --model output/merged --output model.gguf --quant Q4_K_M
```

---

## Next Steps

- [Recipes](recipes/index.md) -- learn about each training recipe in detail
- [Config Reference](api/config.md) -- all configuration options
- [CLI Reference](cli.md) -- complete CLI documentation
