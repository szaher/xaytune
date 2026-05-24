# CLI Reference

xaytune installs a `xaytune` command-line tool for running training, evaluation, export, and component listing from the terminal.

```bash
xaytune [command] [options]
```

## Global Options

| Flag | Description |
|------|-------------|
| `--version` | Print xaytune version and exit |
| `--help` | Show help message |

---

## train

Run a training recipe from a YAML config file.

```bash
xaytune train --config <path> [--override key=value ...] [--resume] [--dry-run]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--config` | Yes | Path to a YAML config file |
| `--override` | No | Config override in dot notation (repeatable) |
| `--resume` | No | Resume training from the last checkpoint |
| `--dry-run` | No | Validate and print the resolved config as JSON without training |

### Examples

```bash
# Basic training run
xaytune train --config configs/examples/lora_finetune.yaml

# Override model and learning rate
xaytune train --config configs/examples/lora_finetune.yaml \
    --override model.name=mistralai/Mistral-7B-v0.3 \
    --override trainer.learning_rate=1e-4

# Validate config without training
xaytune train --config configs/examples/lora_finetune.yaml --dry-run

# Resume from checkpoint
xaytune train --config configs/examples/lora_finetune.yaml --resume
```

---

## eval

Evaluate a model on benchmarks or a custom dataset.

```bash
xaytune eval --model <path> [--benchmarks <list>] [--dataset <path>] [--metrics <list>] [--num-fewshot <n>]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--model` | Yes | Model path or Hugging Face Hub name |
| `--benchmarks` | No | Comma-separated benchmark names (e.g., `mmlu,gsm8k`) |
| `--dataset` | No | Path to a JSONL evaluation dataset |
| `--metrics` | No | Comma-separated metric names (e.g., `loss,perplexity`) |
| `--num-fewshot` | No | Number of few-shot examples for benchmarks |

!!! note
    Provide either `--benchmarks` or `--dataset`, not both.

### Examples

```bash
# Benchmark evaluation
xaytune eval --model output/my-finetune --benchmarks mmlu,gsm8k --num-fewshot 5

# Custom dataset evaluation
xaytune eval --model output/my-finetune --dataset data/eval.jsonl --metrics loss,perplexity
```

---

## export

Export and convert models. Has three subcommands: `merge`, `gguf`, and `push`.

### export merge

Merge LoRA adapters into the base model.

```bash
xaytune export merge --checkpoint <path> --output <path>
```

| Option | Required | Description |
|--------|----------|-------------|
| `--checkpoint` | Yes | Path to a LoRA/QLoRA checkpoint directory |
| `--output` | Yes | Output directory for the merged model |

### export gguf

Convert a model to GGUF format.

```bash
xaytune export gguf --model <path> --output <path> [--quant <type>]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--model` | Yes | -- | Path to model directory |
| `--output` | Yes | -- | Output GGUF file path |
| `--quant` | No | `Q4_K_M` | Quantization type |

### export push

Push a model to the Hugging Face Hub.

```bash
xaytune export push --model <path> --repo <repo>
```

| Option | Required | Description |
|--------|----------|-------------|
| `--model` | Yes | Path to model directory |
| `--repo` | Yes | HF Hub repo name (e.g., `username/model-name`) |

### Examples

```bash
# Full export pipeline
xaytune export merge --checkpoint output/lora-finetune --output output/merged
xaytune export gguf --model output/merged --output model.gguf --quant Q5_K_M
xaytune export push --model output/merged --repo username/my-model
```

---

## compare

Compare two models side-by-side on the same benchmarks.

```bash
xaytune compare <model_a> <model_b> --benchmarks <list> [--num-fewshot <n>]
```

| Option | Required | Description |
|--------|----------|-------------|
| (positional) | Yes | Exactly two model paths |
| `--benchmarks` | Yes | Comma-separated benchmark names |
| `--num-fewshot` | No | Number of few-shot examples |

### Example

```bash
xaytune compare output/model-a output/model-b --benchmarks mmlu,gsm8k,hellaswag
```

Output is a table showing scores for each model across all benchmark metrics.

---

## list

List registered components (recipes, data formats, metrics, reward functions).

```bash
xaytune list [type]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `type` | No | Component type: `recipes`, `formats`, `metrics`, `rewards` |

If no type is given, all registries are listed.

### Examples

```bash
# List everything
xaytune list

# List only data formats
xaytune list formats

# List only metrics
xaytune list metrics
```

---

## Example Config Files

xaytune ships example configs in `configs/examples/`:

| File | Description |
|------|-------------|
| `full_finetune.yaml` | Full parameter fine-tuning |
| `lora_finetune.yaml` | LoRA fine-tuning |
| `qlora_finetune.yaml` | QLoRA fine-tuning |
| `pretrain.yaml` | Pre-training |
| `dpo_align.yaml` | DPO alignment |
| `grpo_align.yaml` | GRPO alignment |
| `orpo_align.yaml` | ORPO alignment |
| `simpo_align.yaml` | SimPO alignment |
| `ppo_align.yaml` | PPO alignment |
| `reinforce_align.yaml` | REINFORCE alignment |
