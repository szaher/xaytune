# Examples

xaytune includes Jupyter notebooks and YAML config files to help you get started quickly.

## Jupyter Notebooks

The `examples/` directory contains step-by-step notebooks:

| Notebook | Description |
|----------|-------------|
| [01_quickstart.ipynb](https://github.com/szaher/xaytune/blob/main/examples/01_quickstart.ipynb) | Your first training run -- LoRA fine-tuning in a few lines |
| [02_finetuning.ipynb](https://github.com/szaher/xaytune/blob/main/examples/02_finetuning.ipynb) | Full, LoRA, and QLoRA fine-tuning compared |
| [03_pretraining.ipynb](https://github.com/szaher/xaytune/blob/main/examples/03_pretraining.ipynb) | Pre-training on a text corpus |
| [04_alignment.ipynb](https://github.com/szaher/xaytune/blob/main/examples/04_alignment.ipynb) | Alignment with DPO, GRPO, and other methods |
| [05_evaluation.ipynb](https://github.com/szaher/xaytune/blob/main/examples/05_evaluation.ipynb) | Evaluating models with metrics and benchmarks |
| [06_advanced.ipynb](https://github.com/szaher/xaytune/blob/main/examples/06_advanced.ipynb) | Custom formats, metrics, rewards, callbacks, and distributed training |

## Example Configs

The `configs/examples/` directory contains ready-to-use YAML config files for every recipe and method:

### Fine-tuning

```bash
# Full fine-tuning
xaytune train --config configs/examples/full_finetune.yaml

# LoRA fine-tuning
xaytune train --config configs/examples/lora_finetune.yaml

# QLoRA fine-tuning
xaytune train --config configs/examples/qlora_finetune.yaml
```

### Pre-training

```bash
xaytune train --config configs/examples/pretrain.yaml
```

### Alignment

```bash
# DPO
xaytune train --config configs/examples/dpo_align.yaml

# GRPO
xaytune train --config configs/examples/grpo_align.yaml

# ORPO
xaytune train --config configs/examples/orpo_align.yaml

# SimPO
xaytune train --config configs/examples/simpo_align.yaml

# PPO
xaytune train --config configs/examples/ppo_align.yaml

# REINFORCE
xaytune train --config configs/examples/reinforce_align.yaml
```

## Quick Recipes

### Fine-tune Llama 3.1 with LoRA

```python
import xaytune

state = xaytune.finetune(
    model="meta-llama/Llama-3.1-8B",
    dataset="data/train.jsonl",
    method="lora",
    format="alpaca",
    num_epochs=3,
    learning_rate=2e-4,
)
```

### Align with DPO

```python
import xaytune

state = xaytune.align(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dataset="data/preferences.jsonl",
    method="dpo",
    format="preference",
)
```

### Evaluate and Compare

```python
from xaytune.eval.benchmarks import benchmark_evaluate

results = benchmark_evaluate(
    model="output/my-model",
    benchmarks=["mmlu", "gsm8k"],
    num_fewshot=5,
)
```

### Export Pipeline

```python
from xaytune.export.merge import merge
from xaytune.export.hub import push_to_hub
from xaytune.export.gguf import to_gguf

# Merge LoRA adapters
merge("output/lora-finetune", save_to="output/merged")

# Push to Hub
push_to_hub("output/merged", repo="username/my-model")

# Convert to GGUF
to_gguf("output/merged", output="model.gguf", quantization="Q4_K_M")
```
