# Examples

## Control plane

[`examples/control_plane/`](https://github.com/szaher/xaytune/tree/main/examples/control_plane)
uses the experiment control plane, first released in `1.0.0a1`. See
[Control-plane getting started](control-plane/getting-started.md) for a walkthrough.

| Script | What it shows | Needs |
|--------|---------------|-------|
| [01_compile_a_candidate.py](https://github.com/szaher/xaytune/blob/main/examples/control_plane/01_compile_a_candidate.py) | Describe a candidate, see it refused with reasons, compile it into a serializable plan | Nothing: no model, no GPU |
| [02_train.py](https://github.com/szaher/xaytune/blob/main/examples/control_plane/02_train.py) | Submit, follow events, wait, read the result (`--compiler native` or `trl`) | A local model and JSONL dataset |
| [03_cancel.py](https://github.com/szaher/xaytune/blob/main/examples/control_plane/03_cancel.py) | Cancel a running experiment | As above |
| [04_restart_and_attach.py](https://github.com/szaher/xaytune/blob/main/examples/control_plane/04_restart_and_attach.py) | Submit, end the process, adopt the running workload from another | As above |
| [05_train_and_evaluate.py](https://github.com/szaher/xaytune/blob/main/examples/control_plane/05_train_and_evaluate.py) | Train, evaluate with the built-in `native` evaluator, and decide against a loss `--target`: the whole loop | As above, plus a held-out JSONL file |

## Legacy trainer API

The notebooks and configs below use the **legacy trainer API**, the
`finetune()` / `align()` / CLI library in the `0.6.0` package, still included
in `1.0.0a1`. They remain supported.

### Jupyter Notebooks

The `examples/` directory contains step-by-step notebooks:

| Notebook | Description |
|----------|-------------|
| [01_quickstart.ipynb](https://github.com/szaher/xaytune/blob/main/examples/01_quickstart.ipynb) | Your first training run -- LoRA fine-tuning in a few lines |
| [02_finetuning.ipynb](https://github.com/szaher/xaytune/blob/main/examples/02_finetuning.ipynb) | Full, LoRA, and QLoRA fine-tuning compared |
| [03_pretraining.ipynb](https://github.com/szaher/xaytune/blob/main/examples/03_pretraining.ipynb) | Pre-training on a text corpus |
| [04_alignment.ipynb](https://github.com/szaher/xaytune/blob/main/examples/04_alignment.ipynb) | Alignment with DPO, GRPO, and other methods |
| [05_evaluation.ipynb](https://github.com/szaher/xaytune/blob/main/examples/05_evaluation.ipynb) | Evaluating models with metrics and benchmarks |
| [06_advanced.ipynb](https://github.com/szaher/xaytune/blob/main/examples/06_advanced.ipynb) | Custom formats, metrics, rewards, callbacks, and distributed training |
| [07_gpu_training.ipynb](https://github.com/szaher/xaytune/blob/main/examples/07_gpu_training.ipynb) | GPU training with single-node and Ray distributed examples |
| [08_data_preparation.ipynb](https://github.com/szaher/xaytune/blob/main/examples/08_data_preparation.ipynb) | Dataset preparation -- dedup, filtering, conversion, synthetic generation |
| [09_callbacks.ipynb](https://github.com/szaher/xaytune/blob/main/examples/09_callbacks.ipynb) | Training callbacks -- loss tracking, timing, early stopping |
| [10_model_merging.ipynb](https://github.com/szaher/xaytune/blob/main/examples/10_model_merging.ipynb) | Model merging with Linear, SLERP, TIES, and DARE algorithms |
| [11_agent_finetuning.ipynb](https://github.com/szaher/xaytune/blob/main/examples/11_agent_finetuning.ipynb) | Agent fine-tuning -- data formats, loss masking, rewards, evaluation, multi-agent |

### Example Configs

The `configs/examples/` directory contains ready-to-use YAML config files for every recipe and method:

#### Fine-tuning

```bash
# Full fine-tuning
xaytune train --config configs/examples/full_finetune.yaml

# LoRA fine-tuning
xaytune train --config configs/examples/lora_finetune.yaml

# QLoRA fine-tuning
xaytune train --config configs/examples/qlora_finetune.yaml
```

#### Pre-training

```bash
xaytune train --config configs/examples/pretrain.yaml
```

#### Alignment

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

### Quick Recipes

#### Fine-tune Llama 3.1 with LoRA

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

#### Align with DPO

```python
import xaytune

state = xaytune.align(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dataset="data/preferences.jsonl",
    method="dpo",
    format="preference",
)
```

#### Evaluate and Compare

```python
from xaytune.eval.benchmarks import benchmark_evaluate

results = benchmark_evaluate(
    model="output/my-model",
    benchmarks=["mmlu", "gsm8k"],
    num_fewshot=5,
)
```

#### Agent Fine-Tuning

```python
import xaytune

# Fine-tune on tool-use conversations with loss masking
state = xaytune.finetune(
    model="meta-llama/Llama-3.1-8B",
    dataset="data/agent_traces.jsonl",
    method="lora",
    format="function_calling",  # or "react", "trajectory", "multi_agent"
    num_epochs=3,
)
```

#### Agent Evaluation

```python
from xaytune.eval.agent_metrics import evaluate_agent

results = evaluate_agent(
    responses=[{"prompt": "...", "response": "..."}],
    expected_tools=["search"],
    success_markers=["Done"],
)
```

#### Export Pipeline

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
