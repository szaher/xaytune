# Fine-tuning

The `finetune` recipe adapts a pre-trained language model to a specific task or domain. It supports three methods: full fine-tuning, LoRA, and QLoRA.

## Methods

| Method | Description | VRAM | When to Use |
|--------|-------------|------|-------------|
| `full` | Updates all model parameters | High | Maximum quality, sufficient GPU memory |
| `lora` | Low-Rank Adaptation -- trains small adapter matrices | Medium | Good balance of quality and efficiency |
| `qlora` | Quantized LoRA -- 4-bit base model with LoRA adapters | Low | Large models on limited hardware |

## Python API

```python
import xaytune

# Full fine-tuning
state = xaytune.finetune(
    model="meta-llama/Llama-3.1-8B",
    dataset="data/train.jsonl",
    method="full",
    format="alpaca",
    num_epochs=3,
    learning_rate=2e-4,
    batch_size=4,
)

# LoRA
state = xaytune.finetune(
    model="meta-llama/Llama-3.1-8B",
    dataset="data/train.jsonl",
    method="lora",
    format="alpaca",
)

# QLoRA (4-bit quantized base + LoRA adapters)
state = xaytune.finetune(
    model="meta-llama/Llama-3.1-8B",
    dataset="data/train.jsonl",
    method="qlora",
    format="alpaca",
)
```

### Function Signature

```python
def finetune(
    *,
    config: TrainConfig | None = None,
    model: str | None = None,
    dataset: str | None = None,
    method: str = "full",
    format: str = "alpaca",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    **kwargs,
) -> TrainState:
```

- **config** -- A full `TrainConfig` object. If provided, all other arguments are ignored.
- **model** -- Model name or path (Hugging Face Hub ID or local path).
- **dataset** -- Path to training data file.
- **method** -- Training method: `"full"`, `"lora"`, or `"qlora"`.
- **format** -- Data format: `"alpaca"`, `"sharegpt"`, `"chat"`, or `"text"`.
- **num_epochs** -- Number of training epochs (default: 3).
- **learning_rate** -- Learning rate (default: 2e-4).
- **batch_size** -- Per-device batch size (default: 4).
- **\*\*kwargs** -- Additional `TrainerConfig` fields (e.g., `gradient_accumulation`, `warmup_steps`).

## YAML Config Examples

### Full Fine-tuning

```yaml
recipe: finetune
method: full

model:
  name: meta-llama/Llama-3.1-8B

data:
  path: data/train.jsonl
  format: alpaca
  eval_split: 0.05

trainer:
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2e-4
  num_epochs: 3
  mixed_precision: bf16

logging:
  backends: [console, tensorboard]

output:
  dir: output/full-finetune
```

### LoRA Fine-tuning

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
  dropout: 0.05
  target_modules: auto

trainer:
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2e-4
  num_epochs: 3

output:
  dir: output/lora-finetune
```

### QLoRA Fine-tuning

```yaml
recipe: finetune
method: qlora

model:
  name: meta-llama/Llama-3.1-8B
  quantization: 4bit

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

output:
  dir: output/qlora-finetune
```

## Data Formats

xaytune has a registry of built-in data formats. Each format function transforms raw samples into the `{"text": "..."}` structure expected by the trainer.

| Format | Fields | Description |
|--------|--------|-------------|
| `alpaca` | `instruction`, `input` (optional), `output` | Stanford Alpaca format |
| `sharegpt` | `conversations[].from`, `conversations[].value` | ShareGPT multi-turn |
| `chat` | `messages[].role`, `messages[].content` | OpenAI-style chat format |
| `text` | `text` or `content` | Raw text for continued pre-training |

### Custom Formats

Register your own format with the `@format_registry.register()` decorator:

```python
from xaytune.data.registry import format_registry

@format_registry.register("my_format")
def format_my_data(sample):
    return {"text": f"Q: {sample['question']}\nA: {sample['answer']}"}
```

## LoRA Configuration

When using `lora` or `qlora` methods, the `lora` section in the config controls adapter parameters:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rank` | int | 16 | LoRA rank (r) |
| `alpha` | int | 32 | LoRA alpha scaling factor |
| `dropout` | float | 0.05 | Dropout probability for LoRA layers |
| `target_modules` | str or list | `"auto"` | Which modules to apply LoRA to |

!!! tip "After training with LoRA"
    Merge adapters back into the base model for inference:
    ```bash
    xaytune export merge --checkpoint output/lora-finetune --output output/merged
    ```
