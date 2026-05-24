# Export

trainlib provides utilities for saving, merging, and converting models after training.

## merge()

Merge LoRA/QLoRA adapters back into the base model, producing a standalone model that can be used without PEFT.

```python
from trainlib.export.merge import merge

merge("output/lora-finetune", save_to="output/merged")
```

### Function Signature

```python
def merge(checkpoint_path: str, *, save_to: str) -> None:
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_path` | `str` | Path to a LoRA/QLoRA checkpoint directory |
| `save_to` | `str` | Output directory for the merged model |

The merged model and tokenizer are saved in Hugging Face format, ready for inference or further export.

!!! warning
    `merge()` only works with PEFT (LoRA/QLoRA) checkpoints. It raises `ValueError` if the checkpoint is not a PEFT model.

---

## save()

Save a model and tokenizer to disk with optional metadata.

```python
from trainlib.export.merge import save

save(
    model,
    tokenizer,
    output_dir="output/my-model",
    metadata={"recipe": "finetune", "method": "lora", "epochs": 3},
)
```

### Function Signature

```python
def save(
    model: Any,
    tokenizer: Any,
    *,
    output_dir: str,
    metadata: dict[str, Any] | None = None,
) -> None:
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | model object | The model to save |
| `tokenizer` | tokenizer object | The tokenizer to save |
| `output_dir` | `str` | Output directory |
| `metadata` | `dict` \| `None` | Optional metadata written to `trainlib_metadata.json` |

---

## push_to_hub()

Push a model and tokenizer to the Hugging Face Hub.

```python
from trainlib.export.hub import push_to_hub

# From a saved directory
push_to_hub("output/merged", repo="username/my-model")

# From model objects
push_to_hub(model, repo="username/my-model", tokenizer=tokenizer)
```

### Function Signature

```python
def push_to_hub(
    model_or_path: Any,
    *,
    repo: str | None = None,
    tokenizer: Any | None = None,
) -> None:
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_or_path` | model object or `str` | A model instance or path to a saved model |
| `repo` | `str` | HF Hub repository (e.g., `"username/model-name"`) |
| `tokenizer` | tokenizer object \| `None` | Tokenizer to push alongside the model (auto-loaded if `model_or_path` is a string) |

!!! note
    You must be authenticated with the Hugging Face Hub. Run `huggingface-cli login` first.

---

## to_gguf()

Convert a model to GGUF format for use with llama.cpp and compatible inference engines.

```python
from trainlib.export.gguf import to_gguf

to_gguf(
    "output/merged",
    output="model.gguf",
    quantization="Q4_K_M",
)
```

### Function Signature

```python
def to_gguf(
    model_path: str,
    *,
    output: str,
    quantization: str = "Q4_K_M",
) -> None:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | *required* | Path to the model directory |
| `output` | `str` | *required* | Output file path for the GGUF file |
| `quantization` | `str` | `"Q4_K_M"` | GGUF quantization type |

Common quantization types: `Q4_0`, `Q4_K_M`, `Q5_K_M`, `Q8_0`, `F16`.

---

## CLI Usage

All export operations are available through the `trainlib export` subcommand:

```bash
# Merge LoRA adapters
trainlib export merge --checkpoint output/lora-finetune --output output/merged

# Convert to GGUF
trainlib export gguf --model output/merged --output model.gguf --quant Q4_K_M

# Push to Hugging Face Hub
trainlib export push --model output/merged --repo username/my-model
```

---

## Typical Export Pipeline

A common post-training workflow:

```bash
# 1. Train with LoRA
trainlib train --config configs/examples/lora_finetune.yaml

# 2. Merge adapters into base model
trainlib export merge --checkpoint output/lora-finetune --output output/merged

# 3a. Push to Hub for cloud inference
trainlib export push --model output/merged --repo username/my-model

# 3b. Or convert to GGUF for local inference
trainlib export gguf --model output/merged --output model.gguf
```

---

## Full API Reference

::: trainlib.export.merge.merge

::: trainlib.export.merge.save

::: trainlib.export.hub.push_to_hub

::: trainlib.export.gguf.to_gguf
