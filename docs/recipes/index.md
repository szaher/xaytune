# Recipes

xaytune organizes training workflows into **recipes** -- high-level functions that encapsulate the full training pipeline for a specific task. Each recipe handles model loading, data preparation, training loop execution, and state management.

## Available Recipes

| Recipe | Function | Methods | Use Case |
|--------|----------|---------|----------|
| **Fine-tune** | `xaytune.finetune()` | `full`, `lora`, `qlora` | Adapt a pre-trained model to a specific task or domain |
| **Pre-train** | `xaytune.pretrain()` | `full` | Train a model from scratch on a large text corpus |
| **Align** | `xaytune.align()` | `dpo`, `grpo`, `orpo`, `simpo`, `ppo`, `reinforce` | Align a model with human preferences |

## How Recipes Work

Every recipe follows the same pattern:

1. **Build or accept a config** -- either from keyword arguments or a `TrainConfig` object
2. **Set up components** -- model, tokenizer, data loaders, trainer, callbacks, logging
3. **Run the training loop** -- delegate to the trainer
4. **Return a `TrainState`** -- final metrics, step count, and training status

```python
import xaytune

# All recipes share the same calling convention:
state = xaytune.finetune(model="...", dataset="...", method="lora")
state = xaytune.pretrain(model="...", dataset="...", format="text")
state = xaytune.align(model="...", dataset="...", method="dpo")
```

## Python API vs. Config

Each recipe can be called in two ways:

=== "Keyword arguments"

    ```python
    state = xaytune.finetune(
        model="meta-llama/Llama-3.1-8B",
        dataset="data/train.jsonl",
        method="lora",
        format="alpaca",
        num_epochs=3,
    )
    ```

=== "TrainConfig object"

    ```python
    from xaytune.config.schema import TrainConfig, ModelConfig, DataConfig

    config = TrainConfig(
        recipe="finetune",
        method="lora",
        model=ModelConfig(name="meta-llama/Llama-3.1-8B"),
        data=DataConfig(path="data/train.jsonl", format="alpaca"),
    )
    state = xaytune.finetune(config=config)
    ```

=== "YAML + CLI"

    ```yaml
    recipe: finetune
    method: lora
    model:
      name: meta-llama/Llama-3.1-8B
    data:
      path: data/train.jsonl
      format: alpaca
    ```

    ```bash
    xaytune train --config config.yaml
    ```

## Recipe Registry

Recipes are registered in a global `recipe_registry`. You can list them:

```python
from xaytune.recipes import recipe_registry

print(recipe_registry.list())
# ['align', 'finetune', 'pretrain']
```

Or from the CLI:

```bash
xaytune list recipes
```

## Next Steps

- [Fine-tuning](finetuning.md) -- full, LoRA, and QLoRA fine-tuning
- [Pre-training](pretraining.md) -- training from scratch
- [Alignment](alignment.md) -- DPO, GRPO, and other alignment methods
