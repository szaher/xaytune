# API Reference

This section documents xaytune's Python API. The library exposes four top-level functions and a supporting ecosystem of configs, callbacks, evaluation, and export utilities.

## Top-Level Functions

These are the primary entry points, importable directly from `xaytune`:

```python
import xaytune

xaytune.finetune(...)   # Fine-tune a model (full, LoRA, QLoRA)
xaytune.pretrain(...)   # Pre-train a model from scratch
xaytune.align(...)      # Align a model with human preferences
xaytune.evaluate(...)   # Evaluate a model on metrics
```

## Module Overview

| Module | Description |
|--------|-------------|
| `xaytune.config.schema` | Pydantic config models (`TrainConfig`, `ModelConfig`, etc.) |
| `xaytune.config.parser` | YAML config loading and override parsing |
| `xaytune.config.validation` | Config validation rules |
| `xaytune.trainer.callbacks` | `CallbackManager`, `TrainState`, event system |
| `xaytune.eval.evaluate` | `evaluate()` function for custom datasets |
| `xaytune.eval.benchmarks` | `benchmark_evaluate()` for lm-eval benchmarks |
| `xaytune.eval.metrics` | Metric registry and built-in metrics |
| `xaytune.export.merge` | `merge()` and `save()` for model persistence |
| `xaytune.export.hub` | `push_to_hub()` for Hugging Face Hub |
| `xaytune.export.gguf` | `to_gguf()` for GGUF conversion |
| `xaytune.data.formats` | Built-in data format functions |
| `xaytune.data.registry` | `format_registry` |
| `xaytune.models.registry` | `model_registry` |
| `xaytune.recipes.align.rewards` | `reward_registry` and reward functions |
| `xaytune.utils.registry` | Generic `Registry` class |

## Registries

xaytune uses a registry pattern to make components extensible. Each registry maps string names to callable objects:

| Registry | Location | Decorator | Purpose |
|----------|----------|-----------|---------|
| `format_registry` | `xaytune.data.registry` | `@format_registry.register("name")` | Data format functions |
| `metric_registry` | `xaytune.eval.metrics` | `@register_metric("name")` | Evaluation metrics |
| `reward_registry` | `xaytune.recipes.align.rewards` | `@register_reward("name")` | Reward functions for alignment |
| `model_registry` | `xaytune.models.registry` | `@model_registry.register("name")` | Model loaders |
| `recipe_registry` | `xaytune.recipes` | `@recipe_registry.register("name")` | Training recipes |

### Registry API

All registries share the same interface:

```python
from xaytune.utils.registry import Registry

registry = Registry("my_registry")

# Register an item
@registry.register("key")
def my_function():
    ...

# Retrieve an item
fn = registry.get("key")

# Check existence
registry.has("key")  # True

# List all registered keys
registry.list()  # ['key']
```

## Detailed References

- [Config](config.md) -- all Pydantic config models and their fields
- [Callbacks](callbacks.md) -- event system and `TrainState`
- [Evaluation](evaluation.md) -- `evaluate()`, `benchmark_evaluate()`, and metrics
- [Export](export.md) -- `merge()`, `save()`, `push_to_hub()`, `to_gguf()`
