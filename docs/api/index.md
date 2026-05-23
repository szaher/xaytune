# API Reference

This section documents trainlib's Python API. The library exposes four top-level functions and a supporting ecosystem of configs, callbacks, evaluation, and export utilities.

## Top-Level Functions

These are the primary entry points, importable directly from `trainlib`:

```python
import trainlib

trainlib.finetune(...)   # Fine-tune a model (full, LoRA, QLoRA)
trainlib.pretrain(...)   # Pre-train a model from scratch
trainlib.align(...)      # Align a model with human preferences
trainlib.evaluate(...)   # Evaluate a model on metrics
```

## Module Overview

| Module | Description |
|--------|-------------|
| `trainlib.config.schema` | Pydantic config models (`TrainConfig`, `ModelConfig`, etc.) |
| `trainlib.config.parser` | YAML config loading and override parsing |
| `trainlib.config.validation` | Config validation rules |
| `trainlib.trainer.callbacks` | `CallbackManager`, `TrainState`, event system |
| `trainlib.eval.evaluate` | `evaluate()` function for custom datasets |
| `trainlib.eval.benchmarks` | `benchmark_evaluate()` for lm-eval benchmarks |
| `trainlib.eval.metrics` | Metric registry and built-in metrics |
| `trainlib.export.merge` | `merge()` and `save()` for model persistence |
| `trainlib.export.hub` | `push_to_hub()` for Hugging Face Hub |
| `trainlib.export.gguf` | `to_gguf()` for GGUF conversion |
| `trainlib.data.formats` | Built-in data format functions |
| `trainlib.data.registry` | `format_registry` |
| `trainlib.models.registry` | `model_registry` |
| `trainlib.recipes.align.rewards` | `reward_registry` and reward functions |
| `trainlib.utils.registry` | Generic `Registry` class |

## Registries

trainlib uses a registry pattern to make components extensible. Each registry maps string names to callable objects:

| Registry | Location | Decorator | Purpose |
|----------|----------|-----------|---------|
| `format_registry` | `trainlib.data.registry` | `@format_registry.register("name")` | Data format functions |
| `metric_registry` | `trainlib.eval.metrics` | `@register_metric("name")` | Evaluation metrics |
| `reward_registry` | `trainlib.recipes.align.rewards` | `@register_reward("name")` | Reward functions for alignment |
| `model_registry` | `trainlib.models.registry` | `@model_registry.register("name")` | Model loaders |
| `recipe_registry` | `trainlib.recipes` | `@recipe_registry.register("name")` | Training recipes |

### Registry API

All registries share the same interface:

```python
from trainlib.utils.registry import Registry

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
