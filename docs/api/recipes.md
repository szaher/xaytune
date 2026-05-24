# Recipes

trainlib's three top-level recipe functions are the primary entry points for training. Each recipe calls `setup_training()` internally, builds a `Trainer`, and runs the loop.

```python
import trainlib

state = trainlib.finetune("config.yaml")
state = trainlib.pretrain("config.yaml")
state = trainlib.align("config.yaml")
```

---

## finetune

::: trainlib.recipes.finetune.finetune

## pretrain

::: trainlib.recipes.pretrain.pretrain

## align

::: trainlib.recipes.align.align

## setup_training

::: trainlib.recipes.base.setup_training

## TrainingComponents

::: trainlib.recipes.base.TrainingComponents
