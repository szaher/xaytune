# Recipes

xaytune's three top-level recipe functions are the primary entry points for training. Each recipe calls `setup_training()` internally, builds a `Trainer`, and runs the loop.

```python
import xaytune

state = xaytune.finetune("config.yaml")
state = xaytune.pretrain("config.yaml")
state = xaytune.align("config.yaml")
```

---

## finetune

::: xaytune.recipes.finetune.finetune

## pretrain

::: xaytune.recipes.pretrain.pretrain

## align

::: xaytune.recipes.align.align

## setup_training

::: xaytune.recipes.base.setup_training

## TrainingComponents

::: xaytune.recipes.base.TrainingComponents
