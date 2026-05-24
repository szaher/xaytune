# Extensibility

trainlib is designed to be extended. You can bring your own models, training logic, recipes, and data formats without modifying the library itself.

## Custom Training Step

Subclass `Trainer` and override `training_step()` to implement custom forward/backward logic while keeping the epoch loop, callbacks, checkpointing, and scheduling:

```python
import torch
from trainlib.trainer import Trainer
from trainlib.config.schema import TrainerConfig

class DistillTrainer(Trainer):
    def __init__(self, config, teacher_model, **kwargs):
        super().__init__(config, **kwargs)
        self.teacher = teacher_model

    def training_step(self, model, batch, optimizer, state):
        # Custom forward pass
        student_out = model(**batch)
        with torch.no_grad():
            teacher_out = self.teacher(**batch)

        # Custom loss
        loss = torch.nn.functional.mse_loss(
            student_out.logits, teacher_out.logits
        )

        # Backward + optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

# Use it
config = TrainerConfig(num_epochs=3, learning_rate=1e-4)
trainer = DistillTrainer(config=config, teacher_model=teacher)
state = trainer.train(model=student, train_dataloader=dataloader)
```

The base `train()` method handles epoch iteration, callback firing (`train_start`, `step_end`, `epoch_end`, etc.), early stopping, and `max_steps` — your override only needs to handle the single-step logic.

You can also override `move_batch_to_device()` if your batches need custom device handling.

## Custom Models

### Passing a pre-built model

All recipe functions (`finetune`, `pretrain`, `align`) accept a model instance directly:

```python
import trainlib

state = trainlib.finetune(
    model=my_custom_model,      # any nn.Module
    tokenizer=my_tokenizer,     # required with raw models
    dataset="data/train.jsonl",
    format="alpaca",
    max_steps=100,
)
```

You can also pass a `ModelResult` for more control:

```python
from trainlib.models import ModelResult

model_result = ModelResult(
    model=my_model,
    tokenizer=my_tokenizer,
    name="my-transformer",
    metadata={"custom_key": "value"},
)

state = trainlib.finetune(
    model=model_result,
    dataset="data/train.jsonl",
)
```

### Registering a model loader

Register a custom model loader so it works with YAML config files:

```python
from trainlib.models import register_model, ModelResult

@register_model("my-transformer")
def load_my_transformer(name_or_path, *, dtype="auto", **kwargs):
    model = MyTransformerArchitecture(hidden_size=768, num_layers=12)
    tokenizer = MyTokenizer.from_pretrained("my-tokenizer")
    return ModelResult(model=model, tokenizer=tokenizer, name=name_or_path)
```

Then reference it in your config:

```yaml
recipe: finetune
model:
  name: my-transformer
data:
  path: data/train.jsonl
  format: alpaca
```

When `load_model("my-transformer")` is called, it checks the model registry first and falls back to HuggingFace `AutoModelForCausalLM` if no match is found.

## Custom Recipes

Register a recipe function with the recipe registry:

```python
from trainlib.recipes import recipe_registry
from trainlib.recipes.base import setup_training

@recipe_registry.register("distill")
def distill(*, config, resume_from=None):
    components = setup_training(config, resume_from=resume_from)

    teacher = load_teacher_model()
    trainer = DistillTrainer(
        config=config.trainer,
        teacher_model=teacher,
        callback_manager=components.trainer.callback_manager,
    )

    return trainer.train(
        model=components.model,
        train_dataloader=components.train_dataloader,
        resume_state=components.resume_state,
    )
```

Use it with the CLI:

```yaml
# distill.yaml
recipe: distill
model:
  name: meta-llama/Llama-3-8B
data:
  path: data/train.jsonl
  format: alpaca
trainer:
  num_epochs: 3
```

```bash
trainlib train --config distill.yaml
```

Custom recipe and method names are accepted by the config system — you are not limited to the built-in `finetune`, `pretrain`, and `align`.

## Custom Data Formats

Register a data format function to handle custom dataset structures:

```python
from trainlib.data import format_registry

@format_registry.register("my_format")
def format_my_data(sample):
    return {"text": f"Question: {sample['q']}\nAnswer: {sample['a']}"}
```

Then use it in your config:

```yaml
data:
  path: data/qa.jsonl
  format: my_format
```

## Plugin Discovery

Third-party packages can register recipes, models, formats, and metrics automatically using Python entry points. trainlib discovers them at import time.

In your package's `pyproject.toml`:

```toml
[project.entry-points."trainlib.recipes"]
distill = "my_package.recipes:distill"

[project.entry-points."trainlib.models"]
my-transformer = "my_package.models:load_my_transformer"

[project.entry-points."trainlib.formats"]
my-format = "my_package.formats:format_my_data"

[project.entry-points."trainlib.metrics"]
my-metric = "my_package.metrics:my_metric_fn"
```

Once the package is installed, its extensions appear in trainlib automatically — no imports or registration code needed. The `trainlib list` command shows all registered items including plugins.

## Available Registries

| Registry | Entry Point Group | List Command |
|----------|-------------------|--------------|
| Recipes | `trainlib.recipes` | `trainlib list recipes` |
| Models | `trainlib.models` | `trainlib list models` |
| Data Formats | `trainlib.formats` | `trainlib list formats` |
| Metrics | `trainlib.metrics` | `trainlib list metrics` |
| Rewards | — | `trainlib list rewards` |
