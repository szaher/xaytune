# Config Schema Reference

xaytune uses [Pydantic](https://docs.pydantic.dev/) models for configuration. The root model is `TrainConfig`, which nests several sub-models for model, data, training, evaluation, logging, and output settings.

All config models live in `xaytune.config.schema`.

## TrainConfig

The top-level configuration object. Every training run is driven by a `TrainConfig`.

```python
from xaytune.config.schema import TrainConfig

config = TrainConfig(
    recipe="finetune",
    method="lora",
    model=ModelConfig(name="meta-llama/Llama-3.1-8B"),
    data=DataConfig(path="data/train.jsonl", format="alpaca"),
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `recipe` | `"finetune"` \| `"pretrain"` \| `"align"` | *required* | Which training recipe to use |
| `method` | `str` | `"full"` | Training method (see below) |
| `base` | `str` \| `None` | `None` | Base config to inherit from |
| `model` | `ModelConfig` | *required* | Model configuration |
| `data` | `DataConfig` | *required* | Data configuration |
| `lora` | `LoraConfig` | `LoraConfig()` | LoRA adapter settings |
| `trainer` | `TrainerConfig` | `TrainerConfig()` | Training hyperparameters |
| `eval` | `EvalConfig` | `EvalConfig()` | Evaluation settings |
| `logging` | `LoggingConfig` | `LoggingConfig()` | Logging backend configuration |
| `output` | `OutputConfig` | `OutputConfig()` | Output directory settings |

**Valid methods by recipe:**

- `finetune`: `full`, `lora`, `qlora`
- `pretrain`: `full`
- `align`: `dpo`, `grpo`, `ppo`, `orpo`, `simpo`

---

## ModelConfig

```python
from xaytune.config.schema import ModelConfig

model = ModelConfig(
    name="meta-llama/Llama-3.1-8B",
    quantization="4bit",
    dtype="auto",
    trust_remote_code=False,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *required* | Model name (HF Hub ID) or local path |
| `quantization` | `"4bit"` \| `"8bit"` \| `None` | `None` | Quantization mode for bitsandbytes |
| `dtype` | `str` | `"auto"` | Model dtype (`"auto"`, `"float16"`, `"bfloat16"`, etc.) |
| `trust_remote_code` | `bool` | `False` | Whether to trust remote code from HF Hub |

---

## DataConfig

```python
from xaytune.config.schema import DataConfig

data = DataConfig(
    path="data/train.jsonl",
    format="alpaca",
    source="local",
    eval_split=0.05,
    packing=True,
    max_seq_length=2048,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `str` | *required* | Path to dataset file or HF Hub dataset name |
| `format` | `str` | *required* | Data format key (must be in `format_registry`) |
| `source` | `"local"` \| `"huggingface"` | `"local"` | Where to load data from |
| `eval_split` | `float` | `0.0` | Fraction of data to hold out for evaluation |
| `eval_path` | `str` \| `None` | `None` | Explicit path to evaluation dataset |
| `packing` | `bool` | `True` | Pack multiple sequences into one training example |
| `max_seq_length` | `int` | `2048` | Maximum sequence length |
| `streaming` | `bool` | `False` | Stream data instead of loading into memory |

---

## LoraConfig

```python
from xaytune.config.schema import LoraConfig

lora = LoraConfig(
    rank=16,
    alpha=32,
    dropout=0.05,
    target_modules="auto",
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rank` | `int` | `16` | LoRA rank (r). Higher = more parameters, more capacity |
| `alpha` | `int` | `32` | LoRA alpha scaling factor. Common rule: alpha = 2 * rank |
| `dropout` | `float` | `0.05` | Dropout probability for LoRA layers |
| `target_modules` | `str` \| `list[str]` | `"auto"` | Which modules to apply LoRA to. `"auto"` selects standard attention layers |

---

## TrainerConfig

```python
from xaytune.config.schema import TrainerConfig

trainer = TrainerConfig(
    strategy="auto",
    mixed_precision="bf16",
    batch_size=4,
    gradient_accumulation=4,
    learning_rate=2e-4,
    num_epochs=3,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | `"auto"` \| `"ddp"` \| `"fsdp"` \| `"deepspeed"` | `"auto"` | Distributed training strategy |
| `mixed_precision` | `"fp16"` \| `"bf16"` \| `"fp32"` | `"bf16"` | Mixed precision mode |
| `batch_size` | `int` | `4` | Per-device batch size |
| `gradient_accumulation` | `int` | `1` | Gradient accumulation steps |
| `learning_rate` | `float` | `2e-4` | Optimizer learning rate |
| `num_epochs` | `int` | `3` | Number of training epochs |
| `max_steps` | `int` | `-1` | Maximum training steps (-1 = unlimited) |
| `warmup_steps` | `int` | `0` | Number of warmup steps |
| `warmup_ratio` | `float` | `0.0` | Warmup as a fraction of total steps |
| `weight_decay` | `float` | `0.01` | Weight decay for optimizer |
| `max_grad_norm` | `float` | `1.0` | Maximum gradient norm for clipping |
| `seed` | `int` | `42` | Random seed |
| `checkpoint_every_n_steps` | `int` | `500` | Save a checkpoint every N steps |
| `save_last` | `bool` | `True` | Always save the final checkpoint |

---

## EvalConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `every_n_steps` | `int` | `500` | Run evaluation every N steps |
| `metrics` | `list[str]` | `["loss", "perplexity"]` | Metrics to compute during evaluation |
| `benchmarks` | `list[str]` | `[]` | lm-eval benchmarks to run |

---

## LoggingConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backends` | `list[str]` | `["console"]` | Logging backends to enable |
| `project` | `str` \| `None` | `None` | Project name for wandb/mlflow |
| `run_name` | `str` \| `None` | `None` | Run name for wandb/mlflow |
| `log_every_n_steps` | `int` | `10` | Log metrics every N steps |

**Available backends:** `console`, `tensorboard`, `wandb`, `mlflow`

---

## OutputConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dir` | `str` | `"output"` | Output directory for checkpoints and artifacts |
| `merge_on_complete` | `bool` | `False` | Automatically merge LoRA adapters after training |

---

## Loading Configs from YAML

```python
from xaytune.config import load_config, validate_config

# Load from YAML file
config = load_config("configs/examples/lora_finetune.yaml")

# Load with overrides
config = load_config(
    "configs/examples/lora_finetune.yaml",
    overrides=["model.name=mistralai/Mistral-7B-v0.3", "trainer.num_epochs=5"],
)

# Validate
validate_config(config)
```

---

## Full API Reference

### Schema Classes

::: xaytune.config.schema.TrainConfig

::: xaytune.config.schema.ModelConfig

::: xaytune.config.schema.DataConfig

::: xaytune.config.schema.LoraConfig

::: xaytune.config.schema.TrainerConfig

::: xaytune.config.schema.EvalConfig

::: xaytune.config.schema.LoggingConfig

::: xaytune.config.schema.OutputConfig

::: xaytune.config.schema.FSDPConfig

::: xaytune.config.schema.DeepSpeedConfig

### Parser

::: xaytune.config.parser.load_config

::: xaytune.config.parser.merge_dicts

::: xaytune.config.parser.apply_overrides

### Validation

::: xaytune.config.validation.validate_config

::: xaytune.config.validation.preflight_check

::: xaytune.config.validation.ConfigValidationError
