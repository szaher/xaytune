# Pre-training

The `pretrain` recipe trains a language model from scratch (or continues pre-training an existing model) on a large text corpus using a standard causal language modeling objective.

## Python API

```python
import trainlib

state = trainlib.pretrain(
    model="meta-llama/Llama-3.1-8B",
    dataset="data/corpus.jsonl",
    format="text",
    num_epochs=1,
    learning_rate=3e-4,
    batch_size=4,
)
```

### Function Signature

```python
def pretrain(
    *,
    config: TrainConfig | None = None,
    model: str | None = None,
    dataset: str | None = None,
    format: str = "text",
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    batch_size: int = 4,
    **kwargs,
) -> TrainState:
```

- **config** -- A full `TrainConfig` object. If provided, all other arguments are ignored.
- **model** -- Model name or path.
- **dataset** -- Path to training data.
- **format** -- Data format (default: `"text"`).
- **num_epochs** -- Number of training epochs (default: 1).
- **learning_rate** -- Learning rate (default: 3e-4).
- **batch_size** -- Per-device batch size (default: 4).
- **\*\*kwargs** -- Additional `TrainerConfig` fields.

!!! note
    Pre-training always uses `method="full"` -- LoRA/QLoRA are not applicable since there are no pre-trained weights to adapt.

## YAML Config

```yaml
recipe: pretrain
method: full

model:
  name: meta-llama/Llama-3.1-8B

data:
  path: data/corpus.jsonl
  format: text
  packing: true
  max_seq_length: 2048
  streaming: true

trainer:
  batch_size: 4
  gradient_accumulation: 8
  learning_rate: 3e-4
  num_epochs: 1
  warmup_steps: 1000
  weight_decay: 0.01
  mixed_precision: bf16
  strategy: fsdp

logging:
  backends: [console, tensorboard, wandb]
  project: pretrain-run
  log_every_n_steps: 10

eval:
  every_n_steps: 500
  metrics: [loss, perplexity]

output:
  dir: output/pretrain
```

## Data Preparation

Pre-training typically uses the `text` format, where each sample has a `text` (or `content`) field containing raw text:

```json
{"text": "The quick brown fox jumps over the lazy dog."}
{"text": "Machine learning is a subset of artificial intelligence..."}
```

### Sequence Packing

By default, trainlib packs multiple short sequences into a single training example up to `max_seq_length` to improve GPU utilization. Disable this with `packing: false` if your data already contains full-length sequences.

### Streaming

For large datasets that do not fit in memory, enable `streaming: true` in the data config. This loads data on the fly rather than materializing the full dataset.

## Distributed Training

Pre-training typically requires multiple GPUs. Configure the distributed strategy in the trainer config:

```yaml
trainer:
  strategy: fsdp  # or "ddp", "deepspeed", "auto"
```

| Strategy | Description |
|----------|-------------|
| `auto` | Automatically selects based on available hardware |
| `ddp` | Distributed Data Parallel -- replicates model on each GPU |
| `fsdp` | Fully Sharded Data Parallel -- shards model across GPUs |
| `deepspeed` | DeepSpeed ZeRO optimization (requires `pip install trainlib[deepspeed]`) |
