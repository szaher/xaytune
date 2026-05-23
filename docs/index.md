# trainlib

**An opinionated LLM training and fine-tuning library.**

trainlib provides a recipe-based approach to LLM training built on PyTorch and Hugging Face Transformers. It covers the full lifecycle from pre-training through fine-tuning to alignment, with a clean Python API and a YAML-driven CLI.

---

## Key Features

- **3 recipes** -- fine-tune, pre-train, and align -- each accessible as a single function call
- **Multiple methods** -- full fine-tuning, LoRA, QLoRA, DPO, GRPO, ORPO, SimPO, PPO, and REINFORCE
- **Pydantic config system** -- type-safe, validated configuration with sensible defaults
- **Registry pattern** -- extend data formats, metrics, rewards, and model loaders via decorators
- **Callback system** -- hook into training events with `@on("step_end")` style decorators
- **Evaluation** -- built-in metrics (loss, perplexity, token accuracy) and lm-eval benchmark integration
- **Export pipeline** -- merge LoRA adapters, save, push to Hugging Face Hub, convert to GGUF
- **Flexible logging** -- console, TensorBoard, Weights & Biases, and MLflow backends
- **Distributed training** -- DDP, FSDP, and DeepSpeed strategies

---

## Quick Example

```python
import trainlib

# Fine-tune with LoRA in one call
state = trainlib.finetune(
    model="meta-llama/Llama-3.1-8B",
    dataset="data/train.jsonl",
    method="lora",
    format="alpaca",
    num_epochs=3,
)
```

Or drive the same workflow from the CLI:

```bash
trainlib train --config configs/examples/lora_finetune.yaml
```

---

## How It Works

1. **Choose a recipe** -- `finetune`, `pretrain`, or `align`
2. **Configure** -- pass keyword arguments in Python, or write a YAML config
3. **Train** -- trainlib loads the model, prepares data, runs the training loop, and returns a `TrainState`
4. **Export** -- merge adapters, push to Hub, or convert to GGUF for local inference

---

## Next Steps

- [Getting Started](getting-started.md) -- installation and your first training run
- [Recipes](recipes/index.md) -- deep dive into each recipe
- [API Reference](api/index.md) -- complete API documentation
- [CLI Reference](cli.md) -- command-line interface
- [Examples](examples.md) -- Jupyter notebooks and sample configs
