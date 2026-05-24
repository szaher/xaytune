# Evaluation

trainlib provides two evaluation paths: custom dataset evaluation with `evaluate()` and benchmark evaluation with `benchmark_evaluate()`.

## evaluate()

Evaluate a model on a custom dataset with configurable metrics.

```python
from trainlib.eval import evaluate

results = evaluate(
    model="output/my-finetune",
    dataset=[{"input_ids": ..., "labels": ...}],
    metrics=["loss", "perplexity"],
)

print(results)
# {'loss': 1.234, 'perplexity': 3.435}
```

### Function Signature

```python
def evaluate(
    *,
    model: Any,
    dataset: list[dict[str, Any]],
    metrics: list[str] | None = None,
) -> dict[str, float]:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | model object or `str` | *required* | A model instance or path to load from |
| `dataset` | `list[dict]` | *required* | List of data batches to evaluate on |
| `metrics` | `list[str]` \| `None` | `["loss", "perplexity"]` | Metric names to compute (must be in `metric_registry`) |

**Returns:** `dict[str, float]` mapping metric names to their computed values.

!!! note
    When `model` is a string path, trainlib automatically loads the model and tokenizer using `trainlib.models.load_model()`.

---

## benchmark_evaluate()

Run standard benchmarks using [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness).

```python
from trainlib.eval.benchmarks import benchmark_evaluate

results = benchmark_evaluate(
    model="meta-llama/Llama-3.1-8B",
    benchmarks=["mmlu", "gsm8k", "hellaswag"],
    num_fewshot=5,
)

for task, metrics in results.items():
    print(f"{task}: {metrics}")
```

### Function Signature

```python
def benchmark_evaluate(
    *,
    model: str,
    benchmarks: list[str],
    num_fewshot: int | None = None,
) -> dict[str, dict[str, Any]]:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *required* | Model path or Hugging Face Hub name |
| `benchmarks` | `list[str]` | *required* | List of benchmark task names |
| `num_fewshot` | `int` \| `None` | `None` | Number of few-shot examples (benchmark default if `None`) |

**Returns:** Nested dict `{task_name: {metric_name: value}}`.

!!! warning "Requires `lm-eval`"
    Install the eval extra to use benchmarks:
    ```bash
    pip install trainlib[eval]
    ```

---

## Built-in Metrics

trainlib ships three metrics, registered in `trainlib.eval.metrics.metric_registry`:

| Metric | Function | Description |
|--------|----------|-------------|
| `loss` | `compute_loss` | Average cross-entropy loss |
| `perplexity` | `compute_perplexity` | Exponentiated average loss: exp(mean_loss) |
| `token_accuracy` | `compute_token_accuracy` | Fraction of correctly predicted tokens |

### Custom Metrics

Register your own metrics with the `@register_metric` decorator:

```python
from trainlib.eval.metrics import register_metric

@register_metric("bleu")
def compute_bleu(predictions, references, **kwargs):
    # Your BLEU implementation here
    ...
    return score
```

Once registered, custom metrics can be used anywhere metrics are accepted:

```python
results = evaluate(model=model, dataset=data, metrics=["loss", "bleu"])
```

Or in YAML config:

```yaml
eval:
  metrics: [loss, perplexity, bleu]
```

---

## CLI Usage

### Benchmark Evaluation

```bash
trainlib eval --model output/my-finetune --benchmarks mmlu,gsm8k --num-fewshot 5
```

### Dataset Evaluation

```bash
trainlib eval --model output/my-finetune --dataset data/eval.jsonl --metrics loss,perplexity
```

### Model Comparison

Compare two models side-by-side on the same benchmarks:

```bash
trainlib compare model-a model-b --benchmarks mmlu,gsm8k
```

This prints a table showing each model's score on every benchmark metric.

---

## Full API Reference

::: trainlib.eval.evaluate.evaluate

::: trainlib.eval.benchmarks.benchmark_evaluate

::: trainlib.eval.metrics.compute_loss

::: trainlib.eval.metrics.compute_perplexity

::: trainlib.eval.metrics.compute_token_accuracy
