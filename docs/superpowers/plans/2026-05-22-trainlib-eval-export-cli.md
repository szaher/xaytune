# xaytune Eval, Export & CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the eval module (metric registry, built-in metrics, evaluate function), export module (LoRA merge, save with metadata, Hub push stub), and complete the CLI (wire `train` to recipes, add `list` and `eval` commands). This is the final plan — completing the v1 library.

**Architecture:** Eval and export follow the same registry/function patterns as models and data. The CLI wires to recipe registry for `train`, metric registry for `eval`, and export functions. The `list` command uses all registries for discoverability.

**Tech Stack:** PyTorch, pytest, unittest.mock, argparse

---

## Plan Sequence

This is **Plan 6 of 6** — depends on Plans 1-5 being complete.

---

### Task 1: Metric Registry & Built-in Metrics

**Files:**
- Create: `xaytune/eval/metrics.py`
- Create: `tests/test_eval/__init__.py`
- Create: `tests/test_eval/test_metrics.py`

Metric registry with `@register_metric` decorator and three built-in metrics: loss, perplexity, token_accuracy.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval/__init__.py` (empty).

Create `tests/test_eval/test_metrics.py`:

```python
import pytest
import torch
from xaytune.eval.metrics import metric_registry, register_metric


class TestMetricRegistry:
    def test_register_and_get(self):
        @register_metric("test-metric")
        def my_metric(predictions, references) -> float:
            return 1.0

        assert metric_registry.has("test-metric")
        fn = metric_registry.get("test-metric")
        assert fn([], []) == 1.0

    def test_register_returns_original(self):
        @register_metric("identity-metric")
        def my_fn(predictions, references) -> float:
            return 0.5

        assert my_fn([], []) == 0.5

    def test_unknown_metric_raises(self):
        with pytest.raises(KeyError, match="not found"):
            metric_registry.get("nonexistent-metric")

    def test_list_metrics(self):
        metrics = metric_registry.list()
        assert "loss" in metrics
        assert "perplexity" in metrics
        assert "token_accuracy" in metrics


class TestBuiltinMetrics:
    def test_loss_metric(self):
        compute_loss = metric_registry.get("loss")
        losses = [0.5, 0.3, 0.4]
        result = compute_loss(losses)
        assert abs(result - 0.4) < 1e-5

    def test_loss_metric_empty(self):
        compute_loss = metric_registry.get("loss")
        result = compute_loss([])
        assert result == 0.0

    def test_perplexity_metric(self):
        compute_ppl = metric_registry.get("perplexity")
        losses = [1.0, 2.0, 3.0]
        result = compute_ppl(losses)
        # perplexity = exp(mean_loss) = exp(2.0) ≈ 7.389
        import math
        assert abs(result - math.exp(2.0)) < 0.01

    def test_perplexity_empty(self):
        compute_ppl = metric_registry.get("perplexity")
        result = compute_ppl([])
        assert result == 0.0

    def test_token_accuracy(self):
        compute_acc = metric_registry.get("token_accuracy")
        predictions = [1, 2, 3, 4, 5]
        references = [1, 2, 0, 4, 0]
        result = compute_acc(predictions, references)
        assert abs(result - 0.6) < 1e-5

    def test_token_accuracy_perfect(self):
        compute_acc = metric_registry.get("token_accuracy")
        predictions = [1, 2, 3]
        references = [1, 2, 3]
        result = compute_acc(predictions, references)
        assert result == 1.0

    def test_token_accuracy_empty(self):
        compute_acc = metric_registry.get("token_accuracy")
        result = compute_acc([], [])
        assert result == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval/test_metrics.py -v`

- [ ] **Step 3: Implement metric registry and built-in metrics**

Create `xaytune/eval/metrics.py`:

```python
from __future__ import annotations

import math
from typing import Any

from xaytune.utils.registry import Registry

metric_registry = Registry("metric")

register_metric = metric_registry.register


@register_metric("loss")
def compute_loss(losses: list[float], *args: Any, **kwargs: Any) -> float:
    if not losses:
        return 0.0
    return sum(losses) / len(losses)


@register_metric("perplexity")
def compute_perplexity(losses: list[float], *args: Any, **kwargs: Any) -> float:
    if not losses:
        return 0.0
    mean_loss = sum(losses) / len(losses)
    return math.exp(mean_loss)


@register_metric("token_accuracy")
def compute_token_accuracy(
    predictions: list[int],
    references: list[int],
    *args: Any,
    **kwargs: Any,
) -> float:
    if not predictions:
        return 0.0
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(predictions)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval/test_metrics.py -v`

Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/eval/metrics.py tests/test_eval/
git commit -m "feat: add metric registry with loss, perplexity, and token accuracy"
```

---

### Task 2: Evaluate Function

**Files:**
- Create: `xaytune/eval/evaluate.py`
- Modify: `xaytune/eval/__init__.py`
- Create: `tests/test_eval/test_evaluate.py`

The `evaluate()` function runs a model against a dataset and computes requested metrics.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval/test_evaluate.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from xaytune.eval.evaluate import evaluate
from xaytune.eval.metrics import metric_registry


class TestEvaluate:
    @patch("xaytune.eval.evaluate.load_model")
    def test_evaluate_with_model_path(self, mock_load_model):
        mock_result = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_result.model = mock_model
        mock_result.tokenizer = mock_tokenizer
        mock_load_model.return_value = mock_result

        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = MagicMock()
        mock_model.return_value.loss.item.return_value = 0.5

        results = evaluate(
            model="output/my-model",
            dataset=[{"input_ids": [1, 2], "labels": [1, 2]}],
            metrics=["loss"],
        )

        assert "loss" in results
        mock_load_model.assert_called_once()

    def test_evaluate_with_model_object(self):
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = MagicMock()
        mock_model.return_value.loss.item.return_value = 0.3

        results = evaluate(
            model=mock_model,
            dataset=[{"input_ids": [1], "labels": [1]}],
            metrics=["loss"],
        )

        assert "loss" in results

    def test_evaluate_multiple_metrics(self):
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = MagicMock()
        mock_model.return_value.loss.item.return_value = 1.0

        results = evaluate(
            model=mock_model,
            dataset=[
                {"input_ids": [1, 2], "labels": [1, 2]},
                {"input_ids": [3, 4], "labels": [3, 4]},
            ],
            metrics=["loss", "perplexity"],
        )

        assert "loss" in results
        assert "perplexity" in results
        assert results["loss"] == 1.0

    def test_evaluate_default_metrics(self):
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = MagicMock()
        mock_model.return_value.loss.item.return_value = 0.5

        results = evaluate(
            model=mock_model,
            dataset=[{"input_ids": [1], "labels": [1]}],
        )

        assert "loss" in results
        assert "perplexity" in results

    def test_evaluate_returns_dict(self):
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = MagicMock()
        mock_model.return_value.loss.item.return_value = 0.5

        results = evaluate(
            model=mock_model,
            dataset=[{"input_ids": [1], "labels": [1]}],
            metrics=["loss"],
        )

        assert isinstance(results, dict)

    def test_evaluate_empty_dataset(self):
        mock_model = MagicMock()
        results = evaluate(model=mock_model, dataset=[], metrics=["loss"])
        assert results["loss"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval/test_evaluate.py -v`

- [ ] **Step 3: Implement evaluate function**

Create `xaytune/eval/evaluate.py`:

```python
from __future__ import annotations

from typing import Any

import torch

from xaytune.eval.metrics import metric_registry


def evaluate(
    *,
    model: Any,
    dataset: list[dict[str, Any]],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    if metrics is None:
        metrics = ["loss", "perplexity"]

    if isinstance(model, str):
        from xaytune.models import load_model
        model_result = load_model(model)
        model = model_result.model

    losses: list[float] = []

    model.eval() if hasattr(model, "eval") else None

    with torch.no_grad():
        for batch in dataset:
            if isinstance(batch, dict):
                outputs = model(**batch)
            else:
                outputs = model(batch)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                losses.append(outputs.loss.item())

    results: dict[str, float] = {}
    for metric_name in metrics:
        compute_fn = metric_registry.get(metric_name)
        if metric_name in ("loss", "perplexity"):
            results[metric_name] = compute_fn(losses)
        else:
            results[metric_name] = compute_fn([], [])

    return results
```

Update `xaytune/eval/__init__.py`:

```python
from xaytune.eval.evaluate import evaluate
from xaytune.eval.metrics import metric_registry, register_metric

__all__ = ["evaluate", "metric_registry", "register_metric"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval/ -v`

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/eval/ tests/test_eval/test_evaluate.py
git commit -m "feat: add evaluate function with loss and perplexity metrics"
```

---

### Task 3: Export Module

**Files:**
- Create: `xaytune/export/merge.py`
- Create: `xaytune/export/hub.py`
- Modify: `xaytune/export/__init__.py`
- Create: `tests/test_export/__init__.py`
- Create: `tests/test_export/test_merge.py`
- Create: `tests/test_export/test_hub.py`

LoRA adapter merging, save with metadata, and Hub push (functional stub).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_export/__init__.py` (empty).

Create `tests/test_export/test_merge.py`:

```python
import json
import tempfile
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock

from xaytune.export.merge import merge, save


class TestMerge:
    @patch("xaytune.export.merge.load_model")
    def test_merge_loads_and_merges(self, mock_load_model):
        mock_result = MagicMock()
        mock_model = MagicMock()
        mock_model.merge_and_unload.return_value = MagicMock()
        mock_result.model = mock_model
        mock_result.tokenizer = MagicMock()
        mock_result.peft_applied = True
        mock_load_model.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            merge("checkpoint/lora", save_to=tmpdir)

        mock_model.merge_and_unload.assert_called_once()

    @patch("xaytune.export.merge.load_model")
    def test_merge_saves_model_and_tokenizer(self, mock_load_model):
        mock_result = MagicMock()
        mock_merged = MagicMock()
        mock_result.model = MagicMock()
        mock_result.model.merge_and_unload.return_value = mock_merged
        mock_result.tokenizer = MagicMock()
        mock_result.peft_applied = True
        mock_load_model.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            merge("checkpoint/lora", save_to=tmpdir)

            mock_merged.save_pretrained.assert_called_once_with(tmpdir)
            mock_result.tokenizer.save_pretrained.assert_called_once_with(tmpdir)

    @patch("xaytune.export.merge.load_model")
    def test_merge_non_peft_raises(self, mock_load_model):
        mock_result = MagicMock()
        mock_result.peft_applied = False
        mock_load_model.return_value = mock_result

        with pytest.raises(ValueError, match="not a PEFT model"):
            merge("checkpoint/full", save_to="output/")


class TestSave:
    def test_save_creates_directory(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "saved-model"
            save(mock_model, mock_tokenizer, output_dir=str(output_dir))

            assert output_dir.exists()

    def test_save_calls_save_pretrained(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            save(mock_model, mock_tokenizer, output_dir=tmpdir)

            mock_model.save_pretrained.assert_called_once_with(tmpdir)
            mock_tokenizer.save_pretrained.assert_called_once_with(tmpdir)

    def test_save_with_metadata(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            save(
                mock_model,
                mock_tokenizer,
                output_dir=tmpdir,
                metadata={"recipe": "finetune", "method": "lora"},
            )

            meta_path = Path(tmpdir) / "xaytune_metadata.json"
            assert meta_path.exists()
            meta = json.loads(meta_path.read_text())
            assert meta["recipe"] == "finetune"
```

Create `tests/test_export/test_hub.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from xaytune.export.hub import push_to_hub


class TestPushToHub:
    @patch("xaytune.export.hub.load_model")
    def test_push_to_hub_with_path(self, mock_load_model):
        mock_result = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_result.model = mock_model
        mock_result.tokenizer = mock_tokenizer
        mock_load_model.return_value = mock_result

        push_to_hub("output/my-model", repo="user/my-model")

        mock_model.push_to_hub.assert_called_once_with("user/my-model")
        mock_tokenizer.push_to_hub.assert_called_once_with("user/my-model")

    def test_push_to_hub_with_objects(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        push_to_hub(mock_model, repo="user/my-model", tokenizer=mock_tokenizer)

        mock_model.push_to_hub.assert_called_once_with("user/my-model")
        mock_tokenizer.push_to_hub.assert_called_once_with("user/my-model")

    def test_push_requires_repo(self):
        with pytest.raises(ValueError, match="repo"):
            push_to_hub("output/model")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_export/ -v`

- [ ] **Step 3: Implement export module**

Create `xaytune/export/merge.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def merge(checkpoint_path: str, *, save_to: str) -> None:
    from xaytune.models import load_model

    model_result = load_model(checkpoint_path)

    if not model_result.peft_applied:
        raise ValueError(
            f"Model at '{checkpoint_path}' is not a PEFT model. "
            f"merge() only works with LoRA/QLoRA checkpoints."
        )

    merged_model = model_result.model.merge_and_unload()

    Path(save_to).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(save_to)
    model_result.tokenizer.save_pretrained(save_to)


def save(
    model: Any,
    tokenizer: Any,
    *,
    output_dir: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if metadata:
        meta_path = path / "xaytune_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
```

Create `xaytune/export/hub.py`:

```python
from __future__ import annotations

from typing import Any


def push_to_hub(
    model_or_path: Any,
    *,
    repo: str | None = None,
    tokenizer: Any | None = None,
) -> None:
    if repo is None:
        raise ValueError("'repo' is required (e.g., 'username/model-name').")

    if isinstance(model_or_path, str):
        from xaytune.models import load_model
        model_result = load_model(model_or_path)
        model = model_result.model
        tokenizer = model_result.tokenizer
    else:
        model = model_or_path

    model.push_to_hub(repo)
    if tokenizer is not None:
        tokenizer.push_to_hub(repo)
```

Update `xaytune/export/__init__.py`:

```python
from xaytune.export.hub import push_to_hub
from xaytune.export.merge import merge, save

__all__ = ["merge", "push_to_hub", "save"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_export/ -v`

Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/export/ tests/test_export/
git commit -m "feat: add export module with LoRA merge, save, and Hub push"
```

---

### Task 4: CLI Completion

**Files:**
- Modify: `xaytune/cli.py`
- Modify: `tests/test_cli.py`

Wire the `train` command to actually run recipes via recipe_registry, add `list` command for discoverability, and `eval` command.

- [ ] **Step 1: Write the failing tests**

Read existing `tests/test_cli.py` first, then append new tests.

Add to `tests/test_cli.py`:

```python
from unittest.mock import patch, MagicMock
from xaytune.cli import main


class TestTrainCommand:
    @patch("xaytune.cli.load_config")
    @patch("xaytune.cli.validate_config")
    @patch("xaytune.cli.recipe_registry")
    def test_train_runs_recipe(self, mock_registry, mock_validate, mock_load):
        mock_config = MagicMock()
        mock_config.recipe = "finetune"
        mock_config.model_dump_json.return_value = "{}"
        mock_load.return_value = mock_config
        mock_recipe = MagicMock()
        mock_registry.get.return_value = mock_recipe

        result = main(["train", "--config", "test.yaml"])

        mock_registry.get.assert_called_once_with("finetune")
        mock_recipe.assert_called_once_with(config=mock_config)
        assert result == 0


class TestListCommand:
    def test_list_recipes(self, capsys):
        result = main(["list", "recipes"])
        captured = capsys.readouterr()
        assert result == 0
        assert "finetune" in captured.out
        assert "pretrain" in captured.out
        assert "align" in captured.out

    def test_list_formats(self, capsys):
        result = main(["list", "formats"])
        captured = capsys.readouterr()
        assert result == 0
        assert "alpaca" in captured.out

    def test_list_metrics(self, capsys):
        result = main(["list", "metrics"])
        captured = capsys.readouterr()
        assert result == 0
        assert "loss" in captured.out

    def test_list_no_type(self, capsys):
        result = main(["list"])
        captured = capsys.readouterr()
        assert result == 0
        assert "recipes" in captured.out.lower() or "Recipes" in captured.out

    def test_list_unknown_type(self, capsys):
        result = main(["list", "unknown"])
        captured = capsys.readouterr()
        assert result == 1
```

- [ ] **Step 2: Run tests to verify the new tests fail**

Run: `pytest tests/test_cli.py -v`

- [ ] **Step 3: Update CLI**

Replace `xaytune/cli.py`:

```python
from __future__ import annotations

import argparse
import sys

import xaytune
from xaytune.config import load_config, validate_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xaytune",
        description="xaytune — An opinionated LLM training and fine-tuning library",
    )
    parser.add_argument(
        "--version", action="version", version=f"xaytune {xaytune.__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Run a training recipe")
    train_parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    train_parser.add_argument(
        "--override", action="append", default=[],
        help="Config overrides in dot notation (e.g., model.name=my-model)",
    )
    train_parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    train_parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and print config without training",
    )

    list_parser = subparsers.add_parser("list", help="List registered components")
    list_parser.add_argument(
        "type", nargs="?", default=None,
        help="Component type: recipes, formats, metrics, rewards",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "train":
        return _handle_train(args)

    if args.command == "list":
        return _handle_list(args)

    return 0


def _handle_train(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config, overrides=args.override or None)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    try:
        validate_config(config)
    except Exception as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(config.model_dump_json(indent=2))
        return 0

    from xaytune.recipes import recipe_registry

    recipe_fn = recipe_registry.get(config.recipe)
    recipe_fn(config=config)
    return 0


def _handle_list(args: argparse.Namespace) -> int:
    from xaytune.recipes import recipe_registry
    from xaytune.data import format_registry
    from xaytune.eval.metrics import metric_registry
    from xaytune.recipes.align.rewards import reward_registry

    registries = {
        "recipes": recipe_registry,
        "formats": format_registry,
        "metrics": metric_registry,
        "rewards": reward_registry,
    }

    if args.type is None:
        for name, registry in registries.items():
            items = registry.list()
            print(f"{name.capitalize()}: {', '.join(items)}")
        return 0

    if args.type not in registries:
        print(f"Unknown type: '{args.type}'. Available: {', '.join(registries)}", file=sys.stderr)
        return 1

    registry = registries[args.type]
    for item in registry.list():
        print(f"  {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/cli.py tests/test_cli.py
git commit -m "feat: wire CLI train to recipe registry, add list command"
```

---

### Task 5: Top-Level API Wire-Up

**Files:**
- Modify: `xaytune/__init__.py`
- Modify: `tests/test_top_level_api.py`

Add `xaytune.evaluate` and `xaytune.export` to the top-level package.

- [ ] **Step 1: Write the failing tests**

Update `tests/test_top_level_api.py` to add:

```python
import xaytune
from xaytune import export


class TestTopLevelAPI:
    def test_version(self):
        assert xaytune.__version__ == "0.1.0"

    def test_finetune_importable(self):
        assert callable(xaytune.finetune)

    def test_pretrain_importable(self):
        assert callable(xaytune.pretrain)

    def test_align_importable(self):
        assert callable(xaytune.align)

    def test_evaluate_importable(self):
        assert callable(xaytune.evaluate)

    def test_export_module(self):
        assert callable(export.merge)
        assert callable(export.save)
        assert callable(export.push_to_hub)

    def test_finetune_is_recipe(self):
        from xaytune.recipes.finetune import finetune
        assert xaytune.finetune is finetune

    def test_pretrain_is_recipe(self):
        from xaytune.recipes.pretrain import pretrain
        assert xaytune.pretrain is pretrain

    def test_align_is_recipe(self):
        from xaytune.recipes.align.align import align
        assert xaytune.align is align

    def test_evaluate_is_eval(self):
        from xaytune.eval.evaluate import evaluate
        assert xaytune.evaluate is evaluate
```

- [ ] **Step 2: Run tests to verify the new tests fail**

- [ ] **Step 3: Update top-level __init__.py**

Replace `xaytune/__init__.py`:

```python
"""xaytune — An opinionated LLM training and fine-tuning library."""

__version__ = "0.1.0"

from xaytune.eval import evaluate
from xaytune.recipes.align import align
from xaytune.recipes.finetune import finetune
from xaytune.recipes.pretrain import pretrain

__all__ = ["__version__", "align", "evaluate", "finetune", "pretrain"]
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/__init__.py tests/test_top_level_api.py
git commit -m "feat: complete top-level API with evaluate, export, and all recipes"
```
