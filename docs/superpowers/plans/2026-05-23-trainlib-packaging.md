# Packaging & Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make xaytune pip-installable with a proper README, example configs, `py.typed` marker, `__main__.py` for `python -m xaytune`, and verify the full package works end-to-end.

**Architecture:** The `pyproject.toml` already has correct dependencies, extras, and entry points. This plan adds the missing distribution artifacts (README, examples, py.typed, __main__.py), creates user-facing example configs in `configs/examples/`, and adds packaging integration tests that verify the installed package works correctly.

**Tech Stack:** hatchling (build), pytest

---

## File Structure

```
xaytune/
├── pyproject.toml          # (exists, minor update) — add py.typed to package data
├── README.md               # (exists, rewrite) — proper README with install, quickstart, API
├── xaytune/
│   ├── py.typed            # (create) — PEP 561 type stub marker
│   └── __main__.py         # (create) — enables `python -m xaytune`
├── configs/
│   └── examples/           # (create dir + files) — user-facing example configs
│       ├── lora_finetune.yaml
│       ├── qlora_finetune.yaml
│       ├── full_finetune.yaml
│       ├── pretrain.yaml
│       └── dpo_align.yaml
└── tests/
    └── test_packaging.py   # (create) — verify package metadata, imports, CLI entry point
```

---

### Task 1: py.typed & __main__.py

**Files:**
- Create: `xaytune/py.typed`
- Create: `xaytune/__main__.py`
- Create: `tests/test_packaging.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_packaging.py`:

```python
import subprocess
import sys

import pytest

import xaytune


class TestPackageMetadata:
    def test_version_is_string(self):
        assert isinstance(xaytune.__version__, "0.1.0".__class__)

    def test_version_matches_pyproject(self):
        assert xaytune.__version__ == "0.1.0"

    def test_all_exports(self):
        expected = {"__version__", "align", "evaluate", "finetune", "pretrain"}
        assert set(xaytune.__all__) == expected


class TestPythonModule:
    def test_python_m_xaytune_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "xaytune", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_python_m_xaytune_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "xaytune", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "train" in result.stdout


class TestPyTyped:
    def test_py_typed_exists(self):
        import importlib.resources as resources
        ref = resources.files("xaytune") / "py.typed"
        assert ref.is_file()


class TestImports:
    def test_top_level_imports(self):
        from xaytune import finetune, pretrain, align, evaluate
        assert callable(finetune)
        assert callable(pretrain)
        assert callable(align)
        assert callable(evaluate)

    def test_submodule_imports(self):
        from xaytune.models import load_model, register_model
        from xaytune.data import load_dataset, register_format
        from xaytune.trainer import Trainer, on
        from xaytune.eval import evaluate, register_metric
        from xaytune.export import merge, save, push_to_hub
        from xaytune.logging import setup_logging
        from xaytune.recipes import recipe_registry

    def test_align_losses_importable(self):
        from xaytune.recipes.align import (
            dpo_loss, grpo_loss, orpo_loss, simpo_loss,
            ppo_clip_loss, reinforce_loss,
        )
```

- [ ] **Step 2: Run tests to verify failures**

Run: `.venv/bin/python -m pytest tests/test_packaging.py -v`
Expected: `TestPythonModule` and `TestPyTyped` tests FAIL

- [ ] **Step 3: Create py.typed marker**

Create `xaytune/py.typed` (empty file — PEP 561 marker):

```
```

- [ ] **Step 4: Create __main__.py**

Create `xaytune/__main__.py`:

```python
from xaytune.cli import main

raise SystemExit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_packaging.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add xaytune/py.typed xaytune/__main__.py tests/test_packaging.py
git commit -m "feat: add py.typed marker, __main__.py, and packaging tests"
```

---

### Task 2: Example Configs

**Files:**
- Create: `configs/examples/lora_finetune.yaml`
- Create: `configs/examples/qlora_finetune.yaml`
- Create: `configs/examples/full_finetune.yaml`
- Create: `configs/examples/pretrain.yaml`
- Create: `configs/examples/dpo_align.yaml`

User-facing example configs that demonstrate full xaytune configs (model, data, trainer, logging, output sections). These inherit from the built-in defaults via `base:` and add the user-specific parts (model name, dataset path).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_packaging.py` (append after existing classes):

```python
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "configs" / "examples"


class TestExampleConfigs:
    def test_examples_dir_exists(self):
        assert EXAMPLES_DIR.is_dir()

    def test_lora_finetune_exists(self):
        assert (EXAMPLES_DIR / "lora_finetune.yaml").is_file()

    def test_qlora_finetune_exists(self):
        assert (EXAMPLES_DIR / "qlora_finetune.yaml").is_file()

    def test_full_finetune_exists(self):
        assert (EXAMPLES_DIR / "full_finetune.yaml").is_file()

    def test_pretrain_exists(self):
        assert (EXAMPLES_DIR / "pretrain.yaml").is_file()

    def test_dpo_align_exists(self):
        assert (EXAMPLES_DIR / "dpo_align.yaml").is_file()

    def test_all_examples_are_valid_yaml(self):
        import yaml
        for f in EXAMPLES_DIR.glob("*.yaml"):
            data = yaml.safe_load(f.read_text())
            assert isinstance(data, dict), f"{f.name} is not a valid YAML mapping"
            assert "recipe" in data, f"{f.name} missing 'recipe' key"
            assert "model" in data, f"{f.name} missing 'model' key"
            assert "data" in data, f"{f.name} missing 'data' key"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_packaging.py::TestExampleConfigs -v`
Expected: FAIL — directory does not exist

- [ ] **Step 3: Create example configs**

Create `configs/examples/lora_finetune.yaml`:

```yaml
# LoRA fine-tuning example
# Usage: xaytune train --config configs/examples/lora_finetune.yaml
base: lora

model:
  name: meta-llama/Llama-3.1-8B

data:
  path: data/train.jsonl
  format: alpaca
  eval_split: 0.05

trainer:
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2e-4
  num_epochs: 3

logging:
  backends: [console, tensorboard]
  project: lora-finetune

output:
  dir: output/lora-finetune
```

Create `configs/examples/qlora_finetune.yaml`:

```yaml
# QLoRA fine-tuning example (4-bit quantized base model)
# Usage: xaytune train --config configs/examples/qlora_finetune.yaml
base: qlora

model:
  name: meta-llama/Llama-3.1-70B
  quantization: 4bit

data:
  path: data/train.jsonl
  format: alpaca
  eval_split: 0.05

trainer:
  batch_size: 4
  gradient_accumulation: 8
  learning_rate: 2e-4
  num_epochs: 3

logging:
  backends: [console, tensorboard]
  project: qlora-finetune

output:
  dir: output/qlora-finetune
```

Create `configs/examples/full_finetune.yaml`:

```yaml
# Full fine-tuning example (all parameters trainable)
# Usage: xaytune train --config configs/examples/full_finetune.yaml
base: full_finetune

model:
  name: meta-llama/Llama-3.1-8B

data:
  path: data/train.jsonl
  format: alpaca
  eval_split: 0.05

trainer:
  strategy: fsdp
  batch_size: 2
  gradient_accumulation: 8
  learning_rate: 5e-5
  num_epochs: 3

logging:
  backends: [console, tensorboard]
  project: full-finetune

output:
  dir: output/full-finetune
  merge_on_complete: true
```

Create `configs/examples/pretrain.yaml`:

```yaml
# Pre-training / continued pre-training example
# Usage: xaytune train --config configs/examples/pretrain.yaml
base: pretrain

model:
  name: meta-llama/Llama-3.1-8B

data:
  path: data/corpus/
  format: text
  packing: true
  streaming: true
  max_seq_length: 4096

trainer:
  strategy: fsdp
  batch_size: 2
  gradient_accumulation: 16
  learning_rate: 3e-4
  max_steps: 100000
  warmup_steps: 2000

logging:
  backends: [console, tensorboard]
  project: pretrain

output:
  dir: output/pretrain
```

Create `configs/examples/dpo_align.yaml`:

```yaml
# DPO alignment example
# Usage: xaytune train --config configs/examples/dpo_align.yaml
recipe: align
method: dpo

model:
  name: output/sft-model

data:
  path: data/preferences.jsonl
  format: preference
  eval_split: 0.05

trainer:
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 5e-6
  num_epochs: 1

logging:
  backends: [console, tensorboard]
  project: dpo-align

output:
  dir: output/dpo-aligned
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_packaging.py::TestExampleConfigs -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add configs/examples/
git commit -m "feat: add example configs for LoRA, QLoRA, full finetune, pretrain, and DPO"
```

---

### Task 3: README

**Files:**
- Modify: `README.md`

Replace the stub README with a proper one covering install, quickstart, all three recipes, config usage, CLI, decorator APIs, and project structure.

- [ ] **Step 1: Replace README.md**

Replace `README.md` with:

````markdown
# xaytune

An opinionated LLM training and fine-tuning library built on PyTorch.

xaytune provides a recipe-based architecture with a layered API: simple one-liners for beginners, full control for experts. Config files and Python API are equal citizens.

## Install

```bash
pip install xaytune
```

Optional extras:

```bash
pip install xaytune[wandb]       # Weights & Biases logging
pip install xaytune[mlflow]      # MLflow logging
pip install xaytune[deepspeed]   # DeepSpeed distributed training
pip install xaytune[eval]        # lm-eval-harness benchmarks
pip install xaytune[all]         # Everything
```

## Quickstart

### Python API

```python
import xaytune

# LoRA fine-tuning
xaytune.finetune(
    model="meta-llama/Llama-3.1-8B",
    dataset="data/train.jsonl",
    method="lora",
    format="alpaca",
    num_epochs=3,
)

# Pre-training
xaytune.pretrain(
    model="meta-llama/Llama-3.1-8B",
    dataset="data/corpus/",
    format="text",
)

# DPO alignment
xaytune.align(
    model="output/sft-model",
    dataset="data/preferences.jsonl",
    method="dpo",
    format="preference",
)

# Evaluation
results = xaytune.evaluate(
    model="output/my-model",
    dataset=[{"input_ids": [1, 2], "labels": [1, 2]}],
    metrics=["loss", "perplexity"],
)
```

### CLI

```bash
# Train with config
xaytune train --config configs/examples/lora_finetune.yaml

# Train with overrides
xaytune train --config configs/examples/lora_finetune.yaml \
    --override model.name=mistralai/Mistral-7B-v0.3

# Dry run (validate and print config)
xaytune train --config configs/examples/lora_finetune.yaml --dry-run

# List registered components
xaytune list recipes
xaytune list formats
xaytune list metrics
```

### Config file

```yaml
recipe: finetune
method: lora

model:
  name: meta-llama/Llama-3.1-8B

data:
  path: data/train.jsonl
  format: alpaca
  eval_split: 0.05

lora:
  rank: 16
  alpha: 32

trainer:
  batch_size: 4
  learning_rate: 2e-4
  num_epochs: 3

logging:
  backends: [console, tensorboard]
```

See `configs/examples/` for more examples.

## Recipes

| Recipe | Methods | Use case |
|--------|---------|----------|
| `finetune` | `full`, `lora`, `qlora` | Supervised fine-tuning on instruction data |
| `pretrain` | `full` | Pre-training or continued pre-training on raw text |
| `align` | `dpo`, `grpo`, `ppo`, `orpo`, `simpo` | Alignment with human preferences |

## Extensibility

Register custom components with decorators:

```python
from xaytune.models import register_model
from xaytune.data import register_format
from xaytune.eval import register_metric
from xaytune.recipes.align import register_reward
from xaytune.trainer import on

@register_format("my-format")
def parse_my_data(sample):
    return {"instruction": sample["q"], "response": sample["a"]}

@register_metric("domain-accuracy")
def domain_accuracy(predictions, references):
    return sum(p == r for p, r in zip(predictions, references)) / len(predictions)

@on("step_end")
def log_memory(state):
    print(f"Step {state.global_step}: loss={state.metrics.get('loss', 'N/A')}")
```

## Export

```python
from xaytune import export

# Merge LoRA adapters into base model
export.merge("output/lora-checkpoint", save_to="output/merged-model")

# Save with metadata
export.save(model, tokenizer, output_dir="output/final", metadata={"recipe": "finetune"})

# Push to Hugging Face Hub
export.push_to_hub("output/merged-model", repo="username/my-model")
```

## Architecture

```
┌─────────────────────────────────────────┐
│           CLI / Config Engine           │  Layer 3 — Interface
├─────────────────────────────────────────┤
│   pretrain │ finetune │ align (recipes) │  Layer 2 — Recipes
├────────┬────────┬─────────┬────────┬────┤
│ models │  data  │ trainer │  eval  │ exp│  Layer 1 — Building Blocks
└────────┴────────┴─────────┴────────┴────┘
         PyTorch / HuggingFace / DeepSpeed
```

## License

Apache 2.0
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README with install, quickstart, and API reference"
```

---

### Task 4: Verify Package Install

**Files:**
- Modify: `tests/test_packaging.py` (append)

Add an integration test that verifies the package can be installed in the current venv and the CLI entry point resolves.

- [ ] **Step 1: Add install verification tests**

Append to `tests/test_packaging.py`:

```python
class TestEntryPoint:
    def test_cli_entry_point_defined(self):
        from importlib.metadata import entry_points
        eps = entry_points()
        console_scripts = eps.get("console_scripts", eps.select(group="console_scripts"))
        names = [ep.name for ep in console_scripts]
        assert "xaytune" in names

    def test_cli_entry_point_resolves(self):
        from importlib.metadata import entry_points
        eps = entry_points()
        console_scripts = eps.get("console_scripts", eps.select(group="console_scripts"))
        xaytune_ep = [ep for ep in console_scripts if ep.name == "xaytune"][0]
        fn = xaytune_ep.load()
        assert callable(fn)
```

- [ ] **Step 2: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_packaging.py
git commit -m "test: add entry point resolution and package install verification"
```
