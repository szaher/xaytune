# xaytune Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the xaytune Python package with project scaffolding, the base registry pattern (powering all `@register_*` decorators), and the full config system (schema, YAML parsing, inheritance, CLI overrides, validation).

**Architecture:** Foundation layer that all other subsystems build on. The registry provides the decorator pattern used by models, data, recipes, eval, and alignment. The config system provides schema definitions, YAML loading with inheritance, CLI overrides, and validation — used by every recipe and the CLI.

**Tech Stack:** Python 3.10+, PyYAML, pydantic (for config validation), pytest, rich (for console output)

---

## Plan Sequence

This is **Plan 1 of 6**:

1. **Foundation** (this plan) — scaffolding, registry, config
2. **Models & Data** — model loading, PEFT, data pipeline, formats
3. **Trainer** — training loop, distributed strategies, callbacks, checkpointing
4. **Recipes** — finetune (full/LoRA/QLoRA) and pretrain recipes
5. **Alignment** — DPO, GRPO, PPO, ORPO, SimPO
6. **Eval, Export & CLI** — evaluation, export, CLI entry point, logging integrations

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `xaytune/__init__.py`
- Create: `.gitignore`
- Create: `.python-version`

- [ ] **Step 1: Initialize git repository**

Run:
```bash
cd /Users/szaher/go/src/github.com/szaher/xaytune
git init
```

Expected: `Initialized empty Git repository`

- [ ] **Step 2: Create .python-version**

```
3.10
```

- [ ] **Step 3: Create .gitignore**

```gitignore
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/
.eggs/
*.egg
.venv/
venv/
env/
.env
.pytest_cache/
.mypy_cache/
.ruff_cache/
htmlcov/
.coverage
*.log
output/
checkpoints/
wandb/
runs/
*.gguf
.DS_Store
```

- [ ] **Step 4: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "xaytune"
version = "0.1.0"
description = "An opinionated LLM training and fine-tuning library"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.10"
authors = [
    { name = "szaher" },
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "torch>=2.0",
    "transformers>=4.40",
    "peft>=0.10",
    "bitsandbytes>=0.43",
    "datasets>=2.18",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "rich>=13.0",
]

[project.optional-dependencies]
deepspeed = ["deepspeed>=0.14"]
eval = ["lm-eval>=0.4"]
wandb = ["wandb>=0.16"]
mlflow = ["mlflow>=2.10"]
all = [
    "xaytune[deepspeed]",
    "xaytune[eval]",
    "xaytune[wandb]",
    "xaytune[mlflow]",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
]

[project.scripts]
xaytune = "xaytune.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

- [ ] **Step 5: Create xaytune/__init__.py**

```python
"""xaytune — An opinionated LLM training and fine-tuning library."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Create empty sub-package __init__.py files**

Create all necessary `__init__.py` files so the package structure exists:

```
xaytune/config/__init__.py          (empty)
xaytune/models/__init__.py          (empty)
xaytune/data/__init__.py            (empty)
xaytune/trainer/__init__.py         (empty)
xaytune/recipes/__init__.py         (empty)
xaytune/recipes/align/__init__.py   (empty)
xaytune/eval/__init__.py            (empty)
xaytune/export/__init__.py          (empty)
xaytune/logging/__init__.py         (empty)
xaytune/utils/__init__.py           (empty)
```

- [ ] **Step 7: Create virtual environment and verify install**

Run:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -c "import xaytune; print(xaytune.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 8: Commit**

```bash
git add .gitignore .python-version pyproject.toml xaytune/
git commit -m "feat: initial project scaffolding"
```

---

### Task 2: Base Registry Pattern

**Files:**
- Create: `xaytune/utils/registry.py`
- Create: `tests/test_utils/__init__.py`
- Create: `tests/test_utils/test_registry.py`

The registry is a generic class that powers all `@register_*` decorators. It stores name→object mappings and provides decorator registration, lookup, and listing.

- [ ] **Step 1: Write the failing tests**

Create `tests/__init__.py` (empty) and `tests/test_utils/__init__.py` (empty).

Create `tests/test_utils/test_registry.py`:

```python
import pytest
from xaytune.utils.registry import Registry


class TestRegistry:
    def setup_method(self):
        self.registry = Registry("test")

    def test_register_and_get(self):
        @self.registry.register("my-item")
        def my_func():
            return 42

        assert self.registry.get("my-item") is my_func

    def test_register_returns_original(self):
        @self.registry.register("item")
        def my_func():
            return 1

        assert my_func() == 1

    def test_get_missing_raises(self):
        with pytest.raises(KeyError, match="not found in test registry"):
            self.registry.get("nonexistent")

    def test_list_registered(self):
        @self.registry.register("a")
        def func_a():
            pass

        @self.registry.register("b")
        def func_b():
            pass

        assert self.registry.list() == ["a", "b"]

    def test_duplicate_raises(self):
        @self.registry.register("dup")
        def first():
            pass

        with pytest.raises(ValueError, match="already registered in test"):
            @self.registry.register("dup")
            def second():
                pass

    def test_register_class(self):
        @self.registry.register("my-class")
        class MyClass:
            pass

        assert self.registry.get("my-class") is MyClass

    def test_has(self):
        @self.registry.register("exists")
        def func():
            pass

        assert self.registry.has("exists") is True
        assert self.registry.has("missing") is False

    def test_register_with_override(self):
        @self.registry.register("item")
        def first():
            return 1

        @self.registry.register("item", override=True)
        def second():
            return 2

        assert self.registry.get("item") is second
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_utils/test_registry.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'xaytune.utils.registry'`

- [ ] **Step 3: Implement the Registry class**

Create `xaytune/utils/registry.py`:

```python
from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    """A generic registry that maps string names to objects (functions, classes, etc.)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Any] = {}

    def register(self, key: str, override: bool = False) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if key in self._items and not override:
                raise ValueError(
                    f"'{key}' is already registered in {self.name}. "
                    f"Use override=True to replace it."
                )
            self._items[key] = obj
            return obj

        return decorator

    def get(self, key: str) -> Any:
        if key not in self._items:
            raise KeyError(
                f"'{key}' not found in {self.name} registry. "
                f"Available: {', '.join(self._items) or '(none)'}"
            )
        return self._items[key]

    def has(self, key: str) -> bool:
        return key in self._items

    def list(self) -> list[str]:
        return sorted(self._items.keys())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_utils/test_registry.py -v`

Expected: All 9 tests PASS

- [ ] **Step 5: Export from utils __init__**

Update `xaytune/utils/__init__.py`:

```python
from xaytune.utils.registry import Registry

__all__ = ["Registry"]
```

- [ ] **Step 6: Commit**

```bash
git add xaytune/utils/ tests/
git commit -m "feat: add base Registry class for decorator pattern"
```

---

### Task 3: Config Schema Definitions

**Files:**
- Create: `xaytune/config/schema.py`
- Create: `tests/test_config/__init__.py`
- Create: `tests/test_config/test_schema.py`

Config schemas are pydantic models that define the structure and defaults for all configuration. Every config section gets its own model. A top-level `TrainConfig` composes them all.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config/__init__.py` (empty).

Create `tests/test_config/test_schema.py`:

```python
import pytest
from xaytune.config.schema import (
    DataConfig,
    EvalConfig,
    LoggingConfig,
    LoraConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainerConfig,
)


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(name="meta-llama/Llama-3.1-8B")
        assert cfg.name == "meta-llama/Llama-3.1-8B"
        assert cfg.quantization is None
        assert cfg.dtype == "auto"
        assert cfg.trust_remote_code is False

    def test_with_quantization(self):
        cfg = ModelConfig(name="my-model", quantization="4bit")
        assert cfg.quantization == "4bit"

    def test_invalid_quantization(self):
        with pytest.raises(ValueError):
            ModelConfig(name="my-model", quantization="3bit")


class TestLoraConfig:
    def test_defaults(self):
        cfg = LoraConfig()
        assert cfg.rank == 16
        assert cfg.alpha == 32
        assert cfg.dropout == 0.05
        assert cfg.target_modules == "auto"

    def test_custom(self):
        cfg = LoraConfig(rank=64, alpha=128, target_modules=["q_proj", "v_proj"])
        assert cfg.rank == 64
        assert cfg.target_modules == ["q_proj", "v_proj"]


class TestDataConfig:
    def test_minimal(self):
        cfg = DataConfig(path="data.jsonl", format="alpaca")
        assert cfg.path == "data.jsonl"
        assert cfg.format == "alpaca"
        assert cfg.eval_split == 0.0
        assert cfg.packing is True
        assert cfg.max_seq_length == 2048

    def test_streaming(self):
        cfg = DataConfig(path="corpus/", format="text", streaming=True)
        assert cfg.streaming is True


class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.strategy == "auto"
        assert cfg.mixed_precision == "bf16"
        assert cfg.batch_size == 4
        assert cfg.gradient_accumulation == 1
        assert cfg.learning_rate == 2e-4
        assert cfg.num_epochs == 3
        assert cfg.max_steps == -1
        assert cfg.warmup_steps == 0
        assert cfg.weight_decay == 0.01
        assert cfg.max_grad_norm == 1.0
        assert cfg.seed == 42

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            TrainerConfig(strategy="invalid")

    def test_checkpoint_defaults(self):
        cfg = TrainerConfig()
        assert cfg.checkpoint_every_n_steps == 500
        assert cfg.save_last is True


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.every_n_steps == 500
        assert cfg.metrics == ["loss", "perplexity"]
        assert cfg.benchmarks == []


class TestLoggingConfig:
    def test_defaults(self):
        cfg = LoggingConfig()
        assert cfg.backends == ["console"]
        assert cfg.project is None
        assert cfg.log_every_n_steps == 10

    def test_with_wandb(self):
        cfg = LoggingConfig(backends=["console", "wandb"], project="my-run")
        assert "wandb" in cfg.backends


class TestOutputConfig:
    def test_defaults(self):
        cfg = OutputConfig()
        assert cfg.dir == "output"
        assert cfg.merge_on_complete is False

    def test_custom(self):
        cfg = OutputConfig(dir="my-output", merge_on_complete=True)
        assert cfg.dir == "my-output"


class TestTrainConfig:
    def test_minimal(self):
        cfg = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="my-model"),
            data=DataConfig(path="data.jsonl", format="alpaca"),
        )
        assert cfg.recipe == "finetune"
        assert cfg.method == "full"
        assert cfg.model.name == "my-model"
        assert cfg.trainer.batch_size == 4
        assert cfg.output.dir == "output"

    def test_full_config(self):
        cfg = TrainConfig(
            recipe="finetune",
            method="lora",
            model=ModelConfig(name="my-model", quantization="4bit"),
            data=DataConfig(path="data.jsonl", format="alpaca"),
            lora=LoraConfig(rank=32),
            trainer=TrainerConfig(num_epochs=5, strategy="ddp"),
            eval=EvalConfig(every_n_steps=100),
            logging=LoggingConfig(backends=["console", "wandb"]),
            output=OutputConfig(dir="my-output"),
        )
        assert cfg.method == "lora"
        assert cfg.lora.rank == 32
        assert cfg.trainer.num_epochs == 5

    def test_invalid_recipe(self):
        with pytest.raises(ValueError):
            TrainConfig(
                recipe="invalid",
                model=ModelConfig(name="m"),
                data=DataConfig(path="d", format="alpaca"),
            )

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            TrainConfig(
                recipe="finetune",
                method="invalid",
                model=ModelConfig(name="m"),
                data=DataConfig(path="d", format="alpaca"),
            )

    def test_to_dict_roundtrip(self):
        cfg = TrainConfig(
            recipe="finetune",
            model=ModelConfig(name="my-model"),
            data=DataConfig(path="data.jsonl", format="alpaca"),
        )
        d = cfg.model_dump()
        cfg2 = TrainConfig(**d)
        assert cfg2.recipe == cfg.recipe
        assert cfg2.model.name == cfg.model.name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config/test_schema.py -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement config schemas**

Create `xaytune/config/schema.py`:

```python
from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, field_validator


class ModelConfig(BaseModel):
    name: str
    quantization: Literal["4bit", "8bit"] | None = None
    dtype: str = "auto"
    trust_remote_code: bool = False

    @field_validator("quantization")
    @classmethod
    def validate_quantization(cls, v: str | None) -> str | None:
        if v is not None and v not in ("4bit", "8bit"):
            raise ValueError(f"quantization must be '4bit' or '8bit', got '{v}'")
        return v


class LoraConfig(BaseModel):
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Union[str, list[str]] = "auto"


class DataConfig(BaseModel):
    path: str
    format: str
    source: Literal["local", "huggingface"] = "local"
    eval_split: float = 0.0
    eval_path: str | None = None
    packing: bool = True
    max_seq_length: int = 2048
    streaming: bool = False


class TrainerConfig(BaseModel):
    strategy: Literal["auto", "ddp", "fsdp", "deepspeed"] = "auto"
    mixed_precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    batch_size: int = 4
    gradient_accumulation: int = 1
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    checkpoint_every_n_steps: int = 500
    save_last: bool = True

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid = {"auto", "ddp", "fsdp", "deepspeed"}
        if v not in valid:
            raise ValueError(f"strategy must be one of {valid}, got '{v}'")
        return v


class EvalConfig(BaseModel):
    every_n_steps: int = 500
    metrics: list[str] = ["loss", "perplexity"]
    benchmarks: list[str] = []


class LoggingConfig(BaseModel):
    backends: list[str] = ["console"]
    project: str | None = None
    run_name: str | None = None
    log_every_n_steps: int = 10


class OutputConfig(BaseModel):
    dir: str = "output"
    merge_on_complete: bool = False


class TrainConfig(BaseModel):
    recipe: Literal["finetune", "pretrain", "align"]
    method: str = "full"
    base: str | None = None
    model: ModelConfig
    data: DataConfig
    lora: LoraConfig = LoraConfig()
    trainer: TrainerConfig = TrainerConfig()
    eval: EvalConfig = EvalConfig()
    logging: LoggingConfig = LoggingConfig()
    output: OutputConfig = OutputConfig()

    @field_validator("recipe")
    @classmethod
    def validate_recipe(cls, v: str) -> str:
        valid = {"finetune", "pretrain", "align"}
        if v not in valid:
            raise ValueError(f"recipe must be one of {valid}, got '{v}'")
        return v

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        valid = {"full", "lora", "qlora", "dpo", "grpo", "ppo", "orpo", "simpo"}
        if v not in valid:
            raise ValueError(f"method must be one of {valid}, got '{v}'")
        return v
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config/test_schema.py -v`

Expected: All 16 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/config/schema.py tests/test_config/
git commit -m "feat: add config schema definitions with pydantic models"
```

---

### Task 4: Config Parser (YAML Loading + Inheritance + CLI Overrides)

**Files:**
- Create: `xaytune/config/parser.py`
- Create: `tests/test_config/test_parser.py`
- Create: `tests/test_config/fixtures/` (test YAML files)

The parser loads YAML configs, resolves `base:` inheritance chains, applies CLI overrides (dot-notation), and returns a validated `TrainConfig`.

- [ ] **Step 1: Create test fixture YAML files**

Create `tests/test_config/fixtures/base_lora.yaml`:

```yaml
recipe: finetune
method: lora

model:
  quantization: 4bit

lora:
  rank: 16
  alpha: 32

trainer:
  strategy: ddp
  mixed_precision: bf16
  learning_rate: 2e-4
  num_epochs: 3
```

Create `tests/test_config/fixtures/child_config.yaml`:

```yaml
base: base_lora.yaml

model:
  name: meta-llama/Llama-3.1-8B

data:
  path: data/train.jsonl
  format: alpaca

trainer:
  num_epochs: 5
```

Create `tests/test_config/fixtures/full_config.yaml`:

```yaml
recipe: finetune
method: lora

model:
  name: my-model
  quantization: 4bit

data:
  path: data.jsonl
  format: alpaca
  eval_split: 0.05

lora:
  rank: 32

trainer:
  batch_size: 8
  num_epochs: 3
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_config/test_parser.py`:

```python
import os
from pathlib import Path

import pytest
import yaml

from xaytune.config.parser import load_config, merge_dicts, apply_overrides

FIXTURES = Path(__file__).parent / "fixtures"


class TestMergeDicts:
    def test_shallow_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_dicts(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge(self):
        base = {"model": {"name": "a", "quantization": "4bit"}}
        override = {"model": {"name": "b"}}
        result = merge_dicts(base, override)
        assert result == {"model": {"name": "b", "quantization": "4bit"}}

    def test_override_does_not_mutate_base(self):
        base = {"model": {"name": "a"}}
        override = {"model": {"name": "b"}}
        merge_dicts(base, override)
        assert base["model"]["name"] == "a"


class TestApplyOverrides:
    def test_dot_notation(self):
        data = {"model": {"name": "a"}, "trainer": {"batch_size": 4}}
        overrides = ["model.name=b", "trainer.batch_size=8"]
        result = apply_overrides(data, overrides)
        assert result["model"]["name"] == "b"
        assert result["trainer"]["batch_size"] == 8

    def test_nested_creation(self):
        data = {}
        overrides = ["model.name=my-model"]
        result = apply_overrides(data, overrides)
        assert result["model"]["name"] == "my-model"

    def test_boolean_parsing(self):
        data = {}
        overrides = ["model.trust_remote_code=true"]
        result = apply_overrides(data, overrides)
        assert result["model"]["trust_remote_code"] is True

    def test_numeric_parsing(self):
        data = {}
        overrides = ["trainer.learning_rate=1e-5", "trainer.batch_size=16"]
        result = apply_overrides(data, overrides)
        assert result["trainer"]["learning_rate"] == 1e-5
        assert result["trainer"]["batch_size"] == 16


class TestLoadConfig:
    def test_load_full_config(self):
        cfg = load_config(str(FIXTURES / "full_config.yaml"))
        assert cfg.recipe == "finetune"
        assert cfg.method == "lora"
        assert cfg.model.name == "my-model"
        assert cfg.model.quantization == "4bit"
        assert cfg.data.path == "data.jsonl"
        assert cfg.lora.rank == 32
        assert cfg.trainer.batch_size == 8

    def test_load_with_inheritance(self):
        cfg = load_config(str(FIXTURES / "child_config.yaml"))
        assert cfg.model.name == "meta-llama/Llama-3.1-8B"
        assert cfg.model.quantization == "4bit"  # inherited from base
        assert cfg.lora.rank == 16  # inherited from base
        assert cfg.trainer.num_epochs == 5  # overridden in child
        assert cfg.trainer.learning_rate == 2e-4  # inherited from base

    def test_load_with_cli_overrides(self):
        cfg = load_config(
            str(FIXTURES / "full_config.yaml"),
            overrides=["model.name=different-model", "trainer.num_epochs=10"],
        )
        assert cfg.model.name == "different-model"
        assert cfg.trainer.num_epochs == 10
        assert cfg.lora.rank == 32  # unchanged

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_resolved_config_is_serializable(self):
        cfg = load_config(str(FIXTURES / "full_config.yaml"))
        d = cfg.model_dump()
        assert isinstance(d, dict)
        assert d["model"]["name"] == "my-model"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_config/test_parser.py -v`

Expected: FAIL — `ImportError`

- [ ] **Step 4: Implement the config parser**

Create `xaytune/config/parser.py`:

```python
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from xaytune.config.schema import TrainConfig


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_value(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null" or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(data)
    for override in overrides:
        key, _, value = override.partition("=")
        parts = key.split(".")
        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = _parse_value(value)
    return result


def _resolve_inheritance(data: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    base_path = data.pop("base", None)
    if base_path is None:
        return data

    full_base_path = config_dir / base_path
    if not full_base_path.exists():
        raise FileNotFoundError(f"Base config not found: {full_base_path}")

    with open(full_base_path) as f:
        base_data = yaml.safe_load(f)

    base_data = _resolve_inheritance(base_data, full_base_path.parent)
    return merge_dicts(base_data, data)


def load_config(
    path: str,
    overrides: list[str] | None = None,
) -> TrainConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    data = _resolve_inheritance(data, config_path.parent)

    if overrides:
        data = apply_overrides(data, overrides)

    return TrainConfig(**data)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config/test_parser.py -v`

Expected: All 10 tests PASS

- [ ] **Step 6: Commit**

```bash
git add xaytune/config/parser.py tests/test_config/
git commit -m "feat: add config parser with YAML loading, inheritance, and CLI overrides"
```

---

### Task 5: Config Validation & Helpful Errors

**Files:**
- Create: `xaytune/config/validation.py`
- Create: `tests/test_config/test_validation.py`

Validation goes beyond pydantic type-checking. It catches logical errors: LoRA config without lora method, QLoRA without 4-bit quantization, eval split > 1.0, etc. Each error includes a suggestion for how to fix it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config/test_validation.py`:

```python
import pytest
from xaytune.config.schema import (
    DataConfig,
    LoraConfig,
    ModelConfig,
    TrainConfig,
    TrainerConfig,
)
from xaytune.config.validation import validate_config, ConfigValidationError


class TestValidateConfig:
    def _make_config(self, **kwargs) -> TrainConfig:
        defaults = {
            "recipe": "finetune",
            "model": ModelConfig(name="test-model"),
            "data": DataConfig(path="data.jsonl", format="alpaca"),
        }
        defaults.update(kwargs)
        return TrainConfig(**defaults)

    def test_valid_config_passes(self):
        cfg = self._make_config(method="lora")
        validate_config(cfg)  # should not raise

    def test_qlora_without_4bit(self):
        cfg = self._make_config(
            method="qlora",
            model=ModelConfig(name="m", quantization=None),
        )
        with pytest.raises(ConfigValidationError, match="4bit quantization"):
            validate_config(cfg)

    def test_qlora_with_8bit(self):
        cfg = self._make_config(
            method="qlora",
            model=ModelConfig(name="m", quantization="8bit"),
        )
        with pytest.raises(ConfigValidationError, match="4bit quantization"):
            validate_config(cfg)

    def test_qlora_with_4bit_passes(self):
        cfg = self._make_config(
            method="qlora",
            model=ModelConfig(name="m", quantization="4bit"),
        )
        validate_config(cfg)  # should not raise

    def test_eval_split_too_large(self):
        cfg = self._make_config(
            data=DataConfig(path="d", format="alpaca", eval_split=1.5),
        )
        with pytest.raises(ConfigValidationError, match="eval_split"):
            validate_config(cfg)

    def test_eval_split_negative(self):
        cfg = self._make_config(
            data=DataConfig(path="d", format="alpaca", eval_split=-0.1),
        )
        with pytest.raises(ConfigValidationError, match="eval_split"):
            validate_config(cfg)

    def test_batch_size_zero(self):
        cfg = self._make_config(
            trainer=TrainerConfig(batch_size=0),
        )
        with pytest.raises(ConfigValidationError, match="batch_size"):
            validate_config(cfg)

    def test_learning_rate_negative(self):
        cfg = self._make_config(
            trainer=TrainerConfig(learning_rate=-1e-4),
        )
        with pytest.raises(ConfigValidationError, match="learning_rate"):
            validate_config(cfg)

    def test_align_recipe_requires_align_method(self):
        cfg = self._make_config(recipe="align", method="lora")
        with pytest.raises(ConfigValidationError, match="alignment method"):
            validate_config(cfg)

    def test_align_recipe_with_dpo_passes(self):
        cfg = self._make_config(recipe="align", method="dpo")
        validate_config(cfg)

    def test_finetune_with_align_method(self):
        cfg = self._make_config(recipe="finetune", method="dpo")
        with pytest.raises(ConfigValidationError, match="fine-tuning method"):
            validate_config(cfg)

    def test_error_includes_suggestion(self):
        cfg = self._make_config(
            method="qlora",
            model=ModelConfig(name="m"),
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(cfg)
        assert "suggestion" in str(exc_info.value).lower() or "set" in str(exc_info.value).lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config/test_validation.py -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement validation**

Create `xaytune/config/validation.py`:

```python
from __future__ import annotations

from xaytune.config.schema import TrainConfig


class ConfigValidationError(Exception):
    pass


_FINETUNE_METHODS = {"full", "lora", "qlora"}
_ALIGN_METHODS = {"dpo", "grpo", "ppo", "orpo", "simpo"}


def validate_config(config: TrainConfig) -> None:
    errors: list[str] = []

    if config.method == "qlora" and config.model.quantization != "4bit":
        errors.append(
            "QLoRA requires 4bit quantization, but model.quantization="
            f"'{config.model.quantization}'. Suggestion: set model.quantization='4bit'."
        )

    if not 0.0 <= config.data.eval_split <= 1.0:
        errors.append(
            f"data.eval_split must be between 0.0 and 1.0, got {config.data.eval_split}. "
            "Suggestion: set eval_split to a value like 0.05 for a 5% eval split."
        )

    if config.trainer.batch_size < 1:
        errors.append(
            f"trainer.batch_size must be >= 1, got {config.trainer.batch_size}. "
            "Suggestion: set batch_size to at least 1."
        )

    if config.trainer.learning_rate <= 0:
        errors.append(
            f"trainer.learning_rate must be positive, got {config.trainer.learning_rate}. "
            "Suggestion: typical values are 1e-5 to 5e-4."
        )

    if config.recipe == "align" and config.method not in _ALIGN_METHODS:
        errors.append(
            f"Recipe 'align' requires an alignment method "
            f"({', '.join(sorted(_ALIGN_METHODS))}), got '{config.method}'. "
            "Suggestion: set method='dpo' or method='grpo'."
        )

    if config.recipe == "finetune" and config.method not in _FINETUNE_METHODS:
        errors.append(
            f"Recipe 'finetune' requires a fine-tuning method "
            f"({', '.join(sorted(_FINETUNE_METHODS))}), got '{config.method}'. "
            "Suggestion: set method='lora' or method='full'."
        )

    if errors:
        raise ConfigValidationError(
            f"Config validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config/test_validation.py -v`

Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/config/validation.py tests/test_config/test_validation.py
git commit -m "feat: add config validation with helpful error messages"
```

---

### Task 6: Config Public API & Default Configs

**Files:**
- Modify: `xaytune/config/__init__.py`
- Create: `xaytune/config/defaults/lora.yaml`
- Create: `xaytune/config/defaults/qlora.yaml`
- Create: `xaytune/config/defaults/full_finetune.yaml`
- Create: `xaytune/config/defaults/pretrain.yaml`
- Create: `tests/test_config/test_defaults.py`

Wire up the config package public API and provide built-in default configs that users can extend via `base:`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config/test_defaults.py`:

```python
from pathlib import Path

import pytest

from xaytune.config import load_config, validate_config, get_defaults_dir
from xaytune.config.schema import TrainConfig


class TestDefaults:
    def test_defaults_dir_exists(self):
        d = get_defaults_dir()
        assert d.is_dir()

    def test_lora_default_exists(self):
        d = get_defaults_dir()
        assert (d / "lora.yaml").exists()

    def test_qlora_default_exists(self):
        d = get_defaults_dir()
        assert (d / "qlora.yaml").exists()

    def test_full_finetune_default_exists(self):
        d = get_defaults_dir()
        assert (d / "full_finetune.yaml").exists()

    def test_pretrain_default_exists(self):
        d = get_defaults_dir()
        assert (d / "pretrain.yaml").exists()


class TestConfigPublicAPI:
    def test_load_config_importable(self):
        from xaytune.config import load_config
        assert callable(load_config)

    def test_validate_config_importable(self):
        from xaytune.config import validate_config
        assert callable(validate_config)

    def test_schema_classes_importable(self):
        from xaytune.config import TrainConfig, ModelConfig, DataConfig
        assert TrainConfig is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config/test_defaults.py -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Create default config files**

Create `xaytune/config/defaults/lora.yaml`:

```yaml
recipe: finetune
method: lora

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: auto

trainer:
  strategy: ddp
  mixed_precision: bf16
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2e-4
  num_epochs: 3
  weight_decay: 0.01

eval:
  every_n_steps: 500
  metrics: [loss, perplexity]
```

Create `xaytune/config/defaults/qlora.yaml`:

```yaml
recipe: finetune
method: qlora

model:
  quantization: 4bit

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: auto

trainer:
  strategy: ddp
  mixed_precision: bf16
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2e-4
  num_epochs: 3
  weight_decay: 0.01

eval:
  every_n_steps: 500
  metrics: [loss, perplexity]
```

Create `xaytune/config/defaults/full_finetune.yaml`:

```yaml
recipe: finetune
method: full

trainer:
  strategy: fsdp
  mixed_precision: bf16
  batch_size: 2
  gradient_accumulation: 8
  learning_rate: 5e-5
  num_epochs: 3
  weight_decay: 0.01

eval:
  every_n_steps: 500
  metrics: [loss, perplexity]
```

Create `xaytune/config/defaults/pretrain.yaml`:

```yaml
recipe: pretrain
method: full

data:
  packing: true
  streaming: true
  max_seq_length: 4096

trainer:
  strategy: fsdp
  mixed_precision: bf16
  batch_size: 2
  gradient_accumulation: 16
  learning_rate: 3e-4
  max_steps: 100000
  warmup_steps: 2000
  weight_decay: 0.1
  max_grad_norm: 1.0

eval:
  every_n_steps: 1000
  metrics: [loss, perplexity]
```

- [ ] **Step 4: Wire up config __init__.py**

Update `xaytune/config/__init__.py`:

```python
from pathlib import Path

from xaytune.config.parser import load_config, merge_dicts, apply_overrides
from xaytune.config.schema import (
    DataConfig,
    EvalConfig,
    LoggingConfig,
    LoraConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainerConfig,
)
from xaytune.config.validation import ConfigValidationError, validate_config


def get_defaults_dir() -> Path:
    return Path(__file__).parent / "defaults"


__all__ = [
    "apply_overrides",
    "ConfigValidationError",
    "DataConfig",
    "EvalConfig",
    "get_defaults_dir",
    "load_config",
    "LoggingConfig",
    "LoraConfig",
    "merge_dicts",
    "ModelConfig",
    "OutputConfig",
    "TrainConfig",
    "TrainerConfig",
    "validate_config",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config/test_defaults.py -v`

Expected: All 8 tests PASS

- [ ] **Step 6: Run all tests to verify nothing broke**

Run: `pytest tests/ -v`

Expected: All tests PASS (registry: 9, schema: 16, parser: 10, validation: 12, defaults: 8 = 55 total)

- [ ] **Step 7: Commit**

```bash
git add xaytune/config/ tests/test_config/
git commit -m "feat: add config public API and built-in default configs"
```

---

### Task 7: CLI Skeleton

**Files:**
- Create: `xaytune/cli.py`
- Create: `tests/test_cli.py`

A minimal CLI entry point using argparse. For now it only supports `xaytune train --config <file>` which loads, validates, and prints the resolved config. This establishes the CLI pattern that later plans will extend with `eval`, `export`, and `list` commands.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "test_config" / "fixtures"


class TestCLI:
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "xaytune.cli", *args],
            capture_output=True,
            text=True,
        )

    def test_no_args_shows_help(self):
        result = self._run()
        assert result.returncode != 0 or "usage" in result.stdout.lower() or "usage" in result.stderr.lower()

    def test_train_requires_config(self):
        result = self._run("train")
        assert result.returncode != 0

    def test_train_with_config(self):
        result = self._run("train", "--config", str(FIXTURES / "full_config.yaml"), "--dry-run")
        assert result.returncode == 0
        assert "my-model" in result.stdout

    def test_train_with_overrides(self):
        result = self._run(
            "train",
            "--config", str(FIXTURES / "full_config.yaml"),
            "--dry-run",
            "--override", "model.name=overridden",
        )
        assert result.returncode == 0
        assert "overridden" in result.stdout

    def test_train_invalid_config(self):
        result = self._run("train", "--config", "nonexistent.yaml")
        assert result.returncode != 0

    def test_version(self):
        result = self._run("--version")
        assert "0.1.0" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`

Expected: FAIL

- [ ] **Step 3: Implement the CLI**

Create `xaytune/cli.py`:

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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "train":
        return _handle_train(args)

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

    print(f"Training with recipe={config.recipe}, method={config.method}")
    print(f"Model: {config.model.name}")
    print("Training loop not yet implemented — use --dry-run to validate config.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`

Expected: All 6 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`

Expected: All tests PASS (61 total)

- [ ] **Step 6: Commit**

```bash
git add xaytune/cli.py tests/test_cli.py
git commit -m "feat: add CLI skeleton with train command and dry-run support"
```

---

## Self-Review

**Spec coverage:** This plan covers spec sections 1 (architecture scaffolding), 8 (config system — schema, parsing, inheritance, overrides, validation), 9 (decorator pattern via Registry, discoverability via CLI `list`), 10 (project structure), and packaging. Sections 2-7 and 11-12 are covered by Plans 2-6.

**Placeholder scan:** No TBDs, TODOs, or incomplete sections. All code is complete and runnable.

**Type consistency:** `Registry`, `TrainConfig`, `load_config`, `validate_config`, `ConfigValidationError` — names are consistent across all tasks. `merge_dicts` and `apply_overrides` are used in both parser tests and implementation.

---

## Remaining Plans (to be detailed when this plan is complete)

**Plan 2: Models & Data** — `xaytune/models/` (loader, peft, registry) + `xaytune/data/` (formats, packing, preferences, registry). Produces: working model loading with quantization/LoRA, dataset loading with all built-in formats.

**Plan 3: Trainer** — `xaytune/trainer/` (loop, distributed, callbacks, checkpointing) + `xaytune/logging/` (console, integrations). Produces: working training loop with DDP/FSDP support, callbacks, rich progress output.

**Plan 4: Recipes** — `xaytune/recipes/` (base, pretrain, finetune). Produces: end-to-end training with `xaytune.finetune()` and `xaytune.pretrain()`.

**Plan 5: Alignment** — `xaytune/recipes/align/` (base, dpo, grpo, ppo). Produces: working alignment training with all RL methods.

**Plan 6: Eval, Export & CLI** — `xaytune/eval/` + `xaytune/export/` + CLI commands. Produces: evaluation with benchmarks, LoRA merging, GGUF export, Hub push, full CLI.
