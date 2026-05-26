# xaytune Recipes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the recipe layer — opinionated workflows that wire building blocks (models, data, trainer) together with sensible defaults. Finetune (full/LoRA/QLoRA), pretrain, and the top-level Python API (`xaytune.finetune(...)`, `xaytune.pretrain(...)`).

**Architecture:** Each recipe is a function registered via the recipe registry. Recipes accept a `TrainConfig` or keyword arguments, set up the full pipeline (model → PEFT → data → dataloader → trainer), and return the `TrainState`. A shared `_setup_training` helper handles common wiring to avoid duplication. The top-level `xaytune` package exposes `finetune()` and `pretrain()` as convenience functions.

**Tech Stack:** PyTorch, pytest, unittest.mock (all tests mock heavy dependencies)

---

## Plan Sequence

This is **Plan 4 of 6** — depends on Plans 1-3 being complete.

---

### Task 1: Recipe Base — Shared Setup Helper

**Files:**
- Create: `xaytune/recipes/base.py`
- Create: `tests/test_recipes/__init__.py`
- Create: `tests/test_recipes/test_base.py`

A shared `_setup_training` function that all recipes use. It handles model loading, optional PEFT, data loading, DataLoader creation, and Trainer instantiation. This is NOT a base class — it's a plain function that returns a named tuple of components.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_recipes/__init__.py` (empty).

Create `tests/test_recipes/test_base.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from xaytune.recipes.base import setup_training, TrainingComponents
from xaytune.config.schema import TrainConfig, ModelConfig, DataConfig, TrainerConfig, LoraConfig


class TestTrainingComponents:
    def test_is_namedtuple(self):
        tc = TrainingComponents(
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataloader=MagicMock(),
            eval_dataloader=None,
            trainer=MagicMock(),
        )
        assert tc.model is not None
        assert tc.tokenizer is not None
        assert tc.trainer is not None
        assert tc.eval_dataloader is None

    def test_fields(self):
        fields = TrainingComponents._fields
        assert "model" in fields
        assert "tokenizer" in fields
        assert "train_dataloader" in fields
        assert "eval_dataloader" in fields
        assert "trainer" in fields


class TestSetupTraining:
    def _make_config(self, method="full", **trainer_kwargs):
        return TrainConfig(
            recipe="finetune",
            method=method,
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="data.jsonl", format="alpaca"),
            trainer=TrainerConfig(**trainer_kwargs),
        )

    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.DataLoader")
    def test_full_finetune_setup(self, mock_dl_cls, mock_load_ds, mock_load_model):
        mock_model_result = MagicMock()
        mock_model_result.model = MagicMock()
        mock_model_result.tokenizer = MagicMock()
        mock_load_model.return_value = mock_model_result
        mock_load_ds.return_value = [{"input_ids": [1, 2, 3]}]
        mock_dl_cls.return_value = MagicMock()

        config = self._make_config(method="full")
        components = setup_training(config)

        mock_load_model.assert_called_once_with(
            "test-model",
            quantization=None,
            dtype="auto",
            trust_remote_code=False,
        )
        assert components.model is mock_model_result.model
        assert components.tokenizer is mock_model_result.tokenizer
        assert components.trainer is not None

    @patch("xaytune.recipes.base.apply_lora")
    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.DataLoader")
    def test_lora_setup_applies_peft(self, mock_dl_cls, mock_load_ds, mock_load_model, mock_apply_lora):
        mock_model_result = MagicMock()
        mock_model_result.model = MagicMock()
        mock_model_result.tokenizer = MagicMock()
        mock_load_model.return_value = mock_model_result

        lora_result = MagicMock()
        lora_result.model = MagicMock()
        lora_result.tokenizer = mock_model_result.tokenizer
        mock_apply_lora.return_value = lora_result

        mock_load_ds.return_value = [{"input_ids": [1]}]
        mock_dl_cls.return_value = MagicMock()

        config = self._make_config(method="lora")
        components = setup_training(config)

        mock_apply_lora.assert_called_once()
        assert components.model is lora_result.model

    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.DataLoader")
    def test_qlora_uses_4bit_quantization(self, mock_dl_cls, mock_load_ds, mock_load_model):
        mock_model_result = MagicMock()
        mock_model_result.model = MagicMock()
        mock_model_result.tokenizer = MagicMock()
        mock_load_model.return_value = mock_model_result
        mock_load_ds.return_value = [{"input_ids": [1]}]
        mock_dl_cls.return_value = MagicMock()

        config = self._make_config(method="qlora")
        with patch("xaytune.recipes.base.apply_lora") as mock_apply_lora:
            mock_apply_lora.return_value = mock_model_result
            components = setup_training(config)

        mock_load_model.assert_called_once_with(
            "test-model",
            quantization="4bit",
            dtype="auto",
            trust_remote_code=False,
        )

    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.DataLoader")
    def test_eval_split_creates_eval_dataloader(self, mock_dl_cls, mock_load_ds, mock_load_model):
        mock_model_result = MagicMock()
        mock_model_result.model = MagicMock()
        mock_model_result.tokenizer = MagicMock()
        mock_load_model.return_value = mock_model_result
        mock_load_ds.return_value = ([{"input_ids": [1]}], [{"input_ids": [2]}])
        mock_dl_cls.return_value = MagicMock()

        config = TrainConfig(
            recipe="finetune",
            method="full",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="data.jsonl", format="alpaca", eval_split=0.1),
            trainer=TrainerConfig(),
        )
        components = setup_training(config)

        assert mock_dl_cls.call_count == 2
        assert components.eval_dataloader is not None

    @patch("xaytune.recipes.base.load_model")
    @patch("xaytune.recipes.base.load_dataset")
    @patch("xaytune.recipes.base.DataLoader")
    def test_no_eval_split_no_eval_dataloader(self, mock_dl_cls, mock_load_ds, mock_load_model):
        mock_model_result = MagicMock()
        mock_model_result.model = MagicMock()
        mock_model_result.tokenizer = MagicMock()
        mock_load_model.return_value = mock_model_result
        mock_load_ds.return_value = [{"input_ids": [1]}]
        mock_dl_cls.return_value = MagicMock()

        config = self._make_config(method="full")
        components = setup_training(config)

        assert mock_dl_cls.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_recipes/test_base.py -v`

- [ ] **Step 3: Implement the setup helper**

Create `xaytune/recipes/base.py`:

```python
from __future__ import annotations

from typing import Any, NamedTuple

from torch.utils.data import DataLoader

from xaytune.config.schema import TrainConfig
from xaytune.data import load_dataset
from xaytune.models import load_model, apply_lora
from xaytune.trainer import Trainer, CallbackManager


class TrainingComponents(NamedTuple):
    model: Any
    tokenizer: Any
    train_dataloader: DataLoader
    eval_dataloader: DataLoader | None
    trainer: Trainer


def setup_training(
    config: TrainConfig,
    callback_manager: CallbackManager | None = None,
) -> TrainingComponents:
    quantization = None
    if config.method == "qlora":
        quantization = "4bit"
    elif config.model.quantization:
        quantization = config.model.quantization

    model_result = load_model(
        config.model.name,
        quantization=quantization,
        dtype=config.model.dtype,
        trust_remote_code=config.model.trust_remote_code,
    )

    if config.method in ("lora", "qlora"):
        model_result = apply_lora(
            model_result,
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
        )

    dataset = load_dataset(
        config.data.path,
        format=config.data.format,
        eval_split=config.data.eval_split,
    )

    if config.data.eval_split > 0:
        train_data, eval_data = dataset
    else:
        train_data = dataset
        eval_data = None

    train_dataloader = DataLoader(
        train_data,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )

    eval_dataloader = None
    if eval_data is not None:
        eval_dataloader = DataLoader(
            eval_data,
            batch_size=config.trainer.batch_size,
            shuffle=False,
        )

    trainer = Trainer(
        config=config.trainer,
        callback_manager=callback_manager or CallbackManager(),
    )

    return TrainingComponents(
        model=model_result.model,
        tokenizer=model_result.tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        trainer=trainer,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_recipes/test_base.py -v`

Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/recipes/base.py tests/test_recipes/
git commit -m "feat: add recipe setup helper with model/data/trainer wiring"
```

---

### Task 2: Finetune Recipe

**Files:**
- Create: `xaytune/recipes/finetune.py`
- Create: `tests/test_recipes/test_finetune.py`

The finetune recipe supports full, LoRA, and QLoRA fine-tuning. It uses `setup_training` from base.py and adds finetune-specific logic.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_recipes/test_finetune.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from xaytune.recipes.finetune import finetune
from xaytune.config.schema import TrainConfig, ModelConfig, DataConfig, TrainerConfig
from xaytune.trainer.callbacks import TrainState


class TestFinetune:
    @patch("xaytune.recipes.finetune.setup_training")
    def test_finetune_with_config(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState(global_step=100)
        mock_setup.return_value = mock_components

        config = TrainConfig(
            recipe="finetune",
            method="lora",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="data.jsonl", format="alpaca"),
        )

        state = finetune(config=config)

        mock_setup.assert_called_once()
        mock_components.trainer.train.assert_called_once()
        assert state.global_step == 100

    @patch("xaytune.recipes.finetune.setup_training")
    def test_finetune_with_kwargs(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState(global_step=50)
        mock_setup.return_value = mock_components

        state = finetune(
            model="meta-llama/Llama-3.1-8B",
            dataset="train.jsonl",
            method="lora",
        )

        call_args = mock_setup.call_args
        config = call_args[0][0] if call_args[0] else call_args[1].get("config")
        assert config.model.name == "meta-llama/Llama-3.1-8B"
        assert config.data.path == "train.jsonl"
        assert config.method == "lora"
        assert state.global_step == 50

    @patch("xaytune.recipes.finetune.setup_training")
    def test_finetune_default_method_is_full(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        finetune(model="test-model", dataset="data.jsonl")

        config = mock_setup.call_args[0][0]
        assert config.method == "full"

    @patch("xaytune.recipes.finetune.setup_training")
    def test_finetune_passes_model_and_dataloader_to_trainer(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        finetune(model="test-model", dataset="data.jsonl")

        train_call = mock_components.trainer.train.call_args
        assert train_call.kwargs["model"] is mock_components.model
        assert train_call.kwargs["train_dataloader"] is mock_components.train_dataloader

    @patch("xaytune.recipes.finetune.setup_training")
    def test_finetune_returns_train_state(self, mock_setup):
        expected_state = TrainState(global_step=75, metrics={"loss": 0.3})
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = expected_state
        mock_setup.return_value = mock_components

        result = finetune(model="test-model", dataset="data.jsonl")

        assert isinstance(result, TrainState)
        assert result.global_step == 75
        assert result.metrics["loss"] == 0.3

    @patch("xaytune.recipes.finetune.setup_training")
    def test_finetune_accepts_trainer_kwargs(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        finetune(
            model="test-model",
            dataset="data.jsonl",
            num_epochs=5,
            learning_rate=1e-5,
            batch_size=8,
        )

        config = mock_setup.call_args[0][0]
        assert config.trainer.num_epochs == 5
        assert config.trainer.learning_rate == 1e-5
        assert config.trainer.batch_size == 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_recipes/test_finetune.py -v`

- [ ] **Step 3: Implement finetune recipe**

Create `xaytune/recipes/finetune.py`:

```python
from __future__ import annotations

from typing import Any

from xaytune.config.schema import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    TrainerConfig,
)
from xaytune.recipes.base import setup_training
from xaytune.trainer.callbacks import TrainState


def finetune(
    *,
    config: TrainConfig | None = None,
    model: str | None = None,
    dataset: str | None = None,
    method: str = "full",
    format: str = "alpaca",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    **kwargs: Any,
) -> TrainState:
    if config is None:
        if model is None or dataset is None:
            raise ValueError("Either 'config' or both 'model' and 'dataset' are required.")

        trainer_fields = {}
        trainer_param_names = {f for f in TrainerConfig.model_fields}
        for k, v in list(kwargs.items()):
            if k in trainer_param_names:
                trainer_fields[k] = kwargs.pop(k)

        config = TrainConfig(
            recipe="finetune",
            method=method,
            model=ModelConfig(name=model),
            data=DataConfig(path=dataset, format=format),
            trainer=TrainerConfig(
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                **trainer_fields,
            ),
        )

    components = setup_training(config)

    state = components.trainer.train(
        model=components.model,
        train_dataloader=components.train_dataloader,
    )

    return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_recipes/test_finetune.py -v`

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/recipes/finetune.py tests/test_recipes/test_finetune.py
git commit -m "feat: add finetune recipe supporting full/LoRA/QLoRA methods"
```

---

### Task 3: Pretrain Recipe

**Files:**
- Create: `xaytune/recipes/pretrain.py`
- Create: `tests/test_recipes/test_pretrain.py`

The pretrain recipe handles pre-training and continued pre-training. Uses text format by default, no PEFT.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_recipes/test_pretrain.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from xaytune.recipes.pretrain import pretrain
from xaytune.config.schema import TrainConfig, ModelConfig, DataConfig, TrainerConfig
from xaytune.trainer.callbacks import TrainState


class TestPretrain:
    @patch("xaytune.recipes.pretrain.setup_training")
    def test_pretrain_with_config(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState(global_step=1000)
        mock_setup.return_value = mock_components

        config = TrainConfig(
            recipe="pretrain",
            method="full",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="corpus.jsonl", format="text"),
        )

        state = pretrain(config=config)

        mock_setup.assert_called_once()
        assert state.global_step == 1000

    @patch("xaytune.recipes.pretrain.setup_training")
    def test_pretrain_with_kwargs(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState(global_step=500)
        mock_setup.return_value = mock_components

        state = pretrain(
            model="my-model",
            dataset="corpus.jsonl",
        )

        config = mock_setup.call_args[0][0]
        assert config.model.name == "my-model"
        assert config.data.path == "corpus.jsonl"
        assert config.recipe == "pretrain"
        assert state.global_step == 500

    @patch("xaytune.recipes.pretrain.setup_training")
    def test_pretrain_default_format_is_text(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        pretrain(model="my-model", dataset="corpus.jsonl")

        config = mock_setup.call_args[0][0]
        assert config.data.format == "text"

    @patch("xaytune.recipes.pretrain.setup_training")
    def test_pretrain_method_is_always_full(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        pretrain(model="my-model", dataset="corpus.jsonl")

        config = mock_setup.call_args[0][0]
        assert config.method == "full"

    @patch("xaytune.recipes.pretrain.setup_training")
    def test_pretrain_accepts_trainer_kwargs(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        pretrain(
            model="my-model",
            dataset="corpus.jsonl",
            num_epochs=1,
            learning_rate=3e-4,
            max_steps=10000,
        )

        config = mock_setup.call_args[0][0]
        assert config.trainer.num_epochs == 1
        assert config.trainer.learning_rate == 3e-4
        assert config.trainer.max_steps == 10000

    @patch("xaytune.recipes.pretrain.setup_training")
    def test_pretrain_returns_train_state(self, mock_setup):
        expected_state = TrainState(global_step=200, metrics={"loss": 2.1})
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = expected_state
        mock_setup.return_value = mock_components

        result = pretrain(model="my-model", dataset="corpus.jsonl")

        assert isinstance(result, TrainState)
        assert result.global_step == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_recipes/test_pretrain.py -v`

- [ ] **Step 3: Implement pretrain recipe**

Create `xaytune/recipes/pretrain.py`:

```python
from __future__ import annotations

from typing import Any

from xaytune.config.schema import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    TrainerConfig,
)
from xaytune.recipes.base import setup_training
from xaytune.trainer.callbacks import TrainState


def pretrain(
    *,
    config: TrainConfig | None = None,
    model: str | None = None,
    dataset: str | None = None,
    format: str = "text",
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    batch_size: int = 4,
    **kwargs: Any,
) -> TrainState:
    if config is None:
        if model is None or dataset is None:
            raise ValueError("Either 'config' or both 'model' and 'dataset' are required.")

        trainer_fields = {}
        trainer_param_names = {f for f in TrainerConfig.model_fields}
        for k in list(kwargs.keys()):
            if k in trainer_param_names:
                trainer_fields[k] = kwargs.pop(k)

        config = TrainConfig(
            recipe="pretrain",
            method="full",
            model=ModelConfig(name=model),
            data=DataConfig(path=dataset, format=format),
            trainer=TrainerConfig(
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                **trainer_fields,
            ),
        )

    components = setup_training(config)

    state = components.trainer.train(
        model=components.model,
        train_dataloader=components.train_dataloader,
    )

    return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_recipes/test_pretrain.py -v`

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/recipes/pretrain.py tests/test_recipes/test_pretrain.py
git commit -m "feat: add pretrain recipe for pre-training and continued pre-training"
```

---

### Task 4: Recipe Registry & Top-Level API Wire-Up

**Files:**
- Modify: `xaytune/recipes/__init__.py`
- Modify: `xaytune/__init__.py`
- Create: `tests/test_recipes/test_init.py`
- Create: `tests/test_top_level_api.py`

Register recipes in the recipe registry, wire up `xaytune/recipes/__init__.py` exports, and add `xaytune.finetune()` and `xaytune.pretrain()` to the top-level package API.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_recipes/test_init.py`:

```python
from xaytune.recipes import finetune, pretrain, recipe_registry


class TestRecipesPublicAPI:
    def test_finetune_importable(self):
        assert callable(finetune)

    def test_pretrain_importable(self):
        assert callable(pretrain)

    def test_recipe_registry_has_finetune(self):
        assert recipe_registry.has("finetune")
        assert recipe_registry.get("finetune") is finetune

    def test_recipe_registry_has_pretrain(self):
        assert recipe_registry.has("pretrain")
        assert recipe_registry.get("pretrain") is pretrain

    def test_recipe_registry_list(self):
        recipes = recipe_registry.list()
        assert "finetune" in recipes
        assert "pretrain" in recipes
```

Create `tests/test_top_level_api.py`:

```python
import xaytune


class TestTopLevelAPI:
    def test_version(self):
        assert xaytune.__version__ == "0.1.0"

    def test_finetune_importable(self):
        assert callable(xaytune.finetune)

    def test_pretrain_importable(self):
        assert callable(xaytune.pretrain)

    def test_finetune_is_recipe(self):
        from xaytune.recipes.finetune import finetune
        assert xaytune.finetune is finetune

    def test_pretrain_is_recipe(self):
        from xaytune.recipes.pretrain import pretrain
        assert xaytune.pretrain is pretrain
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_recipes/test_init.py tests/test_top_level_api.py -v`

- [ ] **Step 3: Wire up recipes package and top-level API**

Update `xaytune/recipes/__init__.py`:

```python
from xaytune.utils.registry import Registry

recipe_registry = Registry("recipe")

from xaytune.recipes.finetune import finetune
from xaytune.recipes.pretrain import pretrain

recipe_registry.register("finetune")(finetune)
recipe_registry.register("pretrain")(pretrain)

__all__ = ["finetune", "pretrain", "recipe_registry"]
```

Update `xaytune/__init__.py`:

```python
"""xaytune — An opinionated LLM training and fine-tuning library."""

__version__ = "0.1.0"

from xaytune.recipes.finetune import finetune
from xaytune.recipes.pretrain import pretrain

__all__ = ["__version__", "finetune", "pretrain"]
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`

Expected: All tests PASS (151 existing + ~17 new)

- [ ] **Step 5: Commit**

```bash
git add xaytune/recipes/__init__.py xaytune/__init__.py tests/test_recipes/test_init.py tests/test_top_level_api.py
git commit -m "feat: wire up recipe registry and top-level xaytune.finetune/pretrain API"
```
