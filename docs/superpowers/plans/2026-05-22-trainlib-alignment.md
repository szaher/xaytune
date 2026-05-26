# xaytune Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the alignment recipe layer — DPO and GRPO loss functions, reward registry, the `align()` recipe function, and top-level API wire-up. ORPO, SimPO, and PPO are deferred to a future plan.

**Architecture:** Each alignment method is a loss function registered in a method registry. The `align()` recipe loads a model and preference dataset via `setup_training`, then runs a custom training loop that computes the method-specific loss. A reward registry supports custom reward functions for GRPO. The align recipe is registered in the recipe registry and exposed at `xaytune.align()`.

**Tech Stack:** PyTorch, pytest, unittest.mock

---

## Plan Sequence

This is **Plan 5 of 6** — depends on Plans 1-4 being complete.

---

### Task 1: Reward Registry

**Files:**
- Create: `xaytune/recipes/align/rewards.py`
- Create: `tests/test_recipes/test_align/__init__.py`
- Create: `tests/test_recipes/test_align/test_rewards.py`

A registry for custom reward functions used by GRPO and other RL methods.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_recipes/test_align/__init__.py` (empty).

Create `tests/test_recipes/test_align/test_rewards.py`:

```python
import pytest
from xaytune.recipes.align.rewards import reward_registry, register_reward


class TestRewardRegistry:
    def test_register_and_get(self):
        @register_reward("test-reward")
        def my_reward(prompt: str, response: str) -> float:
            return 1.0

        assert reward_registry.has("test-reward")
        fn = reward_registry.get("test-reward")
        assert fn("hello", "world") == 1.0

    def test_register_returns_original(self):
        @register_reward("identity-test")
        def my_fn(prompt: str, response: str) -> float:
            return 0.5

        assert my_fn("a", "b") == 0.5

    def test_unknown_reward_raises(self):
        with pytest.raises(KeyError, match="not found"):
            reward_registry.get("nonexistent-reward")

    def test_list_rewards(self):
        @register_reward("list-test-reward")
        def r(prompt: str, response: str) -> float:
            return 0.0

        rewards = reward_registry.list()
        assert "list-test-reward" in rewards

    def test_default_reward_registered(self):
        assert reward_registry.has("default")
        fn = reward_registry.get("default")
        assert fn("prompt", "response") == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_recipes/test_align/test_rewards.py -v`

- [ ] **Step 3: Implement reward registry**

Create `xaytune/recipes/align/rewards.py`:

```python
from __future__ import annotations

from typing import Callable

from xaytune.utils.registry import Registry

reward_registry = Registry("reward")

register_reward = reward_registry.register


@register_reward("default")
def default_reward(prompt: str, response: str) -> float:
    return 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_recipes/test_align/test_rewards.py -v`

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/recipes/align/rewards.py tests/test_recipes/test_align/
git commit -m "feat: add reward registry with decorator-based registration"
```

---

### Task 2: DPO Loss Function

**Files:**
- Create: `xaytune/recipes/align/dpo.py`
- Create: `tests/test_recipes/test_align/test_dpo.py`

Direct Preference Optimization loss. Takes policy log-probs for chosen/rejected responses and a reference model's log-probs, computes the DPO loss.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_recipes/test_align/test_dpo.py`:

```python
import pytest
import torch
from xaytune.recipes.align.dpo import dpo_loss


class TestDPOLoss:
    def test_basic_loss_computation(self):
        policy_chosen_logps = torch.tensor([-1.0, -2.0])
        policy_rejected_logps = torch.tensor([-3.0, -4.0])
        ref_chosen_logps = torch.tensor([-1.5, -2.5])
        ref_rejected_logps = torch.tensor([-3.5, -4.5])

        loss = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_loss_decreases_when_chosen_preferred(self):
        policy_chosen_logps = torch.tensor([-0.5])
        policy_rejected_logps = torch.tensor([-3.0])
        ref_chosen_logps = torch.tensor([-1.0])
        ref_rejected_logps = torch.tensor([-1.0])

        loss = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )

        assert loss.item() < 1.0

    def test_loss_increases_when_rejected_preferred(self):
        policy_chosen_logps = torch.tensor([-3.0])
        policy_rejected_logps = torch.tensor([-0.5])
        ref_chosen_logps = torch.tensor([-1.0])
        ref_rejected_logps = torch.tensor([-1.0])

        loss = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )

        assert loss.item() > 0.5

    def test_custom_beta(self):
        policy_chosen_logps = torch.tensor([-1.0])
        policy_rejected_logps = torch.tensor([-2.0])
        ref_chosen_logps = torch.tensor([-1.5])
        ref_rejected_logps = torch.tensor([-2.5])

        loss_low_beta = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            beta=0.05,
        )

        loss_high_beta = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            beta=0.5,
        )

        assert loss_low_beta.item() != loss_high_beta.item()

    def test_equal_logprobs_gives_log2_loss(self):
        policy_chosen_logps = torch.tensor([-1.0])
        policy_rejected_logps = torch.tensor([-1.0])
        ref_chosen_logps = torch.tensor([-1.0])
        ref_rejected_logps = torch.tensor([-1.0])

        loss = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )

        # When logprob ratios are equal, sigmoid(0) = 0.5, -log(0.5) = log(2)
        assert abs(loss.item() - 0.6931) < 0.01

    def test_batch_dimension(self):
        batch_size = 8
        policy_chosen_logps = torch.randn(batch_size)
        policy_rejected_logps = torch.randn(batch_size)
        ref_chosen_logps = torch.randn(batch_size)
        ref_rejected_logps = torch.randn(batch_size)

        loss = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )

        assert loss.ndim == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_recipes/test_align/test_dpo.py -v`

- [ ] **Step 3: Implement DPO loss**

Create `xaytune/recipes/align/dpo.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    *,
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    logits = chosen_rewards - rejected_rewards

    return -F.logsigmoid(logits).mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_recipes/test_align/test_dpo.py -v`

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/recipes/align/dpo.py tests/test_recipes/test_align/test_dpo.py
git commit -m "feat: add DPO loss function for preference-based alignment"
```

---

### Task 3: GRPO Loss Function

**Files:**
- Create: `xaytune/recipes/align/grpo.py`
- Create: `tests/test_recipes/test_align/test_grpo.py`

Group Relative Policy Optimization. Scores multiple completions per prompt, computes advantages within the group, applies policy gradient with KL penalty. No critic model needed.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_recipes/test_align/test_grpo.py`:

```python
import pytest
import torch
from xaytune.recipes.align.grpo import grpo_loss, compute_group_advantages


class TestComputeGroupAdvantages:
    def test_basic_advantages(self):
        rewards = torch.tensor([1.0, 3.0, 2.0])
        advantages = compute_group_advantages(rewards)

        assert advantages.shape == rewards.shape
        # Mean-centered: highest reward gets positive advantage
        assert advantages[1] > advantages[0]
        assert advantages[1] > advantages[2]

    def test_advantages_are_normalized(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        advantages = compute_group_advantages(rewards)

        assert abs(advantages.mean().item()) < 1e-5
        assert abs(advantages.std().item() - 1.0) < 0.2

    def test_single_reward(self):
        rewards = torch.tensor([5.0])
        advantages = compute_group_advantages(rewards)

        assert advantages.shape == (1,)
        assert advantages[0].item() == 0.0

    def test_equal_rewards(self):
        rewards = torch.tensor([2.0, 2.0, 2.0])
        advantages = compute_group_advantages(rewards)

        for a in advantages:
            assert abs(a.item()) < 1e-5


class TestGRPOLoss:
    def test_basic_loss(self):
        logprobs = torch.tensor([-1.0, -2.0, -1.5])
        ref_logprobs = torch.tensor([-1.2, -2.1, -1.6])
        advantages = torch.tensor([1.0, -0.5, 0.2])

        loss = grpo_loss(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_kl_penalty_increases_loss(self):
        logprobs = torch.tensor([-1.0, -2.0])
        ref_logprobs = torch.tensor([-3.0, -4.0])
        advantages = torch.tensor([1.0, 1.0])

        loss_no_kl = grpo_loss(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            kl_coeff=0.0,
        )

        loss_with_kl = grpo_loss(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            kl_coeff=0.1,
        )

        assert loss_with_kl.item() != loss_no_kl.item()

    def test_zero_advantages_only_kl(self):
        logprobs = torch.tensor([-1.0, -2.0])
        ref_logprobs = torch.tensor([-1.5, -2.5])
        advantages = torch.tensor([0.0, 0.0])

        loss = grpo_loss(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            kl_coeff=0.1,
        )

        assert loss.item() >= 0.0

    def test_custom_kl_coeff(self):
        logprobs = torch.tensor([-1.0])
        ref_logprobs = torch.tensor([-2.0])
        advantages = torch.tensor([1.0])

        loss_low = grpo_loss(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            kl_coeff=0.01,
        )

        loss_high = grpo_loss(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            kl_coeff=0.5,
        )

        assert loss_low.item() != loss_high.item()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_recipes/test_align/test_grpo.py -v`

- [ ] **Step 3: Implement GRPO loss**

Create `xaytune/recipes/align/grpo.py`:

```python
from __future__ import annotations

import torch


def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)

    mean = rewards.mean()
    std = rewards.std()

    if std < 1e-8:
        return torch.zeros_like(rewards)

    return (rewards - mean) / (std + 1e-8)


def grpo_loss(
    *,
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    kl_coeff: float = 0.04,
) -> torch.Tensor:
    policy_loss = -(logprobs * advantages).mean()

    kl = (logprobs - ref_logprobs).mean()

    return policy_loss + kl_coeff * kl
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_recipes/test_align/test_grpo.py -v`

Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/recipes/align/grpo.py tests/test_recipes/test_align/test_grpo.py
git commit -m "feat: add GRPO loss with group advantage computation"
```

---

### Task 4: Align Recipe & Top-Level Wire-Up

**Files:**
- Create: `xaytune/recipes/align/align.py`
- Modify: `xaytune/recipes/align/__init__.py`
- Modify: `xaytune/recipes/__init__.py`
- Modify: `xaytune/__init__.py`
- Create: `tests/test_recipes/test_align/test_align.py`
- Create: `tests/test_recipes/test_align/test_init.py`

The `align()` recipe function that routes to the correct alignment method, plus package wire-up.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_recipes/test_align/test_align.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from xaytune.recipes.align.align import align
from xaytune.config.schema import TrainConfig, ModelConfig, DataConfig, TrainerConfig
from xaytune.trainer.callbacks import TrainState


class TestAlign:
    @patch("xaytune.recipes.align.align.setup_training")
    def test_align_with_config(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState(global_step=100)
        mock_setup.return_value = mock_components

        config = TrainConfig(
            recipe="align",
            method="dpo",
            model=ModelConfig(name="test-model"),
            data=DataConfig(path="prefs.jsonl", format="preference"),
        )

        state = align(config=config)

        mock_setup.assert_called_once()
        assert state.global_step == 100

    @patch("xaytune.recipes.align.align.setup_training")
    def test_align_with_kwargs(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState(global_step=50)
        mock_setup.return_value = mock_components

        state = align(
            model="my-sft-model",
            dataset="prefs.jsonl",
            method="grpo",
        )

        config = mock_setup.call_args[0][0]
        assert config.model.name == "my-sft-model"
        assert config.data.path == "prefs.jsonl"
        assert config.method == "grpo"
        assert config.recipe == "align"
        assert state.global_step == 50

    @patch("xaytune.recipes.align.align.setup_training")
    def test_align_default_method_is_dpo(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        align(model="test-model", dataset="prefs.jsonl")

        config = mock_setup.call_args[0][0]
        assert config.method == "dpo"

    @patch("xaytune.recipes.align.align.setup_training")
    def test_align_default_format_is_preference(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        align(model="test-model", dataset="prefs.jsonl")

        config = mock_setup.call_args[0][0]
        assert config.data.format == "preference"

    @patch("xaytune.recipes.align.align.setup_training")
    def test_align_accepts_trainer_kwargs(self, mock_setup):
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = TrainState()
        mock_setup.return_value = mock_components

        align(
            model="test-model",
            dataset="prefs.jsonl",
            num_epochs=2,
            learning_rate=5e-6,
        )

        config = mock_setup.call_args[0][0]
        assert config.trainer.num_epochs == 2
        assert config.trainer.learning_rate == 5e-6

    @patch("xaytune.recipes.align.align.setup_training")
    def test_align_returns_train_state(self, mock_setup):
        expected_state = TrainState(global_step=200, metrics={"loss": 0.4})
        mock_components = MagicMock()
        mock_components.trainer.train.return_value = expected_state
        mock_setup.return_value = mock_components

        result = align(model="test-model", dataset="prefs.jsonl")

        assert isinstance(result, TrainState)
        assert result.metrics["loss"] == 0.4

    def test_align_requires_model_and_dataset(self):
        with pytest.raises(ValueError, match="required"):
            align(model="test-model")

    def test_align_requires_model_and_dataset_2(self):
        with pytest.raises(ValueError, match="required"):
            align(dataset="prefs.jsonl")
```

Create `tests/test_recipes/test_align/test_init.py`:

```python
from xaytune.recipes.align import align, register_reward, reward_registry, dpo_loss, grpo_loss
import xaytune


class TestAlignPublicAPI:
    def test_align_importable(self):
        assert callable(align)

    def test_register_reward_importable(self):
        assert callable(register_reward)

    def test_reward_registry_importable(self):
        assert reward_registry is not None

    def test_dpo_loss_importable(self):
        assert callable(dpo_loss)

    def test_grpo_loss_importable(self):
        assert callable(grpo_loss)

    def test_align_in_recipe_registry(self):
        from xaytune.recipes import recipe_registry
        assert recipe_registry.has("align")

    def test_top_level_align(self):
        assert callable(xaytune.align)

    def test_top_level_align_is_recipe(self):
        from xaytune.recipes.align.align import align as align_fn
        assert xaytune.align is align_fn
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_recipes/test_align/test_align.py tests/test_recipes/test_align/test_init.py -v`

- [ ] **Step 3: Implement align recipe and wire up packages**

Create `xaytune/recipes/align/align.py`:

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


def align(
    *,
    config: TrainConfig | None = None,
    model: str | None = None,
    dataset: str | None = None,
    method: str = "dpo",
    format: str = "preference",
    num_epochs: int = 1,
    learning_rate: float = 5e-6,
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
            recipe="align",
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

Update `xaytune/recipes/align/__init__.py`:

```python
from xaytune.recipes.align.align import align
from xaytune.recipes.align.dpo import dpo_loss
from xaytune.recipes.align.grpo import compute_group_advantages, grpo_loss
from xaytune.recipes.align.rewards import register_reward, reward_registry

__all__ = [
    "align",
    "compute_group_advantages",
    "dpo_loss",
    "grpo_loss",
    "register_reward",
    "reward_registry",
]
```

Update `xaytune/recipes/__init__.py`:

```python
from xaytune.utils.registry import Registry

recipe_registry = Registry("recipe")

from xaytune.recipes.finetune import finetune
from xaytune.recipes.pretrain import pretrain
from xaytune.recipes.align import align

recipe_registry.register("finetune")(finetune)
recipe_registry.register("pretrain")(pretrain)
recipe_registry.register("align")(align)

__all__ = ["align", "finetune", "pretrain", "recipe_registry"]
```

Update `xaytune/__init__.py`:

```python
"""xaytune — An opinionated LLM training and fine-tuning library."""

__version__ = "0.1.0"

from xaytune.recipes.align import align
from xaytune.recipes.finetune import finetune
from xaytune.recipes.pretrain import pretrain

__all__ = ["__version__", "align", "finetune", "pretrain"]
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/recipes/align/ xaytune/recipes/__init__.py xaytune/__init__.py tests/test_recipes/test_align/ tests/test_top_level_api.py
git commit -m "feat: add align recipe with DPO/GRPO and wire up xaytune.align API"
```
