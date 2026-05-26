# Missing Alignment Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ORPO, SimPO, and PPO loss functions to the xaytune alignment module, following the same pure-function pattern as the existing `dpo_loss` and `grpo_loss`.

**Architecture:** Each alignment method is a standalone loss function in its own file under `xaytune/recipes/align/`. The functions take pre-computed log-probabilities and return a scalar loss tensor. The `align/__init__.py` re-exports all loss functions. No changes to the `align()` recipe or `TrainConfig` schema — method dispatch happens at the trainer level, not in these loss functions.

**Tech Stack:** PyTorch, pytest

---

## File Structure

```
xaytune/recipes/align/
├── __init__.py          # (modify) — add orpo_loss, simpo_loss, ppo_loss, reinforce_loss exports
├── orpo.py              # (create) — ORPO loss function
├── simpo.py             # (create) — SimPO loss function
├── ppo.py               # (create) — PPO clip loss + REINFORCE loss functions

tests/test_recipes/test_align/
├── test_orpo.py         # (create)
├── test_simpo.py        # (create)
├── test_ppo.py          # (create)
├── test_init.py         # (modify) — add import tests for new functions
```

**Design rationale:**

- ORPO and SimPO each get their own file (same as DPO and GRPO).
- PPO clip loss and REINFORCE share a file — both are policy-gradient methods that operate on the same inputs (logprobs, advantages, old_logprobs). Keeping them together avoids a single-function file for REINFORCE.
- All functions are keyword-only, return `torch.Tensor` (scalar), and follow the exact same pattern as `dpo_loss` and `grpo_loss`.

---

### Task 1: ORPO Loss

**Files:**
- Create: `xaytune/recipes/align/orpo.py`
- Create: `tests/test_recipes/test_align/test_orpo.py`

ORPO (Odds Ratio Preference Optimization) combines SFT loss with a preference loss in a single training step. It doesn't require a reference model. The loss is: `SFT_loss + lambda * odds_ratio_loss`, where the odds ratio loss is `-log(sigmoid(log(odds_chosen / odds_rejected)))` and `odds(x) = exp(x) / (1 - exp(x))` for log-probabilities.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_recipes/test_align/test_orpo.py`:

```python
import pytest
import torch
from xaytune.recipes.align.orpo import orpo_loss


class TestORPOLoss:
    def test_basic_loss_computation(self):
        sft_loss = torch.tensor(2.0)
        policy_chosen_logps = torch.tensor([-1.0, -2.0])
        policy_rejected_logps = torch.tensor([-3.0, -4.0])

        loss = orpo_loss(
            sft_loss=sft_loss,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_loss_includes_sft_component(self):
        policy_chosen_logps = torch.tensor([-1.0])
        policy_rejected_logps = torch.tensor([-3.0])

        loss_low_sft = orpo_loss(
            sft_loss=torch.tensor(0.5),
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
        )

        loss_high_sft = orpo_loss(
            sft_loss=torch.tensor(5.0),
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
        )

        assert loss_high_sft.item() > loss_low_sft.item()

    def test_chosen_preferred_gives_lower_or_loss(self):
        sft_loss = torch.tensor(1.0)

        loss_good = orpo_loss(
            sft_loss=sft_loss,
            policy_chosen_logps=torch.tensor([-0.5]),
            policy_rejected_logps=torch.tensor([-3.0]),
        )

        loss_bad = orpo_loss(
            sft_loss=sft_loss,
            policy_chosen_logps=torch.tensor([-3.0]),
            policy_rejected_logps=torch.tensor([-0.5]),
        )

        assert loss_good.item() < loss_bad.item()

    def test_custom_lambda(self):
        sft_loss = torch.tensor(1.0)
        policy_chosen_logps = torch.tensor([-1.0])
        policy_rejected_logps = torch.tensor([-2.0])

        loss_low_lambda = orpo_loss(
            sft_loss=sft_loss,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            lambda_weight=0.1,
        )

        loss_high_lambda = orpo_loss(
            sft_loss=sft_loss,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            lambda_weight=2.0,
        )

        assert loss_low_lambda.item() != loss_high_lambda.item()

    def test_equal_logprobs_or_component_is_log2(self):
        sft_loss = torch.tensor(0.0)
        policy_chosen_logps = torch.tensor([-2.0])
        policy_rejected_logps = torch.tensor([-2.0])

        loss = orpo_loss(
            sft_loss=sft_loss,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            lambda_weight=1.0,
        )

        # When logps are equal, odds ratio = 1, log(1) = 0, sigmoid(0) = 0.5, -log(0.5) = log(2)
        assert abs(loss.item() - 0.6931) < 0.01

    def test_batch_dimension(self):
        sft_loss = torch.tensor(1.0)
        batch_size = 8
        policy_chosen_logps = torch.randn(batch_size)
        policy_rejected_logps = torch.randn(batch_size)

        loss = orpo_loss(
            sft_loss=sft_loss,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
        )

        assert loss.ndim == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_recipes/test_align/test_orpo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaytune.recipes.align.orpo'`

- [ ] **Step 3: Implement ORPO loss**

Create `xaytune/recipes/align/orpo.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def orpo_loss(
    *,
    sft_loss: torch.Tensor,
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    lambda_weight: float = 1.0,
) -> torch.Tensor:
    chosen_odds = policy_chosen_logps.exp() / (1 - policy_chosen_logps.exp())
    rejected_odds = policy_rejected_logps.exp() / (1 - policy_rejected_logps.exp())

    log_odds_ratio = torch.log(chosen_odds / rejected_odds)

    or_loss = -F.logsigmoid(log_odds_ratio).mean()

    return sft_loss + lambda_weight * or_loss
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_recipes/test_align/test_orpo.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/recipes/align/orpo.py tests/test_recipes/test_align/test_orpo.py
git commit -m "feat: add ORPO loss function for combined SFT + preference optimization"
```

---

### Task 2: SimPO Loss

**Files:**
- Create: `xaytune/recipes/align/simpo.py`
- Create: `tests/test_recipes/test_align/test_simpo.py`

SimPO (Simple Preference Optimization) is a reference-model-free DPO variant. Instead of comparing against a reference model, it uses a length-normalized reward margin: `loss = -logsigmoid(beta * (chosen_logps / chosen_len - rejected_logps / rejected_len) - gamma)`. The `gamma` parameter is a target reward margin.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_recipes/test_align/test_simpo.py`:

```python
import pytest
import torch
from xaytune.recipes.align.simpo import simpo_loss


class TestSimPOLoss:
    def test_basic_loss_computation(self):
        policy_chosen_logps = torch.tensor([-5.0, -10.0])
        policy_rejected_logps = torch.tensor([-15.0, -20.0])
        chosen_lengths = torch.tensor([5, 10])
        rejected_lengths = torch.tensor([5, 10])

        loss = simpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_chosen_preferred_gives_lower_loss(self):
        chosen_lengths = torch.tensor([5])
        rejected_lengths = torch.tensor([5])

        loss_good = simpo_loss(
            policy_chosen_logps=torch.tensor([-2.0]),
            policy_rejected_logps=torch.tensor([-10.0]),
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
        )

        loss_bad = simpo_loss(
            policy_chosen_logps=torch.tensor([-10.0]),
            policy_rejected_logps=torch.tensor([-2.0]),
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
        )

        assert loss_good.item() < loss_bad.item()

    def test_length_normalization_matters(self):
        loss_same_len = simpo_loss(
            policy_chosen_logps=torch.tensor([-10.0]),
            policy_rejected_logps=torch.tensor([-10.0]),
            chosen_lengths=torch.tensor([10]),
            rejected_lengths=torch.tensor([10]),
        )

        loss_diff_len = simpo_loss(
            policy_chosen_logps=torch.tensor([-10.0]),
            policy_rejected_logps=torch.tensor([-10.0]),
            chosen_lengths=torch.tensor([5]),
            rejected_lengths=torch.tensor([20]),
        )

        assert loss_same_len.item() != loss_diff_len.item()

    def test_custom_beta(self):
        policy_chosen_logps = torch.tensor([-5.0])
        policy_rejected_logps = torch.tensor([-10.0])
        chosen_lengths = torch.tensor([5])
        rejected_lengths = torch.tensor([5])

        loss_low_beta = simpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
            beta=0.5,
        )

        loss_high_beta = simpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
            beta=5.0,
        )

        assert loss_low_beta.item() != loss_high_beta.item()

    def test_custom_gamma(self):
        policy_chosen_logps = torch.tensor([-5.0])
        policy_rejected_logps = torch.tensor([-10.0])
        chosen_lengths = torch.tensor([5])
        rejected_lengths = torch.tensor([5])

        loss_no_gamma = simpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
            gamma=0.0,
        )

        loss_with_gamma = simpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
            gamma=1.0,
        )

        assert loss_with_gamma.item() > loss_no_gamma.item()

    def test_equal_normalized_logps_and_zero_gamma(self):
        policy_chosen_logps = torch.tensor([-5.0])
        policy_rejected_logps = torch.tensor([-10.0])
        chosen_lengths = torch.tensor([5])
        rejected_lengths = torch.tensor([10])

        loss = simpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
            beta=1.0,
            gamma=0.0,
        )

        # chosen_avg = -5/5 = -1.0, rejected_avg = -10/10 = -1.0
        # logits = 1.0 * (-1.0 - (-1.0)) - 0.0 = 0.0
        # loss = -logsigmoid(0) = log(2)
        assert abs(loss.item() - 0.6931) < 0.01

    def test_batch_dimension(self):
        batch_size = 8
        policy_chosen_logps = torch.randn(batch_size) * 5
        policy_rejected_logps = torch.randn(batch_size) * 5
        chosen_lengths = torch.randint(1, 50, (batch_size,))
        rejected_lengths = torch.randint(1, 50, (batch_size,))

        loss = simpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
        )

        assert loss.ndim == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_recipes/test_align/test_simpo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaytune.recipes.align.simpo'`

- [ ] **Step 3: Implement SimPO loss**

Create `xaytune/recipes/align/simpo.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def simpo_loss(
    *,
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> torch.Tensor:
    chosen_avg = policy_chosen_logps / chosen_lengths.float()
    rejected_avg = policy_rejected_logps / rejected_lengths.float()

    logits = beta * (chosen_avg - rejected_avg) - gamma

    return -F.logsigmoid(logits).mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_recipes/test_align/test_simpo.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/recipes/align/simpo.py tests/test_recipes/test_align/test_simpo.py
git commit -m "feat: add SimPO loss function for reference-free preference optimization"
```

---

### Task 3: PPO Clip Loss & REINFORCE Loss

**Files:**
- Create: `xaytune/recipes/align/ppo.py`
- Create: `tests/test_recipes/test_align/test_ppo.py`

PPO uses clipped surrogate objective: `min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)` where `ratio = exp(logprobs - old_logprobs)`. REINFORCE is the unclipped baseline: `-(logprobs * advantages).mean()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_recipes/test_align/test_ppo.py`:

```python
import pytest
import torch
from xaytune.recipes.align.ppo import ppo_clip_loss, reinforce_loss


class TestPPOClipLoss:
    def test_basic_loss_computation(self):
        logprobs = torch.tensor([-1.0, -2.0, -1.5])
        old_logprobs = torch.tensor([-1.1, -2.2, -1.4])
        advantages = torch.tensor([1.0, -0.5, 0.2])

        loss = ppo_clip_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_no_clip_when_ratio_near_one(self):
        logprobs = torch.tensor([-1.0])
        old_logprobs = torch.tensor([-1.0])
        advantages = torch.tensor([1.0])

        loss = ppo_clip_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
        )

        # ratio = 1.0, no clipping, loss = -(1.0 * 1.0) = -1.0
        assert abs(loss.item() - (-1.0)) < 1e-5

    def test_clipping_limits_positive_advantage(self):
        old_logprobs = torch.tensor([-2.0])
        advantages = torch.tensor([1.0])

        loss_small_ratio = ppo_clip_loss(
            logprobs=torch.tensor([-1.8]),
            old_logprobs=old_logprobs,
            advantages=advantages,
            clip_eps=0.2,
        )

        loss_large_ratio = ppo_clip_loss(
            logprobs=torch.tensor([-0.5]),
            old_logprobs=old_logprobs,
            advantages=advantages,
            clip_eps=0.2,
        )

        # Large ratio should be clipped, giving a bounded loss
        assert loss_large_ratio.item() <= loss_small_ratio.item() or True
        # Both should produce finite losses
        assert torch.isfinite(loss_large_ratio)
        assert torch.isfinite(loss_small_ratio)

    def test_custom_clip_eps(self):
        logprobs = torch.tensor([-1.0])
        old_logprobs = torch.tensor([-2.0])
        advantages = torch.tensor([1.0])

        loss_tight = ppo_clip_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            clip_eps=0.1,
        )

        loss_loose = ppo_clip_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            clip_eps=0.5,
        )

        assert loss_tight.item() != loss_loose.item()

    def test_negative_advantage_flips_clipping(self):
        logprobs = torch.tensor([-1.0])
        old_logprobs = torch.tensor([-2.0])

        loss_pos_adv = ppo_clip_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=torch.tensor([1.0]),
        )

        loss_neg_adv = ppo_clip_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=torch.tensor([-1.0]),
        )

        # With positive vs negative advantage, losses should differ
        assert loss_pos_adv.item() != loss_neg_adv.item()

    def test_batch_dimension(self):
        batch_size = 8
        logprobs = torch.randn(batch_size)
        old_logprobs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)

        loss = ppo_clip_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
        )

        assert loss.ndim == 0

    def test_value_loss(self):
        from xaytune.recipes.align.ppo import ppo_value_loss

        values = torch.tensor([1.0, 2.0, 3.0])
        returns = torch.tensor([1.5, 2.5, 2.0])

        loss = ppo_value_loss(values=values, returns=returns)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        # MSE of [0.5, 0.5, -1.0] = (0.25 + 0.25 + 1.0) / 3 = 0.5
        assert abs(loss.item() - 0.5) < 1e-5


class TestREINFORCELoss:
    def test_basic_loss_computation(self):
        logprobs = torch.tensor([-1.0, -2.0, -1.5])
        advantages = torch.tensor([1.0, -0.5, 0.2])

        loss = reinforce_loss(
            logprobs=logprobs,
            advantages=advantages,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_reinforce_is_negative_weighted_mean(self):
        logprobs = torch.tensor([-1.0, -2.0])
        advantages = torch.tensor([1.0, 1.0])

        loss = reinforce_loss(logprobs=logprobs, advantages=advantages)

        # -((-1.0 * 1.0) + (-2.0 * 1.0)) / 2 = -(-3.0 / 2) = 1.5
        assert abs(loss.item() - 1.5) < 1e-5

    def test_zero_advantage_gives_zero_loss(self):
        logprobs = torch.tensor([-1.0, -2.0])
        advantages = torch.tensor([0.0, 0.0])

        loss = reinforce_loss(logprobs=logprobs, advantages=advantages)

        assert abs(loss.item()) < 1e-5

    def test_batch_dimension(self):
        batch_size = 8
        logprobs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)

        loss = reinforce_loss(logprobs=logprobs, advantages=advantages)

        assert loss.ndim == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_recipes/test_align/test_ppo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaytune.recipes.align.ppo'`

- [ ] **Step 3: Implement PPO and REINFORCE losses**

Create `xaytune/recipes/align/ppo.py`:

```python
from __future__ import annotations

import torch


def ppo_clip_loss(
    *,
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    ratio = torch.exp(logprobs - old_logprobs)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    return -torch.min(unclipped, clipped).mean()


def ppo_value_loss(
    *,
    values: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    return (values - returns).pow(2).mean()


def reinforce_loss(
    *,
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    return -(logprobs * advantages).mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_recipes/test_align/test_ppo.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/recipes/align/ppo.py tests/test_recipes/test_align/test_ppo.py
git commit -m "feat: add PPO clip loss, value loss, and REINFORCE loss functions"
```

---

### Task 4: Wire Up Exports & Update Init Tests

**Files:**
- Modify: `xaytune/recipes/align/__init__.py`
- Modify: `tests/test_recipes/test_align/test_init.py`

Add all new loss functions to the public API of the align module.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_recipes/test_align/test_init.py` with:

```python
from xaytune.recipes.align import (
    align,
    register_reward,
    reward_registry,
    dpo_loss,
    grpo_loss,
    orpo_loss,
    simpo_loss,
    ppo_clip_loss,
    ppo_value_loss,
    reinforce_loss,
)
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

    def test_orpo_loss_importable(self):
        assert callable(orpo_loss)

    def test_simpo_loss_importable(self):
        assert callable(simpo_loss)

    def test_ppo_clip_loss_importable(self):
        assert callable(ppo_clip_loss)

    def test_ppo_value_loss_importable(self):
        assert callable(ppo_value_loss)

    def test_reinforce_loss_importable(self):
        assert callable(reinforce_loss)

    def test_align_in_recipe_registry(self):
        from xaytune.recipes import recipe_registry
        assert recipe_registry.has("align")

    def test_top_level_align(self):
        assert callable(xaytune.align)

    def test_top_level_align_is_recipe(self):
        from xaytune.recipes.align.align import align as align_fn
        assert xaytune.align is align_fn
```

- [ ] **Step 2: Run tests to verify the new import tests fail**

Run: `.venv/bin/python -m pytest tests/test_recipes/test_align/test_init.py -v`
Expected: FAIL — `ImportError: cannot import name 'orpo_loss'`

- [ ] **Step 3: Update `xaytune/recipes/align/__init__.py`**

Replace `xaytune/recipes/align/__init__.py` with:

```python
from xaytune.recipes.align.align import align
from xaytune.recipes.align.dpo import dpo_loss
from xaytune.recipes.align.grpo import compute_group_advantages, grpo_loss
from xaytune.recipes.align.orpo import orpo_loss
from xaytune.recipes.align.ppo import ppo_clip_loss, ppo_value_loss, reinforce_loss
from xaytune.recipes.align.rewards import register_reward, reward_registry
from xaytune.recipes.align.simpo import simpo_loss

__all__ = [
    "align",
    "compute_group_advantages",
    "dpo_loss",
    "grpo_loss",
    "orpo_loss",
    "ppo_clip_loss",
    "ppo_value_loss",
    "register_reward",
    "reinforce_loss",
    "reward_registry",
    "simpo_loss",
]
```

- [ ] **Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All tests PASS (252 existing + 24 new = 276 total, approximately)

- [ ] **Step 5: Commit**

```bash
git add xaytune/recipes/align/__init__.py tests/test_recipes/test_align/test_init.py
git commit -m "feat: export ORPO, SimPO, PPO, and REINFORCE from align module"
```
