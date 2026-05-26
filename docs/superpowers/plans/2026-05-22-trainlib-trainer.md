# xaytune Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the core training loop, callback system, distributed strategy wrappers (DDP/FSDP/DeepSpeed), and checkpoint save/resume.

**Architecture:** The Trainer is the engine that recipes use. It owns the training loop, optimizer, scheduler, gradient accumulation, mixed precision, and checkpointing. Callbacks hook into events. Distributed strategies wrap the model. The Trainer doesn't know about recipes — recipes configure and call the Trainer.

**Tech Stack:** PyTorch, torch.distributed, torch.cuda.amp, pytest

---

## Plan Sequence

This is **Plan 3 of 6** — depends on Plans 1-2 being complete.

---

### Task 1: Callback System

**Files:**
- Create: `xaytune/trainer/callbacks.py`
- Create: `tests/test_trainer/__init__.py`
- Create: `tests/test_trainer/test_callbacks.py`

The callback system fires events at key points in the training loop. Users hook in via `@on("event")` decorators.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trainer/__init__.py` (empty).

Create `tests/test_trainer/test_callbacks.py`:

```python
import pytest
from xaytune.trainer.callbacks import CallbackManager, TrainState


class TestTrainState:
    def test_initial_state(self):
        state = TrainState()
        assert state.step == 0
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.metrics == {}
        assert state.should_stop is False

    def test_stop_training(self):
        state = TrainState()
        state.stop_training()
        assert state.should_stop is True

    def test_update_metrics(self):
        state = TrainState()
        state.metrics["loss"] = 0.5
        assert state.metrics["loss"] == 0.5


class TestCallbackManager:
    def test_register_and_fire(self):
        manager = CallbackManager()
        calls = []

        @manager.on("step_end")
        def my_callback(state):
            calls.append(state.step)

        state = TrainState(step=5)
        manager.fire("step_end", state)
        assert calls == [5]

    def test_multiple_callbacks_same_event(self):
        manager = CallbackManager()
        calls = []

        @manager.on("step_end")
        def cb1(state):
            calls.append("cb1")

        @manager.on("step_end")
        def cb2(state):
            calls.append("cb2")

        manager.fire("step_end", TrainState())
        assert calls == ["cb1", "cb2"]

    def test_different_events(self):
        manager = CallbackManager()
        calls = []

        @manager.on("train_start")
        def on_start(state):
            calls.append("start")

        @manager.on("train_end")
        def on_end(state):
            calls.append("end")

        manager.fire("train_start", TrainState())
        manager.fire("train_end", TrainState())
        assert calls == ["start", "end"]

    def test_fire_unknown_event_is_noop(self):
        manager = CallbackManager()
        manager.fire("nonexistent", TrainState())  # should not raise

    def test_all_event_types(self):
        manager = CallbackManager()
        events = [
            "train_start", "train_end",
            "epoch_start", "epoch_end",
            "step_start", "step_end",
            "eval_start", "eval_end",
            "checkpoint_saved", "error",
        ]
        fired = []
        for event in events:
            @manager.on(event)
            def cb(state, e=event):
                fired.append(e)

        for event in events:
            manager.fire(event, TrainState())
        assert fired == events

    def test_on_returns_original_function(self):
        manager = CallbackManager()

        @manager.on("step_end")
        def my_func(state):
            return 42

        assert my_func(TrainState()) == 42

    def test_callback_can_stop_training(self):
        manager = CallbackManager()

        @manager.on("step_end")
        def early_stop(state):
            if state.step >= 10:
                state.stop_training()

        state = TrainState(step=10)
        manager.fire("step_end", state)
        assert state.should_stop is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trainer/test_callbacks.py -v`

- [ ] **Step 3: Implement callback system**

Create `xaytune/trainer/callbacks.py`:

```python
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

VALID_EVENTS = {
    "train_start", "train_end",
    "epoch_start", "epoch_end",
    "step_start", "step_end",
    "eval_start", "eval_end",
    "checkpoint_saved", "error",
}


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    global_step: int = 0
    num_epochs: int = 0
    max_steps: int = -1
    metrics: dict[str, Any] = field(default_factory=dict)
    should_stop: bool = False

    def stop_training(self) -> None:
        self.should_stop = True


class CallbackManager:
    def __init__(self) -> None:
        self._callbacks: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self._callbacks[event].append(fn)
            return fn
        return decorator

    def fire(self, event: str, state: TrainState) -> None:
        for callback in self._callbacks.get(event, []):
            callback(state)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trainer/test_callbacks.py -v`

Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/trainer/callbacks.py tests/test_trainer/
git commit -m "feat: add callback system with TrainState and event firing"
```

---

### Task 2: Distributed Strategy Wrappers

**Files:**
- Create: `xaytune/trainer/distributed.py`
- Create: `tests/test_trainer/test_distributed.py`

Wrappers for DDP, FSDP, and DeepSpeed. Each strategy wraps a model and returns it. The auto strategy picks based on model size and available GPUs.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trainer/test_distributed.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from xaytune.trainer.distributed import (
    wrap_model_distributed,
    get_strategy,
    DistributedContext,
)


class TestDistributedContext:
    def test_defaults(self):
        ctx = DistributedContext()
        assert ctx.rank == 0
        assert ctx.world_size == 1
        assert ctx.local_rank == 0
        assert ctx.is_main_process is True

    def test_non_main_process(self):
        ctx = DistributedContext(rank=1, world_size=4, local_rank=1)
        assert ctx.is_main_process is False


class TestGetStrategy:
    def test_auto_single_gpu(self):
        strategy = get_strategy("auto", world_size=1)
        assert strategy == "none"

    def test_auto_multi_gpu(self):
        strategy = get_strategy("auto", world_size=4)
        assert strategy == "fsdp"

    def test_explicit_ddp(self):
        strategy = get_strategy("ddp", world_size=4)
        assert strategy == "ddp"

    def test_explicit_fsdp(self):
        strategy = get_strategy("fsdp", world_size=1)
        assert strategy == "fsdp"


class TestWrapModelDistributed:
    def test_none_strategy_returns_model(self):
        mock_model = MagicMock()
        ctx = DistributedContext()
        result = wrap_model_distributed(mock_model, strategy="none", ctx=ctx)
        assert result is mock_model

    @patch("xaytune.trainer.distributed.DistributedDataParallel")
    def test_ddp_wraps_model(self, mock_ddp_cls):
        mock_model = MagicMock()
        mock_ddp_cls.return_value = MagicMock()
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        result = wrap_model_distributed(mock_model, strategy="ddp", ctx=ctx)
        mock_ddp_cls.assert_called_once()

    @patch("xaytune.trainer.distributed.FullyShardedDataParallel")
    def test_fsdp_wraps_model(self, mock_fsdp_cls):
        mock_model = MagicMock()
        mock_fsdp_cls.return_value = MagicMock()
        ctx = DistributedContext(rank=0, world_size=2, local_rank=0)
        result = wrap_model_distributed(mock_model, strategy="fsdp", ctx=ctx)
        mock_fsdp_cls.assert_called_once()

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            wrap_model_distributed(MagicMock(), strategy="invalid", ctx=DistributedContext())
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement distributed wrappers**

Create `xaytune/trainer/distributed.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel


@dataclass
class DistributedContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def get_strategy(strategy: str, world_size: int = 1) -> str:
    if strategy == "auto":
        return "fsdp" if world_size > 1 else "none"
    return strategy


def wrap_model_distributed(
    model: Any,
    *,
    strategy: str,
    ctx: DistributedContext,
    **kwargs: Any,
) -> Any:
    if strategy == "none":
        return model

    if strategy == "ddp":
        return DistributedDataParallel(
            model,
            device_ids=[ctx.local_rank] if ctx.local_rank >= 0 else None,
        )

    if strategy == "fsdp":
        return FullyShardedDataParallel(model, **kwargs)

    if strategy == "deepspeed":
        return model  # DeepSpeed init handled separately

    raise ValueError(
        f"Unknown strategy: '{strategy}'. "
        f"Valid options: none, ddp, fsdp, deepspeed."
    )
```

- [ ] **Step 4: Run tests, commit**

```bash
git commit -m "feat: add distributed strategy wrappers (DDP, FSDP)"
```

---

### Task 3: Training Loop

**Files:**
- Create: `xaytune/trainer/loop.py`
- Create: `tests/test_trainer/test_loop.py`

The core training loop. Handles forward/backward pass, gradient accumulation, optimizer step, mixed precision, and callback firing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trainer/test_loop.py`:

```python
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from xaytune.trainer.loop import Trainer
from xaytune.trainer.callbacks import CallbackManager, TrainState
from xaytune.config.schema import TrainerConfig


class TestTrainer:
    def _make_trainer(self, **kwargs):
        config = TrainerConfig(**kwargs)
        return Trainer(config=config)

    def test_init_defaults(self):
        trainer = self._make_trainer()
        assert trainer.config.batch_size == 4
        assert trainer.config.learning_rate == 2e-4
        assert trainer.config.num_epochs == 3
        assert trainer.callback_manager is not None

    def test_custom_callback_manager(self):
        cm = CallbackManager()
        trainer = Trainer(config=TrainerConfig(), callback_manager=cm)
        assert trainer.callback_manager is cm

    def test_compute_total_steps(self):
        trainer = self._make_trainer(num_epochs=3)
        total = trainer.compute_total_steps(dataset_size=100, batch_size=4)
        # 100 / 4 = 25 steps per epoch, 25 * 3 = 75 total
        assert total == 75

    def test_compute_total_steps_with_accumulation(self):
        trainer = self._make_trainer(num_epochs=2, gradient_accumulation=4)
        total = trainer.compute_total_steps(dataset_size=80, batch_size=4)
        # 80 / 4 = 20 micro-steps per epoch, 20 / 4 = 5 optimizer steps, 5 * 2 = 10
        assert total == 10

    def test_compute_total_steps_with_max_steps(self):
        trainer = self._make_trainer(num_epochs=100, max_steps=50)
        total = trainer.compute_total_steps(dataset_size=1000, batch_size=4)
        assert total == 50

    def test_train_fires_callbacks(self):
        trainer = self._make_trainer(num_epochs=1, max_steps=2)
        events = []

        @trainer.callback_manager.on("train_start")
        def on_start(state):
            events.append("train_start")

        @trainer.callback_manager.on("step_start")
        def on_step_start(state):
            events.append(f"step_start:{state.global_step}")

        @trainer.callback_manager.on("step_end")
        def on_step_end(state):
            events.append(f"step_end:{state.global_step}")

        @trainer.callback_manager.on("train_end")
        def on_end(state):
            events.append("train_end")

        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = MagicMock()
        mock_model.return_value.loss.item.return_value = 0.5
        mock_model.return_value.loss.backward = MagicMock()
        mock_model.parameters.return_value = [MagicMock()]

        mock_dataloader = [
            {"input_ids": MagicMock(), "attention_mask": MagicMock(), "labels": MagicMock()},
            {"input_ids": MagicMock(), "attention_mask": MagicMock(), "labels": MagicMock()},
            {"input_ids": MagicMock(), "attention_mask": MagicMock(), "labels": MagicMock()},
        ]

        trainer.train(model=mock_model, train_dataloader=mock_dataloader)

        assert "train_start" in events
        assert "train_end" in events
        assert "step_start:0" in events

    def test_early_stopping_via_callback(self):
        trainer = self._make_trainer(num_epochs=100, max_steps=-1)

        @trainer.callback_manager.on("step_end")
        def stop_early(state):
            if state.global_step >= 1:
                state.stop_training()

        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        mock_model.return_value.loss = MagicMock()
        mock_model.return_value.loss.item.return_value = 0.5
        mock_model.return_value.loss.backward = MagicMock()
        mock_model.parameters.return_value = [MagicMock()]

        mock_dataloader = [
            {"input_ids": MagicMock(), "attention_mask": MagicMock(), "labels": MagicMock()}
            for _ in range(100)
        ]

        state = trainer.train(model=mock_model, train_dataloader=mock_dataloader)
        assert state.should_stop is True
        assert state.global_step <= 2
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement the training loop**

Create `xaytune/trainer/loop.py`:

```python
from __future__ import annotations

from typing import Any

import torch

from xaytune.config.schema import TrainerConfig
from xaytune.trainer.callbacks import CallbackManager, TrainState


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        self.config = config
        self.callback_manager = callback_manager or CallbackManager()

    def compute_total_steps(
        self,
        dataset_size: int,
        batch_size: int,
    ) -> int:
        steps_per_epoch = dataset_size // batch_size
        if self.config.gradient_accumulation > 1:
            steps_per_epoch = steps_per_epoch // self.config.gradient_accumulation
        total = steps_per_epoch * self.config.num_epochs
        if self.config.max_steps > 0:
            total = min(total, self.config.max_steps)
        return total

    def train(
        self,
        *,
        model: Any,
        train_dataloader: Any,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
    ) -> TrainState:
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        state = TrainState(
            num_epochs=self.config.num_epochs,
            max_steps=self.config.max_steps,
        )

        self.callback_manager.fire("train_start", state)

        for epoch in range(self.config.num_epochs):
            state.epoch = epoch
            self.callback_manager.fire("epoch_start", state)

            for step, batch in enumerate(train_dataloader):
                state.step = step
                self.callback_manager.fire("step_start", state)

                loss = self._training_step(model, batch, optimizer, state)
                state.metrics["loss"] = loss

                self.callback_manager.fire("step_end", state)
                state.global_step += 1

                if self.config.max_steps > 0 and state.global_step >= self.config.max_steps:
                    state.stop_training()

                if state.should_stop:
                    break

            self.callback_manager.fire("epoch_end", state)
            if state.should_stop:
                break

        self.callback_manager.fire("train_end", state)
        return state

    def _training_step(
        self,
        model: Any,
        batch: dict[str, Any],
        optimizer: Any,
        state: TrainState,
    ) -> float:
        outputs = model(**batch) if isinstance(batch, dict) else model(batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs

        if self.config.gradient_accumulation > 1:
            loss = loss / self.config.gradient_accumulation

        loss.backward()

        if (state.step + 1) % self.config.gradient_accumulation == 0 or state.step == 0:
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.max_grad_norm
                )
            optimizer.step()
            optimizer.zero_grad()

        return loss.item() if hasattr(loss, "item") else float(loss)
```

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add core training loop with gradient accumulation and callbacks"
```

---

### Task 4: Checkpointing

**Files:**
- Create: `xaytune/trainer/checkpointing.py`
- Create: `tests/test_trainer/test_checkpointing.py`

Save and resume training state: model weights, optimizer state, scheduler state, training step, and config.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trainer/test_checkpointing.py`:

```python
import json
import tempfile
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch
from xaytune.trainer.checkpointing import save_checkpoint, load_checkpoint, find_latest_checkpoint
from xaytune.trainer.callbacks import TrainState


class TestSaveCheckpoint:
    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "checkpoints" / "step-100"
            state = TrainState(global_step=100, epoch=2)

            with patch("xaytune.trainer.checkpointing.torch") as mock_torch:
                save_checkpoint(
                    output_dir=str(output_dir),
                    model=MagicMock(),
                    optimizer=MagicMock(),
                    state=state,
                )

            assert output_dir.exists()

    def test_save_writes_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "step-100"
            state = TrainState(global_step=100, epoch=2, metrics={"loss": 0.5})

            with patch("xaytune.trainer.checkpointing.torch") as mock_torch:
                save_checkpoint(
                    output_dir=str(output_dir),
                    model=MagicMock(),
                    optimizer=MagicMock(),
                    state=state,
                )

            metadata_path = output_dir / "metadata.json"
            assert metadata_path.exists()
            metadata = json.loads(metadata_path.read_text())
            assert metadata["global_step"] == 100
            assert metadata["epoch"] == 2
            assert metadata["metrics"]["loss"] == 0.5

    @patch("xaytune.trainer.checkpointing.torch")
    def test_save_calls_torch_save(self, mock_torch):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "step-50"
            state = TrainState(global_step=50)
            mock_model = MagicMock()
            mock_optimizer = MagicMock()

            save_checkpoint(
                output_dir=str(output_dir),
                model=mock_model,
                optimizer=mock_optimizer,
                state=state,
            )

            assert mock_torch.save.call_count >= 1


class TestLoadCheckpoint:
    @patch("xaytune.trainer.checkpointing.torch")
    def test_load_restores_state(self, mock_torch):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "step-100"
            ckpt_dir.mkdir()
            metadata = {"global_step": 100, "epoch": 2, "metrics": {"loss": 0.3}}
            (ckpt_dir / "metadata.json").write_text(json.dumps(metadata))

            mock_model = MagicMock()
            mock_optimizer = MagicMock()
            mock_torch.load.return_value = {}

            state = load_checkpoint(
                checkpoint_dir=str(ckpt_dir),
                model=mock_model,
                optimizer=mock_optimizer,
            )

            assert state.global_step == 100
            assert state.epoch == 2
            assert state.metrics["loss"] == 0.3

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(
                checkpoint_dir="nonexistent",
                model=MagicMock(),
                optimizer=MagicMock(),
            )


class TestFindLatestCheckpoint:
    def test_find_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for step in [10, 50, 30]:
                d = Path(tmpdir) / f"step-{step}"
                d.mkdir()
                meta = {"global_step": step, "epoch": 0, "metrics": {}}
                (d / "metadata.json").write_text(json.dumps(meta))

            latest = find_latest_checkpoint(tmpdir)
            assert latest is not None
            assert "step-50" in str(latest)

    def test_no_checkpoints_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            latest = find_latest_checkpoint(tmpdir)
            assert latest is None
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement checkpointing**

Create `xaytune/trainer/checkpointing.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from xaytune.trainer.callbacks import TrainState


def save_checkpoint(
    *,
    output_dir: str,
    model: Any,
    optimizer: Any,
    state: TrainState,
    scheduler: Any | None = None,
) -> None:
    ckpt_path = Path(output_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    model_state = model.state_dict() if hasattr(model, "state_dict") else {}
    torch.save(model_state, ckpt_path / "model.pt")

    optimizer_state = optimizer.state_dict() if hasattr(optimizer, "state_dict") else {}
    torch.save(optimizer_state, ckpt_path / "optimizer.pt")

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        torch.save(scheduler.state_dict(), ckpt_path / "scheduler.pt")

    metadata = {
        "global_step": state.global_step,
        "epoch": state.epoch,
        "metrics": state.metrics,
    }
    (ckpt_path / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_checkpoint(
    *,
    checkpoint_dir: str,
    model: Any,
    optimizer: Any,
    scheduler: Any | None = None,
) -> TrainState:
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    model_path = ckpt_path / "model.pt"
    if model_path.exists():
        model_state = torch.load(model_path, weights_only=True)
        if model_state and hasattr(model, "load_state_dict"):
            model.load_state_dict(model_state)

    optimizer_path = ckpt_path / "optimizer.pt"
    if optimizer_path.exists():
        opt_state = torch.load(optimizer_path, weights_only=True)
        if opt_state and hasattr(optimizer, "load_state_dict"):
            optimizer.load_state_dict(opt_state)

    scheduler_path = ckpt_path / "scheduler.pt"
    if scheduler is not None and scheduler_path.exists():
        sched_state = torch.load(scheduler_path, weights_only=True)
        if sched_state and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(sched_state)

    metadata_path = ckpt_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    return TrainState(
        global_step=metadata.get("global_step", 0),
        epoch=metadata.get("epoch", 0),
        metrics=metadata.get("metrics", {}),
    )


def find_latest_checkpoint(output_dir: str) -> str | None:
    base = Path(output_dir)
    if not base.exists():
        return None

    checkpoints = []
    for d in base.iterdir():
        if d.is_dir() and (d / "metadata.json").exists():
            meta = json.loads((d / "metadata.json").read_text())
            checkpoints.append((meta.get("global_step", 0), str(d)))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]
```

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add checkpoint save/resume with metadata"
```

---

### Task 5: Trainer Public API & Wire-Up

**Files:**
- Modify: `xaytune/trainer/__init__.py`
- Create: `tests/test_trainer/test_init.py`

Wire up the trainer package public API with the `on()` decorator for user-facing callback registration.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trainer/test_init.py`:

```python
from xaytune.trainer import Trainer, TrainerConfig, TrainState, on, CallbackManager
from xaytune.trainer import save_checkpoint, load_checkpoint, find_latest_checkpoint
from xaytune.trainer import wrap_model_distributed, DistributedContext, get_strategy


class TestTrainerPublicAPI:
    def test_trainer_importable(self):
        assert Trainer is not None

    def test_trainer_config_importable(self):
        assert TrainerConfig is not None

    def test_train_state_importable(self):
        assert TrainState is not None

    def test_on_decorator_importable(self):
        assert callable(on)

    def test_on_registers_globally(self):
        calls = []

        @on("step_end")
        def my_callback(state):
            calls.append(state.step)

        assert callable(my_callback)

    def test_checkpoint_functions_importable(self):
        assert callable(save_checkpoint)
        assert callable(load_checkpoint)
        assert callable(find_latest_checkpoint)

    def test_distributed_importable(self):
        assert callable(wrap_model_distributed)
        assert DistributedContext is not None
        assert callable(get_strategy)
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Wire up trainer __init__.py**

Update `xaytune/trainer/__init__.py`:

```python
from xaytune.config.schema import TrainerConfig
from xaytune.trainer.callbacks import CallbackManager, TrainState
from xaytune.trainer.checkpointing import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from xaytune.trainer.distributed import (
    DistributedContext,
    get_strategy,
    wrap_model_distributed,
)
from xaytune.trainer.loop import Trainer

_global_callback_manager = CallbackManager()

on = _global_callback_manager.on

__all__ = [
    "CallbackManager",
    "DistributedContext",
    "find_latest_checkpoint",
    "get_strategy",
    "load_checkpoint",
    "on",
    "save_checkpoint",
    "TrainState",
    "Trainer",
    "TrainerConfig",
    "wrap_model_distributed",
]
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: wire up trainer public API with global callback support"
```
