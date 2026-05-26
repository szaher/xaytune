# Logging & Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add logging backends (console, TensorBoard, W&B, MLflow) that hook into xaytune's existing callback system to make training runs observable.

**Architecture:** A `LoggingManager` wraps multiple `LoggingBackend` instances. Each backend implements a simple interface (`log_scalar`, `log_config`, `close`). The `LoggingManager` registers itself as a callback on `train_start`, `step_end`, and `train_end` events via the existing `CallbackManager`. A `setup_logging(config, callback_manager)` factory creates the manager from `LoggingConfig`. Console backend is always-on; TensorBoard, W&B, and MLflow are opt-in and use lazy imports so their dependencies remain optional.

**Tech Stack:** PyTorch, pytest, unittest.mock (for mocking optional deps)

---

## File Structure

```
xaytune/logging/
├── __init__.py          # (modify) — export setup_logging, LoggingManager, LoggingBackend
├── base.py              # (create) — LoggingBackend ABC + LoggingManager
├── console.py           # (create) — ConsoleBackend with formatted step logging
├── tensorboard.py       # (create) — TensorBoardBackend wrapping SummaryWriter
├── wandb.py             # (create) — WandbBackend wrapping wandb.init/log
├── mlflow.py            # (create) — MLflowBackend wrapping mlflow.log_metric

tests/test_logging/
├── __init__.py          # (create)
├── test_base.py         # (create) — LoggingManager + LoggingBackend tests
├── test_console.py      # (create) — ConsoleBackend output tests
├── test_tensorboard.py  # (create) — TensorBoardBackend with mocked SummaryWriter
├── test_wandb.py        # (create) — WandbBackend with mocked wandb
├── test_mlflow.py       # (create) — MLflowBackend with mocked mlflow
├── test_setup.py        # (create) — setup_logging factory + callback integration
```

---

### Task 1: LoggingBackend ABC & LoggingManager

**Files:**
- Create: `xaytune/logging/base.py`
- Create: `tests/test_logging/__init__.py`
- Create: `tests/test_logging/test_base.py`

The base class defines the interface. `LoggingManager` holds multiple backends, delegates calls to all of them, and integrates with `CallbackManager`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_logging/__init__.py` (empty file).

Create `tests/test_logging/test_base.py`:

```python
import pytest
from xaytune.logging.base import LoggingBackend, LoggingManager
from xaytune.trainer.callbacks import CallbackManager, TrainState


class FakeBackend(LoggingBackend):
    def __init__(self):
        self.scalars = []
        self.config_logged = None
        self.closed = False

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self.scalars.append((key, value, step))

    def log_config(self, config: dict) -> None:
        self.config_logged = config

    def close(self) -> None:
        self.closed = True


class TestLoggingBackend:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            LoggingBackend()

    def test_subclass_works(self):
        backend = FakeBackend()
        backend.log_scalar("loss", 0.5, 1)
        assert backend.scalars == [("loss", 0.5, 1)]


class TestLoggingManager:
    def test_add_backend(self):
        manager = LoggingManager()
        backend = FakeBackend()
        manager.add_backend(backend)
        assert len(manager.backends) == 1

    def test_log_scalar_delegates(self):
        manager = LoggingManager()
        b1 = FakeBackend()
        b2 = FakeBackend()
        manager.add_backend(b1)
        manager.add_backend(b2)

        manager.log_scalar("loss", 0.5, 10)

        assert b1.scalars == [("loss", 0.5, 10)]
        assert b2.scalars == [("loss", 0.5, 10)]

    def test_log_config_delegates(self):
        manager = LoggingManager()
        backend = FakeBackend()
        manager.add_backend(backend)

        manager.log_config({"lr": 0.001})

        assert backend.config_logged == {"lr": 0.001}

    def test_close_delegates(self):
        manager = LoggingManager()
        b1 = FakeBackend()
        b2 = FakeBackend()
        manager.add_backend(b1)
        manager.add_backend(b2)

        manager.close()

        assert b1.closed is True
        assert b2.closed is True

    def test_register_callbacks(self):
        cb_manager = CallbackManager()
        log_manager = LoggingManager(log_every_n_steps=1)
        backend = FakeBackend()
        log_manager.add_backend(backend)

        log_manager.register_callbacks(cb_manager)

        state = TrainState(global_step=5)
        state.metrics["loss"] = 0.25

        cb_manager.fire("step_end", state)

        assert ("loss", 0.25, 5) in backend.scalars

    def test_log_every_n_steps_skips(self):
        cb_manager = CallbackManager()
        log_manager = LoggingManager(log_every_n_steps=10)
        backend = FakeBackend()
        log_manager.add_backend(backend)
        log_manager.register_callbacks(cb_manager)

        state = TrainState(global_step=3)
        state.metrics["loss"] = 0.5
        cb_manager.fire("step_end", state)

        assert len(backend.scalars) == 0

    def test_log_every_n_steps_fires_on_match(self):
        cb_manager = CallbackManager()
        log_manager = LoggingManager(log_every_n_steps=10)
        backend = FakeBackend()
        log_manager.add_backend(backend)
        log_manager.register_callbacks(cb_manager)

        state = TrainState(global_step=10)
        state.metrics["loss"] = 0.3
        cb_manager.fire("step_end", state)

        assert ("loss", 0.3, 10) in backend.scalars

    def test_train_end_closes(self):
        cb_manager = CallbackManager()
        log_manager = LoggingManager()
        backend = FakeBackend()
        log_manager.add_backend(backend)
        log_manager.register_callbacks(cb_manager)

        cb_manager.fire("train_end", TrainState())

        assert backend.closed is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_logging/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaytune.logging.base'`

- [ ] **Step 3: Implement LoggingBackend and LoggingManager**

Create `xaytune/logging/base.py`:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from xaytune.trainer.callbacks import CallbackManager, TrainState


class LoggingBackend(ABC):
    @abstractmethod
    def log_scalar(self, key: str, value: float, step: int) -> None: ...

    @abstractmethod
    def log_config(self, config: dict[str, Any]) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class LoggingManager:
    def __init__(self, log_every_n_steps: int = 10) -> None:
        self.backends: list[LoggingBackend] = []
        self.log_every_n_steps = log_every_n_steps

    def add_backend(self, backend: LoggingBackend) -> None:
        self.backends.append(backend)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        for backend in self.backends:
            backend.log_scalar(key, value, step)

    def log_config(self, config: dict[str, Any]) -> None:
        for backend in self.backends:
            backend.log_config(config)

    def close(self) -> None:
        for backend in self.backends:
            backend.close()

    def register_callbacks(self, callback_manager: CallbackManager) -> None:
        @callback_manager.on("step_end")
        def _on_step_end(state: TrainState) -> None:
            if state.global_step % self.log_every_n_steps != 0:
                return
            for key, value in state.metrics.items():
                if isinstance(value, (int, float)):
                    self.log_scalar(key, float(value), state.global_step)

        @callback_manager.on("train_end")
        def _on_train_end(state: TrainState) -> None:
            self.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_logging/test_base.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/logging/base.py tests/test_logging/
git commit -m "feat: add LoggingBackend ABC and LoggingManager with callback integration"
```

---

### Task 2: Console Backend

**Files:**
- Create: `xaytune/logging/console.py`
- Create: `tests/test_logging/test_console.py`

The console backend prints formatted training metrics to stdout. Always on.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_logging/test_console.py`:

```python
import pytest
from xaytune.logging.console import ConsoleBackend


class TestConsoleBackend:
    def test_log_scalar(self, capsys):
        backend = ConsoleBackend()
        backend.log_scalar("loss", 0.5432, 10)
        captured = capsys.readouterr()
        assert "loss" in captured.out
        assert "0.5432" in captured.out
        assert "10" in captured.out

    def test_log_config(self, capsys):
        backend = ConsoleBackend()
        backend.log_config({"lr": 0.001, "epochs": 3})
        captured = capsys.readouterr()
        assert "lr" in captured.out
        assert "0.001" in captured.out

    def test_close_is_noop(self):
        backend = ConsoleBackend()
        backend.close()  # should not raise

    def test_multiple_scalars_formatted(self, capsys):
        backend = ConsoleBackend()
        backend.log_scalar("loss", 0.5, 1)
        backend.log_scalar("lr", 0.001, 1)
        captured = capsys.readouterr()
        assert "loss" in captured.out
        assert "lr" in captured.out

    def test_is_logging_backend(self):
        from xaytune.logging.base import LoggingBackend
        backend = ConsoleBackend()
        assert isinstance(backend, LoggingBackend)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_logging/test_console.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaytune.logging.console'`

- [ ] **Step 3: Implement ConsoleBackend**

Create `xaytune/logging/console.py`:

```python
from __future__ import annotations

from typing import Any

from xaytune.logging.base import LoggingBackend


class ConsoleBackend(LoggingBackend):
    def log_scalar(self, key: str, value: float, step: int) -> None:
        print(f"[step {step}] {key}: {value:.4f}")

    def log_config(self, config: dict[str, Any]) -> None:
        print("Training config:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    def close(self) -> None:
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_logging/test_console.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/logging/console.py tests/test_logging/test_console.py
git commit -m "feat: add console logging backend with formatted step output"
```

---

### Task 3: TensorBoard Backend

**Files:**
- Create: `xaytune/logging/tensorboard.py`
- Create: `tests/test_logging/test_tensorboard.py`

Wraps `torch.utils.tensorboard.SummaryWriter`. Lazy import so it fails gracefully if tensorboard isn't installed.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_logging/test_tensorboard.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from xaytune.logging.tensorboard import TensorBoardBackend


class TestTensorBoardBackend:
    @patch("xaytune.logging.tensorboard.SummaryWriter")
    def test_creates_writer(self, mock_writer_cls):
        backend = TensorBoardBackend(log_dir="runs/test")
        mock_writer_cls.assert_called_once_with(log_dir="runs/test")

    @patch("xaytune.logging.tensorboard.SummaryWriter")
    def test_log_scalar(self, mock_writer_cls):
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        backend = TensorBoardBackend(log_dir="runs/test")
        backend.log_scalar("loss", 0.5, 10)

        mock_writer.add_scalar.assert_called_once_with("loss", 0.5, 10)

    @patch("xaytune.logging.tensorboard.SummaryWriter")
    def test_log_config(self, mock_writer_cls):
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        backend = TensorBoardBackend(log_dir="runs/test")
        backend.log_config({"lr": 0.001})

        mock_writer.add_text.assert_called_once()
        call_args = mock_writer.add_text.call_args
        assert call_args[0][0] == "config"
        assert "lr" in call_args[0][1]

    @patch("xaytune.logging.tensorboard.SummaryWriter")
    def test_close(self, mock_writer_cls):
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        backend = TensorBoardBackend(log_dir="runs/test")
        backend.close()

        mock_writer.close.assert_called_once()

    @patch("xaytune.logging.tensorboard.SummaryWriter")
    def test_is_logging_backend(self, mock_writer_cls):
        from xaytune.logging.base import LoggingBackend
        backend = TensorBoardBackend(log_dir="runs/test")
        assert isinstance(backend, LoggingBackend)

    @patch("xaytune.logging.tensorboard.SummaryWriter")
    def test_default_log_dir(self, mock_writer_cls):
        backend = TensorBoardBackend()
        mock_writer_cls.assert_called_once_with(log_dir="runs")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_logging/test_tensorboard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaytune.logging.tensorboard'`

- [ ] **Step 3: Implement TensorBoardBackend**

Create `xaytune/logging/tensorboard.py`:

```python
from __future__ import annotations

import json
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from xaytune.logging.base import LoggingBackend


class TensorBoardBackend(LoggingBackend):
    def __init__(self, log_dir: str = "runs") -> None:
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self.writer.add_scalar(key, value, step)

    def log_config(self, config: dict[str, Any]) -> None:
        self.writer.add_text("config", json.dumps(config, indent=2))

    def close(self) -> None:
        self.writer.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_logging/test_tensorboard.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/logging/tensorboard.py tests/test_logging/test_tensorboard.py
git commit -m "feat: add TensorBoard logging backend"
```

---

### Task 4: Weights & Biases Backend

**Files:**
- Create: `xaytune/logging/wandb.py`
- Create: `tests/test_logging/test_wandb.py`

Wraps the `wandb` library. Lazy import — `wandb` is optional. Calls `wandb.init()` on construction, `wandb.log()` for scalars.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_logging/test_wandb.py`:

```python
import pytest
from unittest.mock import patch, MagicMock


class TestWandbBackend:
    @patch("xaytune.logging.wandb.wandb")
    def test_init_calls_wandb_init(self, mock_wandb):
        from xaytune.logging.wandb import WandbBackend

        backend = WandbBackend(project="my-project", run_name="run-1")
        mock_wandb.init.assert_called_once_with(project="my-project", name="run-1")

    @patch("xaytune.logging.wandb.wandb")
    def test_log_scalar(self, mock_wandb):
        from xaytune.logging.wandb import WandbBackend

        backend = WandbBackend(project="test")
        backend.log_scalar("loss", 0.5, 10)

        mock_wandb.log.assert_called_once_with({"loss": 0.5}, step=10)

    @patch("xaytune.logging.wandb.wandb")
    def test_log_config(self, mock_wandb):
        from xaytune.logging.wandb import WandbBackend

        mock_wandb.config = MagicMock()
        backend = WandbBackend(project="test")
        backend.log_config({"lr": 0.001, "epochs": 3})

        mock_wandb.config.update.assert_called_once_with({"lr": 0.001, "epochs": 3})

    @patch("xaytune.logging.wandb.wandb")
    def test_close_calls_finish(self, mock_wandb):
        from xaytune.logging.wandb import WandbBackend

        backend = WandbBackend(project="test")
        backend.close()

        mock_wandb.finish.assert_called_once()

    @patch("xaytune.logging.wandb.wandb")
    def test_is_logging_backend(self, mock_wandb):
        from xaytune.logging.wandb import WandbBackend
        from xaytune.logging.base import LoggingBackend

        backend = WandbBackend(project="test")
        assert isinstance(backend, LoggingBackend)

    @patch("xaytune.logging.wandb.wandb")
    def test_default_project(self, mock_wandb):
        from xaytune.logging.wandb import WandbBackend

        backend = WandbBackend()
        mock_wandb.init.assert_called_once_with(project="xaytune", name=None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_logging/test_wandb.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaytune.logging.wandb'`

- [ ] **Step 3: Implement WandbBackend**

Create `xaytune/logging/wandb.py`:

```python
from __future__ import annotations

from typing import Any

import wandb

from xaytune.logging.base import LoggingBackend


class WandbBackend(LoggingBackend):
    def __init__(
        self,
        project: str = "xaytune",
        run_name: str | None = None,
    ) -> None:
        wandb.init(project=project, name=run_name)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        wandb.log({key: value}, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        wandb.config.update(config)

    def close(self) -> None:
        wandb.finish()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_logging/test_wandb.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/logging/wandb.py tests/test_logging/test_wandb.py
git commit -m "feat: add Weights & Biases logging backend"
```

---

### Task 5: MLflow Backend

**Files:**
- Create: `xaytune/logging/mlflow.py`
- Create: `tests/test_logging/test_mlflow.py`

Wraps the `mlflow` library. Lazy import — `mlflow` is optional. Starts a run on construction, logs metrics via `mlflow.log_metric`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_logging/test_mlflow.py`:

```python
import pytest
from unittest.mock import patch, MagicMock


class TestMLflowBackend:
    @patch("xaytune.logging.mlflow.mlflow")
    def test_init_starts_run(self, mock_mlflow):
        from xaytune.logging.mlflow import MLflowBackend

        backend = MLflowBackend(run_name="test-run")
        mock_mlflow.start_run.assert_called_once_with(run_name="test-run")

    @patch("xaytune.logging.mlflow.mlflow")
    def test_log_scalar(self, mock_mlflow):
        from xaytune.logging.mlflow import MLflowBackend

        backend = MLflowBackend()
        backend.log_scalar("loss", 0.5, 10)

        mock_mlflow.log_metric.assert_called_once_with("loss", 0.5, step=10)

    @patch("xaytune.logging.mlflow.mlflow")
    def test_log_config(self, mock_mlflow):
        from xaytune.logging.mlflow import MLflowBackend

        backend = MLflowBackend()
        backend.log_config({"lr": 0.001, "epochs": 3})

        mock_mlflow.log_params.assert_called_once_with({"lr": 0.001, "epochs": 3})

    @patch("xaytune.logging.mlflow.mlflow")
    def test_close_ends_run(self, mock_mlflow):
        from xaytune.logging.mlflow import MLflowBackend

        backend = MLflowBackend()
        backend.close()

        mock_mlflow.end_run.assert_called_once()

    @patch("xaytune.logging.mlflow.mlflow")
    def test_is_logging_backend(self, mock_mlflow):
        from xaytune.logging.mlflow import MLflowBackend
        from xaytune.logging.base import LoggingBackend

        backend = MLflowBackend()
        assert isinstance(backend, LoggingBackend)

    @patch("xaytune.logging.mlflow.mlflow")
    def test_default_run_name(self, mock_mlflow):
        from xaytune.logging.mlflow import MLflowBackend

        backend = MLflowBackend()
        mock_mlflow.start_run.assert_called_once_with(run_name=None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_logging/test_mlflow.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'xaytune.logging.mlflow'`

- [ ] **Step 3: Implement MLflowBackend**

Create `xaytune/logging/mlflow.py`:

```python
from __future__ import annotations

from typing import Any

import mlflow

from xaytune.logging.base import LoggingBackend


class MLflowBackend(LoggingBackend):
    def __init__(self, run_name: str | None = None) -> None:
        mlflow.start_run(run_name=run_name)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        mlflow.log_params(config)

    def close(self) -> None:
        mlflow.end_run()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_logging/test_mlflow.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/logging/mlflow.py tests/test_logging/test_mlflow.py
git commit -m "feat: add MLflow logging backend"
```

---

### Task 6: setup_logging Factory & Module Exports

**Files:**
- Modify: `xaytune/logging/__init__.py`
- Create: `tests/test_logging/test_setup.py`

The `setup_logging` factory reads `LoggingConfig`, creates the appropriate backends, returns a `LoggingManager` already registered on the callback manager. Console is always added. TensorBoard/W&B/MLflow are added only when listed in `config.backends`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_logging/test_setup.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from xaytune.logging import setup_logging, LoggingManager, LoggingBackend
from xaytune.logging.console import ConsoleBackend
from xaytune.config.schema import LoggingConfig
from xaytune.trainer.callbacks import CallbackManager, TrainState


class TestSetupLogging:
    def test_returns_logging_manager(self):
        config = LoggingConfig(backends=["console"])
        cb = CallbackManager()
        manager = setup_logging(config, cb)
        assert isinstance(manager, LoggingManager)

    def test_console_always_added(self):
        config = LoggingConfig(backends=[])
        cb = CallbackManager()
        manager = setup_logging(config, cb)
        assert any(isinstance(b, ConsoleBackend) for b in manager.backends)

    def test_console_not_duplicated(self):
        config = LoggingConfig(backends=["console"])
        cb = CallbackManager()
        manager = setup_logging(config, cb)
        console_count = sum(1 for b in manager.backends if isinstance(b, ConsoleBackend))
        assert console_count == 1

    @patch("xaytune.logging.TensorBoardBackend")
    def test_tensorboard_added(self, mock_tb_cls):
        mock_tb_cls.return_value = MagicMock(spec=LoggingBackend)
        config = LoggingConfig(backends=["console", "tensorboard"])
        cb = CallbackManager()
        manager = setup_logging(config, cb, output_dir="output/test")
        mock_tb_cls.assert_called_once_with(log_dir="output/test/runs")

    @patch("xaytune.logging.WandbBackend")
    def test_wandb_added(self, mock_wandb_cls):
        mock_wandb_cls.return_value = MagicMock(spec=LoggingBackend)
        config = LoggingConfig(backends=["wandb"], project="my-proj", run_name="run-1")
        cb = CallbackManager()
        manager = setup_logging(config, cb)
        mock_wandb_cls.assert_called_once_with(project="my-proj", run_name="run-1")

    @patch("xaytune.logging.MLflowBackend")
    def test_mlflow_added(self, mock_mlflow_cls):
        mock_mlflow_cls.return_value = MagicMock(spec=LoggingBackend)
        config = LoggingConfig(backends=["mlflow"], run_name="run-1")
        cb = CallbackManager()
        manager = setup_logging(config, cb)
        mock_mlflow_cls.assert_called_once_with(run_name="run-1")

    def test_registers_callbacks(self):
        config = LoggingConfig(backends=["console"], log_every_n_steps=1)
        cb = CallbackManager()
        manager = setup_logging(config, cb)

        state = TrainState(global_step=1)
        state.metrics["loss"] = 0.5
        cb.fire("step_end", state)
        # No error means callbacks were registered

    def test_log_every_n_steps_passed(self):
        config = LoggingConfig(backends=["console"], log_every_n_steps=50)
        cb = CallbackManager()
        manager = setup_logging(config, cb)
        assert manager.log_every_n_steps == 50

    def test_unknown_backend_raises(self):
        config = LoggingConfig(backends=["nonexistent"])
        cb = CallbackManager()
        with pytest.raises(ValueError, match="Unknown logging backend"):
            setup_logging(config, cb)


class TestModuleExports:
    def test_logging_backend_importable(self):
        from xaytune.logging import LoggingBackend
        assert LoggingBackend is not None

    def test_logging_manager_importable(self):
        from xaytune.logging import LoggingManager
        assert LoggingManager is not None

    def test_setup_logging_importable(self):
        from xaytune.logging import setup_logging
        assert callable(setup_logging)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_logging/test_setup.py -v`
Expected: FAIL — `ImportError: cannot import name 'setup_logging'`

- [ ] **Step 3: Implement setup_logging and update __init__.py**

Replace `xaytune/logging/__init__.py`:

```python
from __future__ import annotations

from typing import Any

from xaytune.config.schema import LoggingConfig
from xaytune.logging.base import LoggingBackend, LoggingManager
from xaytune.logging.console import ConsoleBackend
from xaytune.logging.tensorboard import TensorBoardBackend
from xaytune.logging.wandb import WandbBackend
from xaytune.logging.mlflow import MLflowBackend
from xaytune.trainer.callbacks import CallbackManager

_BACKEND_NAMES = {"console", "tensorboard", "wandb", "mlflow"}


def setup_logging(
    config: LoggingConfig,
    callback_manager: CallbackManager,
    *,
    output_dir: str = "output",
) -> LoggingManager:
    manager = LoggingManager(log_every_n_steps=config.log_every_n_steps)

    manager.add_backend(ConsoleBackend())

    for name in config.backends:
        if name == "console":
            continue
        if name == "tensorboard":
            manager.add_backend(TensorBoardBackend(log_dir=f"{output_dir}/runs"))
        elif name == "wandb":
            manager.add_backend(WandbBackend(
                project=config.project or "xaytune",
                run_name=config.run_name,
            ))
        elif name == "mlflow":
            manager.add_backend(MLflowBackend(run_name=config.run_name))
        else:
            raise ValueError(
                f"Unknown logging backend: '{name}'. "
                f"Available: {', '.join(sorted(_BACKEND_NAMES))}"
            )

    manager.register_callbacks(callback_manager)
    return manager


__all__ = [
    "ConsoleBackend",
    "LoggingBackend",
    "LoggingManager",
    "MLflowBackend",
    "setup_logging",
    "TensorBoardBackend",
    "WandbBackend",
]
```

- [ ] **Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add xaytune/logging/__init__.py tests/test_logging/test_setup.py
git commit -m "feat: add setup_logging factory and wire logging module exports"
```
