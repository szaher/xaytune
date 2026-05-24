# Callbacks

xaytune provides an event-driven callback system for hooking into the training loop. You can log custom metrics, implement early stopping, send notifications, or run any arbitrary code at specific training events.

## TrainState

`TrainState` is a dataclass that tracks the current state of training. It is passed to every callback and returned by recipe functions.

```python
from xaytune.trainer.callbacks import TrainState
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `step` | `int` | `0` | Current step within the epoch |
| `epoch` | `int` | `0` | Current epoch number |
| `global_step` | `int` | `0` | Total steps across all epochs |
| `num_epochs` | `int` | `0` | Total number of epochs |
| `max_steps` | `int` | `-1` | Maximum steps limit (-1 = unlimited) |
| `metrics` | `dict[str, Any]` | `{}` | Current metrics (loss, learning rate, etc.) |
| `should_stop` | `bool` | `False` | Set to `True` to stop training early |

### Stopping Training Early

Call `state.stop_training()` from any callback to stop after the current step:

```python
@callbacks.on("step_end")
def early_stop(state):
    if state.metrics.get("loss", float("inf")) < 0.01:
        print("Loss target reached, stopping early.")
        state.stop_training()
```

## CallbackManager

`CallbackManager` manages callback registration and event firing. Use its `on()` decorator to register callbacks.

```python
from xaytune.trainer.callbacks import CallbackManager

callbacks = CallbackManager()

@callbacks.on("step_end")
def log_loss(state):
    if state.global_step % 100 == 0:
        print(f"Step {state.global_step}: loss={state.metrics.get('loss', 'N/A')}")

@callbacks.on("train_end")
def on_complete(state):
    print(f"Training complete after {state.global_step} steps.")
```

### Available Events

| Event | Fired When |
|-------|-----------|
| `train_start` | Training begins |
| `train_end` | Training finishes |
| `epoch_start` | An epoch begins |
| `epoch_end` | An epoch finishes |
| `step_start` | A training step begins |
| `step_end` | A training step finishes |
| `eval_start` | Evaluation begins |
| `eval_end` | Evaluation finishes |
| `checkpoint_saved` | A checkpoint is saved to disk |
| `error` | An error occurs during training |

### Firing Events

The trainer fires events automatically. If you are building custom training loops, fire events manually:

```python
callbacks.fire("step_end", state)
```

## Examples

### Gradient Logging

```python
@callbacks.on("step_end")
def log_gradients(state):
    if state.global_step % 50 == 0:
        grad_norm = state.metrics.get("grad_norm")
        if grad_norm is not None:
            print(f"Step {state.global_step}: grad_norm={grad_norm:.4f}")
```

### Checkpoint Notification

```python
@callbacks.on("checkpoint_saved")
def notify_checkpoint(state):
    print(f"Checkpoint saved at step {state.global_step}")
```

### Error Handling

```python
@callbacks.on("error")
def handle_error(state):
    print(f"Error at step {state.global_step}: {state.metrics.get('error')}")
```

---

## Full API Reference

::: xaytune.trainer.callbacks.TrainState

::: xaytune.trainer.callbacks.CallbackManager
