"""NativeWorker: what it reports, and what it refuses to do.

The callbacks are driven here by a synthetic event sequence rather than a real
trainer, because the decisions being pinned are about *which number means
what* -- and the real loop only exercises one accumulation setting, one
schedule, and no divergence.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from xaytune.runtimes.worker import ObservationWriter
from xaytune.workers.native import (
    _publish_model,
    register_native_observation_callbacks,
)


class _Callbacks:
    """Just the part of CallbackManager the worker uses: ``on`` and ``fire``."""

    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}

    def on(self, event: str):
        def decorator(fn):
            self._handlers.setdefault(event, []).append(fn)
            return fn

        return decorator

    def fire(self, event: str, state) -> None:
        for handler in self._handlers.get(event, []):
            handler(state)


def _state(**metrics: object) -> SimpleNamespace:
    return SimpleNamespace(global_step=0, metrics=dict(metrics))


def _written(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line)["observation"] for line in path.read_text().splitlines()]


def _wired(tmp_path: Path, *, accumulation: int = 1) -> tuple[_Callbacks, Path]:
    path = tmp_path / "observations.jsonl"
    callbacks = _Callbacks()
    register_native_observation_callbacks(
        callbacks, ObservationWriter(path), gradient_accumulation=accumulation
    )
    return callbacks, path


def _optimizer_step(callbacks: _Callbacks, state: SimpleNamespace, *, loss: float) -> None:
    """One optimizer step as the real loop orders it.

    ``step_start`` sees the metrics left by the *previous* step; the loop then
    trains, records the new loss and -- after ``scheduler.step()`` -- the rate
    for the step after this one; then ``step_end`` fires.
    """
    callbacks.fire("step_start", state)
    state.metrics["loss"] = loss
    state.global_step += 1
    state.metrics["learning_rate"] = 1.0 / (state.global_step + 1)  # the *next* rate
    callbacks.fire("step_end", state)


# ---- the learning rate is the one the step used ---------------------------


def test_the_reported_learning_rate_is_the_one_the_step_used(tmp_path: Path) -> None:
    """The loop records the rate *after* stepping the scheduler.

    Read at ``step_end`` it would be the next step's rate -- every value one
    step early, and a warmup curve drawn wrongly. It is read at ``step_start``,
    where it is still the rate this step trained with.
    """
    callbacks, path = _wired(tmp_path)
    state = _state()

    callbacks.fire("train_start", state)
    for loss in (2.0, 1.5, 1.0):
        _optimizer_step(callbacks, state, loss=loss)

    metrics = [o for o in _written(path) if o["type"] == "TrainingMetricObserved"]
    assert [m["learning_rate"] for m in metrics] == [None, 1 / 2, 1 / 3]
    #                                                 ^ unknown, not zero:
    # nothing had recorded a rate before the first step.


# ---- loss under gradient accumulation -------------------------------------


def test_without_accumulation_the_loss_is_the_steps_loss(tmp_path: Path) -> None:
    callbacks, path = _wired(tmp_path, accumulation=1)
    _optimizer_step(callbacks, _state(), loss=0.75)

    (metric,) = [o for o in _written(path) if o["type"] == "TrainingMetricObserved"]
    assert metric["loss"] == 0.75


def test_under_accumulation_the_loss_is_not_claimed_as_the_steps(tmp_path: Path) -> None:
    """The loop overwrites ``loss`` every micro-batch.

    With accumulation, the value at ``step_end`` is only the last micro-batch's.
    Labelling it ``loss`` would claim a number the trainer never computed;
    averaging would invent one. It is reported under its true name instead.
    """
    callbacks, path = _wired(tmp_path, accumulation=4)
    _optimizer_step(callbacks, _state(), loss=0.75)

    (metric,) = [o for o in _written(path) if o["type"] == "TrainingMetricObserved"]
    assert metric["loss"] is None
    assert metric["metadata"] == {"final_micro_batch_loss": 0.75}


# ---- divergence is evidence, not a bad value ------------------------------


@pytest.mark.parametrize(
    ("loss", "reported"),
    [
        (float("nan"), "nan"),
        (float("inf"), "positive-infinity"),
        (float("-inf"), "negative-infinity"),
    ],
)
def test_a_nonfinite_loss_is_reported_as_instability(
    tmp_path: Path, loss: float, reported: str
) -> None:
    """The one signal that a run diverged must not be dropped as invalid.

    A NaN cannot be a JSON number, so ``TrainingMetricObserved`` refuses it --
    and rejecting it as a bad metric would silently lose the divergence. The
    vocabulary has a symbolic event for exactly this.
    """
    callbacks, path = _wired(tmp_path)
    _optimizer_step(callbacks, _state(), loss=loss)

    (observation,) = _written(path)
    assert observation["type"] == "NumericalInstabilityObserved"
    assert observation["quantity"] == "loss"
    assert observation["observation"] == reported


# ---- failure domains -------------------------------------------------------


def test_one_bad_value_is_dropped_and_training_continues(tmp_path: Path) -> None:
    """The observation channel is not authoritative; one bad metric is no
    reason to lose a run."""
    callbacks, path = _wired(tmp_path)
    state = _state()

    # A negative rate is refused by the vocabulary.
    callbacks.fire("step_start", SimpleNamespace(global_step=0, metrics={"learning_rate": -1.0}))
    state.metrics["loss"] = 1.0
    state.global_step = 1
    callbacks.fire("step_end", state)  # must not raise

    callbacks.fire("train_end", state)
    assert [o["type"] for o in _written(path)] == ["TrainingCompleted"]


def test_a_broken_channel_is_not_swallowed(tmp_path: Path) -> None:
    """Writing is the runtime's channel. Failing to write is a contract
    failure, and training blind would be worse than stopping."""
    callbacks, _ = _wired(tmp_path / "does-not-exist" / "nested")

    with pytest.raises(OSError):
        callbacks.fire("train_start", _state())


def test_the_callbacks_only_observe(tmp_path: Path) -> None:
    """Nothing here may steer the run.

    A telemetry callback able to stop training or change a rate would be a
    way to alter a run that bypasses ``Action → TrainingIntervention``.
    """
    callbacks, _ = _wired(tmp_path)
    state = _state(loss=1.0, learning_rate=0.1)
    state.should_stop = False
    before = copy.deepcopy(vars(state))

    for event in ("train_start", "step_start", "step_end", "train_end"):
        callbacks.fire(event, state)

    assert vars(state) == before


# ---- publication -----------------------------------------------------------


def test_a_failed_save_is_a_publication_failure_not_a_training_failure(
    tmp_path: Path,
) -> None:
    """Training had completed. Saying it failed would be false.

    The process still exits non-zero -- the plan's declared output does not
    exist -- but the evidence names the step that actually went wrong.
    """

    class _Unsaveable:
        def save_pretrained(self, _path: object) -> None:
            raise PermissionError("read-only filesystem")

    from xaytune.workers.native_schema import (
        NativeData,
        NativeModel,
        NativeOptimization,
        NativeRealization,
        NativeSftConfig,
    )

    config = NativeSftConfig(
        model=NativeModel(uri="/m"),
        data=NativeData(path="/d", format="text", max_seq_length=8, packing=False),
        optimization=NativeOptimization(
            learning_rate=1e-3,
            micro_batch_size=1,
            gradient_accumulation=1,
            epochs=1,
            max_steps=None,
            max_grad_norm=1.0,
            weight_decay=0.0,
            scheduler="constant",
            warmup_steps=None,
            warmup_ratio=None,
            mixed_precision="fp32",
        ),
        realization=NativeRealization(
            seed=1, output_dir=str(tmp_path), checkpoint_every_optimizer_steps=0
        ),
    )
    path = tmp_path / "observations.jsonl"

    with pytest.raises(PermissionError):
        _publish_model(_Unsaveable(), None, config, ObservationWriter(path))

    (observation,) = _written(path)
    assert observation["type"] == "IncidentObserved"
    assert observation["reason"] == "artifact-publication-failed"
    assert "ArtifactProduced" not in [o["type"] for o in _written(path)], (
        "an artifact announced before it existed would be a claim with nothing behind it"
    )


# ---- the worker's own preconditions ---------------------------------------


def test_no_config_path_means_no_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside a runtime there is nothing to run and nowhere to report."""
    from xaytune.workers.native import main

    monkeypatch.delenv("XAYTUNE_WORKER_CONFIG_PATH", raising=False)
    monkeypatch.delenv("XAYTUNE_OBSERVATIONS_PATH", raising=False)

    with pytest.raises(RuntimeError, match="XAYTUNE_WORKER_CONFIG_PATH"):
        main()


def test_an_unwritable_channel_fails_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail at the start, not an hour in on the first report.

    A runtime that promised an observation channel and delivered a broken one
    is an execution-contract failure. The check happens before the model is
    loaded, so the cost of finding out is nothing.
    """
    from xaytune.workers.native import main

    monkeypatch.setenv("XAYTUNE_WORKER_CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setenv("XAYTUNE_OBSERVATIONS_PATH", str(tmp_path / "missing" / "o.jsonl"))

    with pytest.raises(OSError):
        main()
    assert not (tmp_path / "config.json").exists(), "it failed before reading anything"


# ---- a declared precision is honoured or refused, never degraded ----------


def _cpu_model():
    import torch

    return torch.nn.Linear(2, 2)


def test_full_precision_needs_nothing_from_the_device() -> None:
    from xaytune.workers.native import require_honourable_precision

    require_honourable_precision(_cpu_model(), "fp32")


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_half_precision_on_cpu_is_refused_not_trained_in_fp32(precision: str) -> None:
    """The loop switches autocast off on CPU and says nothing; this says no.

    torch's CPU autocast would accept either dtype -- the refusal is about what
    the native loop does, which is to not use it.
    """
    from xaytune.workers.native import UnsupportedPrecisionError, require_honourable_precision

    with pytest.raises(UnsupportedPrecisionError, match=rf"{precision}.*cpu.*fp32"):
        require_honourable_precision(_cpu_model(), precision)


def test_bf16_on_a_cuda_device_without_bf16_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU that runs autocast does not necessarily have bf16."""
    import torch

    import xaytune.trainer.device as device
    from xaytune.workers.native import UnsupportedPrecisionError, require_honourable_precision

    monkeypatch.setattr(device, "detect_device_type_from_model", lambda _model: "cuda")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: False)

    with pytest.raises(UnsupportedPrecisionError, match="does not support bf16"):
        require_honourable_precision(_cpu_model(), "bf16")


def test_a_backend_that_accepts_the_request_but_computes_otherwise_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe checks the dtype produced, not that autocast did not raise."""
    import contextlib

    import torch

    import xaytune.trainer.device as device
    from xaytune.workers.native import UnsupportedPrecisionError, require_honourable_precision

    monkeypatch.setattr(device, "supports_amp", lambda _device_type: True)
    monkeypatch.setattr(torch.amp, "autocast", lambda *a, **k: contextlib.nullcontext())

    with pytest.raises(UnsupportedPrecisionError, match="computes in torch.float32"):
        require_honourable_precision(_cpu_model(), "bf16")


def test_an_unhonourable_precision_fails_the_run_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At the worker boundary: after the model is placed, before ``train()``.

    Reported as ``TrainingFailed`` -- the run did not train, and it did not
    train in something else either.
    """
    import xaytune.recipes.base as recipes
    from xaytune.compilation import CompilationContext
    from xaytune.compilation.native import NativeCompiler
    from xaytune.core.domain.candidate import (
        CandidateSpec,
        DataSpec,
        LRScheduleSpec,
        ModelSpec,
        OptimizationSpec,
        OptimizerSpec,
        PrecisionSpec,
        TrainingKind,
        TrainingSpec,
    )
    from xaytune.core.immutable import thaw
    from xaytune.core.refs import DatasetRef, ModelRef
    from xaytune.workers.native import UnsupportedPrecisionError, main

    candidate = CandidateSpec(
        model=ModelSpec(model=ModelRef(uri="/models/m")),
        data=DataSpec(
            dataset=DatasetRef(uri="/data/d.jsonl"), format="text", max_seq_length=8, packing=False
        ),
        training=TrainingSpec(
            kind=TrainingKind.SFT,
            optimization=OptimizationSpec(
                optimizer=OptimizerSpec(name="adamw", weight_decay=0.0),
                lr_schedule=LRScheduleSpec(name="constant"),
                learning_rate=1e-3,
                micro_batch_size=1,
                gradient_accumulation=1,
                epochs=1,
                max_grad_norm=1.0,
            ),
            precision=PrecisionSpec(dtype="fp16"),
        ),
    )
    spec = NativeCompiler().compile(
        candidate, CompilationContext(run_id="r", seed=1, output_uri=str(tmp_path / "out"))
    )
    config_path = tmp_path / "worker-config.json"
    config_path.write_text(json.dumps(thaw(spec.config)))
    observations = tmp_path / "observations.jsonl"
    observations.touch()
    monkeypatch.setenv("XAYTUNE_WORKER_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("XAYTUNE_OBSERVATIONS_PATH", str(observations))

    trained: list[bool] = []
    placed = SimpleNamespace(
        model=_cpu_model(),
        tokenizer=None,
        train_dataloader=[],
        resume_state=None,
        trainer=SimpleNamespace(train=lambda **_: trained.append(True)),
    )
    monkeypatch.setattr(recipes, "setup_training", lambda *_a, **_k: placed)

    with pytest.raises(UnsupportedPrecisionError):
        main()

    assert trained == [], "train() must not run"
    (failed,) = _written(observations)
    assert failed["type"] == "TrainingFailed"
    assert failed["reason"] == "unsupported-precision-error"
    assert not (tmp_path / "out").exists(), "and nothing was published"
