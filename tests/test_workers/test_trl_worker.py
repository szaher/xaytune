"""TRLWorker: every SFTConfig field decided, and a tripwire when one is not.

The classification is the worker's claim that nothing TRL does by default can
change this run behind the candidate's back. Two kinds of test pin it:

- **it holds for the installed TRL** -- every field is controlled or inert,
  every inert default is the one recorded, and ``SFTConfig`` keeps every
  controlled value it was given;
- **it bites** -- a new field, a moved default, or a controlled value rewritten
  after construction each stop the run. These are what make a TRL upgrade fail
  until someone has decided what changed.

The observation mapping is driven synthetically, as the native worker's is,
because the real loop exercises no divergence.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from xaytune.runtimes.worker import ObservationWriter
from xaytune.workers.trl import (
    INERT_FIELDS,
    TRLObservations,
    UnclassifiedBehaviourError,
    _read_text_dataset,
    sft_arguments,
    verify_classification,
)
from xaytune.workers.trl_schema import (
    TRLData,
    TRLModel,
    TRLOptimization,
    TRLRealization,
    TRLSftConfig,
)

pytestmark = pytest.mark.trl


def _spec(**optimization: object) -> TRLSftConfig:
    fields: dict[str, object] = {
        "learning_rate": 1e-3,
        "micro_batch_size": 2,
        "gradient_accumulation": 1,
        "epochs": 1,
        "max_steps": 2,
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "scheduler": "constant",
        "warmup_steps": 0,
        "mixed_precision": "fp32",
    }
    fields.update(optimization)
    return TRLSftConfig(
        model=TRLModel(uri="/m"),
        data=TRLData(path="/d.jsonl", max_length=32),
        optimization=TRLOptimization(**fields),  # type: ignore[arg-type]
        realization=TRLRealization(seed=7, output_dir="/o"),
    )


def _written(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line)["observation"] for line in path.read_text().splitlines()]


# ---- the classification is total and disjoint ------------------------------


def test_no_field_is_both_controlled_and_inert() -> None:
    controlled = sft_arguments(_spec(), trainer_dir="/t")
    assert not (set(controlled) & set(INERT_FIELDS))


def test_every_inert_field_says_why() -> None:
    assert all(reason.strip() for _default, reason in INERT_FIELDS.values())


# ---- it holds for the installed TRL ---------------------------------------


@pytest.fixture
def sft_config_type():
    trl = pytest.importorskip("trl")
    return trl.SFTConfig


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_every_sft_config_field_is_classified(sft_config_type, tmp_path, precision) -> None:
    """The tripwire, armed: passing means nothing is left to a TRL default."""
    controlled = sft_arguments(_spec(mixed_precision=precision), trainer_dir=str(tmp_path))

    verify_classification(sft_config_type(**controlled), controlled)


# ---- and it bites ---------------------------------------------------------


def test_a_field_added_by_an_upgrade_stops_the_run(sft_config_type, tmp_path) -> None:
    @dataclasses.dataclass
    class Upgraded(sft_config_type):  # type: ignore[misc, valid-type]
        new_behaviour: bool = True

    controlled = sft_arguments(_spec(), trainer_dir=str(tmp_path))

    with pytest.raises(UnclassifiedBehaviourError, match="new_behaviour"):
        verify_classification(Upgraded(**controlled), controlled)


def test_a_moved_inert_default_stops_the_run(sft_config_type, tmp_path) -> None:
    """Inert *at a value*: a new default is a new decision, not the old one."""

    @dataclasses.dataclass
    class Upgraded(sft_config_type):  # type: ignore[misc, valid-type]
        eval_on_start: bool = True

    controlled = sft_arguments(_spec(), trainer_dir=str(tmp_path))

    with pytest.raises(UnclassifiedBehaviourError, match="eval_on_start"):
        verify_classification(Upgraded(**controlled), controlled)


def test_a_controlled_value_rewritten_after_construction_stops_the_run(
    sft_config_type, tmp_path
) -> None:
    """``__post_init__`` can override what was passed; the run must notice."""

    @dataclasses.dataclass
    class Upgraded(sft_config_type):  # type: ignore[misc, valid-type]
        def __post_init__(self) -> None:
            super().__post_init__()
            self.gradient_checkpointing = True

    controlled = sft_arguments(_spec(), trainer_dir=str(tmp_path))

    with pytest.raises(UnclassifiedBehaviourError, match="gradient_checkpointing"):
        verify_classification(Upgraded(**controlled), controlled)


def test_a_classified_field_that_disappears_stops_the_run(sft_config_type, tmp_path) -> None:
    """A removed field may mean its behaviour moved somewhere unclassified."""
    controlled = sft_arguments(_spec(), trainer_dir=str(tmp_path))
    config = sft_config_type(**controlled)

    with pytest.raises(UnclassifiedBehaviourError, match="no longer has: phantom_field"):
        verify_classification(config, {**controlled, "phantom_field": 1})


# ---- observations ---------------------------------------------------------


def test_a_step_is_reported_with_the_numbers_the_trainer_logged(tmp_path) -> None:
    path = tmp_path / "observations.jsonl"
    observations = TRLObservations(ObservationWriter(path))

    observations.train_begin(0)
    observations.log(1, {"loss": 1.5, "learning_rate": 1e-3, "grad_norm": 2.0, "epoch": 0.5})
    observations.train_end(1)

    written = _written(path)
    assert [w["type"] for w in written] == [
        "TrainingStarted",
        "TrainingMetricObserved",
        "TrainingCompleted",
    ]
    metric = written[1]
    assert (metric["optimizer_step"], metric["loss"], metric["learning_rate"]) == (1, 1.5, 1e-3)
    assert metric["gradient_norm"] == 2.0


def test_the_end_of_training_summary_is_not_a_step(tmp_path) -> None:
    """``Trainer`` logs a summary as ``train_loss``; it is not a step's loss."""
    path = tmp_path / "observations.jsonl"
    TRLObservations(ObservationWriter(path)).log(2, {"train_loss": 1.4, "train_runtime": 0.1})

    assert _written(path) == []


@pytest.mark.parametrize(
    ("value", "observation"),
    [(float("nan"), "nan"), (float("inf"), "positive-infinity")],
)
def test_a_nonfinite_loss_is_reported_as_instability(tmp_path, value, observation) -> None:
    path = tmp_path / "observations.jsonl"
    TRLObservations(ObservationWriter(path)).log(3, {"loss": value, "learning_rate": 1e-3})

    (written,) = _written(path)
    assert written["type"] == "NumericalInstabilityObserved"
    assert (written["quantity"], written["observation"]) == ("loss", observation)
    assert written["optimizer_step"] == 3


def test_a_nonfinite_gradient_norm_is_instability_and_the_loss_still_counts(tmp_path) -> None:
    path = tmp_path / "observations.jsonl"
    TRLObservations(ObservationWriter(path)).log(
        4, {"loss": 1.0, "learning_rate": 1e-3, "grad_norm": float("inf")}
    )

    instability, metric = _written(path)
    assert instability["type"] == "NumericalInstabilityObserved"
    assert instability["quantity"] == "gradient_norm"
    assert metric["type"] == "TrainingMetricObserved"
    assert metric["loss"] == 1.0
    assert metric.get("gradient_norm") is None


def test_the_callback_never_steers_the_trainer(tmp_path) -> None:
    """Observational only: every hook returns ``None``, so ``control`` is untouched."""
    pytest.importorskip("transformers")
    from xaytune.workers.trl import _callback

    callback = _callback(TRLObservations(ObservationWriter(tmp_path / "o.jsonl")))
    state = type("State", (), {"global_step": 1})()
    control = object()

    assert callback.on_train_begin(None, state, control) is None
    assert callback.on_log(None, state, control, logs={"loss": 1.0}) is None
    assert callback.on_train_end(None, state, control) is None


# ---- data -----------------------------------------------------------------


def test_a_record_without_text_is_refused_by_line(tmp_path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text('{"text": "fine"}\n{"prompt": "a", "completion": "b"}\n')

    with pytest.raises(ValueError, match=r"d\.jsonl:2"):
        _read_text_dataset(str(path))


def test_an_empty_dataset_is_refused(tmp_path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text("\n")

    with pytest.raises(ValueError, match="no records"):
        _read_text_dataset(str(path))


def test_extra_fields_do_not_reach_the_trainer(tmp_path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text('{"text": "hello", "label": 3}\n')

    assert _read_text_dataset(str(path)) == [{"text": "hello"}]


def test_the_worker_refuses_to_run_without_its_runtime(monkeypatch) -> None:
    from xaytune.runtimes.worker import WORKER_CONFIG_PATH_ENV
    from xaytune.workers.trl import main

    monkeypatch.delenv(WORKER_CONFIG_PATH_ENV, raising=False)

    with pytest.raises(RuntimeError, match=WORKER_CONFIG_PATH_ENV):
        main()
