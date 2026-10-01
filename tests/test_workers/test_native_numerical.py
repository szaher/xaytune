"""Managed numerical recovery in the Native worker: fail-fast guard and LR directives."""

from __future__ import annotations

import math

import pytest
import torch

from xaytune.core.domain.incident import AttemptContext, IncidentCategory
from xaytune.core.domain.intervention import (
    InterventionDirective,
    InterventionDirectiveKind,
    LearningRateMutation,
)
from xaytune.core.ids import InterventionApplicationId, InterventionId, RunAttemptId
from xaytune.core.telemetry import (
    InterventionAppliedPayload,
    NumericalInstabilityObserved,
    TrainingFailedPayload,
)
from xaytune.resilience import NaNInfDetector
from xaytune.trainer.callbacks import CallbackManager, TrainState
from xaytune.trainer.scheduler import create_scheduler
from xaytune.workers.native import (
    ManagedNumericalInstabilityError,
    register_managed_numerical_guard,
    register_native_observation_callbacks,
)
from xaytune.workers.native_checkpoint import NativeCheckpointAdapter


class Writer:
    def __init__(self) -> None:
        self.written: list[object] = []

    def write(self, payload: object) -> None:
        self.written.append(payload)


def wired(*, armed: bool):
    """Observation callbacks, then (if armed) the guard, then a capture spy -- main()'s order."""
    writer, captured = Writer(), []
    callbacks = CallbackManager()
    register_native_observation_callbacks(callbacks, writer, gradient_accumulation=1)  # type: ignore[arg-type]
    if armed:
        register_managed_numerical_guard(callbacks)

    @callbacks.on("step_end")
    def _capture(state: TrainState) -> None:
        captured.append(state.global_step)

    return callbacks, writer, captured


def step(loss: float, global_step: int = 3) -> TrainState:
    state = TrainState(step=0, epoch=0, global_step=global_step)
    state.metrics["loss"] = loss
    return state


@pytest.mark.parametrize(
    ("loss", "reason"), [(math.nan, "numerical-nan"), (math.inf, "numerical-inf")]
)
def test_armed_guard_reports_first_then_fails_and_nothing_is_captured(loss, reason):
    callbacks, writer, captured = wired(armed=True)
    with pytest.raises(ManagedNumericalInstabilityError) as raised:
        callbacks.fire("step_end", step(loss))
    assert raised.value.reason == reason
    (observation,) = writer.written
    assert isinstance(observation, NumericalInstabilityObserved)
    assert observation.optimizer_step == 3
    assert captured == [], "the unsafe step must never become a checkpoint"


def test_unarmed_managed_run_keeps_report_and_continue():
    callbacks, writer, captured = wired(armed=False)
    callbacks.fire("step_end", step(math.nan))
    assert [type(item) for item in writer.written] == [NumericalInstabilityObserved]
    assert captured == [3], "managed checkpoints alone do not arm fail-fast"


def test_finite_loss_is_untouched_by_the_guard():
    callbacks, _, captured = wired(armed=True)
    callbacks.fire("step_end", step(1.25))
    assert captured == [3]


@pytest.mark.parametrize(
    ("reason", "category"),
    [
        ("numerical-nan", IncidentCategory.NUMERICAL_NAN),
        ("numerical-inf", IncidentCategory.NUMERICAL_INF),
    ],
)
def test_the_guard_failure_confirms_the_observation_diagnosis(reason, category):
    detector = NaNInfDetector()
    assert detector.version == "2"
    context = AttemptContext.model_construct()
    candidate = detector.inspect(TrainingFailedPayload(reason=reason), context)
    assert candidate is not None and candidate.category is category
    assert detector.inspect(TrainingFailedPayload(reason="value-error"), context) is None


def directive(previous: float, applied: float, ordinal: int = 0) -> InterventionDirective:
    return InterventionDirective(
        application_id=InterventionApplicationId.generate(),
        intervention_id=InterventionId.generate(),
        attempt_id=RunAttemptId.generate(),
        ordinal=ordinal,
        kind=InterventionDirectiveKind.INITIAL,
        mutation=LearningRateMutation(learning_rate=applied),
        expected_previous_value=previous,
    )


def restored_optimizer(lr: float = 2e-4, steps: int = 5):
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW([parameter], lr=lr)
    scheduler = create_scheduler(optimizer, "linear", total_steps=100, warmup_steps=0)
    for _ in range(steps):
        optimizer.step()
        scheduler.step()
    return optimizer, scheduler


def adapter_with(writer: Writer, embodied: list[str]) -> NativeCheckpointAdapter:
    adapter = object.__new__(NativeCheckpointAdapter)
    adapter.writer = writer  # type: ignore[assignment]
    adapter._embodied = list(embodied)
    return adapter


def test_directive_replaces_the_base_rate_and_keeps_the_schedule_position():
    optimizer, scheduler = restored_optimizer()
    factor = optimizer.param_groups[0]["lr"] / 2e-4
    writer = Writer()
    adapter = adapter_with(writer, ["intapp_restored"])
    first = directive(2e-4, 1e-4)
    adapter.apply_directives((first,), optimizer, scheduler, 5)

    assert scheduler.base_lrs == [1e-4]
    assert optimizer.param_groups[0]["initial_lr"] == 1e-4
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4 * factor)
    (payload,) = writer.written
    assert payload == InterventionAppliedPayload(
        application_id=str(first.application_id),
        intervention_id=str(first.intervention_id),
        optimizer_step=5,
        previous_value=2e-4,
        applied_value=1e-4,
    )
    assert adapter._embodied == ["intapp_restored", str(first.application_id)]
    # The schedule continues from its position at the new base.
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4 * (100 - 6) / 100)


def test_directives_chain_in_order():
    optimizer, scheduler = restored_optimizer()
    adapter = adapter_with(Writer(), [])
    adapter.apply_directives(
        (directive(2e-4, 1e-4, 0), directive(1e-4, 5e-5, 1)), optimizer, scheduler, 5
    )
    assert scheduler.base_lrs == [5e-5]
    assert len(adapter._embodied) == 2


def test_restored_rate_that_disagrees_with_the_directive_is_refused_before_any_effect():
    optimizer, scheduler = restored_optimizer(lr=3e-4)
    writer = Writer()
    adapter = adapter_with(writer, [])
    with pytest.raises(ValueError, match="expected"):
        adapter.apply_directives((directive(2e-4, 1e-4),), optimizer, scheduler, 5)
    assert scheduler.base_lrs == [3e-4]
    assert writer.written == [] and adapter._embodied == []
