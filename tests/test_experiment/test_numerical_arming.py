"""Arming managed numerical recovery: explicit, durable, refused where it cannot be honoured."""

from __future__ import annotations

import asyncio

import pytest

from tests.test_experiment.test_host_behaviour import _experiments, _spec
from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
from xaytune.compilation.native import NativeCompiler
from xaytune.core.domain.candidate import CheckpointIntent
from xaytune.core.domain.numerical_recovery import NumericalRecoveryPolicyV1
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.core.domain.run import RunAttempt
from xaytune.core.execution_controls import MANAGED_NUMERICAL_RECOVERY
from xaytune.core.ids import RunAttemptId
from xaytune.experiment import (
    CompilerSpec,
    EmbeddedControllerHost,
    UnsupportedNumericalRecoveryError,
)
from xaytune.runtimes.local import LocalRuntime
from xaytune.runtimes.local.runtime import _refuse

HALVE = NumericalRecoveryPolicyV1(learning_rate_multiplier=0.5, minimum_learning_rate=None)


def managed(spec):
    training = spec.candidate.training.model_copy(
        update={"checkpoint": CheckpointIntent(every_optimizer_steps=1)}
    )
    return spec.model_copy(
        update={"candidate": spec.candidate.model_copy(update={"training": training})}
    )


def host_for(tmp_path, *, manager=True, request=True):
    return EmbeddedControllerHost(
        tmp_path / "state.db",
        checkpoint_manager=CheckpointManager(
            SerializedStateCodec(), LocalCheckpointStore(tmp_path / "artifacts" / "checkpoints")
        )
        if manager
        else None,
        recovery_request_for_incident=(lambda _incident: RecoveryRequest()) if request else None,
    )


@pytest.mark.parametrize(
    ("variant", "reason"),
    [
        ("no-manager", "checkpoint manager"),
        ("no-request", "recovery_request_for_incident"),
        ("unmanaged", "no managed Native checkpoints"),
        ("trl", "no managed Native checkpoints"),
    ],
)
def test_unhonourable_arming_is_refused_before_anything_is_recorded(tmp_path, variant, reason):
    spec = _spec(tmp_path, numerical_recovery=HALVE)
    if variant not in ("unmanaged", "trl"):
        spec = managed(spec)
    if variant == "trl":
        spec = spec.model_copy(update={"compiler": CompilerSpec(name="trl")})

    async def scenario():
        host = host_for(tmp_path, manager=variant != "no-manager", request=variant != "no-request")
        try:
            with pytest.raises(UnsupportedNumericalRecoveryError, match=reason):
                await host.submit(spec)
            assert _experiments(host) == []
        finally:
            await host.close()

    asyncio.run(scenario())


def test_arming_is_recorded_with_the_experiment_and_only_it_adds_the_control(tmp_path):
    """Unarmed and armed experiments of one candidate differ only by the control."""

    async def scenario():
        host = host_for(tmp_path)
        try:
            plans = {}
            for armed in (False, True):
                spec = managed(_spec(tmp_path, numerical_recovery=HALVE if armed else None))
                experiment = host._record_experiment(
                    spec, NativeCompiler(), LocalRuntime(tmp_path / "runtime"), None
                )
                assert (experiment.numerical_recovery is not None) is armed
                recorded = host.repository.aggregates.load_experiment(str(experiment.id))
                assert recorded.numerical_recovery == (HALVE if armed else None)
                node = host._record_node(experiment, spec)
                run = host._record_run(node, spec.seed)
                plans[armed] = host._plan(
                    experiment, run, RunAttemptId.generate(), NativeCompiler()
                )
            assert MANAGED_NUMERICAL_RECOVERY not in plans[False].runtime_options
            assert MANAGED_NUMERICAL_RECOVERY in plans[True].runtime_options
            assert plans[True].spec.candidate_fingerprint == plans[False].spec.candidate_fingerprint
        finally:
            await host.close()

    asyncio.run(scenario())


def test_local_runtime_accepts_worker_requests_only_for_the_managed_native_worker(tmp_path):
    from tests.test_compilation.test_attempt_resolution import _candidate
    from tests.test_compilation.test_numerical_resolution import successor
    from xaytune.compilation import CompilationContext
    from xaytune.compilation.attempt_resolution import resolve_training_attempt
    from xaytune.compilation.trl import TRLCompiler

    attempt = successor()
    context = CompilationContext(
        run_id="r",
        seed=7,
        output_uri=str(tmp_path / "out"),
        checkpoint_store_uri=str(tmp_path / "checkpoints"),
    )
    candidate = managed(_spec(tmp_path)).candidate
    native = resolve_training_attempt(
        NativeCompiler().compile(candidate, context), attempt, "local", numerical_recovery=True
    )
    assert _refuse(native) is None
    trl = resolve_training_attempt(
        TRLCompiler().compile(_candidate(tmp_path), context),
        RunAttempt(id=RunAttemptId.generate(), run_id=attempt.run_id, attempt_number=1),
        "local",
        numerical_recovery=True,
    )
    assert "managed Native worker" in (_refuse(trl) or "")
    capabilities = LocalRuntime(tmp_path / "rt").capabilities()
    assert set(capabilities.extensions["worker_requests"]["xaytune.workers.native"]) == {
        "managed_numerical_recovery",
        "training_interventions",
    }
