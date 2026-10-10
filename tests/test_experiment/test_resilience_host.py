"""The host delegates in-attempt recovery through the resilience boundary (PR-035).

```text
ExperimentSpec.resilience → bound at submission (version + engine recorded)
every training plan       → resolve_training_attempt → augment_execution_plan → submit
every evaluation plan     → unchanged
a restarted host          → rebinds the record, rebuilds the same requests -- or refuses
```

Refused before anything is recorded when the runtime or compiler does not
declare the provider's request schema. An experiment without a resilience spec
gets exactly the plans it got before.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation_fixtures import EVALUATORS
from tests.test_experiment.adaptive_fixtures import AdaptiveRuntime, LoRACompiler, adaptive_spec
from tests.test_resilience.resilience_support import (
    ENGINE,
    KIND,
    HonouringCompiler,
    HostingRuntime,
    ReplicaRecoveryProvider,
    providers,
    resilience_spec,
)
from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
from xaytune.compilation.attempt_resolution import training_execution_fingerprint
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.domain.policy import PolicyVerdict
from xaytune.core.domain.run import RunAttempt
from xaytune.core.execution_controls import RESILIENCE, ResilienceRequest
from xaytune.core.ids import RunAttemptId
from xaytune.core.immutable import FrozenDict, thaw
from xaytune.decision import AdaptiveThresholdDecisionEngine
from xaytune.experiment import EmbeddedControllerHost
from xaytune.experiment.host import ImplementationMismatchError, UnknownImplementationError
from xaytune.planning import PLANNERS
from xaytune.policy import RulePolicyEngine
from xaytune.resilience.provider import (
    ResilienceProviderConfigurationError,
    UnsupportedResilienceError,
)

_TIMEOUT = 30


def _sync(test: Callable[..., Coroutine[Any, Any, None]]) -> Callable[..., None]:
    """Run an async scenario as a plain test, the way the other host tests do."""

    @functools.wraps(test)
    def run(*args: Any, **kwargs: Any) -> None:
        asyncio.run(test(*args, **kwargs))

    return run


def _host(
    tmp_path: Path,
    *,
    runtime: Any = None,
    compiler: type[LoRACompiler] = HonouringCompiler,
    registry: Any = None,
) -> tuple[EmbeddedControllerHost, Any]:
    manager = CheckpointManager(SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles"))
    runtime = runtime or HostingRuntime(manager, tmp_path, oom_rank=None)
    host = EmbeddedControllerHost(
        tmp_path / "state.db",
        compilers={"native": compiler},
        runtimes={"local": lambda config: runtime},
        evaluators=EVALUATORS,
        decision_engine=AdaptiveThresholdDecisionEngine(),
        policy=RulePolicyEngine(default=PolicyVerdict.ALLOW),
        checkpoint_manager=manager,
        planners=PLANNERS,
        resilience_providers=providers() if registry is None else registry,
    )
    return host, runtime


def _spec(tmp_path: Path, *, resilience: bool = True) -> Any:
    spec = adaptive_spec(tmp_path, budget=BudgetSpec(max_runs=2))
    return spec.model_copy(update={"resilience": resilience_spec() if resilience else None})


async def _run(host: EmbeddedControllerHost, spec: Any) -> Any:
    handle = await host.submit(spec)
    await asyncio.wait_for(handle.wait(), timeout=_TIMEOUT)
    return handle.experiment_id


def _attempts(host: EmbeddedControllerHost, experiment_id: Any) -> list[tuple[Any, Any, Any]]:
    aggregates = host.repository.aggregates
    return [
        (node, run, attempt)
        for node in aggregates.nodes_for_experiment(str(experiment_id))
        for run in aggregates.runs_for_node(str(node.id))
        for attempt in aggregates.attempts_for_run(str(run.id))
    ]


def _submitted(host: EmbeddedControllerHost, attempt: Any) -> Any:
    (submission,) = [
        operation
        for operation in host.repository.operations.for_target("training-attempt", str(attempt.id))
        if operation.type == "submit"
    ]
    return submission


@_sync
async def test_every_training_plan_carries_the_request_and_no_evaluation_plan_does(
    tmp_path: Path,
) -> None:
    host, runtime = _host(tmp_path)
    try:
        experiment_id = await _run(host, _spec(tmp_path))
        experiment = host.repository.aggregates.load_experiment(str(experiment_id))
        attempts = _attempts(host, experiment_id)
        submitted = {str(attempt.id): _submitted(host, attempt) for _, _, attempt in attempts}
    finally:
        await host.close()

    bound = experiment.resilience
    assert bound is not None
    assert (bound.version, bound.engine) == ("1.0.0", FrozenDict(ENGINE))
    assert len(runtime.training_plans) == len(attempts) == 2  # the root, and a planned child
    requests = {
        ResilienceRequest.model_validate(thaw(plan.runtime_options[RESILIENCE]))
        for plan in runtime.training_plans
    }
    assert len(requests) == 1
    (request,) = requests
    assert (request.provider, request.engine) == ("test-replica-recovery", FrozenDict(ENGINE))
    assert runtime.evaluation_plans
    assert all(RESILIENCE not in plan.runtime_options for plan in runtime.evaluation_plans)
    by_target = {plan.target.id: plan for plan in runtime.training_plans}
    for _, _, attempt in attempts:
        plan = by_target[str(attempt.id)]
        assert attempt.execution_fingerprint == training_execution_fingerprint(plan)
        assert submitted[str(attempt.id)].request_digest == plan.request_digest("submit")


@_sync
async def test_a_restarted_host_rebuilds_every_recorded_request_exactly(tmp_path: Path) -> None:
    first, _ = _host(tmp_path)
    try:
        experiment_id = await _run(first, _spec(tmp_path))
    finally:
        await first.close()

    second, _ = _host(tmp_path)
    try:
        experiment = second.repository.aggregates.load_experiment(str(experiment_id))
        for _, run, attempt in _attempts(second, experiment_id):
            rebuilt = second._plan(experiment, run, attempt, HonouringCompiler())
            assert rebuilt.request_digest("submit") == _submitted(second, attempt).request_digest
    finally:
        await second.close()


@pytest.mark.parametrize(
    ("registry", "message"),
    [
        (providers(installed={"replica-engine": "2.4.0"}), "engine"),
        ({}, "no resilience provider"),
        (providers(installed=None), "not installed"),
    ],
)
@_sync
async def test_a_host_without_the_recorded_provider_and_engine_builds_nothing(
    tmp_path: Path, registry: Any, message: str
) -> None:
    first, _ = _host(tmp_path)
    try:
        experiment_id = await _run(first, _spec(tmp_path))
    finally:
        await first.close()

    second, _ = _host(tmp_path, registry=registry)
    try:
        experiment = second.repository.aggregates.load_experiment(str(experiment_id))
        _, run, attempt = _attempts(second, experiment_id)[0]
        with pytest.raises(
            (
                ImplementationMismatchError,
                UnknownImplementationError,
                ResilienceProviderConfigurationError,
            ),
            match=message,
        ):
            second._plan(experiment, run, attempt, HonouringCompiler())
    finally:
        await second.close()


class _PlainRuntime(AdaptiveRuntime):
    """Hosts no resilience request: the default for every runtime."""


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"runtime": "plain"}, UnsupportedResilienceError, "runtime does not declare hosting"),
        ({"compiler": LoRACompiler}, UnsupportedResilienceError, "compiler does not declare"),
        ({"registry": {}}, UnknownImplementationError, "no resilience provider"),
        (
            {"registry": providers(installed=None)},
            ResilienceProviderConfigurationError,
            "installed",
        ),
    ],
)
@_sync
async def test_an_unsupported_combination_is_refused_before_anything_is_recorded(
    tmp_path: Path, kwargs: dict[str, Any], error: type[Exception], message: str
) -> None:
    if kwargs.get("runtime") == "plain":
        manager = CheckpointManager(
            SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles")
        )
        kwargs = {"runtime": _PlainRuntime(manager, tmp_path, oom_rank=None)}
    host, runtime = _host(tmp_path, **kwargs)
    try:
        with pytest.raises(error, match=message):
            await host.submit(_spec(tmp_path))
        count = host.repository._connection.execute("SELECT COUNT(*) FROM experiments").fetchone()
        assert count[0] == 0
        assert runtime.submitted == {}
    finally:
        await host.close()


@_sync
async def test_without_a_resilience_spec_every_plan_is_what_it_was(tmp_path: Path) -> None:
    spec = _spec(tmp_path, resilience=False)
    assert "resilience" not in spec.submission_payload()

    with_providers, runtime = _host(tmp_path / "a")
    without, plain = _host(tmp_path / "b", runtime=None, compiler=LoRACompiler, registry={})
    try:
        await _run(with_providers, spec)
        await _run(without, spec)
    finally:
        await with_providers.close()
        await without.close()

    # Run ids differ between the two records, so the plans are compared by
    # what resilience could have touched: their runtime options.
    assert runtime.training_plans and len(runtime.training_plans) == len(plain.training_plans)
    assert all(RESILIENCE not in plan.runtime_options for plan in runtime.training_plans)
    assert [plan.runtime_options for plan in runtime.training_plans] == [
        plan.runtime_options for plan in plain.training_plans
    ]


def test_a_caller_cannot_claim_a_provider_version_or_engine(tmp_path: Path) -> None:
    for bound in ({"version": "1.0.0"}, {"engine": FrozenDict(ENGINE)}):
        with pytest.raises(ValueError, match="resolved by the host"):
            _spec(tmp_path, resilience=False).model_validate(
                {
                    **_spec(tmp_path, resilience=False).model_dump(),
                    "resilience": resilience_spec().model_copy(update=bound).model_dump(),
                }
            )


class _Impostor(ReplicaRecoveryProvider):
    descriptor = ReplicaRecoveryProvider.descriptor.model_copy(update={"provider": "elsewhere"})


class _OtherSchema(ReplicaRecoveryProvider):
    request_schema = "xaytune.test-replica-recovery/v2alpha1"


@_sync
@pytest.mark.parametrize("substitute", [_Impostor, _OtherSchema])
async def test_a_substituted_implementation_builds_no_new_plan(
    tmp_path: Path, substitute: type[ReplicaRecoveryProvider]
) -> None:
    """Same kind, version and engine, another descriptor or schema: refused before any plan."""
    first, _ = _host(tmp_path)
    try:
        experiment_id = await _run(first, _spec(tmp_path))
    finally:
        await first.close()

    registry = {KIND: lambda spec: substitute(spec, ENGINE)}
    second, runtime = _host(tmp_path, registry=registry)
    try:
        experiment = second.repository.aggregates.load_experiment(str(experiment_id))
        _, run, _ = _attempts(second, experiment_id)[0]
        new_attempt = RunAttempt(id=RunAttemptId.generate(), run_id=run.id, attempt_number=9)
        with pytest.raises(ImplementationMismatchError, match="elsewhere|v2alpha1"):
            second._plan(experiment, run, new_attempt, HonouringCompiler())
        assert runtime.training_plans == []
    finally:
        await second.close()
