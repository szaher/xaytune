"""TorchFTResilienceProvider: a versioned TorchFT request, and nothing else (PR-035, commit 2).

The engine is read from distribution metadata, faked here; the ``torchft``
marker's tests read the real installed release (CI's ``torchft`` job, Linux
x86_64). No built-in runtime or compiler hosts the request in PR-035, so
every built-in combination is refused -- hosting it in Ray Train is PR-036.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

import xaytune
from tests.evaluation_fixtures import EVALUATORS
from tests.test_experiment.adaptive_fixtures import LoRACompiler, adaptive_spec
from tests.test_resilience.resilience_support import hosting
from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
from xaytune.compilation.attempt_resolution import training_execution_fingerprint
from xaytune.compilation.native import NativeCompiler
from xaytune.compilation.trl import TRLCompiler
from xaytune.core import execution_controls
from xaytune.core.capabilities import CapabilityDocument, CheckpointCapabilities
from xaytune.core.domain.objective import BudgetSpec
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.domain.policy import PolicyVerdict
from xaytune.core.domain.resilience import ResiliencePolicy, ResilienceSpec
from xaytune.core.execution import (
    CheckpointExecutionContract,
    CompilerIdentity,
    PythonModuleEntrypoint,
    ResolvedExecutionPlan,
    ResourceRequirements,
    TrainingExecutionSpec,
)
from xaytune.core.execution_controls import RESILIENCE, ResilienceRequest
from xaytune.core.immutable import FrozenDict, thaw
from xaytune.decision import AdaptiveThresholdDecisionEngine
from xaytune.experiment import EmbeddedControllerHost
from xaytune.planning import PLANNERS
from xaytune.policy import RulePolicyEngine
from xaytune.ray.runtime.jobs import RayJobsConfig, RayJobsRuntime
from xaytune.ray.runtime.train import RayTrainConfig, RayTrainRuntime
from xaytune.resilience import torchft as torchft_module
from xaytune.resilience.provider import (
    ExecutionCapabilities,
    ResilienceProviderConfigurationError,
    UnsupportedResilienceError,
    augment_execution_plan,
    bind_resilience_provider,
)
from xaytune.resilience.torchft import (
    MANAGER_PARAMETERS,
    SUPPORTED_TORCHFT,
    TORCHFT_REQUEST_SCHEMA,
    TorchFTResilienceProvider,
    torchft_resilience_provider,
)
from xaytune.runtimes.local import LocalRuntime

POLICY = ResiliencePolicy(delegate=("per-step-worker-recovery",))
CONFIG = {
    "lighthouse_address": "http://lighthouse.internal:29510",
    "min_replica_size": 2,
    "quorum_timeout_seconds": 120.0,
    "timeout_seconds": 60.0,
    "use_async_quorum": True,
}
REGISTRY = {"torchft": torchft_resilience_provider}
HOSTED = ExecutionCapabilities(
    runtime=hosting(
        CapabilityDocument(checkpoint=CheckpointCapabilities(atomic_commit=True)),
        TORCHFT_REQUEST_SCHEMA,
    ),
    compiler=hosting(CapabilityDocument(), TORCHFT_REQUEST_SCHEMA),
)


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The distributions "installed", by name: torchft 0.2.0 unless a test changes it."""
    distributions = {"torchft": SUPPORTED_TORCHFT}
    monkeypatch.setattr(torchft_module, "_installed", distributions.get)
    return distributions


def _spec(**config: Any) -> ResilienceSpec:
    return ResilienceSpec(kind="torchft", config=FrozenDict({**CONFIG, **config}), policy=POLICY)


def _plan(workers: int = 4, **options: Any) -> ResolvedExecutionPlan:
    checkpoint = options.pop("checkpoint", CheckpointExecutionContract())
    return ResolvedExecutionPlan(
        spec=TrainingExecutionSpec(
            compiler=CompilerIdentity(name="fake", version="0.1.0"),
            candidate_fingerprint="sha256:" + "0" * 64,
            entrypoint=PythonModuleEntrypoint(module="my.torchft_worker"),
            resources=ResourceRequirements(workers=workers),
            checkpoint=checkpoint,
        ),
        runtime="ray-train",
        target=RuntimeOperationTarget(kind="training-attempt", id="ra_1"),
        runtime_options=FrozenDict(options),
    )


def _augment(plan: ResolvedExecutionPlan, capabilities: Any = HOSTED, **config: Any) -> Any:
    provider = bind_resilience_provider(_spec(**config), REGISTRY)
    return augment_execution_plan(provider, plan, capabilities=capabilities)


# ---- identity --------------------------------------------------------------------


def test_binding_records_the_exact_installed_release(installed: dict[str, str]) -> None:
    bound = bind_resilience_provider(_spec(), REGISTRY).spec
    assert (bound.version, bound.engine) == ("1.0.0", FrozenDict({"torchft": "0.2.0"}))
    assert bind_resilience_provider(bound, REGISTRY).spec == bound


@pytest.mark.parametrize(
    ("distributions", "message"),
    [
        ({}, "torchft is not installed"),
        ({"torchft": "0.2.1"}, "torchft 0.2.1 is installed; this provider translates for exactly"),
        ({"torchft": "0.1.1"}, "exactly 0.2.0"),
        ({"torchft-nightly": "2026.10.9"}, "torchft-nightly 2026.10.9 is installed"),
        ({"torchft": "0.2.0", "torchft-nightly": "2026.10.9"}, "only the stable torchft"),
    ],
)
def test_an_unsupported_or_missing_engine_is_refused_at_binding(
    installed: dict[str, str], distributions: dict[str, str], message: str
) -> None:
    installed.clear()
    installed.update(distributions)
    with pytest.raises(ResilienceProviderConfigurationError, match=message):
        bind_resilience_provider(_spec(), REGISTRY)


def test_the_same_plan_policy_and_engine_make_the_same_request(installed: Any) -> None:
    first, again = _augment(_plan()), _augment(_plan())
    assert first == again and first.model_dump_json() == again.model_dump_json()

    request = ResilienceRequest.model_validate(thaw(first.runtime_options[RESILIENCE]))
    assert request.provider == "torchft"
    assert request.engine == FrozenDict({"torchft": "0.2.0"})
    assert request.request_schema == "xaytune.torchft/v1alpha1"
    assert request.delegate == ("per-step-worker-recovery",)
    assert dict(request.parameters) == {
        **CONFIG,
        "replica_group_size": 1,
        "replica_groups": 4,
    }
    assert first.spec == _plan().spec, "the candidate and its training settings are untouched"


def test_the_request_changes_the_digest_and_the_execution_identity(installed: Any) -> None:
    plan = _plan()
    augmented = _augment(plan)
    longer = _augment(plan, quorum_timeout_seconds=300.0)
    assert augmented.request_digest("submit") != plan.request_digest("submit")
    assert training_execution_fingerprint(augmented) != training_execution_fingerprint(plan)
    assert longer.request_digest("submit") != augmented.request_digest("submit")


def test_every_request_parameter_is_a_manager_argument_or_placement(installed: Any) -> None:
    parameters = set(
        ResilienceRequest.model_validate(
            thaw(_augment(_plan()).runtime_options[RESILIENCE])
        ).parameters
    )
    assert parameters == set(MANAGER_PARAMETERS) | {"replica_group_size", "replica_groups"}


# ---- refusals ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plan", "config", "message"),
    [
        (_plan(workers=1), {}, "1 workers make 1 replica group"),
        (_plan(workers=4), {"min_replica_size": 3, "replica_group_size": 2}, "need"),
        (_plan(workers=6), {"replica_group_size": 4}, "do not divide"),
        (_plan(workers=3), {"min_replica_size": 4}, "at least 4"),
        (
            _plan(checkpoint=CheckpointExecutionContract(boundary="mid-accumulation")),
            {},
            "optimizer-step boundaries",
        ),
        (
            _plan(checkpoint=CheckpointExecutionContract(require_atomic_commit=False)),
            {},
            "committed atomically",
        ),
        (_plan(checkpoint_restore={"id": "ck"}), {}, "'checkpoint_restore' address one"),
        (_plan(training_interventions={}), {}, "'training_interventions' address one"),
        (_plan(managed_numerical_recovery={}), {}, "'managed_numerical_recovery' address one"),
    ],
)
def test_a_plan_outside_what_v1_translates_is_refused(
    installed: Any, plan: ResolvedExecutionPlan, config: dict[str, Any], message: str
) -> None:
    with pytest.raises(UnsupportedResilienceError, match=message):
        _augment(plan, **config)


def test_checkpoints_need_a_runtime_that_commits_atomically(installed: Any) -> None:
    plan = _plan(checkpoint=CheckpointExecutionContract(store_uri="/ckpt"))
    assert _augment(plan) is not None
    unsafe = HOSTED.model_copy(
        update={"runtime": hosting(CapabilityDocument(), TORCHFT_REQUEST_SCHEMA)}
    )
    with pytest.raises(UnsupportedResilienceError, match="does not declare atomic commit"):
        _augment(plan, unsafe)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"lighthouse_address": "http://user:secret@lighthouse:29510"}, "credentials"),
        ({"lighthouse_address": "lighthouse:29510"}, "http"),
        ({"lighthouse_address": "http://lighthouse:29510/path"}, "host and port"),
        ({"min_replica_size": 0}, "min_replica_size"),
        ({"min_replica_size": "2"}, "min_replica_size"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"quorum_timeout_seconds": -1.0}, "quorum_timeout_seconds"),
        ({"use_async_quorum": "yes"}, "use_async_quorum"),
        ({"max_retries": 3}, "max_retries"),
    ],
)
def test_a_configuration_that_cannot_be_translated_is_refused(
    installed: Any, config: dict[str, Any], message: str
) -> None:
    with pytest.raises(ResilienceProviderConfigurationError, match=message):
        bind_resilience_provider(_spec(**config), REGISTRY)


def test_every_torchft_setting_is_stated(installed: Any) -> None:
    for name in ("lighthouse_address", "min_replica_size", "timeout_seconds", "use_async_quorum"):
        config = {key: value for key, value in CONFIG.items() if key != name}
        spec = ResilienceSpec(kind="torchft", config=FrozenDict(config), policy=POLICY)
        with pytest.raises(ResilienceProviderConfigurationError, match=name):
            bind_resilience_provider(spec, REGISTRY)


def _built_in_runtimes(tmp_path: Path) -> dict[str, Any]:
    ray = {"address": "http://ray.test:8265", "runtime_env": {}, "shared_state_root": str(tmp_path)}
    return {
        "local": LocalRuntime(tmp_path / "local"),
        "ray-jobs": RayJobsRuntime(RayJobsConfig(**ray)),
        "ray-train": RayTrainRuntime(RayTrainConfig(**ray)),
    }


def test_no_built_in_runtime_or_compiler_hosts_torchft_yet(installed: Any, tmp_path: Path) -> None:
    """PR-036 is where Ray Train learns to host it; until then every built-in refuses."""
    for name, runtime in _built_in_runtimes(tmp_path).items():
        capabilities = HOSTED.model_copy(update={"runtime": runtime.capabilities()})
        with pytest.raises(UnsupportedResilienceError, match="runtime does not declare"):
            _augment(_plan(), capabilities)
        # And were a request to reach one anyway, the runtime refuses it as an
        # option it does not implement, rather than dropping it.
        augmented = _augment(_plan().model_copy(update={"runtime": name}))
        refusal = inspect.getmodule(type(runtime))._refuse(augmented)  # type: ignore[union-attr]
        assert refusal is not None and "'resilience'" in refusal, name
    for compiler in (NativeCompiler(), TRLCompiler()):
        capabilities = HOSTED.model_copy(update={"compiler": compiler.capabilities()})
        with pytest.raises(UnsupportedResilienceError, match="compiler does not declare"):
            _augment(_plan(), capabilities)


def test_an_experiment_on_a_built_in_runtime_is_refused_before_anything_is_recorded(
    installed: Any, tmp_path: Path
) -> None:
    spec = adaptive_spec(tmp_path, budget=BudgetSpec(max_runs=2)).model_copy(
        update={"resilience": _spec()}
    )
    manager = CheckpointManager(SerializedStateCodec(), LocalCheckpointStore(tmp_path / "bundles"))
    host = EmbeddedControllerHost(
        tmp_path / "state.db",
        compilers={"native": LoRACompiler},
        runtimes={"local": lambda config: LocalRuntime(tmp_path / "local")},
        evaluators=EVALUATORS,
        decision_engine=AdaptiveThresholdDecisionEngine(),
        policy=RulePolicyEngine(default=PolicyVerdict.ALLOW),
        checkpoint_manager=manager,
        planners=PLANNERS,
        resilience_providers=REGISTRY,
    )

    async def scenario() -> None:
        try:
            with pytest.raises(UnsupportedResilienceError, match="runtime does not declare"):
                await host.submit(spec)
            count = host.repository._connection.execute("SELECT COUNT(*) FROM experiments")
            assert count.fetchone()[0] == 0
        finally:
            await host.close()

    asyncio.run(scenario())


# ---- the boundary ------------------------------------------------------------------


def test_nothing_in_xaytune_imports_torchft() -> None:
    """The provider reads metadata; no runtime or worker imports TorchFT before PR-036."""
    root = Path(xaytune.__file__).parent
    importers = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            if any(name == "torchft" or name.startswith("torchft.") for name in names):
                importers.append(str(path.relative_to(root)))
    assert importers == []


def test_the_request_envelope_lives_in_core_and_the_provider_below_it() -> None:
    assert "torchft" not in Path(execution_controls.__file__).read_text(encoding="utf-8")
    assert TorchFTResilienceProvider.__module__ == "xaytune.resilience.torchft"


# ---- the real release (CI's torchft job) ----------------------------------------------


def _real_torchft() -> str:
    try:
        return metadata.version("torchft")
    except metadata.PackageNotFoundError:
        pytest.skip("torchft is not installed (the torchft extra, Linux x86_64 only)")


@pytest.mark.torchft
def test_the_installed_release_is_the_one_the_provider_binds() -> None:
    assert _real_torchft() == SUPPORTED_TORCHFT
    assert TorchFTResilienceProvider.engine_versions() == {"torchft": SUPPORTED_TORCHFT}
    bound = bind_resilience_provider(_spec(), REGISTRY).spec
    assert bound.engine == FrozenDict({"torchft": SUPPORTED_TORCHFT})


@pytest.mark.torchft
def test_every_manager_parameter_is_an_argument_of_the_installed_manager() -> None:
    _real_torchft()
    from torchft.manager import Manager  # the test reads the API; xaytune never imports it

    arguments = inspect.signature(Manager.__init__).parameters
    assert set(MANAGER_PARAMETERS.values()) <= set(arguments)


def test_the_extra_pins_the_release_the_provider_supports() -> None:
    pyproject = (Path(xaytune.__file__).parent.parent / "pyproject.toml").read_text()
    assert (
        f"torchft = [\"torchft=={SUPPORTED_TORCHFT}; sys_platform == 'linux' and "
        f"platform_machine == 'x86_64'\"]"
    ) in pyproject
