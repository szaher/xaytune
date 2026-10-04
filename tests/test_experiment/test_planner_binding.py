"""A planner is a durable spec, bound at submission (ADR-016), never a live object.

```text
ExperimentSpec.planner = PlannerSpec(kind, version=None, config)
   ↓ host: resolve kind → check plugin API → validate config → record version
Experiment.planner = PlannerSpec(kind, version="1.0.0", canonical config)
```

Nothing invokes the planner yet: proposals are consumed from PR-025/026.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from tests.test_experiment.test_host_behaviour import _experiments, _spec
from xaytune.compilation.native import NativeCompiler
from xaytune.core.domain.experiment import Experiment
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.immutable import FrozenDict
from xaytune.experiment import (
    EmbeddedControllerHost,
    ImplementationMismatchError,
    UnknownImplementationError,
)
from xaytune.planning import PlannerConfigurationError, RuleBasedPlanner
from xaytune.runtimes.local import LocalRuntime

RULES = FrozenDict({"rules": [{"kind": "increase-lora-rank", "factor": 2, "max_rank": 64}]})


def _host(tmp_path):
    return EmbeddedControllerHost(tmp_path / "state.db")


def test_the_caller_may_not_supply_the_version() -> None:
    from pathlib import Path

    with pytest.raises(ValidationError, match="resolved by the host"):
        _spec(Path("/tmp/x"), planner=PlannerSpec(kind="rule-based", version="1.0.0"))


def test_the_bound_spec_is_recorded_with_its_version(tmp_path) -> None:
    async def scenario():
        host = _host(tmp_path)
        try:
            spec = _spec(tmp_path, planner=PlannerSpec(kind="rule-based", config=RULES))
            bound = host._planner(spec.planner).spec
            experiment = host._record_experiment(
                spec, NativeCompiler(), LocalRuntime(tmp_path / "runtime"), None, bound
            )
            recorded = host.repository.aggregates.load_experiment(str(experiment.id))
            assert recorded.planner == bound
            assert recorded.planner.version == RuleBasedPlanner.descriptor.plugin_version
            assert recorded.planner.config == RULES

            # A restarted host rebuilds the planner from the record alone.
            rebuilt = EmbeddedControllerHost(tmp_path / "state.db")
            try:
                planner = rebuilt._recorded_planner(recorded)
                assert planner is not None and planner.spec == recorded.planner
            finally:
                await rebuilt.close()
        finally:
            await host.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("planner", "error"),
    [
        (PlannerSpec(kind="llm"), UnknownImplementationError),
        (
            PlannerSpec(kind="rule-based", config=FrozenDict({"rules": [{"kind": "nope"}]})),
            PlannerConfigurationError,
        ),
        (PlannerSpec(kind="no-op", config=RULES), PlannerConfigurationError),
    ],
    ids=["unknown-kind", "malformed-rules", "config-for-no-op"],
)
def test_an_unbindable_planner_is_refused_before_anything_is_recorded(
    tmp_path, planner, error
) -> None:
    async def scenario():
        host = _host(tmp_path)
        try:
            with pytest.raises(error):
                await host.submit(_spec(tmp_path, planner=planner))
            assert _experiments(host) == []
        finally:
            await host.close()

    asyncio.run(scenario())


def test_an_experiment_recorded_before_planners_still_loads(tmp_path) -> None:
    async def scenario():
        host = _host(tmp_path)
        try:
            spec = _spec(tmp_path)
            experiment = host._record_experiment(
                spec, NativeCompiler(), LocalRuntime(tmp_path / "runtime"), None
            )
            payload = experiment.model_dump(mode="json")
            payload.pop("planner")
            assert Experiment.model_validate(payload).planner is None
            recorded = host.repository.aggregates.load_experiment(str(experiment.id))
            assert recorded.planner is None
            assert host._recorded_planner(recorded) is None
        finally:
            await host.close()

    asyncio.run(scenario())


def test_a_recorded_planner_at_another_version_is_refused(tmp_path) -> None:
    async def scenario():
        host = _host(tmp_path)
        try:
            spec = _spec(tmp_path)
            experiment = host._record_experiment(
                spec,
                NativeCompiler(),
                LocalRuntime(tmp_path / "runtime"),
                None,
                PlannerSpec(kind="rule-based", version="0.0.1", config=RULES),
            )
            with pytest.raises(ImplementationMismatchError, match="planner 'rule-based'"):
                host._recorded_planner(experiment)
        finally:
            await host.close()

    asyncio.run(scenario())


def _llm_config(model) -> FrozenDict:
    return FrozenDict(
        {
            "model": model.model_dump(),
            "prompt_version": "xaytune.llm-planner/v1",
            "allowed_actions": [{"type": "reject-candidate"}],
        }
    )


def test_an_llm_planner_binds_only_through_an_explicitly_supplied_factory(tmp_path) -> None:
    """``kind="llm"`` has no default: the host binds it with the agent model it was given."""
    from xaytune.agent import AgentModelIdentity, ScriptedAgentModel
    from xaytune.planning import PLANNERS
    from xaytune.planning.llm import LLMPlanner, llm_planner_factory

    identity = AgentModelIdentity(provider="vendor", name="planner-large", revision="r1")
    model = ScriptedAgentModel([], model=identity)
    planners = {**PLANNERS, "llm": llm_planner_factory(model)}

    async def scenario():
        host = EmbeddedControllerHost(tmp_path / "state.db", planners=planners)
        try:
            spec = _spec(tmp_path, planner=PlannerSpec(kind="llm", config=_llm_config(identity)))
            bound = host._planner(spec.planner).spec
            experiment = host._record_experiment(
                spec, NativeCompiler(), LocalRuntime(tmp_path / "runtime"), None, bound
            )
            recorded = host.repository.aggregates.load_experiment(str(experiment.id))
            assert recorded.planner == bound
            assert recorded.planner.config["model"] == identity.model_dump()
            assert recorded.planner.config["allowed_actions"][0]["contract_fingerprint"]
        finally:
            await host.close()

        # A restarted host given the same model rebuilds the same planner from the record.
        rebuilt = EmbeddedControllerHost(tmp_path / "state.db", planners=planners)
        try:
            planner = rebuilt._recorded_planner(recorded)
            assert isinstance(planner, LLMPlanner) and planner.spec == recorded.planner
        finally:
            await rebuilt.close()

        # One given another model refuses it; one given none does not know the kind.
        other = AgentModelIdentity(provider="vendor", name="planner-large", revision="r2")
        swapped = EmbeddedControllerHost(
            tmp_path / "state.db",
            planners={**PLANNERS, "llm": llm_planner_factory(ScriptedAgentModel([], model=other))},
        )
        try:
            with pytest.raises(PlannerConfigurationError, match="agent model supplied"):
                swapped._recorded_planner(recorded)
        finally:
            await swapped.close()
        default = EmbeddedControllerHost(tmp_path / "state.db")
        try:
            with pytest.raises(UnknownImplementationError):
                await default.submit(
                    _spec(tmp_path, planner=PlannerSpec(kind="llm", config=_llm_config(identity)))
                )
        finally:
            await default.close()

    asyncio.run(scenario())
