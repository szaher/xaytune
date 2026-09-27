"""Typed action intent: schemas, mutation classes, the durable envelope (PR-022).

```text
spec (validated)  ─action_from_spec─▶  Action {type, target, payload}  ─spec_of─▶  spec
payload           {"schema_version": "1", "parameters": {...}[, "provider": {...}]}
cancel-*          payload {} = schema version 1, unchanged since 1.0.0a1
```
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

import pytest
from pydantic import Field, StrictInt, ValidationError

from xaytune.core.capabilities import PluginDescriptor
from xaytune.core.domain.action import (
    Action,
    ActionTarget,
    ActionTargetKind,
    UnknownActionTypeError,
    register_action_type,
    registered_action_types,
)
from xaytune.core.domain.actions import (
    BUILTIN_ACTION_SPECS,
    ActionDescriptor,
    ActionPayloadError,
    ActionRegistrationError,
    ActionSpec,
    CancelAttempt,
    CancelExperiment,
    CancelRun,
    ChangeCheckpointInterval,
    ChangeGradientAccumulation,
    ChangeLearningRate,
    ChangeScheduler,
    ChangeWarmup,
    ChangeWorkerCount,
    MutationClass,
    PromoteCandidate,
    RejectCandidate,
    ResizeMicrobatch,
    UnsupportedActionError,
    action_descriptor,
    action_from_spec,
    contract,
    encode_payload,
    register_action,
    spec_of,
    validate_intent,
)
from xaytune.core.errors import DomainError, IncompatiblePluginError
from xaytune.core.ids import ExperimentId
from xaytune.core.immutable import FrozenDict
from xaytune.core.refs import Actor
from xaytune.core.support import SupportResult

ACTOR = Actor(type="llm_agent", id="planner")
EXPERIMENT = ExperimentId("exp_01J9ZQ00000000000000000EXP")


def _target(kind: ActionTargetKind) -> ActionTarget:
    return ActionTarget(kind=kind, id=f"{kind}_1")


RUN = _target("run")

BUILT_IN: dict[str, ActionSpec] = {
    "cancel-attempt": CancelAttempt(target=_target("training-attempt")),
    "cancel-run": CancelRun(target=RUN),
    "cancel-experiment": CancelExperiment(target=_target("experiment")),
    "resize-microbatch": ResizeMicrobatch(target=RUN, micro_batch_size=4, gradient_accumulation=8),
    "change-gradient-accumulation": ChangeGradientAccumulation(
        target=RUN, gradient_accumulation=16
    ),
    "change-worker-count": ChangeWorkerCount(target=RUN, workers=2),
    "change-checkpoint-interval": ChangeCheckpointInterval(target=RUN, every_steps=500),
    "change-learning-rate": ChangeLearningRate(target=RUN, learning_rate=5e-5),
    "change-scheduler": ChangeScheduler(
        target=RUN, name="cosine", params=FrozenDict({"min_lr": 1e-6})
    ),
    "change-warmup": ChangeWarmup(target=RUN, warmup_steps=100),
    "reject-candidate": RejectCandidate(target=_target("node")),
    "promote-candidate": PromoteCandidate(target=_target("node")),
}


def _action(spec: ActionSpec) -> Action:
    return action_from_spec(spec, experiment_id=EXPERIMENT, proposed_by=ACTOR, reason="because")


def _with_payload(action: Action, payload: Any) -> Action:
    return Action.model_validate({**action.model_dump(mode="python"), "payload": payload})


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registrations made by a test are undone after it."""
    monkeypatch.setattr(contract, "_DESCRIPTORS", dict(contract._DESCRIPTORS))


# ---- the vocabulary ----------------------------------------------------------------------


def test_every_built_in_type_declares_its_mutation_class() -> None:
    operational, scientific, experiment = (
        MutationClass.OPERATIONAL,
        MutationClass.SCIENTIFIC_INTERVENTION,
        MutationClass.EXPERIMENT,
    )
    assert {name: action_descriptor(name).mutation_class for name in BUILT_IN} == {
        "resize-microbatch": operational,
        "change-gradient-accumulation": operational,
        "change-worker-count": operational,
        "change-checkpoint-interval": operational,
        "cancel-attempt": operational,
        "cancel-run": operational,
        "change-learning-rate": scientific,
        "change-scheduler": scientific,
        "change-warmup": scientific,
        "cancel-experiment": experiment,
        "reject-candidate": experiment,
        "promote-candidate": experiment,
    }
    assert registered_action_types() == set(BUILT_IN)


@pytest.mark.parametrize(
    "deferred",
    ["change-reward-coefficient", "stop-experiment", "change-dataset", "change-optimizer"],
)
def test_what_has_no_honest_semantics_yet_is_not_registered(deferred: str) -> None:
    with pytest.raises(UnknownActionTypeError, match=deferred):
        action_descriptor(deferred)


def test_a_schema_less_type_can_no_longer_be_registered() -> None:
    with pytest.raises(DomainError, match="no longer permits schema-less actions"):
        register_action_type("do-something")
    from xaytune.core.domain import register_action_type as still_importable

    assert still_importable is register_action_type
    assert "do-something" not in registered_action_types()


# ---- a malformed spec is never built -----------------------------------------------------


@pytest.mark.parametrize("name", sorted(BUILT_IN))
def test_every_built_in_refuses_a_target_it_does_not_act_on(name: str) -> None:
    spec = BUILT_IN[name]
    kinds: tuple[ActionTargetKind, ...] = ("experiment", "node", "run", "training-attempt")
    wrong = next(k for k in kinds if k not in type(spec).target_kinds)
    with pytest.raises(ValidationError, match="acts on"):
        type(spec).model_validate({**spec.model_dump(), "target": _target(wrong)})


@pytest.mark.parametrize(
    ("spec", "fields"),
    [
        (ResizeMicrobatch, {"micro_batch_size": 0}),
        (ResizeMicrobatch, {"micro_batch_size": True}),
        (ResizeMicrobatch, {"micro_batch_size": "4"}),
        (ResizeMicrobatch, {"micro_batch_size": 4.0}),
        (ResizeMicrobatch, {"micro_batch_size": 4, "gradient_accumulation": 0}),
        (ChangeGradientAccumulation, {"gradient_accumulation": 0}),
        (ChangeWorkerCount, {"workers": 0}),
        (ChangeCheckpointInterval, {"every_steps": 0}),
        (ChangeLearningRate, {"learning_rate": 0.0}),
        (ChangeLearningRate, {"learning_rate": -1e-4}),
        (ChangeLearningRate, {"learning_rate": float("nan")}),
        (ChangeLearningRate, {"learning_rate": float("inf")}),
        (ChangeLearningRate, {"learning_rate": "1e-4"}),
        (ChangeScheduler, {"name": ""}),
        (ChangeScheduler, {"name": " cosine"}),
        (ChangeWarmup, {"warmup_steps": -1}),
        (ChangeWarmup, {"warmup_steps": 10, "warmup_ratio": 0.1}),
        (CancelRun, {"force": True}),
    ],
    ids=lambda value: value.__name__ if isinstance(value, type) else "-".join(map(str, value)),
)
def test_a_value_the_schema_does_not_allow_is_refused(spec: type[ActionSpec], fields: dict) -> None:
    with pytest.raises(ValidationError):
        spec(target=RUN, **fields)


def test_a_planner_output_parses_into_its_type_or_not_at_all() -> None:
    parsed = BUILTIN_ACTION_SPECS.validate_python(
        {
            "type": "change-learning-rate",
            "target": {"kind": "run", "id": "run_1"},
            "learning_rate": 1e-4,
        }
    )
    assert isinstance(parsed, ChangeLearningRate)
    for bad in (
        {"type": "change-dataset", "target": {"kind": "run", "id": "run_1"}},
        {"type": "change-learning-rate", "target": {"kind": "run", "id": "run_1"}},
        {"target": {"kind": "run", "id": "run_1"}, "learning_rate": 1e-4},
    ):
        with pytest.raises(ValidationError):
            BUILTIN_ACTION_SPECS.validate_python(bad)


# ---- the durable form --------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(BUILT_IN))
def test_every_built_in_round_trips_through_its_durable_form(name: str) -> None:
    spec = BUILT_IN[name]
    action = _action(spec)
    stored = json.dumps(action.model_dump(mode="json"), sort_keys=True)

    loaded = Action.model_validate_json(stored)

    assert (loaded.type, loaded.target) == (name, spec.target)
    assert spec_of(loaded) == spec
    assert json.dumps(loaded.model_dump(mode="json"), sort_keys=True) == stored


def test_the_payload_is_an_envelope_that_repeats_neither_type_nor_target() -> None:
    assert encode_payload(BUILT_IN["resize-microbatch"]) == {
        "schema_version": "1",
        "parameters": {"micro_batch_size": 4, "gradient_accumulation": 8},
    }


def test_the_payload_has_one_spelling_whatever_order_it_was_given_in() -> None:
    one = ChangeScheduler(
        target=RUN, name="cosine", params=FrozenDict({"b": 1, "a": {"z": [1, 2], "y": 2}})
    )
    two = ChangeScheduler(
        target=RUN, name="cosine", params=FrozenDict({"a": {"y": 2, "z": [1, 2]}, "b": 1})
    )

    # No sort_keys here: the envelope itself is canonical, not only its storage.
    assert json.dumps(encode_payload(one)) == json.dumps(encode_payload(two))
    assert json.dumps(encode_payload(one)) == (
        '{"parameters": {"name": "cosine", "params": {"a": {"y": 2, "z": [1, 2]}, "b": 1}}, '
        '"schema_version": "1"}'
    )


def test_a_cancellation_keeps_the_payload_it_has_always_had() -> None:
    assert all(encode_payload(BUILT_IN[name]) == {} for name in contract.CANCELLATION_TYPES)


# A cancel-attempt as 1.0.0a1 (5cdf350) writes it, generated on that commit.
_LEGACY_CANCELLATION = (
    '{"created_at": "2026-09-20T12:00:00Z", "experiment_id": "exp_01J9ZQ00000000000000000EXP", '
    '"id": "act_01J9ZQ0000000000000000CANC", "outcome": "superseded", '
    '"parent_action_id": "act_01J9ZQ0000000000000000PRNT", "payload": {}, '
    '"policy_decision_id": null, "proposed_by": {"id": "controller", "metadata": {}, '
    '"type": "system"}, "reason": "the user asked", "revision": 2, "status": "succeeded", '
    '"target": {"id": "att_01J9ZQ0000000000000000ATT1", "kind": "training-attempt"}, '
    '"type": "cancel-attempt", "updated_at": "2026-09-20T12:00:00Z"}'
)


def test_a_cancellation_written_before_typed_actions_reads_as_version_one_unchanged() -> None:
    action = Action.model_validate_json(_LEGACY_CANCELLATION)

    assert spec_of(action) == CancelAttempt(target=action.target)
    assert json.dumps(action.model_dump(mode="json"), sort_keys=True) == _LEGACY_CANCELLATION
    assert action.created_at == datetime(2026, 9, 20, 12, tzinfo=timezone.utc)


# ---- reading back fails closed -----------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"parameters": {"micro_batch_size": 4}}, "must be {schema_version, parameters"),
        ({"schema_version": "1"}, "must be {schema_version, parameters"),
        (
            {"schema_version": "1", "parameters": {"micro_batch_size": 4}, "type": "x"},
            "must be {schema_version, parameters",
        ),
        ({"micro_batch_size": 4}, "must be {schema_version, parameters"),
        ({"schema_version": 1, "parameters": {"micro_batch_size": 4}}, "schema_version must"),
        ({"schema_version": "1", "parameters": [4]}, "parameters an object"),
        ({"schema_version": "1", "parameters": {"micro_batch_size": 0}}, "do not validate"),
        ({"schema_version": "1", "parameters": {"micro_batch_size": 4, "x": 1}}, "do not validate"),
        (
            {
                "schema_version": "1",
                "parameters": {"micro_batch_size": 4},
                "provider": {"name": "a"},
            },
            "recorded as defined by a plugin",
        ),
    ],
    ids=[
        "no-version",
        "no-parameters",
        "extra-key",
        "bare-parameters",
        "version-not-string",
        "parameters-not-object",
        "bad-value",
        "unknown-parameter",
        "built-in-claimed-by-plugin",
    ],
)
def test_a_payload_its_schema_does_not_accept_is_refused(payload: dict, reason: str) -> None:
    action = _with_payload(_action(BUILT_IN["resize-microbatch"]), payload)
    with pytest.raises(ActionPayloadError, match=reason.replace("[", r"\[").replace("{", r"\{")):
        spec_of(action)


def test_a_version_nothing_registered_is_refused() -> None:
    action = _with_payload(
        _action(BUILT_IN["change-warmup"]),
        {"schema_version": "2", "parameters": {"warmup_steps": 1}},
    )
    with pytest.raises(UnknownActionTypeError, match="change-warmup' at schema version 2"):
        spec_of(action)


def test_a_cancellation_with_a_payload_is_read_through_the_envelope() -> None:
    action = _with_payload(_action(BUILT_IN["cancel-run"]), {"force": True})
    with pytest.raises(ActionPayloadError, match="must be {schema_version"):
        spec_of(action)


def test_a_target_the_type_does_not_act_on_is_refused_on_reading() -> None:
    action = Action.model_validate(
        {**_action(BUILT_IN["change-learning-rate"]).model_dump(), "target": _target("experiment")}
    )
    with pytest.raises(ActionPayloadError, match="acts on run"):
        spec_of(action)


# ---- plugin actions ----------------------------------------------------------------------

ACME = PluginDescriptor(
    api_version="xaytune.plugins/v1alpha1",
    name="acme-actions",
    plugin_version="0.1.0",
    provider="acme",
    xaytune_version="1.0.0a1",
)


class ReduceSequenceLength(ActionSpec):
    """A plugin's action: truncate what a run trains on."""

    mutation_class: ClassVar[MutationClass] = MutationClass.SCIENTIFIC_INTERVENTION
    target_kinds: ClassVar[tuple[ActionTargetKind, ...]] = ("run",)
    type: Literal["acme/reduce-sequence-length"] = "acme/reduce-sequence-length"
    max_length: StrictInt = Field(ge=16)


def test_a_plugin_action_is_first_class(isolated_registry: None) -> None:
    register_action(ActionDescriptor.for_spec(ReduceSequenceLength, provider=ACME))
    register_action(ActionDescriptor.for_spec(ReduceSequenceLength, provider=ACME))  # a restart

    spec = ReduceSequenceLength(target=RUN, max_length=512)
    action = _action(spec)

    assert action.payload == {
        "schema_version": "1",
        "parameters": {"max_length": 512},
        "provider": {
            "api_version": "xaytune.plugins/v1alpha1",
            "name": "acme-actions",
            "plugin_version": "0.1.0",
        },
    }
    loaded = Action.model_validate_json(json.dumps(action.model_dump(mode="json")))
    assert spec_of(loaded) == spec
    assert action_descriptor("acme/reduce-sequence-length").mutation_class is (
        MutationClass.SCIENTIFIC_INTERVENTION
    )
    with pytest.raises(ValidationError):
        ReduceSequenceLength(target=RUN, max_length=8)


def test_history_outlives_an_uninstalled_plugin_but_its_spec_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = dict(contract._DESCRIPTORS)
    monkeypatch.setattr(contract, "_DESCRIPTORS", dict(before))
    register_action(ActionDescriptor.for_spec(ReduceSequenceLength, provider=ACME))
    stored = json.dumps(
        _action(ReduceSequenceLength(target=RUN, max_length=512)).model_dump(mode="json")
    )

    monkeypatch.setattr(contract, "_DESCRIPTORS", dict(before))  # uninstalled

    loaded = Action.model_validate_json(stored)
    assert loaded.type == "acme/reduce-sequence-length"
    with pytest.raises(UnknownActionTypeError, match="acme/reduce-sequence-length"):
        spec_of(loaded)


def test_a_plugin_upgrade_reads_what_the_older_version_wrote(isolated_registry: None) -> None:
    register_action(ActionDescriptor.for_spec(ReduceSequenceLength, provider=ACME))
    written = _action(ReduceSequenceLength(target=RUN, max_length=512))
    upgraded = ACME.model_copy(update={"plugin_version": "0.2.0"})
    contract._DESCRIPTORS[("acme/reduce-sequence-length", "1")] = ActionDescriptor.for_spec(
        ReduceSequenceLength, provider=upgraded
    )

    assert spec_of(written).max_length == 512  # type: ignore[attr-defined]


def test_an_action_another_plugin_defined_is_not_read_as_this_ones(
    isolated_registry: None,
) -> None:
    register_action(ActionDescriptor.for_spec(ReduceSequenceLength, provider=ACME))
    action = _action(ReduceSequenceLength(target=RUN, max_length=512))
    impostor = {**action.payload, "provider": {**action.payload["provider"], "name": "other"}}

    with pytest.raises(ActionPayloadError, match="registered by 'acme-actions'"):
        spec_of(_with_payload(action, impostor))


class _OtherSchema(ReduceSequenceLength):
    max_length: StrictInt = Field(ge=1)


class _ClaimsABuiltIn(ActionSpec):
    mutation_class: ClassVar[MutationClass] = MutationClass.OPERATIONAL
    target_kinds: ClassVar[tuple[ActionTargetKind, ...]] = ("run",)
    type: Literal["cancel-run"] = "cancel-run"
    version: Literal["2"] = "2"  # type: ignore[assignment]


class _Unclassified(ActionSpec):
    target_kinds: ClassVar[tuple[ActionTargetKind, ...]] = ("run",)
    type: Literal["acme/unclassified"] = "acme/unclassified"


def test_a_registration_that_would_make_a_type_ambiguous_is_refused(
    isolated_registry: None,
) -> None:
    register_action(ActionDescriptor.for_spec(ReduceSequenceLength, provider=ACME))

    with pytest.raises(ActionRegistrationError, match="already registered"):
        register_action(ActionDescriptor.for_spec(_OtherSchema, provider=ACME))
    with pytest.raises(ActionRegistrationError, match="belongs to xaytune"):
        register_action(ActionDescriptor.for_spec(_ClaimsABuiltIn, provider=ACME))
    with pytest.raises(ActionRegistrationError, match="does not declare mutation_class"):
        ActionDescriptor.for_spec(_Unclassified, provider=ACME)
    with pytest.raises(IncompatiblePluginError):
        register_action(
            ActionDescriptor.for_spec(
                ReduceSequenceLength, provider=ACME.model_copy(update={"api_version": "v9"})
            )
        )


def test_a_descriptor_cannot_disagree_with_its_schema() -> None:
    with pytest.raises(ValidationError, match="mutation_class"):
        ActionDescriptor(
            type="acme/reduce-sequence-length",
            version="1",
            mutation_class=MutationClass.OPERATIONAL,
            target_kinds=("run",),
            spec=ReduceSequenceLength,
            provider=ACME,
        )


def test_only_the_registered_class_encodes_a_type(isolated_registry: None) -> None:
    register_action(ActionDescriptor.for_spec(ReduceSequenceLength, provider=ACME))
    with pytest.raises(ActionRegistrationError, match="not the registered schema"):
        encode_payload(_OtherSchema(target=RUN, max_length=4))


# ---- intent, not execution ---------------------------------------------------------------

_EXECUTORS = ("compilation", "runtimes", "trainer", "workers", "evaluation", "experiment")
_HEAVY = ("torch", "transformers", "trl", "ray", "kubernetes")


def test_typed_actions_reach_nothing_that_could_carry_them_out() -> None:
    code = (
        "import sys, xaytune.core.domain.actions\n"
        f"hit = sorted(m for m in sys.modules if m.split('.')[:2] in "
        f"[['xaytune', e] for e in {_EXECUTORS!r}] or m.split('.')[0] in {_HEAVY!r})\n"
        "print(hit)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_nothing_that_executes_reads_a_typed_action_yet() -> None:
    root = Path(__file__).resolve().parents[2] / "xaytune"
    readers = sorted(
        str(path.relative_to(root))
        for executor in _EXECUTORS
        for path in (root / executor).rglob("*.py")
        if "domain.actions" in path.read_text()
    )
    assert readers == [], "PR-022 records intent; carrying it out is a later PR"


def test_a_spec_holds_data_only() -> None:
    """No callables, no runtime handles: every field of every built-in is plain data."""
    for spec in BUILT_IN.values():
        assert json.loads(json.dumps(spec.model_dump(mode="json"))) == spec.model_dump(mode="json")
    assert isinstance(BUILT_IN["change-scheduler"].params, FrozenDict)  # type: ignore[attr-defined]


def test_worker_count_is_logical_and_runtime_neutral() -> None:
    """Workers, as in ResourceRequirements.workers -- not pods, actors, replicas or GPUs."""
    assert set(ChangeWorkerCount.model_fields) == {"type", "version", "target", "workers"}


# ---- versions and static plugin validation -----------------------------------------------


def test_a_spec_carries_its_schema_version_and_only_its_own() -> None:
    spec = BUILT_IN["change-warmup"]
    assert spec.version == "1"
    with pytest.raises(ValidationError):
        ChangeWarmup(target=RUN, warmup_steps=1, version="2")  # type: ignore[arg-type]


def _no_long_sequences(spec: ActionSpec) -> SupportResult:
    too_long = spec.max_length > 8192  # type: ignore[attr-defined]
    return SupportResult(
        supported=not too_long, reasons=("max_length above 8192",) if too_long else ()
    )


def test_a_plugin_may_refuse_a_spec_statically(isolated_registry: None) -> None:
    register_action(
        ActionDescriptor.for_spec(ReduceSequenceLength, provider=ACME, validator=_no_long_sequences)
    )
    assert _action(ReduceSequenceLength(target=RUN, max_length=512)).type == (
        "acme/reduce-sequence-length"
    )

    with pytest.raises(UnsupportedActionError, match="max_length above 8192"):
        _action(ReduceSequenceLength(target=RUN, max_length=100_000))

    # Built around action_from_spec, the durable check still runs the plugin.
    allowed = _action(ReduceSequenceLength(target=RUN, max_length=512))
    smuggled = _with_payload(allowed, {**allowed.payload, "parameters": {"max_length": 100_000}})
    assert spec_of(smuggled).max_length == 100_000  # type: ignore[attr-defined]
    with pytest.raises(UnsupportedActionError):
        validate_intent(smuggled)


def test_support_result_lives_in_core_and_compilers_still_import_it() -> None:
    from xaytune import compilation

    assert compilation.SupportResult is SupportResult


def test_a_descriptor_may_not_repeat_a_target_kind() -> None:
    with pytest.raises(ValidationError, match="repeat"):
        ActionDescriptor(
            type="acme/reduce-sequence-length",
            version="1",
            mutation_class=MutationClass.SCIENTIFIC_INTERVENTION,
            target_kinds=("run", "run"),
            spec=ReduceSequenceLength,
            provider=ACME,
        )
