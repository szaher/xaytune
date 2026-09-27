"""The typed-action contract: spec, descriptor, registry and durable envelope (PR-022).

```text
ActionSpec (typed, validated)  ──action_from_spec──▶  Action (durable)
        ▲                                                   │
        └─────────────────────── spec_of ───────────────────┘
```

See :mod:`xaytune.core.domain.actions` for what the layer is for; this module
is the machinery built-in and plugin actions share.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, ClassVar, Literal, NoReturn, get_args

from pydantic import ConfigDict, Field, model_validator

from xaytune.core.capabilities import PluginDescriptor, require_supported_plugin
from xaytune.core.domain.action import (
    CANCELLATION_TYPES,
    Action,
    ActionTarget,
    ActionTargetKind,
    UnknownActionTypeError,
)
from xaytune.core.errors import DomainError
from xaytune.core.ids import ActionId, ExperimentId
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.refs import Actor
from xaytune.core.support import SupportResult

__all__ = [
    "ActionDescriptor",
    "ActionPayloadError",
    "ActionRegistrationError",
    "ActionSpec",
    "ActionValidator",
    "MutationClass",
    "UnsupportedActionError",
    "action_descriptor",
    "action_descriptors",
    "action_from_spec",
    "encode_payload",
    "register_action",
    "spec_of",
    "validate_intent",
]


class MutationClass(str, Enum):
    """What kind of change an action is. Declared by each type, never inferred."""

    OPERATIONAL = "operational"
    """Changes execution or control without changing scientific identity.

    Not a promise of an ``ExecutionOverride``: cancelling a run is
    operational and overrides nothing. PR-022 only classifies.
    """

    SCIENTIFIC_INTERVENTION = "scientific-intervention"
    """A scientifically meaningful change to a run that is still going (ADR-011)."""

    EXPERIMENT = "experiment"
    """Changes the experiment's or a candidate's control state."""


class ActionRegistrationError(DomainError):
    """A type cannot be registered, or encoded, as declared."""


class ActionPayloadError(DomainError):
    """A durable payload does not match what its type and version declare."""

    def __init__(self, action_id: str, action_type: str, reason: str) -> None:
        self.action_id = action_id
        self.action_type = action_type
        self.reason = reason
        super().__init__(f"action {action_id} ({action_type}): {reason}")


class UnsupportedActionError(DomainError):
    """A plugin's static validation refused a spec. Carries every reason."""

    def __init__(self, action_type: str, reasons: tuple[str, ...]) -> None:
        self.action_type = action_type
        self.reasons = reasons
        super().__init__(f"{action_type} is refused: " + "; ".join(reasons or ("no reason given",)))


class ActionSpec(FrozenDomainModel):
    """One typed intent. Subclass it to define an action type.

    A subclass declares:

    - as fields, ``type: Literal["its-name"] = "its-name"``, its typed
      parameters, and -- for a schema after the first --
      ``version: Literal["2"] = "2"``;
    - as class attributes, ``mutation_class`` and ``target_kinds``.

    Unknown fields are refused, NaN and infinity are refused, and the target
    must be one of ``target_kinds``. Checks that need no context belong in
    pydantic validators on the subclass; they run whenever a spec is built,
    including when :func:`spec_of` reads one back. A spec holds data only: no
    callables, no runtime handles.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    mutation_class: ClassVar[MutationClass]
    target_kinds: ClassVar[tuple[ActionTargetKind, ...]]

    version: Literal["1"] = "1"
    target: ActionTarget

    @model_validator(mode="after")
    def _target_is_one_this_type_acts_on(self) -> ActionSpec:
        kinds = type(self).target_kinds
        if self.target.kind not in kinds:
            raise ValueError(
                f"{_literal(type(self), 'type')} acts on {', '.join(kinds)}, "
                f"not on a {self.target.kind}"
            )
        return self

    def parameters(self) -> dict[str, Any]:
        """The typed parameters alone, canonical and JSON-ready."""
        dumped = self.model_dump(mode="json", exclude={"type", "version", "target"})
        return _canonical_mapping(dumped)


def _literal(spec: builtins.type[ActionSpec], name: str) -> str:
    """The single string a spec's ``Literal`` field *name* allows, and defaults to."""
    field = spec.model_fields.get(name)
    if field is None:
        raise ActionRegistrationError(f"{spec.__name__} declares no `{name}` field")
    choices = get_args(field.annotation)
    if len(choices) != 1 or not isinstance(choices[0], str) or field.default != choices[0]:
        raise ActionRegistrationError(
            f"{spec.__name__}.{name} must be `Literal[<value>] = <value>`: "
            f"one value, and its own default"
        )
    return choices[0]


def _canonical(value: Any) -> Any:
    """Keys sorted at every depth, sequences as lists: one spelling per value."""
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _canonical_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}


# ---- the descriptor and the registry -----------------------------------------------------

ActionValidator = Callable[["ActionSpec"], SupportResult]
"""A plugin's static check of a spec, beyond its schema. Sees the spec and nothing else."""


class ActionDescriptor(FrozenDomainModel):
    """Everything the control plane knows about one action type at one schema version.

    Built with :meth:`for_spec`, which reads the spec class's declarations;
    one constructed by hand is checked against the class, so a descriptor
    cannot say one thing while its schema says another.

    Attributes:
        provider: The plugin that defines the type, or ``None`` for one built
            into Xaytune. Its identity is persisted with each action.
        validator: Optional static validation beyond the schema. It may not
            look at live state: whether the target exists, is running, or
            whether policy allows the action is PR-023's to decide.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    type: str = Field(min_length=1)
    version: str = Field(min_length=1)
    mutation_class: MutationClass
    target_kinds: tuple[ActionTargetKind, ...] = Field(min_length=1)
    spec: builtins.type[ActionSpec]
    provider: PluginDescriptor | None = None
    validator: ActionValidator | None = None

    @model_validator(mode="after")
    def _matches_its_spec(self) -> ActionDescriptor:
        if len(set(self.target_kinds)) != len(self.target_kinds):
            raise ValueError(f"target kinds repeat: {', '.join(self.target_kinds)}")
        spec = self.spec
        declared = {
            "type": _literal(spec, "type"),
            "version": _literal(spec, "version"),
            "mutation_class": getattr(spec, "mutation_class", None),
            "target_kinds": getattr(spec, "target_kinds", None),
        }
        differing = [name for name, value in declared.items() if getattr(self, name) != value]
        if differing:
            raise ValueError(
                f"descriptor for {spec.__name__} disagrees with the class on {', '.join(differing)}"
            )
        return self

    @classmethod
    def for_spec(
        cls,
        spec: builtins.type[ActionSpec],
        provider: PluginDescriptor | None = None,
        validator: ActionValidator | None = None,
    ) -> ActionDescriptor:
        missing = [name for name in ("mutation_class", "target_kinds") if not hasattr(spec, name)]
        if missing:
            raise ActionRegistrationError(f"{spec.__name__} does not declare {', '.join(missing)}")
        return cls(
            type=_literal(spec, "type"),
            version=_literal(spec, "version"),
            mutation_class=spec.mutation_class,
            target_kinds=spec.target_kinds,
            spec=spec,
            provider=provider,
            validator=validator,
        )


_DESCRIPTORS: dict[tuple[str, str], ActionDescriptor] = {}


def register_action(descriptor: ActionDescriptor) -> None:
    """Register an action type at one schema version, so actions of it may be recorded.

    Explicit, and nothing else registers: a missing plugin is visible at
    startup, not at its first action. Registering the same descriptor again is
    a no-op, so a plugin may register on every start.

    Raises:
        IncompatiblePluginError: If the provider speaks a plugin API this
            build does not implement.
        ActionRegistrationError: If the type at that version is registered
            differently, or the type belongs to another provider.
    """
    if descriptor.provider is not None:
        require_supported_plugin(descriptor.provider)
    key = (descriptor.type, descriptor.version)
    existing = _DESCRIPTORS.get(key)
    if existing is not None:
        if existing == descriptor:
            return
        raise ActionRegistrationError(
            f"action type {descriptor.type!r} version {descriptor.version} is already "
            f"registered, by {_provider_name(existing)}, with {existing.spec.__qualname__}"
        )
    owners = {
        _provider_name(other)
        for (name, _), other in _DESCRIPTORS.items()
        if name == descriptor.type
    }
    if owners and owners != {_provider_name(descriptor)}:
        raise ActionRegistrationError(
            f"action type {descriptor.type!r} belongs to {', '.join(sorted(owners))}; "
            f"{_provider_name(descriptor)} cannot define it"
        )
    _DESCRIPTORS[key] = descriptor


def _provider_name(descriptor: ActionDescriptor) -> str:
    return "xaytune" if descriptor.provider is None else f"plugin {descriptor.provider.name!r}"


def action_descriptor(action_type: str, version: str = "1") -> ActionDescriptor:
    """The registered descriptor for *action_type* at *version*.

    Raises:
        UnknownActionTypeError: If nothing registered it.
    """
    found = _DESCRIPTORS.get((action_type, version))
    if found is None:
        raise UnknownActionTypeError(action_type, version=version)
    return found


def action_descriptors() -> tuple[ActionDescriptor, ...]:
    """Every registered descriptor, ordered by type then version."""
    return tuple(_DESCRIPTORS[key] for key in sorted(_DESCRIPTORS))


def _descriptor_of(spec: ActionSpec) -> ActionDescriptor:
    descriptor = action_descriptor(_literal(type(spec), "type"), spec.version)
    if descriptor.spec is not type(spec):
        raise ActionRegistrationError(
            f"{type(spec).__qualname__} is not the registered schema for "
            f"{descriptor.type!r} version {descriptor.version} "
            f"({descriptor.spec.__qualname__} is)"
        )
    return descriptor


def _statically_valid(descriptor: ActionDescriptor, spec: ActionSpec) -> None:
    if descriptor.validator is None:
        return
    result = descriptor.validator(spec)
    if not result.supported:
        raise UnsupportedActionError(descriptor.type, result.reasons)


# ---- the durable form --------------------------------------------------------------------


def encode_payload(spec: ActionSpec) -> dict[str, Any]:
    """The payload an action of *spec* is recorded with. Canonical and deterministic.

    ``{"schema_version": ..., "parameters": {...}}``, plus ``provider`` for a
    plugin's type. A version-1 cancellation is ``{}``, as it always was.

    Raises:
        UnknownActionTypeError: If the spec's type and version are not registered.
        ActionRegistrationError: If *spec* is not the registered class for them.
    """
    descriptor = _descriptor_of(spec)
    if descriptor.type in CANCELLATION_TYPES and descriptor.version == "1":
        return {}
    envelope: dict[str, Any] = {
        "schema_version": descriptor.version,
        "parameters": spec.parameters(),
    }
    if descriptor.provider is not None:
        envelope["provider"] = _provider_identity(descriptor.provider)
    return _canonical_mapping(envelope)


def _provider_identity(provider: PluginDescriptor) -> dict[str, str]:
    return {
        "api_version": provider.api_version,
        "name": provider.name,
        "plugin_version": provider.plugin_version,
    }


def action_from_spec(
    spec: ActionSpec,
    *,
    experiment_id: ExperimentId,
    proposed_by: Actor,
    reason: str,
    action_id: ActionId | None = None,
    parent_action_id: ActionId | None = None,
) -> Action:
    """The durable ``Action`` recording *spec*: proposed, not yet authorized.

    Builds it and writes nothing; the repository records it with whatever it
    causes, as ADR-005 requires.

    Raises:
        UnknownActionTypeError: If the spec's type and version are not registered.
        ActionRegistrationError: If *spec* is not the registered class for them.
        UnsupportedActionError: If the plugin's static validation refuses it.
    """
    descriptor = _descriptor_of(spec)
    _statically_valid(descriptor, spec)
    return Action(
        id=action_id or ActionId.generate(),
        experiment_id=experiment_id,
        type=descriptor.type,
        target=spec.target,
        proposed_by=proposed_by,
        reason=reason,
        payload=FrozenDict(encode_payload(spec)),
        parent_action_id=parent_action_id,
    )


def spec_of(action: Action) -> ActionSpec:
    """The typed spec a durable action records. Fails closed.

    Needs the type registered at the recorded schema version, by the recorded
    provider. Loading the ``Action`` itself never does, so an uninstalled
    plugin makes only its own actions' specs unavailable, not the history.

    Raises:
        UnknownActionTypeError: If the type at its schema version is not
            registered -- a plugin that is no longer installed, say.
        ActionPayloadError: If the payload is not a well-formed envelope, names
            another provider, or does not validate against the schema.
    """

    def refuse(reason: str) -> NoReturn:
        raise ActionPayloadError(str(action.id), action.type, reason)

    if not any(name == action.type for name, _ in _DESCRIPTORS):
        raise UnknownActionTypeError(action.type)
    payload = action.payload
    provider: Any
    if action.type in CANCELLATION_TYPES and not payload:
        version: Any = "1"
        parameters: Any = {}
        provider = None
    else:
        keys = set(payload)
        if (
            not {"schema_version", "parameters"}
            <= keys
            <= {
                "schema_version",
                "parameters",
                "provider",
            }
        ):
            refuse(
                "the payload must be {schema_version, parameters[, provider]}; "
                f"it has {', '.join(sorted(keys)) or 'nothing'}"
            )
        version = payload["schema_version"]
        parameters = payload["parameters"]
        provider = payload.get("provider")
        if not isinstance(version, str) or not isinstance(parameters, Mapping):
            refuse("schema_version must be a string and parameters an object")

    descriptor = action_descriptor(action.type, version)
    expected = None if descriptor.provider is None else _provider_identity(descriptor.provider)
    if (provider is None) != (expected is None):
        refuse(
            f"recorded as {'defined by a plugin' if provider is not None else 'built in'}, "
            f"but {action.type!r} version {version} is registered by "
            f"{_provider_name(descriptor)}"
        )
    if provider is not None and expected is not None:
        recorded = _thaw(provider)
        if not isinstance(recorded, dict) or any(
            recorded.get(key) != expected[key] for key in ("name", "api_version")
        ):
            refuse(
                f"recorded as defined by {recorded!r}; registered by "
                f"{expected['name']!r} at {expected['api_version']}"
            )
    try:
        return descriptor.spec.model_validate(
            {
                **_thaw(parameters),
                "type": action.type,
                "version": version,
                "target": action.target,
            }
        )
    except ValueError as invalid:
        refuse(f"the parameters do not validate against version {version}: {invalid}")


def validate_intent(action: Action) -> ActionSpec:
    """Everything a new intent must pass before it is durable: its schema, then its plugin.

    Raises:
        UnknownActionTypeError: See :func:`spec_of`.
        ActionPayloadError: See :func:`spec_of`.
        UnsupportedActionError: If the plugin's static validation refuses it.
    """
    spec = spec_of(action)
    _statically_valid(_descriptor_of(spec), spec)
    return spec


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value
