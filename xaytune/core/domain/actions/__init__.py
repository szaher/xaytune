"""Typed, versioned action intent (PR-022).

An :class:`~xaytune.core.domain.action.Action` records ``type``, ``target``
and ``payload``. This package gives every type a **schema**, so a payload is
never a dictionary nobody can check, and a **mutation class**, so policy never
has to guess what kind of change it is looking at.

**Intent, not execution.** Nothing here changes training, applies an
``ExecutionOverride``, creates a ``TrainingIntervention`` or decides whether
an action is allowed. PolicyEngine is PR-023; carrying actions out comes after.

**Mutation class** is declared by each type, never inferred from its name or
parameters: ``OPERATIONAL`` (does not change scientific identity),
``SCIENTIFIC_INTERVENTION`` (changes a continuing run's trajectory, ADR-011),
``EXPERIMENT`` (changes the experiment or a candidate's standing).

**The durable payload** is a canonical envelope; ``type`` and ``target`` stay
the Action's own fields and are not repeated::

    {"schema_version": "1", "parameters": {...}}
    {"schema_version": "1", "parameters": {...},
     "provider": {"name": ..., "api_version": ..., "plugin_version": ...}}   # a plugin's

The cancellation types predate the envelope. Their payload is ``{}``, read as
schema version 1, and new cancellations keep writing ``{}``, so every row
1.0.0a1 wrote stays byte-for-byte what it was.

**History does not depend on plugins.** An ``Action`` row always loads.
:func:`spec_of` needs the type registered at the recorded version by the
recorded provider, and fails closed otherwise.

**Registration is explicit**: :func:`register_action`. There is no entry-point
discovery yet; one shared mechanism for every kind of plugin is planned rather
than one for actions alone.
"""

from __future__ import annotations

from xaytune.core.domain.actions.builtin import (
    BUILTIN_ACTION_SPECS,
    BuiltinActionSpec,
    CancelAttempt,
    CancelExperiment,
    CancelRun,
    ChangeCheckpointInterval,
    ChangeGradientAccumulation,
    ChangeLearningRate,
    ChangeScheduler,
    ChangeWarmup,
    ChangeWorkerCount,
    PromoteCandidate,
    RejectCandidate,
    ResizeMicrobatch,
)
from xaytune.core.domain.actions.contract import (
    ActionDescriptor,
    ActionPayloadError,
    ActionRegistrationError,
    ActionSpec,
    ActionValidator,
    MutationClass,
    UnsupportedActionError,
    action_descriptor,
    action_descriptors,
    action_from_spec,
    encode_payload,
    register_action,
    spec_of,
    validate_intent,
)

__all__ = [
    "BUILTIN_ACTION_SPECS",
    "ActionDescriptor",
    "ActionPayloadError",
    "ActionRegistrationError",
    "ActionSpec",
    "ActionValidator",
    "BuiltinActionSpec",
    "CancelAttempt",
    "CancelExperiment",
    "CancelRun",
    "ChangeCheckpointInterval",
    "ChangeGradientAccumulation",
    "ChangeLearningRate",
    "ChangeScheduler",
    "ChangeWarmup",
    "ChangeWorkerCount",
    "MutationClass",
    "PromoteCandidate",
    "RejectCandidate",
    "ResizeMicrobatch",
    "UnsupportedActionError",
    "action_descriptor",
    "action_descriptors",
    "action_from_spec",
    "encode_payload",
    "register_action",
    "spec_of",
    "validate_intent",
]
