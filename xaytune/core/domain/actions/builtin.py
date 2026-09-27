"""The action types built into Xaytune (PR-022).

```text
OPERATIONAL                   resize-microbatch, change-gradient-accumulation,
                              change-worker-count, change-checkpoint-interval,
                              cancel-attempt, cancel-run
SCIENTIFIC_INTERVENTION       change-learning-rate, change-scheduler, change-warmup
EXPERIMENT                    cancel-experiment, reject-candidate, promote-candidate
```

Deliberately not here yet, because the domain cannot say what they would mean:
``change-reward-coefficient`` (``RewardSpec`` declares graders, not weights),
``stop-experiment`` (no durable draining state; cancelling is not stopping),
and changes to the dataset, base model, algorithm, optimizer or adapter.

Every parameter is runtime-neutral: ``workers`` is a count of logical training
workers, never pods, nodes, actors, replicas or GPUs. A runtime translates it
into a topology.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import Field, StrictFloat, StrictInt, TypeAdapter, field_validator

from xaytune.core.domain.action import ActionTargetKind
from xaytune.core.domain.actions.contract import (
    ActionDescriptor,
    ActionSpec,
    MutationClass,
    register_action,
)
from xaytune.core.immutable import FrozenDict

__all__ = [
    "BUILTIN_ACTION_SPECS",
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
    "PromoteCandidate",
    "RejectCandidate",
    "ResizeMicrobatch",
]

_RUN: tuple[ActionTargetKind, ...] = ("run",)
_NODE: tuple[ActionTargetKind, ...] = ("node",)


# ---- operational -------------------------------------------------------------------------


class CancelAttempt(ActionSpec):
    """Stop one attempt, training or evaluation (ADR-013). Payload ``{}``, as always."""

    mutation_class = MutationClass.OPERATIONAL
    target_kinds = ("training-attempt", "evaluation-attempt")
    type: Literal["cancel-attempt"] = "cancel-attempt"


class CancelRun(ActionSpec):
    """Stop a run and its live attempt (ADR-013). Payload ``{}``, as always."""

    mutation_class = MutationClass.OPERATIONAL
    target_kinds = _RUN
    type: Literal["cancel-run"] = "cancel-run"


class ResizeMicrobatch(ActionSpec):
    """Change the micro-batch size of the run's next attempt.

    ``gradient_accumulation``, when given, is a compensating execution
    adjustment -- the usual answer to running out of memory while keeping the
    effective batch.
    """

    mutation_class = MutationClass.OPERATIONAL
    target_kinds = _RUN
    type: Literal["resize-microbatch"] = "resize-microbatch"
    micro_batch_size: StrictInt = Field(ge=1)
    gradient_accumulation: StrictInt | None = Field(default=None, ge=1)


class ChangeGradientAccumulation(ActionSpec):
    """Change how many micro-batches the run accumulates per optimizer step."""

    mutation_class = MutationClass.OPERATIONAL
    target_kinds = _RUN
    type: Literal["change-gradient-accumulation"] = "change-gradient-accumulation"
    gradient_accumulation: StrictInt = Field(ge=1)


class ChangeWorkerCount(ActionSpec):
    """Change the logical training-worker count for the run's next attempt.

    Logical workers, as in ``ResourceRequirements.workers``: never pods,
    nodes, Ray actors, JobSet or KubeRay replicas, GPUs, machines or placement
    groups. The runtime maps workers onto a concrete distributed topology.
    """

    mutation_class = MutationClass.OPERATIONAL
    target_kinds = _RUN
    type: Literal["change-worker-count"] = "change-worker-count"
    workers: StrictInt = Field(ge=1)


class ChangeCheckpointInterval(ActionSpec):
    """Change how often the run checkpoints, in optimizer steps."""

    mutation_class = MutationClass.OPERATIONAL
    target_kinds = _RUN
    type: Literal["change-checkpoint-interval"] = "change-checkpoint-interval"
    every_steps: StrictInt = Field(ge=1)


# ---- scientific intervention -------------------------------------------------------------


class ChangeLearningRate(ActionSpec):
    """Change the learning rate of a run that is still going."""

    mutation_class = MutationClass.SCIENTIFIC_INTERVENTION
    target_kinds = _RUN
    type: Literal["change-learning-rate"] = "change-learning-rate"
    learning_rate: StrictFloat = Field(gt=0)


class ChangeScheduler(ActionSpec):
    """Change a running run's learning-rate schedule.

    ``params`` is frozen and stored with its keys sorted at every depth.
    Whether a trainer supports the schedule is not checked here.
    """

    mutation_class = MutationClass.SCIENTIFIC_INTERVENTION
    target_kinds = _RUN
    type: Literal["change-scheduler"] = "change-scheduler"
    name: str = Field(min_length=1)
    params: FrozenDict = Field(default_factory=FrozenDict)

    @field_validator("name")
    @classmethod
    def _no_surrounding_whitespace(cls, name: str) -> str:
        if name != name.strip():
            raise ValueError("a scheduler name has no surrounding whitespace")
        return name


class ChangeWarmup(ActionSpec):
    """Change a running run's warmup, in optimizer steps.

    Steps only: trainers round a ratio differently, which is why a candidate
    that declares one is refused too.
    """

    mutation_class = MutationClass.SCIENTIFIC_INTERVENTION
    target_kinds = _RUN
    type: Literal["change-warmup"] = "change-warmup"
    warmup_steps: StrictInt = Field(ge=0)


# ---- experiment --------------------------------------------------------------------------


class CancelExperiment(ActionSpec):
    """Stop everything the experiment owns and end it (ADR-013 §6). Payload ``{}``."""

    mutation_class = MutationClass.EXPERIMENT
    target_kinds = ("experiment",)
    type: Literal["cancel-experiment"] = "cancel-experiment"


class RejectCandidate(ActionSpec):
    """Judge a node's candidate not worth pursuing."""

    mutation_class = MutationClass.EXPERIMENT
    target_kinds = _NODE
    type: Literal["reject-candidate"] = "reject-candidate"


class PromoteCandidate(ActionSpec):
    """Judge a node's candidate the one to keep."""

    mutation_class = MutationClass.EXPERIMENT
    target_kinds = _NODE
    type: Literal["promote-candidate"] = "promote-candidate"


_BUILT_IN: tuple[type[ActionSpec], ...] = (
    CancelAttempt,
    CancelRun,
    ResizeMicrobatch,
    ChangeGradientAccumulation,
    ChangeWorkerCount,
    ChangeCheckpointInterval,
    ChangeLearningRate,
    ChangeScheduler,
    ChangeWarmup,
    CancelExperiment,
    RejectCandidate,
    PromoteCandidate,
)

BuiltinActionSpec = Annotated[
    Union[  # noqa: UP007 -- a runtime union, which the discriminator needs
        CancelAttempt,
        CancelRun,
        ResizeMicrobatch,
        ChangeGradientAccumulation,
        ChangeWorkerCount,
        ChangeCheckpointInterval,
        ChangeLearningRate,
        ChangeScheduler,
        ChangeWarmup,
        CancelExperiment,
        RejectCandidate,
        PromoteCandidate,
    ],
    Field(discriminator="type"),
]
"""Any built-in action, told apart by ``type``: what a planner's output parses into."""

BUILTIN_ACTION_SPECS: TypeAdapter[Any] = TypeAdapter(BuiltinActionSpec)

for _spec in _BUILT_IN:
    register_action(ActionDescriptor.for_spec(_spec))
