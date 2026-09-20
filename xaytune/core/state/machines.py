"""Transition tables and validation for each aggregate lifecycle.

Transitions are declared as data so they can be inspected, tested and rendered
without executing anything. Validation lives here; persistence (revision
guards, events, outbox) is the repository's job.

The tables below encode exactly the transitions drawn in the architecture
specification, and nothing beyond them. Where the specification is silent — for
example a node failing while ``ACTIVE`` rather than at ``DECIDING`` — the
transition is rejected rather than quietly invented, so the gap surfaces as a
loud error instead of an undocumented behaviour.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Generic, TypeVar

from xaytune.core.errors import InvalidTransitionError
from xaytune.core.state.status import (
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
)

__all__ = [
    "ATTEMPT_MACHINE",
    "EXPERIMENT_MACHINE",
    "NODE_MACHINE",
    "RUN_MACHINE",
    "StateMachine",
]

S = TypeVar("S", bound=Enum)


class StateMachine(Generic[S]):
    """A validated transition table for one aggregate.

    Args:
        aggregate: Human-readable aggregate name, used in error messages.
        initial: The state a newly created aggregate starts in.
        transitions: Allowed successor states, keyed by source state. Every
            member of the status enum must appear as a key; terminal states map
            to an empty set.

    Raises:
        ValueError: If the table omits a state or names an unknown successor.
    """

    def __init__(
        self,
        aggregate: str,
        initial: S,
        transitions: Mapping[S, Iterable[S]],
    ) -> None:
        self.aggregate = aggregate
        self.initial = initial
        self._transitions: dict[S, frozenset[S]] = {
            state: frozenset(targets) for state, targets in transitions.items()
        }

        enum_type = type(initial)
        declared = set(self._transitions)
        missing = set(enum_type) - declared
        if missing:
            raise ValueError(
                f"{aggregate} transition table is missing states: "
                f"{sorted(s.value for s in missing)}"
            )
        for state, targets in self._transitions.items():
            unknown = {t for t in targets if not isinstance(t, enum_type)}
            if unknown:
                raise ValueError(f"{aggregate} state {state!r} names foreign targets: {unknown!r}")

    @property
    def states(self) -> frozenset[S]:
        """Every state in this machine."""
        return frozenset(self._transitions)

    @property
    def terminal_states(self) -> frozenset[S]:
        """States with no outgoing transitions."""
        return frozenset(s for s, targets in self._transitions.items() if not targets)

    def allowed_from(self, state: S) -> frozenset[S]:
        """Return the states reachable in one step from *state*."""
        return self._transitions[state]

    def is_terminal(self, state: S) -> bool:
        """Return whether *state* has no outgoing transitions."""
        return not self._transitions[state]

    def can(self, current: S, requested: S) -> bool:
        """Return whether ``current -> requested`` is permitted."""
        return requested in self._transitions[current]

    def validate(self, current: S, requested: S) -> None:
        """Raise unless ``current -> requested`` is permitted.

        Raises:
            InvalidTransitionError: If the transition is not in the table.
        """
        if not self.can(current, requested):
            raise InvalidTransitionError(self.aggregate, current, requested)


EXPERIMENT_MACHINE: StateMachine[ExperimentStatus] = StateMachine(
    "Experiment",
    ExperimentStatus.CREATED,
    {
        ExperimentStatus.CREATED: {ExperimentStatus.ACTIVE, ExperimentStatus.CANCELLED},
        ExperimentStatus.ACTIVE: {
            ExperimentStatus.PAUSED,
            ExperimentStatus.SUCCEEDED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.BUDGET_EXHAUSTED,
        },
        ExperimentStatus.PAUSED: {
            ExperimentStatus.ACTIVE,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.FAILED,
        },
        ExperimentStatus.SUCCEEDED: set(),
        ExperimentStatus.FAILED: set(),
        ExperimentStatus.CANCELLED: set(),
        ExperimentStatus.BUDGET_EXHAUSTED: set(),
    },
)

NODE_MACHINE: StateMachine[ExperimentNodeStatus] = StateMachine(
    "ExperimentNode",
    ExperimentNodeStatus.CREATED,
    {
        ExperimentNodeStatus.CREATED: {ExperimentNodeStatus.PLANNED},
        ExperimentNodeStatus.PLANNED: {ExperimentNodeStatus.READY},
        ExperimentNodeStatus.READY: {ExperimentNodeStatus.ACTIVE},
        ExperimentNodeStatus.ACTIVE: {ExperimentNodeStatus.EVALUATING},
        ExperimentNodeStatus.EVALUATING: {ExperimentNodeStatus.DECIDING},
        # A decision may send the node back to ACTIVE for another
        # run/replicate, or close it out.
        ExperimentNodeStatus.DECIDING: {
            ExperimentNodeStatus.ACTIVE,
            ExperimentNodeStatus.COMPLETED,
            ExperimentNodeStatus.REJECTED,
            ExperimentNodeStatus.FAILED,
        },
        ExperimentNodeStatus.COMPLETED: set(),
        ExperimentNodeStatus.REJECTED: set(),
        ExperimentNodeStatus.FAILED: set(),
    },
)

RUN_MACHINE: StateMachine[RunStatus] = StateMachine(
    "Run",
    RunStatus.CREATED,
    {
        RunStatus.CREATED: {RunStatus.ACTIVE},
        RunStatus.ACTIVE: {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED},
        RunStatus.SUCCEEDED: set(),
        RunStatus.FAILED: set(),
        RunStatus.CANCELLED: set(),
    },
)

ATTEMPT_MACHINE: StateMachine[RunAttemptStatus] = StateMachine(
    "RunAttempt",
    RunAttemptStatus.CREATED,
    {
        RunAttemptStatus.CREATED: {RunAttemptStatus.QUEUED},
        RunAttemptStatus.QUEUED: {RunAttemptStatus.STARTING},
        RunAttemptStatus.STARTING: {RunAttemptStatus.RUNNING},
        RunAttemptStatus.RUNNING: {
            RunAttemptStatus.CHECKPOINTING,
            RunAttemptStatus.RECOVERING,
            RunAttemptStatus.SUCCEEDED,
            RunAttemptStatus.FAILED,
            RunAttemptStatus.PREEMPTED,
            RunAttemptStatus.CANCELLED,
        },
        RunAttemptStatus.CHECKPOINTING: {RunAttemptStatus.RUNNING},
        RunAttemptStatus.RECOVERING: {RunAttemptStatus.RUNNING},
        RunAttemptStatus.SUCCEEDED: set(),
        RunAttemptStatus.FAILED: set(),
        RunAttemptStatus.PREEMPTED: set(),
        RunAttemptStatus.CANCELLED: set(),
    },
)
