"""Actions: the durable record of intent behind every mutation (ADR-011, ADR-013).

An `Action` is *what the controller decided to do*; a
:class:`~xaytune.core.domain.operation.RuntimeOperation` is *what actually
happened outside the process*. ADR-005 §5 requires them to commit together, so
a crash can never leave an intent that nothing will act on, or an external
effect whose cause is unrecorded.

This module is the **substrate only**. `PolicyEngine`, approvals, budget
authorization and every mutating action type arrive in Phase 4; PR-006a ships
the aggregate, its state machine and the three cancellation types, because
ADR-013 cancellation needs a durable owner for its intent two phases before
policy exists.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field

from xaytune.core.clock import utc_now
from xaytune.core.errors import DomainError
from xaytune.core.ids import ActionId, ExperimentId
from xaytune.core.immutable import AggregateModel, FrozenDict, FrozenDomainModel
from xaytune.core.refs import Actor
from xaytune.core.state.machines import ACTION_MACHINE
from xaytune.core.state.status import ActionStatus

__all__ = [
    "Action",
    "ActionOutcome",
    "ActionTarget",
    "ActionTargetKind",
    "ActionType",
    "CANCELLATION_TYPES",
    "UnknownActionTypeError",
    "register_action_type",
    "registered_action_types",
]

ActionTargetKind = Literal[
    "experiment",
    "node",
    "run",
    "training-attempt",
    "evaluation-run",
    "evaluation-attempt",
]
"""What an action acts on.

The attempt kinds match ``RuntimeOperationTarget`` on purpose. An action's
target and the target of the operation it causes are the same subject, and two
spellings would mean translating between two contracts that are meant to be
shared.
"""


class ActionOutcome(str, Enum):
    """How a **successful** action resolved.

    Every member describes a way of succeeding, so this pairs with
    ``SUCCEEDED`` and with nothing else. ``FAILED`` and ``REJECTED`` carry their
    reasons in the transition event and the policy decision; inventing an
    outcome for them would make the field mean two different things.
    """

    APPLIED = "applied"
    """The change was made."""

    SUPERSEDED = "superseded"
    """Overtaken by events; there was nothing left to do.

    ADR-013 §5: a cancellation that loses the race against natural completion
    did exactly what it was asked, and the answer was that the workload had
    already finished. Recording that as ``FAILED`` would make a routine race
    look like a defect.
    """

    NOOP = "noop"
    """Already in the requested state, so nothing changed."""


class ActionTarget(FrozenDomainModel):
    """The aggregate an action acts on."""

    kind: ActionTargetKind
    id: str


ActionType = str

CANCELLATION_TYPES: tuple[str, ...] = (
    "cancel-attempt",
    "cancel-run",
    "cancel-experiment",
)
"""The only action types band B creates.

``cancel-attempt`` is workload-neutral: it targets a training or an evaluation
attempt. Splitting it per workload would duplicate one operation on one kind of
subject, differing only in which table the target lives in.
"""

_REGISTRY: set[str] = set(CANCELLATION_TYPES)


class UnknownActionTypeError(DomainError):
    """An action names a type no one registered.

    The vocabulary lives here rather than in a database ``CHECK`` because
    SQLite cannot alter a ``CHECK`` in place: freezing the list in the schema
    would schedule a table rebuild for Phase 4, which adds many types, and for
    the plugin-defined types intended after it. Structural invariants belong to
    the database; an extensible vocabulary belongs to the layer that can
    actually change.
    """

    def __init__(self, action_type: str) -> None:
        self.action_type = action_type
        super().__init__(
            f"unknown action type {action_type!r}; registered types are "
            f"{', '.join(sorted(_REGISTRY))}"
        )


def register_action_type(action_type: str) -> None:
    """Register an action type so actions of it may be created."""
    _REGISTRY.add(action_type)


def registered_action_types() -> frozenset[str]:
    """Return every currently registered action type."""
    return frozenset(_REGISTRY)


class Action(AggregateModel):
    """One typed, durable intent to mutate the experiment.

    Attributes:
        outcome: Set **exactly when** ``status`` is ``SUCCEEDED``, and never
            otherwise. See :class:`ActionOutcome`.
        policy_decision_id: ``None`` means no policy applied, which is
            deliberately distinguishable from a policy having applied and
            allowed it. Unused until PR-023.
    """

    id: ActionId
    experiment_id: ExperimentId

    type: ActionType
    status: ActionStatus = ActionStatus.PROPOSED
    outcome: ActionOutcome | None = None

    target: ActionTarget

    proposed_by: Actor
    reason: str
    payload: FrozenDict = Field(default_factory=FrozenDict)

    policy_decision_id: str | None = None

    # The action this one carries out part of: a cancel-attempt under a
    # cancel-experiment saga (ADR-013 §6). ``None`` for a top-level intent.
    parent_action_id: ActionId | None = None

    revision: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def model_post_init(self, _context: object) -> None:
        if self.type not in _REGISTRY:
            raise UnknownActionTypeError(self.type)
        if (self.status is ActionStatus.SUCCEEDED) != (self.outcome is not None):
            raise DomainError(
                f"an outcome is present exactly when an action has SUCCEEDED; "
                f"got status={self.status.value!r} outcome="
                f"{self.outcome.value if self.outcome else None!r}"
            )

    @property
    def is_terminal(self) -> bool:
        """Whether this action has reached a final state."""
        return ACTION_MACHINE.is_terminal(self.status)

    def with_status(
        self,
        new_status: ActionStatus,
        *,
        outcome: ActionOutcome | None = None,
    ) -> Action:
        """Return a copy in *new_status*, with the revision bumped.

        *outcome* is required when moving to ``SUCCEEDED`` and refused
        otherwise, so the pairing cannot drift from the schema's ``CHECK``.

        Raises:
            InvalidTransitionError: If the transition is not permitted.
            DomainError: If the outcome does not pair with the status.
        """
        ACTION_MACHINE.validate(self.status, new_status)

        if new_status is ActionStatus.SUCCEEDED and outcome is None:
            raise DomainError(
                "a succeeding action must say how it resolved: APPLIED, "
                "SUPERSEDED (overtaken by events) or NOOP (already in the "
                "requested state)"
            )
        if new_status is not ActionStatus.SUCCEEDED and outcome is not None:
            raise DomainError(
                f"only a SUCCEEDED action carries an outcome; {new_status.value} "
                f"does not. Failure and rejection reasons belong to the "
                f"transition event and the policy decision"
            )

        return self._validated_copy(
            {
                "status": new_status,
                "outcome": outcome,
                "revision": self.revision + 1,
                "updated_at": utc_now(),
            }
        )
