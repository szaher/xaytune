"""Which interventions a restored successor must apply, decided from durable lineage.

Pure: no I/O, no clock, no ids. The executor assigns application ids to the
planned directives; the repository re-derives the plan under its write lock and
refuses a successor whose directives disagree.

```text
restore checkpoint C, which embodies applications E
for each intervention that has taken effect on this run, in first-effect order:
    some application of it is in E      -> its effect survives; nothing to do
    APPLY_ONCE                          -> deliberately not re-applied
    REAPPLY_AFTER_ROLLBACK              -> re-apply: a new application, same decision
    REARM_TRIGGER                       -> refused: re-arming is not supported yet
then the new intervention, if any       -> its first application
```

Unknown or ambiguous lineage fails closed: a checkpoint that does not say which
applications it embodies, or names one this run never recorded, cannot
establish what the restored trajectory already contains.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from xaytune.core.domain.intervention import (
    InterventionApplication,
    InterventionDirectiveKind,
    InterventionReplayPolicy,
    TrainingIntervention,
    TrainingMutation,
)
from xaytune.core.errors import DomainError
from xaytune.core.ids import InterventionId
from xaytune.core.immutable import FrozenDomainModel

__all__ = [
    "InterventionReplayError",
    "PlannedDirective",
    "ReplayPlan",
    "plan_intervention_directives",
]


class InterventionReplayError(DomainError):
    """The restored trajectory's intervention lineage cannot be established safely."""


class PlannedDirective(FrozenDomainModel):
    intervention_id: InterventionId
    kind: InterventionDirectiveKind
    mutation: TrainingMutation
    expected_previous_value: float = Field(gt=0, allow_inf_nan=False)


class ReplayPlan(FrozenDomainModel):
    """The restored base learning rate, then each directive in application order."""

    restored_learning_rate: float = Field(gt=0, allow_inf_nan=False)
    directives: tuple[PlannedDirective, ...]

    @property
    def final_learning_rate(self) -> float:
        return (
            self.directives[-1].mutation.learning_rate
            if self.directives
            else self.restored_learning_rate
        )


def plan_intervention_directives(
    *,
    declared_learning_rate: float,
    interventions: Sequence[TrainingIntervention],
    applications: Sequence[InterventionApplication],
    restored: bool,
    embodied_application_ids: Sequence[str] | None,
    initial: TrainingIntervention | None = None,
) -> ReplayPlan:
    """Plan the directives for one successor attempt.

    Args:
        declared_learning_rate: The candidate's base learning rate.
        interventions: Every intervention on the run, in recorded order.
        applications: Every application on the run (any order).
        restored: Whether the successor restores a checkpoint.
        embodied_application_ids: The restore checkpoint's embodied applications;
            ``None`` when it does not say.
        initial: The intervention whose first effect this successor carries.

    Raises:
        InterventionReplayError: When lineage is unknown or ambiguous, a
            re-armable trigger would need re-evaluation, or *initial* already
            took effect.
    """
    by_id = {application.id: application for application in applications}
    first_effect: dict[InterventionId, int] = {}
    for application in applications:
        sequence = first_effect.get(application.intervention_id)
        if sequence is None or application.event_sequence < sequence:
            first_effect[application.intervention_id] = application.event_sequence
    known = {intervention.id for intervention in interventions}
    if any(intervention_id not in known for intervention_id in first_effect):
        raise InterventionReplayError("an application names an intervention not on this run")

    if not restored:
        if first_effect or initial is not None:
            raise InterventionReplayError(
                "a successor that restores no checkpoint cannot carry or re-apply interventions"
            )
        return ReplayPlan(restored_learning_rate=declared_learning_rate, directives=())

    if embodied_application_ids is None:
        if first_effect or initial is not None:
            raise InterventionReplayError(
                "the restore checkpoint does not record which applications it embodies"
            )
        embodied: frozenset[str] = frozenset()
    else:
        embodied = frozenset(str(item) for item in embodied_application_ids)
        if len(embodied) != len(embodied_application_ids) or any(
            item not in {str(key) for key in by_id} for item in embodied
        ):
            raise InterventionReplayError(
                "the restore checkpoint names applications this run never recorded"
            )

    retained = sorted(
        (application for application in applications if str(application.id) in embodied),
        key=lambda application: application.event_sequence,
    )
    current = retained[-1].applied_value if retained else declared_learning_rate
    restored_rate = current
    surviving = {application.intervention_id for application in retained}

    planned: list[PlannedDirective] = []
    decisions = {intervention.id: intervention for intervention in interventions}
    for intervention_id in sorted(first_effect, key=first_effect.__getitem__):
        if intervention_id in surviving:
            continue
        intervention = decisions[intervention_id]
        policy = intervention.replay_policy
        if policy is InterventionReplayPolicy.APPLY_ONCE:
            continue
        if policy is InterventionReplayPolicy.REARM_TRIGGER:
            raise InterventionReplayError(
                f"intervention {intervention_id} would need its trigger re-armed, "
                f"which this release does not support"
            )
        planned.append(
            PlannedDirective(
                intervention_id=intervention_id,
                kind=InterventionDirectiveKind.REAPPLY_AFTER_ROLLBACK,
                mutation=intervention.mutation,
                expected_previous_value=current,
            )
        )
        current = intervention.mutation.learning_rate

    if initial is not None:
        if initial.id in first_effect:
            raise InterventionReplayError(f"intervention {initial.id} has already taken effect")
        planned.append(
            PlannedDirective(
                intervention_id=initial.id,
                kind=InterventionDirectiveKind.INITIAL,
                mutation=initial.mutation,
                expected_previous_value=current,
            )
        )
    return ReplayPlan(restored_learning_rate=restored_rate, directives=tuple(planned))
