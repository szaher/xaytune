"""RunRealization: a pure projection of what a run actually did (ADR-011).

Never a source of truth. It is recomputed from durable intervention and
application records plus the immutable attempt/checkpoint ancestry, and a
stored copy that disagrees with a rebuild is a provenance bug.

```text
RunHistoryFingerprint        every application, rolled-back work included  (audit)
ArtifactLineageFingerprint   only applications on the retained trajectory  (reuse)
```

Both are provisional until the run is terminal. Neither is candidate identity:
applying a reactive intervention never changes ``CandidateFingerprint``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from xaytune.core.domain.event import DomainEvent
from xaytune.core.domain.intervention import (
    InterventionApplication,
    TrainingIntervention,
    TrainingMutation,
    TrainingPosition,
)
from xaytune.core.domain.run import Run, artifact_lineage_fingerprint, run_history_fingerprint
from xaytune.core.ids import (
    CheckpointId,
    InterventionApplicationId,
    InterventionId,
    RunAttemptId,
    RunId,
)
from xaytune.core.immutable import FrozenDomainModel, thaw

__all__ = [
    "AttemptAncestry",
    "CheckpointAncestry",
    "InterventionApplicationRef",
    "RetainedTrajectory",
    "RunRealization",
    "project_run_realization",
    "rebuild_run_realization",
    "retained_trajectory",
]


class AttemptAncestry(FrozenDomainModel):
    """The immutable facts of one attempt that lineage needs."""

    attempt_id: RunAttemptId
    attempt_number: StrictInt = Field(ge=1)
    restored_from: CheckpointId | None = None


class CheckpointAncestry(FrozenDomainModel):
    """A committed checkpoint's producer and the applications it embodies.

    ``embodied_application_ids`` is ``None`` when the producer did not say,
    which makes every trajectory through the checkpoint unknown rather than
    silently free of interventions.
    """

    checkpoint_id: CheckpointId
    producer_attempt_id: RunAttemptId
    embodied_application_ids: tuple[str, ...] | None


class RetainedTrajectory(FrozenDomainModel):
    head_attempt_id: RunAttemptId
    checkpoint_ancestry: tuple[CheckpointId, ...]
    application_ids: tuple[InterventionApplicationId, ...]


class InterventionApplicationRef(FrozenDomainModel):
    """One application as the realization records it, in canonical order."""

    application_id: InterventionApplicationId
    intervention_id: InterventionId
    attempt_id: RunAttemptId
    event_sequence: StrictInt = Field(ge=1)
    position: TrainingPosition
    mutation: TrainingMutation
    previous_value: float
    applied_value: float
    checkpoint_ancestor_id: CheckpointId | None = None

    def history_identity(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    def lineage_identity(self) -> dict[str, object]:
        """What the trajectory experienced, without run-local identifiers."""
        return self.model_dump(
            mode="json", include={"position", "mutation", "previous_value", "applied_value"}
        )


class RunRealization(FrozenDomainModel):
    schema_version: Literal["xaytune.run-realization/v1alpha1"] = "xaytune.run-realization/v1alpha1"
    run_id: RunId
    candidate_fingerprint: str
    seed: int | None
    replicate: int | None
    interventions: tuple[InterventionId, ...]
    applied_interventions: tuple[InterventionApplicationRef, ...]
    history_fingerprint: str
    trajectory: RetainedTrajectory | None
    artifact_lineage_fingerprint: str | None

    @model_validator(mode="after")
    def _canonical(self) -> RunRealization:
        sequences = [ref.event_sequence for ref in self.applied_interventions]
        if sequences != sorted(set(sequences)):
            raise ValueError("applications are in canonical, unique event-sequence order")
        return self


def retained_trajectory(
    head: RunAttemptId,
    attempts: Iterable[AttemptAncestry],
    checkpoints: Iterable[CheckpointAncestry],
    applications: Sequence[InterventionApplication],
) -> RetainedTrajectory | None:
    """The applications and checkpoint ancestry that *head*'s model state descends from.

    ``None`` when any link is unknown: a restore from an unrecorded checkpoint,
    or a checkpoint that does not say which applications it embodies.
    """
    by_attempt = {a.attempt_id: a for a in attempts}
    by_checkpoint = {c.checkpoint_id: c for c in checkpoints}
    known = {application.id for application in applications}

    ancestry: list[CheckpointId] = []
    embodied: tuple[str, ...] = ()
    current = by_attempt.get(head)
    seen: set[RunAttemptId] = set()
    first = True
    while current is not None and current.restored_from is not None:
        if current.attempt_id in seen:
            return None
        seen.add(current.attempt_id)
        checkpoint = by_checkpoint.get(current.restored_from)
        if checkpoint is None or checkpoint.embodied_application_ids is None:
            return None
        if first:
            embodied = checkpoint.embodied_application_ids
            first = False
        ancestry.append(checkpoint.checkpoint_id)
        current = by_attempt.get(checkpoint.producer_attempt_id)
        if current is None:
            return None
    if head not in by_attempt or any(item not in known for item in embodied):
        return None
    retained = set(embodied)
    ordered = tuple(
        application.id
        for application in sorted(applications, key=lambda a: a.event_sequence)
        if application.id in retained or application.attempt_id == head
    )
    return RetainedTrajectory(
        head_attempt_id=head,
        checkpoint_ancestry=tuple(reversed(ancestry)),
        application_ids=ordered,
    )


def project_run_realization(
    run: Run,
    interventions: Iterable[TrainingIntervention],
    applications: Iterable[InterventionApplication],
    attempts: Iterable[AttemptAncestry],
    checkpoints: Iterable[CheckpointAncestry],
) -> RunRealization:
    """Pure: the same durable records always give the same realization.

    *interventions* arrive in their recorded order; *applications* are ordered
    here by event sequence, never by training position.
    """
    decisions: Mapping[InterventionId, TrainingIntervention] = {
        item.id: item for item in interventions
    }
    ordered = tuple(sorted(applications, key=lambda a: a.event_sequence))
    refs = tuple(
        InterventionApplicationRef(
            application_id=application.id,
            intervention_id=application.intervention_id,
            attempt_id=application.attempt_id,
            event_sequence=application.event_sequence,
            position=application.position,
            mutation=decisions[application.intervention_id].mutation,
            previous_value=application.previous_value,
            applied_value=application.applied_value,
            checkpoint_ancestor_id=None
            if application.checkpoint_ancestor is None
            else application.checkpoint_ancestor.id,
        )
        for application in ordered
    )
    attempt_list = tuple(attempts)
    head = max(attempt_list, key=lambda a: a.attempt_number, default=None)
    trajectory = (
        None
        if head is None
        else retained_trajectory(head.attempt_id, attempt_list, checkpoints, ordered)
    )
    lineage = None
    if trajectory is not None:
        retained = set(trajectory.application_ids)
        lineage = artifact_lineage_fingerprint(
            run,
            [ref.lineage_identity() for ref in refs if ref.application_id in retained],
            [str(item) for item in trajectory.checkpoint_ancestry],
        )
    return RunRealization(
        run_id=run.id,
        candidate_fingerprint=run.candidate_fingerprint,
        seed=run.seed,
        replicate=run.replicate,
        interventions=tuple(decisions),
        applied_interventions=refs,
        history_fingerprint=run_history_fingerprint(run, [ref.history_identity() for ref in refs]),
        trajectory=trajectory,
        artifact_lineage_fingerprint=lineage,
    )


def rebuild_run_realization(
    run: Run,
    events: Iterable[DomainEvent],
    attempts: Iterable[AttemptAncestry],
    checkpoints: Iterable[CheckpointAncestry],
) -> RunRealization:
    """Rebuild the realization from the event log, which stays authoritative.

    ``TrainingInterventionRecorded`` and ``InterventionApplied`` carry the full
    records; each application's canonical sequence is its event's sequence.
    A repository projection that disagrees with this is a provenance bug.
    """
    interventions: list[TrainingIntervention] = []
    applications: list[InterventionApplication] = []
    for event in sorted(events, key=lambda e: e.sequence or 0):
        if event.sequence is None:
            raise ValueError("a rebuild reads committed events, which carry a sequence")
        if event.event_type == "TrainingInterventionRecorded" and event.aggregate_id == str(run.id):
            interventions.append(
                TrainingIntervention.model_validate(thaw(event.payload["training_intervention"]))
            )
        elif event.event_type == "InterventionApplied" and event.payload.get("run_id") == str(
            run.id
        ):
            applications.append(
                InterventionApplication.model_validate(
                    {
                        **thaw(event.payload["intervention_application"]),
                        "event_sequence": event.sequence,
                    }
                )
            )
    return project_run_realization(run, interventions, applications, attempts, checkpoints)
