"""Append-only stores for numerical Action bindings, interventions and applications."""

from __future__ import annotations

import sqlite3

from xaytune.core.domain.intervention import InterventionApplication, TrainingIntervention
from xaytune.core.domain.numerical_recovery import NumericalRecoveryActionBinding
from xaytune.storage.journal import _require_transaction

__all__ = [
    "InterventionApplicationStore",
    "NumericalRecoveryActionBindingStore",
    "TrainingInterventionStore",
]


class NumericalRecoveryActionBindingStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def _one(self, column: str, value: str) -> NumericalRecoveryActionBinding | None:
        row = self._connection.execute(
            f"SELECT payload_json FROM numerical_recovery_action_bindings WHERE {column} = ?",
            (value,),
        ).fetchone()
        return (
            None
            if row is None
            else NumericalRecoveryActionBinding.model_validate_json(row["payload_json"])
        )

    def for_action(self, action_id: str) -> NumericalRecoveryActionBinding | None:
        return self._one("action_id", action_id)

    def for_plan(self, plan_id: str) -> NumericalRecoveryActionBinding | None:
        return self._one("plan_id", plan_id)

    def _insert(self, binding: NumericalRecoveryActionBinding) -> None:
        _require_transaction(self._connection, "numerical recovery Action bindings")
        assert binding.proposal.trigger.incident_id is not None
        self._connection.execute(
            "INSERT INTO numerical_recovery_action_bindings (action_id, episode_id, plan_id, "
            "plan_sequence, proposal_fingerprint, input_fingerprint, "
            "source_execution_state_fingerprint, trigger_incident_id, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(binding.action_id),
                str(binding.episode_id),
                str(binding.plan_id),
                binding.plan_sequence,
                binding.proposal_fingerprint,
                binding.input_fingerprint,
                binding.source_execution_state_fingerprint,
                str(binding.proposal.trigger.incident_id),
                binding.model_dump_json(),
                binding.created_at.isoformat(),
            ),
        )


class TrainingInterventionStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def _one(self, column: str, value: str) -> TrainingIntervention | None:
        row = self._connection.execute(
            f"SELECT payload_json FROM training_interventions WHERE {column} = ?", (value,)
        ).fetchone()
        return (
            None if row is None else TrainingIntervention.model_validate_json(row["payload_json"])
        )

    def get(self, intervention_id: str) -> TrainingIntervention | None:
        return self._one("id", intervention_id)

    def for_action(self, action_id: str) -> TrainingIntervention | None:
        return self._one("action_id", action_id)

    def for_run(self, run_id: str) -> tuple[TrainingIntervention, ...]:
        """Every intervention on *run_id*, in recorded (event-sequence) order."""
        rows = self._connection.execute(
            "SELECT payload_json FROM training_interventions WHERE run_id = ? "
            "ORDER BY recorded_event_sequence",
            (run_id,),
        ).fetchall()
        return tuple(TrainingIntervention.model_validate_json(row["payload_json"]) for row in rows)

    def _insert(
        self, intervention: TrainingIntervention, *, experiment_id: str, event_sequence: int
    ) -> None:
        _require_transaction(self._connection, "training interventions")
        trigger = intervention.trigger
        incident_id = getattr(trigger, "incident_id", None)
        self._connection.execute(
            "INSERT INTO training_interventions (id, run_id, experiment_id, action_id, origin, "
            "trigger_type, trigger_incident_id, replay_policy, schedule_ref, derived_from, "
            "mutation_type, recorded_event_sequence, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(intervention.id),
                str(intervention.run_id),
                experiment_id,
                str(intervention.action_id),
                intervention.origin.value,
                trigger.type,
                None if incident_id is None else str(incident_id),
                intervention.replay_policy.value,
                intervention.schedule_ref,
                None if intervention.derived_from is None else str(intervention.derived_from),
                intervention.mutation.type,
                event_sequence,
                intervention.model_dump_json(),
                intervention.created_at.isoformat(),
            ),
        )


class InterventionApplicationStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, application_id: str) -> InterventionApplication | None:
        row = self._connection.execute(
            "SELECT payload_json FROM intervention_applications WHERE id = ?", (application_id,)
        ).fetchone()
        return (
            None
            if row is None
            else InterventionApplication.model_validate_json(row["payload_json"])
        )

    def for_run(self, run_id: str) -> tuple[InterventionApplication, ...]:
        """Every application on *run_id*, in canonical event-sequence order."""
        rows = self._connection.execute(
            "SELECT payload_json FROM intervention_applications WHERE run_id = ? "
            "ORDER BY event_sequence",
            (run_id,),
        ).fetchall()
        return tuple(
            InterventionApplication.model_validate_json(row["payload_json"]) for row in rows
        )

    def for_intervention(self, intervention_id: str) -> tuple[InterventionApplication, ...]:
        rows = self._connection.execute(
            "SELECT payload_json FROM intervention_applications WHERE intervention_id = ? "
            "ORDER BY event_sequence",
            (intervention_id,),
        ).fetchall()
        return tuple(
            InterventionApplication.model_validate_json(row["payload_json"]) for row in rows
        )

    def _insert(self, application: InterventionApplication, *, run_id: str) -> None:
        _require_transaction(self._connection, "intervention applications")
        checkpoint = application.checkpoint_ancestor
        self._connection.execute(
            "INSERT INTO intervention_applications (id, intervention_id, attempt_id, run_id, "
            "event_sequence, checkpoint_id, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(application.id),
                str(application.intervention_id),
                str(application.attempt_id),
                run_id,
                application.event_sequence,
                None if checkpoint is None else str(checkpoint.id),
                application.model_dump_json(),
                application.created_at.isoformat(),
            ),
        )
