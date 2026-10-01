"""Append-only recovery records; effective/freshness/reservation state is queried."""

from __future__ import annotations

import sqlite3

from xaytune.core.domain.incident import Incident
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.domain.recovery import (
    RecoveryEpisode,
    RecoveryEpisodeIncident,
    RecoveryEvidenceDisposition,
    RecoveryPlan,
    incident_signature_v1,
)
from xaytune.core.refs import Actor
from xaytune.storage.journal import _require_transaction


class RecoveryEpisodeStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, episode_id: str) -> RecoveryEpisode | None:
        row = self._connection.execute(
            "SELECT payload_json FROM recovery_episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        return None if row is None else RecoveryEpisode.model_validate_json(row["payload_json"])

    def for_attempt(self, target: RuntimeOperationTarget) -> RecoveryEpisode | None:
        row = self._connection.execute(
            "SELECT payload_json FROM recovery_episodes WHERE target_kind = ? AND target_id = ?",
            (target.kind, target.id),
        ).fetchone()
        return None if row is None else RecoveryEpisode.model_validate_json(row["payload_json"])

    def for_experiment(self, experiment_id: str) -> tuple[RecoveryEpisode, ...]:
        rows = self._connection.execute(
            "SELECT payload_json FROM recovery_episodes WHERE experiment_id = ? "
            "ORDER BY target_kind, run_id, attempt_number",
            (experiment_id,),
        ).fetchall()
        return tuple(RecoveryEpisode.model_validate_json(row["payload_json"]) for row in rows)

    def membership(self, incident_id: str) -> RecoveryEpisodeIncident | None:
        row = self._connection.execute(
            "SELECT payload_json FROM recovery_episode_incidents WHERE incident_id = ?",
            (incident_id,),
        ).fetchone()
        return (
            None
            if row is None
            else RecoveryEpisodeIncident.model_validate_json(row["payload_json"])
        )

    def memberships(self, episode_id: str) -> tuple[RecoveryEpisodeIncident, ...]:
        rows = self._connection.execute(
            "SELECT payload_json FROM recovery_episode_incidents WHERE episode_id = ? "
            "ORDER BY membership_sequence",
            (episode_id,),
        ).fetchall()
        return tuple(
            RecoveryEpisodeIncident.model_validate_json(row["payload_json"]) for row in rows
        )

    def is_open(self, episode_id: str) -> bool:
        row = self._connection.execute(
            "SELECT successor_exists FROM recovery_episode_closure WHERE id = ?", (episode_id,)
        ).fetchone()
        return row is not None and not row["successor_exists"]

    def usage_excluding(self, experiment_id: str, episode_id: str) -> int:
        return int(
            self._connection.execute(
                "SELECT COUNT(*) FROM recovery_episodes e JOIN recovery_effective_plans p "
                "ON p.episode_id = e.id WHERE e.experiment_id = ? AND e.id != ? "
                "AND (p.strategy IN ('RETRY', 'RESUME') OR EXISTS ("
                "SELECT 1 FROM recovery_execution_receipts receipt "
                "WHERE receipt.episode_id = e.id AND receipt.outcome = 'EXECUTED') "
                "OR EXISTS (SELECT 1 FROM numerical_recovery_executions numerical "
                "WHERE numerical.episode_id = e.id AND numerical.outcome = 'EXECUTED'))",
                (experiment_id, episode_id),
            ).fetchone()[0]
        )

    def pending_excluding(
        self, target: RuntimeOperationTarget, run_id: str, episode_id: str
    ) -> int:
        return int(
            self._connection.execute(
                "SELECT COUNT(*) FROM recovery_episodes e JOIN recovery_effective_plans p "
                "ON p.episode_id = e.id JOIN recovery_episode_closure c ON c.id = e.id "
                "WHERE e.target_kind = ? AND e.run_id = ? AND e.id != ? "
                "AND c.successor_exists = 0 AND p.strategy IN ('RETRY', 'RESUME')",
                (target.kind, run_id, episode_id),
            ).fetchone()[0]
        )

    def prior_matching(self, episode: RecoveryEpisode, signature: str) -> int:
        return int(
            self._connection.execute(
                "SELECT COUNT(DISTINCT e.id) FROM recovery_episodes e "
                "JOIN recovery_episode_incidents m ON m.episode_id = e.id "
                "WHERE e.target_kind = ? AND e.run_id = ? AND e.attempt_number < ? "
                "AND m.disposition = 'ACCEPTED_FOR_DECISION' AND m.incident_signature = ?",
                (
                    episode.context.target.kind,
                    episode.context.run_id,
                    episode.attempt_number,
                    signature,
                ),
            ).fetchone()[0]
        )

    def _insert(self, episode: RecoveryEpisode) -> None:
        _require_transaction(self._connection, "recovery episodes")
        c = episode.context
        self._connection.execute(
            "INSERT INTO recovery_episodes (id, experiment_id, node_id, run_id, target_kind, "
            "target_id, attempt_number, candidate_fingerprint, request_fingerprint, payload_json, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(episode.id),
                str(c.experiment_id),
                str(c.node_id),
                c.run_id,
                c.target.kind,
                c.target.id,
                episode.attempt_number,
                episode.candidate_fingerprint,
                episode.request_fingerprint,
                episode.model_dump_json(),
                episode.created_at.isoformat(),
            ),
        )

    def _attach(
        self, episode: RecoveryEpisode, incident: Incident, actor: Actor
    ) -> RecoveryEpisodeIncident:
        _require_transaction(self._connection, "recovery memberships")
        existing = self.membership(str(incident.id))
        if existing is not None:
            if existing.episode_id != episode.id:
                raise ValueError("incident already belongs to another episode")
            return existing
        sequence = len(self.memberships(str(episode.id))) + 1
        membership = RecoveryEpisodeIncident(
            episode_id=episode.id,
            incident_id=incident.id,
            membership_sequence=sequence,
            disposition=(
                RecoveryEvidenceDisposition.ACCEPTED_FOR_DECISION
                if self.is_open(str(episode.id))
                else RecoveryEvidenceDisposition.LATE_AFTER_CLOSURE
            ),
            incident_signature=incident_signature_v1(incident, episode.candidate_fingerprint),
            evidence_fingerprint=incident.evidence_fingerprint,
            attached_by=actor,
        )
        self._connection.execute(
            "INSERT INTO recovery_episode_incidents (incident_id, episode_id, membership_sequence, "
            "disposition, incident_signature, evidence_fingerprint, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(incident.id),
                str(episode.id),
                sequence,
                membership.disposition.value,
                membership.incident_signature,
                membership.evidence_fingerprint,
                membership.model_dump_json(),
            ),
        )
        return membership


class RecoveryPlanStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, plan_id: str) -> RecoveryPlan | None:
        row = self._connection.execute(
            "SELECT payload_json FROM recovery_plans WHERE id = ?", (plan_id,)
        ).fetchone()
        return None if row is None else RecoveryPlan.model_validate_json(row["payload_json"])

    def effective_for_episode(self, episode_id: str) -> RecoveryPlan | None:
        row = self._connection.execute(
            "SELECT payload_json FROM recovery_effective_plans WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        return None if row is None else RecoveryPlan.model_validate_json(row["payload_json"])

    def for_incident(self, incident_id: str) -> RecoveryPlan | None:
        row = self._connection.execute(
            "SELECT p.payload_json FROM recovery_effective_plans p "
            "JOIN recovery_episode_incidents m ON m.episode_id = p.episode_id "
            "WHERE m.incident_id = ?",
            (incident_id,),
        ).fetchone()
        return None if row is None else RecoveryPlan.model_validate_json(row["payload_json"])

    def revisions_for_episode(self, episode_id: str) -> tuple[RecoveryPlan, ...]:
        rows = self._connection.execute(
            "SELECT payload_json FROM recovery_plans WHERE episode_id = ? ORDER BY sequence",
            (episode_id,),
        ).fetchall()
        return tuple(RecoveryPlan.model_validate_json(row["payload_json"]) for row in rows)

    def for_experiment(self, experiment_id: str) -> tuple[RecoveryPlan, ...]:
        rows = self._connection.execute(
            "SELECT p.payload_json FROM recovery_plans p JOIN "
            "recovery_episodes e ON e.id = p.episode_id "
            "WHERE e.experiment_id = ? ORDER BY e.target_kind, e.run_id, "
            "e.attempt_number, p.sequence",
            (experiment_id,),
        ).fetchall()
        return tuple(RecoveryPlan.model_validate_json(row["payload_json"]) for row in rows)

    def is_effective_and_fresh(self, plan_id: str) -> bool:
        """Execution freshness predicate, not execution authorization.

        Future execution must run this check in its successor-creation transaction
        and also revalidate limits/governance/checkpoint bytes.
        """
        row = self._connection.execute(
            "SELECT 1 FROM recovery_effective_plans p JOIN recovery_episodes "
            "e ON e.id = p.episode_id "
            "JOIN recovery_episode_closure c ON c.id = e.id WHERE p.id = ? "
            "AND c.successor_exists = 0 "
            "AND p.accepted_through_sequence = (SELECT MAX(m.membership_sequence) "
            "FROM recovery_episode_incidents m WHERE m.episode_id = e.id "
            "AND m.disposition = 'ACCEPTED_FOR_DECISION') AND NOT EXISTS ("
            "SELECT 1 FROM incidents i WHERE i.target_kind = e.target_kind "
            "AND i.target_id = e.target_id "
            "AND NOT EXISTS (SELECT 1 FROM recovery_episode_incidents m WHERE "
            "m.incident_id = i.id))",
            (plan_id,),
        ).fetchone()
        return row is not None

    def _insert(self, plan: RecoveryPlan) -> None:
        _require_transaction(self._connection, "recovery plans")
        self._connection.execute(
            "INSERT INTO recovery_plans (id, episode_id, sequence, supersedes_plan_id, "
            "accepted_through_sequence, accepted_evidence_fingerprint, strategy, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(plan.id),
                str(plan.episode_id),
                plan.sequence,
                None if plan.supersedes_plan_id is None else str(plan.supersedes_plan_id),
                plan.accepted_through_sequence,
                plan.accepted_evidence_fingerprint,
                plan.strategy.value,
                plan.model_dump_json(),
            ),
        )
