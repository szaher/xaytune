-- PR-021 executor: checkpoint-backed numerical recovery and intervention replay.
--
--   numerical_recovery_executions   immutable receipt: one governed ChangeLearningRate
--                                   consumed as a successor attempt + submit intent
--   intervention_directives         durable intent, recorded with a successor, to apply
--                                   one intervention there under a pre-assigned
--                                   application id; the application is recorded only
--                                   when the worker confirms the effect
--
-- Directives replace migration 014's successor-only trigger: a numerically bound
-- intervention is applied only through a directive, and directives encode both the
-- first effect (episode successor N+1) and re-application after a rollback.

CREATE TABLE numerical_recovery_executions (
  id TEXT PRIMARY KEY,
  episode_id TEXT NOT NULL REFERENCES recovery_episodes(id),
  plan_id TEXT NOT NULL REFERENCES recovery_plans(id),
  plan_sequence INTEGER NOT NULL CHECK (plan_sequence >= 1),
  action_id TEXT NOT NULL UNIQUE REFERENCES actions(id),
  outcome TEXT NOT NULL CHECK (outcome IN ('EXECUTED', 'ABANDONED', 'SUPERSEDED')),
  intervention_id TEXT UNIQUE REFERENCES training_interventions(id),
  successor_attempt_id TEXT UNIQUE REFERENCES run_attempts(id),
  runtime_operation_id TEXT UNIQUE REFERENCES runtime_operations(id),
  checkpoint_id TEXT REFERENCES checkpoints(id),
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  CHECK (
    (outcome = 'EXECUTED' AND intervention_id IS NOT NULL AND successor_attempt_id IS NOT NULL
      AND runtime_operation_id IS NOT NULL AND checkpoint_id IS NOT NULL)
    OR (outcome != 'EXECUTED' AND successor_attempt_id IS NULL
      AND runtime_operation_id IS NULL AND checkpoint_id IS NULL)
  )
);
CREATE INDEX idx_numerical_recovery_executions_episode
  ON numerical_recovery_executions(episode_id, created_at, id);
CREATE UNIQUE INDEX one_executed_numerical_recovery_per_episode
  ON numerical_recovery_executions(episode_id) WHERE outcome = 'EXECUTED';

CREATE TRIGGER numerical_recovery_execution_ownership
BEFORE INSERT ON numerical_recovery_executions
WHEN NOT EXISTS (
  SELECT 1 FROM numerical_recovery_action_bindings b
  JOIN recovery_episodes e ON e.id = b.episode_id
  JOIN recovery_plans p ON p.id = b.plan_id
  WHERE b.action_id = NEW.action_id AND e.id = NEW.episode_id
    AND p.id = NEW.plan_id AND p.sequence = NEW.plan_sequence
    AND e.target_kind = 'training-attempt'
)
BEGIN SELECT RAISE(ABORT, 'numerical receipt does not match its bound decision'); END;

CREATE TRIGGER numerical_recovery_executed_effect
BEFORE INSERT ON numerical_recovery_executions
WHEN NEW.outcome = 'EXECUTED' AND NOT EXISTS (
  SELECT 1 FROM recovery_episodes e
  JOIN recovery_effective_plans p ON p.episode_id = e.id
  JOIN training_interventions t ON t.id = NEW.intervention_id
  JOIN run_attempts successor ON successor.id = NEW.successor_attempt_id
  JOIN runtime_operations operation ON operation.id = NEW.runtime_operation_id
  JOIN intervention_directives d
    ON d.attempt_id = successor.id AND d.intervention_id = t.id AND d.kind = 'initial'
  WHERE e.id = NEW.episode_id AND p.id = NEW.plan_id
    AND t.action_id = NEW.action_id AND t.run_id = e.run_id
    AND p.accepted_through_sequence = (
      SELECT MAX(m.membership_sequence) FROM recovery_episode_incidents m
      WHERE m.episode_id = e.id AND m.disposition = 'ACCEPTED_FOR_DECISION'
    )
    AND NOT EXISTS (
      SELECT 1 FROM incidents i WHERE i.target_kind = e.target_kind
        AND i.target_id = e.target_id AND NOT EXISTS (
          SELECT 1 FROM recovery_episode_incidents m WHERE m.incident_id = i.id
        )
    )
    AND successor.run_id = e.run_id
    AND successor.attempt_number = e.attempt_number + 1
    AND json_extract(successor.payload_json, '$.checkpoint_ref.id') = NEW.checkpoint_id
    AND EXISTS (
      SELECT 1 FROM checkpoints ck WHERE ck.id = NEW.checkpoint_id AND ck.run_id = e.run_id
    )
    AND operation.target_kind = 'training-attempt'
    AND operation.target_id = successor.id AND operation.type = 'submit'
    AND operation.state = 'intended'
    AND operation.caused_by_action_id = NEW.action_id
)
BEGIN SELECT RAISE(ABORT, 'numerical receipt lacks a fresh decision or its bound effect'); END;

CREATE TRIGGER numerical_recovery_executions_immutable
BEFORE UPDATE ON numerical_recovery_executions
BEGIN SELECT RAISE(ABORT, 'numerical recovery executions are append-only'); END;
CREATE TRIGGER numerical_recovery_executions_permanent
BEFORE DELETE ON numerical_recovery_executions
BEGIN SELECT RAISE(ABORT, 'numerical recovery executions are append-only'); END;

-- Directives. Ordinals are the application order within the successor:
-- re-applications first (by first effect), then the episode's own intervention.
CREATE TABLE intervention_directives (
  application_id TEXT PRIMARY KEY,
  attempt_id TEXT NOT NULL REFERENCES run_attempts(id),
  run_id TEXT NOT NULL REFERENCES runs(id),
  intervention_id TEXT NOT NULL REFERENCES training_interventions(id),
  ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
  kind TEXT NOT NULL CHECK (kind IN ('initial', 'reapply-after-rollback')),
  expected_previous_value REAL NOT NULL CHECK (expected_previous_value > 0),
  applied_value REAL NOT NULL CHECK (applied_value > 0),
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (attempt_id, ordinal),
  UNIQUE (attempt_id, intervention_id)
);
CREATE UNIQUE INDEX one_initial_directive_per_intervention
  ON intervention_directives(intervention_id) WHERE kind = 'initial';

-- Every directive: same Run, a checkpoint-backed attempt, the decided value,
-- contiguous ordinals, and an application id never used before.
CREATE TRIGGER intervention_directive_ownership
BEFORE INSERT ON intervention_directives
WHEN NOT EXISTS (
  SELECT 1 FROM training_interventions t
  JOIN run_attempts a ON a.id = NEW.attempt_id
  WHERE t.id = NEW.intervention_id
    AND t.run_id = NEW.run_id AND a.run_id = NEW.run_id
    AND json_extract(a.payload_json, '$.checkpoint_ref.id') IS NOT NULL
    AND EXISTS (
      SELECT 1 FROM checkpoints ck
      WHERE ck.id = json_extract(a.payload_json, '$.checkpoint_ref.id')
        AND ck.run_id = NEW.run_id
    )
    AND NEW.applied_value = json_extract(t.payload_json, '$.mutation.learning_rate')
    AND NEW.ordinal = (
      SELECT COUNT(*) FROM intervention_directives d WHERE d.attempt_id = NEW.attempt_id
    )
    AND NOT EXISTS (SELECT 1 FROM intervention_applications ia WHERE ia.id = NEW.application_id)
)
BEGIN SELECT RAISE(ABORT, 'directive lacks its run, checkpoint-backed attempt or value'); END;

-- First effect: only a numerically bound intervention, only on its episode's
-- successor N+1, and only if it has never taken effect.
CREATE TRIGGER intervention_directive_initial
BEFORE INSERT ON intervention_directives
WHEN NEW.kind = 'initial' AND NOT EXISTS (
  SELECT 1 FROM training_interventions t
  JOIN numerical_recovery_action_bindings b ON b.action_id = t.action_id
  JOIN recovery_episodes e ON e.id = b.episode_id
  JOIN run_attempts a ON a.id = NEW.attempt_id
  WHERE t.id = NEW.intervention_id
    AND a.run_id = e.run_id AND a.attempt_number = e.attempt_number + 1
    AND a.id <> e.target_id
    AND NOT EXISTS (
      SELECT 1 FROM intervention_applications ia WHERE ia.intervention_id = t.id
    )
)
BEGIN SELECT RAISE(ABORT, 'initial directive is not on its episode successor'); END;

-- Re-application: only REAPPLY_AFTER_ROLLBACK, only after it took effect, and
-- only when the attempt's restore checkpoint records its embodied applications
-- and none of them is this intervention's.
CREATE TRIGGER intervention_directive_reapply
BEFORE INSERT ON intervention_directives
WHEN NEW.kind = 'reapply-after-rollback' AND NOT EXISTS (
  SELECT 1 FROM training_interventions t
  JOIN run_attempts a ON a.id = NEW.attempt_id
  JOIN checkpoints ck ON ck.id = json_extract(a.payload_json, '$.checkpoint_ref.id')
  WHERE t.id = NEW.intervention_id
    AND t.replay_policy = 'reapply-after-rollback'
    AND EXISTS (SELECT 1 FROM intervention_applications ia WHERE ia.intervention_id = t.id)
    AND json_type(
      ck.payload_json, '$.payload.state_manifest.applied_intervention_application_ids'
    ) = 'array'
    AND NOT EXISTS (
      SELECT 1 FROM intervention_applications ia,
        json_each(
          ck.payload_json, '$.payload.state_manifest.applied_intervention_application_ids'
        ) j
      WHERE ia.intervention_id = t.id AND j.value = ia.id
    )
)
BEGIN SELECT RAISE(ABORT, 're-application is not justified by the restored lineage'); END;

CREATE TRIGGER intervention_directives_immutable
BEFORE UPDATE ON intervention_directives
BEGIN SELECT RAISE(ABORT, 'intervention directives are append-only'); END;
CREATE TRIGGER intervention_directives_permanent
BEFORE DELETE ON intervention_directives
BEGIN SELECT RAISE(ABORT, 'intervention directives are append-only'); END;

-- Applications: a directive's application must be exactly what it directed, and
-- a numerically bound intervention takes effect only through a directive. This
-- supersedes migration 014's N+1-only rule, which could not express replay.
DROP TRIGGER numerical_intervention_application_successor;

CREATE TRIGGER intervention_application_directive
BEFORE INSERT ON intervention_applications
WHEN (
  EXISTS (SELECT 1 FROM intervention_directives d WHERE d.application_id = NEW.id)
  OR EXISTS (
    SELECT 1 FROM training_interventions t
    JOIN numerical_recovery_action_bindings b ON b.action_id = t.action_id
    WHERE t.id = NEW.intervention_id
  )
) AND NOT EXISTS (
  SELECT 1 FROM intervention_directives d
  WHERE d.application_id = NEW.id
    AND d.attempt_id = NEW.attempt_id AND d.intervention_id = NEW.intervention_id
    AND json_extract(NEW.payload_json, '$.previous_value') = d.expected_previous_value
    AND json_extract(NEW.payload_json, '$.applied_value') = d.applied_value
)
BEGIN SELECT RAISE(ABORT, 'application does not match its directive'); END;

-- A Run succeeds only on a trajectory the control plane can vouch for: every
-- directive its final attempt carried was applied and confirmed by the worker.
-- A successor that exits cleanly without confirming one followed an unknown
-- learning-rate trajectory, so its Run cannot become a scientific result.
CREATE TRIGGER run_success_requires_confirmed_directives
BEFORE UPDATE OF status ON runs
WHEN NEW.status = 'succeeded' AND OLD.status != 'succeeded' AND EXISTS (
  SELECT 1 FROM intervention_directives d
  WHERE d.attempt_id = (
      SELECT a.id FROM run_attempts a WHERE a.run_id = NEW.id
      ORDER BY a.attempt_number DESC LIMIT 1
    )
    AND NOT EXISTS (SELECT 1 FROM intervention_applications ia WHERE ia.id = d.application_id)
)
BEGIN SELECT RAISE(ABORT, 'run cannot succeed with unconfirmed intervention directives'); END;
