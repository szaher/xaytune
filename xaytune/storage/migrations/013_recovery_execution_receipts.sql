-- PR-020: immutable accounting for governed recovery intent.
-- An EXECUTED receipt is committed with the successor attempt and INTENDED
-- submit operation, before any external runtime call. Episode/plan stay immutable.
CREATE TABLE recovery_execution_receipts (
  id TEXT PRIMARY KEY,
  episode_id TEXT NOT NULL REFERENCES recovery_episodes(id),
  plan_id TEXT NOT NULL REFERENCES recovery_plans(id),
  plan_sequence INTEGER NOT NULL CHECK (plan_sequence >= 1),
  action_id TEXT NOT NULL UNIQUE REFERENCES actions(id),
  outcome TEXT NOT NULL CHECK (outcome IN ('EXECUTED', 'ABANDONED', 'SUPERSEDED')),
  successor_attempt_id TEXT UNIQUE REFERENCES run_attempts(id),
  runtime_operation_id TEXT UNIQUE REFERENCES runtime_operations(id),
  checkpoint_id TEXT REFERENCES checkpoints(id),
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  CHECK (
    (outcome = 'EXECUTED' AND successor_attempt_id IS NOT NULL
      AND runtime_operation_id IS NOT NULL)
    OR (outcome != 'EXECUTED' AND successor_attempt_id IS NULL
      AND runtime_operation_id IS NULL AND checkpoint_id IS NULL)
  )
);
CREATE INDEX idx_recovery_execution_receipts_episode
  ON recovery_execution_receipts(episode_id, created_at, id);
CREATE UNIQUE INDEX one_executed_recovery_per_episode
  ON recovery_execution_receipts(episode_id) WHERE outcome = 'EXECUTED';

CREATE TRIGGER recovery_execution_receipt_ownership
BEFORE INSERT ON recovery_execution_receipts
WHEN NOT EXISTS (
  SELECT 1 FROM recovery_episodes e
  JOIN recovery_plans p ON p.id = NEW.plan_id
  JOIN actions a ON a.id = NEW.action_id
  WHERE e.id = NEW.episode_id AND e.target_kind = 'training-attempt'
    AND p.episode_id = e.id AND p.sequence = NEW.plan_sequence
    AND a.experiment_id = e.experiment_id AND a.target_kind = 'run'
    AND a.target_id = e.run_id AND a.type = 'resize-microbatch'
)
BEGIN SELECT RAISE(ABORT, 'receipt plan/action does not belong to training episode'); END;

CREATE TRIGGER recovery_executed_receipt_bindings
BEFORE INSERT ON recovery_execution_receipts
WHEN NEW.outcome = 'EXECUTED' AND NOT EXISTS (
  SELECT 1 FROM recovery_episodes e
  JOIN recovery_effective_plans p ON p.episode_id = e.id
  JOIN run_attempts successor ON successor.id = NEW.successor_attempt_id
  JOIN runtime_operations operation ON operation.id = NEW.runtime_operation_id
  WHERE e.id = NEW.episode_id AND p.id = NEW.plan_id
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
    AND operation.target_kind = 'training-attempt'
    AND operation.target_id = successor.id AND operation.type = 'submit'
    AND operation.state = 'intended'
    AND operation.caused_by_action_id = NEW.action_id
    AND (NEW.checkpoint_id IS NULL OR EXISTS (
      SELECT 1 FROM checkpoints checkpoint
      WHERE checkpoint.id = NEW.checkpoint_id AND checkpoint.run_id = e.run_id
    ))
)
BEGIN SELECT RAISE(ABORT, 'executed receipt lacks current decision or bound effect'); END;

CREATE TRIGGER recovery_execution_receipts_immutable
BEFORE UPDATE ON recovery_execution_receipts
BEGIN SELECT RAISE(ABORT, 'recovery execution receipts are append-only'); END;
CREATE TRIGGER recovery_execution_receipts_permanent
BEFORE DELETE ON recovery_execution_receipts
BEGIN SELECT RAISE(ABORT, 'recovery execution receipts are append-only'); END;
