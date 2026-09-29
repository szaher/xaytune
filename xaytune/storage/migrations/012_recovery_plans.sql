-- Unshipped PR-019: immutable attempt episodes, evidence, and decision revisions.
-- No execution/consumption behavior. A future receipt can reference episode/plan
-- without updating these rows; successor existence remains closure authority.
CREATE TABLE recovery_episodes (
  id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  node_id TEXT NOT NULL REFERENCES experiment_nodes(id),
  run_id TEXT NOT NULL,
  target_kind TEXT NOT NULL CHECK (target_kind IN ('training-attempt', 'evaluation-attempt')),
  target_id TEXT NOT NULL,
  attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
  candidate_fingerprint TEXT NOT NULL,
  request_fingerprint TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (target_kind, target_id),
  UNIQUE (target_kind, run_id, attempt_number)
);
CREATE TABLE recovery_episode_incidents (
  incident_id TEXT PRIMARY KEY REFERENCES incidents(id),
  episode_id TEXT NOT NULL REFERENCES recovery_episodes(id),
  membership_sequence INTEGER NOT NULL CHECK (membership_sequence >= 1),
  disposition TEXT NOT NULL CHECK (disposition IN ('ACCEPTED_FOR_DECISION', 'LATE_AFTER_CLOSURE')),
  incident_signature TEXT NOT NULL,
  evidence_fingerprint TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  UNIQUE (episode_id, membership_sequence)
);
CREATE TABLE recovery_plans (
  id TEXT PRIMARY KEY,
  episode_id TEXT NOT NULL REFERENCES recovery_episodes(id),
  sequence INTEGER NOT NULL CHECK (sequence >= 1),
  supersedes_plan_id TEXT UNIQUE REFERENCES recovery_plans(id),
  accepted_through_sequence INTEGER NOT NULL CHECK (accepted_through_sequence >= 1),
  accepted_evidence_fingerprint TEXT NOT NULL,
  strategy TEXT NOT NULL CHECK (strategy IN ('FAIL','RETRY','RESUME','ROLLBACK','RUNTIME_RECOVER',
    'EXECUTION_OVERRIDE','NEW_EXPERIMENT_NODE','PAUSE_FOR_APPROVAL')),
  payload_json TEXT NOT NULL,
  UNIQUE (episode_id, sequence),
  UNIQUE (episode_id, accepted_through_sequence),
  UNIQUE (episode_id, accepted_evidence_fingerprint),
  CHECK ((sequence = 1 AND supersedes_plan_id IS NULL) OR
         (sequence > 1 AND supersedes_plan_id IS NOT NULL))
);
CREATE INDEX idx_recovery_episodes_experiment
  ON recovery_episodes(experiment_id, target_kind, run_id, attempt_number);
CREATE INDEX idx_recovery_episode_signatures
  ON recovery_episode_incidents(disposition, incident_signature, episode_id);

CREATE VIEW recovery_episode_closure AS
SELECT e.id, CASE WHEN e.target_kind = 'training-attempt' THEN EXISTS (
  SELECT 1 FROM run_attempts a WHERE a.run_id = e.run_id AND a.attempt_number > e.attempt_number
) ELSE EXISTS (
  SELECT 1 FROM evaluation_attempts a WHERE a.evaluation_run_id = e.run_id
    AND a.attempt_number > e.attempt_number
) END AS successor_exists FROM recovery_episodes e;
CREATE VIEW recovery_effective_plans AS
SELECT p.* FROM recovery_plans p WHERE NOT EXISTS (
  SELECT 1 FROM recovery_plans later WHERE later.episode_id = p.episode_id
    AND later.sequence > p.sequence
);

CREATE TRIGGER recovery_episode_ownership BEFORE INSERT ON recovery_episodes
WHEN NOT (
 (NEW.target_kind = 'training-attempt' AND EXISTS (
  SELECT 1 FROM run_attempts a JOIN runs r ON r.id = a.run_id
  WHERE a.id = NEW.target_id AND a.run_id = NEW.run_id
    AND a.attempt_number = NEW.attempt_number AND r.experiment_id = NEW.experiment_id
    AND r.node_id = NEW.node_id AND r.candidate_fingerprint = NEW.candidate_fingerprint
    AND NOT EXISTS (SELECT 1 FROM run_attempts later WHERE later.run_id = r.id
      AND later.attempt_number > a.attempt_number)
 )) OR (NEW.target_kind = 'evaluation-attempt' AND EXISTS (
  SELECT 1 FROM evaluation_attempts a JOIN evaluation_runs r ON r.id = a.evaluation_run_id
    JOIN experiment_nodes n ON n.id = r.node_id
  WHERE a.id = NEW.target_id AND a.evaluation_run_id = NEW.run_id
    AND a.attempt_number = NEW.attempt_number AND r.experiment_id = NEW.experiment_id
    AND r.node_id = NEW.node_id AND n.candidate_fingerprint = NEW.candidate_fingerprint
    AND NOT EXISTS (SELECT 1 FROM evaluation_attempts later WHERE later.evaluation_run_id = r.id
      AND later.attempt_number > a.attempt_number)
 )))
BEGIN SELECT RAISE(ABORT, 'episode has invalid ownership or a successor'); END;

CREATE TRIGGER recovery_membership_ownership BEFORE INSERT ON recovery_episode_incidents
WHEN NOT EXISTS (
 SELECT 1 FROM incidents i JOIN recovery_episodes e ON e.id = NEW.episode_id
 WHERE i.id = NEW.incident_id AND i.target_kind = e.target_kind AND i.target_id = e.target_id
   AND i.run_id = e.run_id AND i.experiment_id = e.experiment_id
   AND i.evidence_fingerprint = NEW.evidence_fingerprint
)
BEGIN SELECT RAISE(ABORT, 'membership does not belong to episode incident'); END;
CREATE TRIGGER recovery_membership_sequence BEFORE INSERT ON recovery_episode_incidents
WHEN NEW.membership_sequence != COALESCE((SELECT MAX(membership_sequence)
 FROM recovery_episode_incidents WHERE episode_id = NEW.episode_id), 0) + 1
BEGIN SELECT RAISE(ABORT, 'membership sequence must be contiguous'); END;
CREATE TRIGGER recovery_membership_closure BEFORE INSERT ON recovery_episode_incidents
WHEN (NEW.disposition = 'LATE_AFTER_CLOSURE') != (
 SELECT successor_exists FROM recovery_episode_closure WHERE id = NEW.episode_id
)
BEGIN SELECT RAISE(ABORT, 'membership disposition disagrees with successor existence'); END;

CREATE TRIGGER recovery_plan_chain BEFORE INSERT ON recovery_plans
WHEN NEW.sequence != COALESCE((SELECT MAX(sequence) FROM recovery_plans
 WHERE episode_id = NEW.episode_id), 0) + 1 OR (
 NEW.sequence > 1 AND NOT EXISTS (SELECT 1 FROM recovery_plans prev
 WHERE prev.id = NEW.supersedes_plan_id AND prev.episode_id = NEW.episode_id
   AND prev.sequence = NEW.sequence - 1)
)
BEGIN SELECT RAISE(ABORT, 'plan predecessor/sequence must continue the episode'); END;
CREATE TRIGGER recovery_plan_closure BEFORE INSERT ON recovery_plans
WHEN (SELECT successor_exists FROM recovery_episode_closure WHERE id = NEW.episode_id)
BEGIN SELECT RAISE(ABORT, 'closed episode cannot gain a decision'); END;
CREATE TRIGGER recovery_plan_coverage BEFORE INSERT ON recovery_plans
WHEN NOT EXISTS (SELECT 1 FROM recovery_episode_incidents m
 WHERE m.episode_id = NEW.episode_id AND m.membership_sequence = NEW.accepted_through_sequence
   AND m.disposition = 'ACCEPTED_FOR_DECISION') OR (
 NEW.sequence > 1 AND NEW.accepted_through_sequence != (
  SELECT accepted_through_sequence + 1 FROM recovery_plans WHERE id = NEW.supersedes_plan_id
 )) OR (NEW.sequence = 1 AND NEW.accepted_through_sequence != (
 SELECT MAX(membership_sequence) FROM recovery_episode_incidents
 WHERE episode_id = NEW.episode_id AND disposition = 'ACCEPTED_FOR_DECISION'
))
BEGIN SELECT RAISE(ABORT, 'plan must cover the next accepted evidence prefix'); END;

CREATE TRIGGER recovery_episodes_immutable BEFORE UPDATE ON recovery_episodes
BEGIN SELECT RAISE(ABORT, 'recovery episodes are append-only'); END;
CREATE TRIGGER recovery_episodes_permanent BEFORE DELETE ON recovery_episodes
BEGIN SELECT RAISE(ABORT, 'recovery episodes are append-only'); END;
CREATE TRIGGER recovery_memberships_immutable BEFORE UPDATE ON recovery_episode_incidents
BEGIN SELECT RAISE(ABORT, 'recovery memberships are append-only'); END;
CREATE TRIGGER recovery_memberships_permanent BEFORE DELETE ON recovery_episode_incidents
BEGIN SELECT RAISE(ABORT, 'recovery memberships are append-only'); END;
CREATE TRIGGER recovery_plans_immutable BEFORE UPDATE ON recovery_plans
BEGIN SELECT RAISE(ABORT, 'recovery plans are append-only'); END;
CREATE TRIGGER recovery_plans_permanent BEFORE DELETE ON recovery_plans
BEGIN SELECT RAISE(ABORT, 'recovery plans are append-only'); END;
