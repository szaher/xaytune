-- PR-019: decisions only. No Action, attempt or runtime operation is created.
CREATE TABLE recovery_plans (
  id TEXT PRIMARY KEY,
  incident_id TEXT NOT NULL UNIQUE REFERENCES incidents(id),
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  run_id TEXT NOT NULL,
  target_kind TEXT NOT NULL,
  target_id TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE INDEX idx_recovery_plans_experiment ON recovery_plans(experiment_id);

CREATE TRIGGER recovery_plans_belong_to_their_incident
BEFORE INSERT ON recovery_plans
WHEN NOT EXISTS (
  SELECT 1 FROM incidents WHERE id = NEW.incident_id
    AND experiment_id = NEW.experiment_id AND run_id = NEW.run_id
    AND target_kind = NEW.target_kind AND target_id = NEW.target_id
)
BEGIN SELECT RAISE(ABORT, 'recovery plan does not belong to its incident'); END;

CREATE TRIGGER recovery_plans_are_immutable BEFORE UPDATE ON recovery_plans
BEGIN SELECT RAISE(ABORT, 'recovery plans are append-only'); END;
CREATE TRIGGER recovery_plans_are_permanent BEFORE DELETE ON recovery_plans
BEGIN SELECT RAISE(ABORT, 'recovery plans are append-only'); END;
