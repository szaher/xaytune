-- PR-017: a diagnosis of one authoritative telemetry observation, not a
-- recovery decision or a recovery-loop signature. The observation key is
-- ADR-014's (target, generation, sequence), independent of diagnosis/version.

CREATE TABLE incidents (
  id TEXT PRIMARY KEY,
  observation_key TEXT NOT NULL UNIQUE,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  run_id TEXT NOT NULL,
  target_kind TEXT NOT NULL CHECK (target_kind IN ('training-attempt', 'evaluation-attempt')),
  target_id TEXT NOT NULL,
  stream_generation INTEGER NOT NULL CHECK (stream_generation >= 0),
  sequence INTEGER NOT NULL CHECK (sequence >= 0),
  category TEXT NOT NULL,
  evidence_fingerprint TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (target_kind, target_id, stream_generation, sequence)
);

CREATE INDEX idx_incidents_experiment ON incidents(experiment_id, created_at);

CREATE TRIGGER incidents_belong_to_their_attempt
BEFORE INSERT ON incidents
FOR EACH ROW
WHEN NOT (
  (NEW.target_kind = 'training-attempt' AND EXISTS (
    SELECT 1 FROM run_attempts AS attempt JOIN runs AS run ON run.id = attempt.run_id
    WHERE attempt.id = NEW.target_id AND run.id = NEW.run_id
      AND run.experiment_id = NEW.experiment_id
  )) OR
  (NEW.target_kind = 'evaluation-attempt' AND EXISTS (
    SELECT 1 FROM evaluation_attempts AS attempt
    JOIN evaluation_runs AS run ON run.id = attempt.evaluation_run_id
    WHERE attempt.id = NEW.target_id AND run.id = NEW.run_id
      AND run.experiment_id = NEW.experiment_id
  ))
)
BEGIN
  SELECT RAISE(ABORT, 'incident names a run or experiment its attempt does not belong to');
END;

CREATE TRIGGER incidents_are_immutable
BEFORE UPDATE ON incidents
BEGIN
  SELECT RAISE(ABORT, 'incidents are append-only');
END;

CREATE TRIGGER incidents_are_permanent
BEFORE DELETE ON incidents
BEGIN
  SELECT RAISE(ABORT, 'incidents are append-only');
END;
