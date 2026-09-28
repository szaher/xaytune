-- Committed checkpoint reports are observations, not restore operations.
CREATE TABLE checkpoints (
  id TEXT PRIMARY KEY,
  attempt_id TEXT NOT NULL REFERENCES run_attempts(id),
  run_id TEXT NOT NULL REFERENCES runs(id),
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  optimizer_step INTEGER NOT NULL CHECK (optimizer_step >= 0),
  payload_json TEXT NOT NULL
);
CREATE INDEX idx_checkpoints_attempt ON checkpoints(attempt_id, optimizer_step);

CREATE TABLE checkpoint_receipts (
  attempt_id TEXT NOT NULL REFERENCES run_attempts(id),
  stream_generation INTEGER NOT NULL CHECK (stream_generation >= 0),
  sequence INTEGER NOT NULL CHECK (sequence >= 0),
  checkpoint_id TEXT NOT NULL REFERENCES checkpoints(id),
  evidence_digest TEXT NOT NULL,
  PRIMARY KEY (attempt_id, stream_generation, sequence)
);

CREATE TRIGGER checkpoints_belong_to_their_attempt
BEFORE INSERT ON checkpoints
WHEN NOT EXISTS (
  SELECT 1 FROM run_attempts AS attempt JOIN runs AS run ON run.id = attempt.run_id
  WHERE attempt.id = NEW.attempt_id AND run.id = NEW.run_id
    AND run.experiment_id = NEW.experiment_id
)
BEGIN
  SELECT RAISE(ABORT, 'checkpoint names a run or experiment its attempt does not belong to');
END;

CREATE TRIGGER checkpoint_receipts_belong_to_their_attempt
BEFORE INSERT ON checkpoint_receipts
WHEN NOT EXISTS (
  SELECT 1 FROM checkpoints WHERE id = NEW.checkpoint_id AND attempt_id = NEW.attempt_id
)
BEGIN
  SELECT RAISE(ABORT, 'checkpoint receipt names another attempt');
END;

CREATE TRIGGER checkpoints_are_immutable BEFORE UPDATE ON checkpoints
BEGIN SELECT RAISE(ABORT, 'checkpoint reports are append-only'); END;
CREATE TRIGGER checkpoints_are_permanent BEFORE DELETE ON checkpoints
BEGIN SELECT RAISE(ABORT, 'checkpoint reports are append-only'); END;
CREATE TRIGGER checkpoint_receipts_are_immutable BEFORE UPDATE ON checkpoint_receipts
BEGIN SELECT RAISE(ABORT, 'checkpoint receipts are append-only'); END;
CREATE TRIGGER checkpoint_receipts_are_permanent BEFORE DELETE ON checkpoint_receipts
BEGIN SELECT RAISE(ABORT, 'checkpoint receipts are append-only'); END;
