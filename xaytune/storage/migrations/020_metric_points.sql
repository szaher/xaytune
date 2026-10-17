-- Track B (ADR-023 §S12): the metric-point store.
--
-- One row per scalar a training attempt's telemetry reported: the loss,
-- learning rate and throughput of a TrainingMetricObserved, a resource or
-- data measurement, a named custom metric. Written by the existing telemetry
-- ingestion path, inside a fenced write transaction, keyed by the event's
-- position in the attempt's stream. Telemetry is delivered at least once
-- (ADR-014 §4), so a replayed event finds its rows already here: the same
-- event at the same position is a no-op, a different one is a conflict.
--
-- Provenance is the event: its id, the telemetry protocol it was written
-- against, and its stream position. A point is evidence, not a transition:
-- no domain event is emitted per point, and nothing here moves an attempt.
-- Points are append-only, like the checkpoint reports they sit beside.
CREATE TABLE metric_points (
  attempt_id TEXT NOT NULL REFERENCES run_attempts(id),
  run_id TEXT NOT NULL REFERENCES runs(id),
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  stream_generation INTEGER NOT NULL CHECK (stream_generation >= 0),
  sequence INTEGER NOT NULL CHECK (sequence >= 0),
  name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 200),
  value REAL NOT NULL,
  unit TEXT,
  optimizer_step INTEGER CHECK (optimizer_step IS NULL OR optimizer_step >= 0),
  worker TEXT,
  rank INTEGER CHECK (rank IS NULL OR rank >= 0),
  observation TEXT NOT NULL,
  event_id TEXT NOT NULL,
  protocol_version TEXT NOT NULL,
  emitted_at TEXT,
  recorded_at TEXT NOT NULL,
  PRIMARY KEY (attempt_id, stream_generation, sequence, name),
  -- SQLite stores NaN as NULL, and NOT NULL already refuses it; infinities
  -- are refused here. Telemetry types refuse both before this point.
  CHECK (value BETWEEN -1.7976931348623157e308 AND 1.7976931348623157e308)
);
CREATE INDEX idx_metric_points_series
  ON metric_points(attempt_id, name, stream_generation, sequence);

CREATE TRIGGER metric_points_belong_to_their_attempt
BEFORE INSERT ON metric_points
WHEN NOT EXISTS (
  SELECT 1 FROM run_attempts AS attempt JOIN runs AS run ON run.id = attempt.run_id
  WHERE attempt.id = NEW.attempt_id AND run.id = NEW.run_id
    AND run.experiment_id = NEW.experiment_id
)
BEGIN
  SELECT RAISE(ABORT, 'metric point names a run or experiment its attempt does not belong to');
END;

CREATE TRIGGER metric_points_are_immutable BEFORE UPDATE ON metric_points
BEGIN SELECT RAISE(ABORT, 'metric points are append-only'); END;
CREATE TRIGGER metric_points_are_permanent BEFORE DELETE ON metric_points
BEGIN SELECT RAISE(ABORT, 'metric points are append-only'); END;
