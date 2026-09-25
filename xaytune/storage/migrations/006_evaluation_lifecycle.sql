-- Migration 006 — the durable evaluation lifecycle (ADR-015, PR-013).
--
-- Evaluation is a workload, so it gets training's run/attempt split in tables
-- of its own: the same lifecycle principles over different states, not
-- training's tables with evaluation squeezed in (ADR-015 §2). The operation
-- journal (002) and the action target vocabulary (003) already name
-- 'evaluation-attempt'; this is what those names resolve against.
--
-- Shape, as in 001: identity, foreign keys, status, revision and whatever the
-- repository queries or constrains are columns; the rest is payload_json.

CREATE TABLE evaluation_runs (
  id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  node_id TEXT NOT NULL REFERENCES experiment_nodes(id),
  -- Which of the node's evaluation rounds this run belongs to (ADR-015 §5). A
  -- node can evaluate, decide, return to ACTIVE and evaluate again; reconciling
  -- it considers only runs of its current round, so the first round's results
  -- can never satisfy the second.
  evaluation_cycle INTEGER NOT NULL CHECK (evaluation_cycle >= 1),
  -- The subject and identity a reuse lookup keys on (ADR-015 §3): artifact
  -- digest and EvaluationFingerprint, plus the run's seed for a SEEDED
  -- evaluator. Columns rather than payload, because the lookup queries them.
  subject_artifact_id TEXT NOT NULL,
  subject_digest TEXT,
  evaluation_fingerprint TEXT NOT NULL,
  seed INTEGER,
  replicate INTEGER,
  status TEXT NOT NULL,
  revision INTEGER NOT NULL CHECK (revision >= 0),
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- "Which runs does this round of evaluation require?"
CREATE INDEX idx_evaluation_runs_cycle
  ON evaluation_runs(node_id, evaluation_cycle, created_at);

-- "Has this subject been evaluated this way before?" Status is included
-- because only a terminal run may ever be matched: an in-flight one is not an
-- answer yet (ADR-015 §3).
CREATE INDEX idx_evaluation_runs_reuse
  ON evaluation_runs(subject_digest, evaluation_fingerprint, seed, status);

CREATE TABLE evaluation_attempts (
  id TEXT PRIMARY KEY,
  evaluation_run_id TEXT NOT NULL REFERENCES evaluation_runs(id),
  attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
  status TEXT NOT NULL,
  -- ADR-014's durable telemetry cursor, from the first row: an evaluation
  -- streams through the same envelope as training, so it needs the same
  -- (generation, sequence) pair -- advanced only in the commit of the effect
  -- an event caused, exactly as 005 defines it for run_attempts.
  telemetry_generation INTEGER NOT NULL DEFAULT 0 CHECK (telemetry_generation >= 0),
  telemetry_sequence INTEGER NOT NULL DEFAULT -1 CHECK (telemetry_sequence >= -1),
  -- An EvaluationCompleted the controller has received but cannot yet turn
  -- into a result, because the workload has not ended: its body and its
  -- (generation, sequence). Written in the commit that advances the cursor
  -- past it, so the completion is part of the record from then on -- a
  -- stream that dies and moves the attempt to a new generation (ADR-014 §1a),
  -- or a controller that dies, cannot lose it. NULL until one arrives.
  pending_completion_json TEXT,
  revision INTEGER NOT NULL CHECK (revision >= 0),
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- Attempt 2 of an evaluation run is one row, as for training (001).
CREATE UNIQUE INDEX idx_evaluation_attempts_number
  ON evaluation_attempts(evaluation_run_id, attempt_number);

-- Reconciliation's first question after a restart, as for training.
CREATE INDEX idx_evaluation_attempts_unresolved
  ON evaluation_attempts(status, updated_at)
  WHERE status NOT IN ('succeeded', 'failed', 'cancelled', 'preempted');

-- A first-class record, not a field inside a run's payload: results are what
-- decisions and reuse read, and they are queried by node and fingerprint.
CREATE TABLE evaluation_results (
  id TEXT PRIMARY KEY,
  -- UNIQUE: one run produces at most one result. A second would be two
  -- answers to one execution, and nothing could say which was the sample.
  evaluation_run_id TEXT NOT NULL UNIQUE REFERENCES evaluation_runs(id),
  node_id TEXT NOT NULL REFERENCES experiment_nodes(id),
  evaluation_fingerprint TEXT NOT NULL,
  -- The subject is compared by identity AND digest: two artifacts can share
  -- a digest, and a result naming another artifact with the same bytes would
  -- still describe an evaluation its run did not perform.
  subject_artifact_id TEXT NOT NULL,
  subject_digest TEXT,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX idx_evaluation_results_node
  ON evaluation_results(node_id, created_at);

-- A result's provenance cannot disagree with the run that produced it. The
-- repository checks this before writing; the trigger makes it a property of
-- the database, so no writer -- present or future -- can file a result under
-- one run while describing another's node, evaluation or subject.
CREATE TRIGGER evaluation_results_match_their_run
BEFORE INSERT ON evaluation_results
FOR EACH ROW
WHEN NOT EXISTS (
  SELECT 1 FROM evaluation_runs AS run
  WHERE run.id = NEW.evaluation_run_id
    AND run.node_id = NEW.node_id
    AND run.evaluation_fingerprint = NEW.evaluation_fingerprint
    AND run.subject_artifact_id = NEW.subject_artifact_id
    AND run.subject_digest IS NEW.subject_digest
)
BEGIN
  SELECT RAISE(ABORT, 'evaluation result provenance disagrees with its run');
END;

-- A result is a historical fact. Correcting one is a new run, not an edit.
CREATE TRIGGER evaluation_results_are_immutable
BEFORE UPDATE ON evaluation_results
BEGIN
  SELECT RAISE(ABORT, 'evaluation results are immutable');
END;
