-- Migration 001 — core aggregates.
--
-- Implements the core-aggregate half of
-- `xaytune-training-harness-spec/schemas/sqlite-schema-migration-001.sql`:
-- experiments, nodes, edges, runs and attempts. The operation journal from that
-- file belongs to PR-005 and arrives with the transactional event/outbox work,
-- because an operation record is only meaningful once intent can be committed
-- atomically with its events (ADR-005 §4).
--
-- Two adaptations from the spec file, both deliberate:
--
--   * `runs.candidate_fingerprint` is added. The spec's Run carries the
--     fingerprint but its table did not expose it, and a fingerprint that can
--     only be read by parsing every payload cannot serve the reuse lookups
--     ADR-006 and ADR-011 define. Adding a column later is a migration.
--
--   * The `candidate_fingerprint` columns keep the spec's name while the Python
--     field is still `training_fingerprint`. PR-007 renames the field when
--     CandidateSpec lands; the column already has its final name, so that
--     rename costs nothing here.
--
--   * `run_attempts.attempt_number` is promoted to a column so that "attempt 2
--     of this run" can be constrained to one row. Two rows claiming to be the
--     same attempt is a provenance failure, not a duplicate to reconcile later.
--
-- Shape: identity, foreign keys, status, revision and timestamps are columns
-- because the repository queries and constrains them. Everything else lives in
-- `payload_json`, so adding a domain field does not require a migration.

CREATE TABLE experiments (
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  revision INTEGER NOT NULL CHECK (revision >= 0),
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE experiment_nodes (
  id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  status TEXT NOT NULL,
  revision INTEGER NOT NULL CHECK (revision >= 0),
  candidate_fingerprint TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX idx_experiment_nodes_experiment
  ON experiment_nodes(experiment_id, created_at);

-- "Has this hypothesis been explored?" (ADR-011). Not unique: the same
-- candidate may legitimately be explored in several experiments.
CREATE INDEX idx_experiment_nodes_fingerprint
  ON experiment_nodes(candidate_fingerprint);

-- Lineage is an edge table rather than a parent column because ADR-011 allows a
-- candidate to derive from more than one predecessor.
CREATE TABLE experiment_edges (
  parent_id TEXT NOT NULL REFERENCES experiment_nodes(id),
  child_id TEXT NOT NULL REFERENCES experiment_nodes(id),
  reason TEXT,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (parent_id, child_id),
  -- Self-parenthood is the one cycle a single row can express, so it is the one
  -- the schema can reject. Longer cycles are PR-006's job (experiment graph).
  CHECK (parent_id <> child_id)
);

CREATE INDEX idx_experiment_edges_child
  ON experiment_edges(child_id);

CREATE TABLE runs (
  id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  node_id TEXT NOT NULL REFERENCES experiment_nodes(id),
  status TEXT NOT NULL,
  revision INTEGER NOT NULL CHECK (revision >= 0),
  candidate_fingerprint TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX idx_runs_node
  ON runs(node_id, created_at);

CREATE TABLE run_attempts (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id),
  attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
  status TEXT NOT NULL,
  -- ADR-014 §1a: which telemetry stream of this attempt is current. Advanced
  -- when a supervisor dies and its history cannot be replayed while the
  -- workload keeps running -- a new attempt is NOT created for that. A column
  -- rather than a payload field because the controller assigns it and
  -- reconciliation queries it.
  telemetry_generation INTEGER NOT NULL DEFAULT 0 CHECK (telemetry_generation >= 0),
  revision INTEGER NOT NULL CHECK (revision >= 0),
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- Attempt numbers are dense and unique within a run: attempt 2 of a run is one
-- thing, and two rows claiming to be it is a provenance failure rather than a
-- duplicate to deduplicate later.
CREATE UNIQUE INDEX idx_run_attempts_number
  ON run_attempts(run_id, attempt_number);

-- Reconciliation's first question after a restart: which attempts are still
-- live? Partial, so it stays small as terminal attempts accumulate.
CREATE INDEX idx_run_attempts_unresolved
  ON run_attempts(status, updated_at)
  WHERE status NOT IN ('succeeded', 'failed', 'cancelled', 'preempted');
