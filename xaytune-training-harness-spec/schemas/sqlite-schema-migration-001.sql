-- Migration 001 — INITIAL SUBSET, NOT THE TARGET SCHEMA.
--
-- This file covers the core aggregates: experiments, nodes, edges, runs,
-- attempts, runtime operations, events and the outbox. PR-005 implements the
-- operation journal before PR-009 can submit. This is not the schema the spec
-- as a whole calls for.
--
-- Migration 002 adds `actions` (PR-006a), and BOTH are required before Phase 2:
-- ADR-013 cancellation needs a durable Action to hold the intent while the
-- operation carries the effect, and `handle.cancel()` is public API from Phase 2
-- onwards. The split is sequencing, not optionality -- 001 is written by PR-005
-- and 002 by PR-006a, which is the order those PRs land in.
--
-- Still to come after that, each in its own migration, and each gated on the ADR
-- that defines its shape:
--
--   decisions                   09-agent-planner-policy-budget.md
--   incidents                   08-resilience-and-recovery.md
--   evaluation_runs             ADR-015
--   evaluation_attempts         ADR-015 (carries telemetry_generation, per ADR-014)
--   evaluations (results)       ADR-007, ADR-015 cache key
--   checkpoints                 ADR-009, ADR-012 (carries the DataCursor)
--   artifacts                   ADR-006
--   budget_ledger               09-agent-planner-policy-budget.md
--   controller_leases           ADR-004
--   worker_events               ADR-014 (the telemetry stream; the `events`
--                               table below is the controller's domain event
--                               log and is a different thing)
--
-- Do not read the absence of a table here as a decision that it is not needed.

PRAGMA foreign_keys = ON;

CREATE TABLE experiments (
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  revision INTEGER NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE experiment_nodes (
  id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  status TEXT NOT NULL,
  revision INTEGER NOT NULL,
  candidate_fingerprint TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE experiment_edges (
  parent_id TEXT NOT NULL REFERENCES experiment_nodes(id),
  child_id TEXT NOT NULL REFERENCES experiment_nodes(id),
  reason TEXT,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE runs (
  id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  node_id TEXT NOT NULL REFERENCES experiment_nodes(id),
  status TEXT NOT NULL,
  revision INTEGER NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE run_attempts (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id),
  status TEXT NOT NULL,
  -- ADR-014: which telemetry stream of this attempt is current. Advanced when a
  -- supervisor dies and its history cannot be replayed while the workload keeps
  -- running -- a new attempt is NOT created for that.
  telemetry_generation INTEGER NOT NULL DEFAULT 0,
  revision INTEGER NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- ADR-013: intent is committed with the attempt before a runtime call.
-- Repository APIs validate transitions and revision CAS; operation transition
-- history is appended atomically to events/outbox, not a separate table.
--
-- The target is typed rather than a foreign key, because evaluation attempts use
-- this same journal (ADR-015 section 4) and live in a different table. There is
-- deliberately no generic Execution aggregate: what training and evaluation
-- share is the external side effect, not the domain object. The repository
-- enforces that target_id exists in the table named by target_kind.
CREATE TABLE runtime_operations (
  id TEXT PRIMARY KEY NOT NULL,
  target_kind TEXT NOT NULL
    CHECK (target_kind IN ('training-attempt', 'evaluation-attempt')),
  target_id TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('submit', 'cancel')),
  request_digest TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('intended', 'sent', 'confirmed', 'failed')),
  runtime_ref_json TEXT,
  revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX idx_runtime_operations_target
  ON runtime_operations(target_kind, target_id);

CREATE INDEX idx_runtime_operations_unresolved
  ON runtime_operations(state, updated_at)
  WHERE state IN ('intended', 'sent');

CREATE TABLE events (
  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT NOT NULL UNIQUE,
  experiment_id TEXT NOT NULL,
  aggregate_type TEXT NOT NULL,
  aggregate_id TEXT NOT NULL,
  aggregate_revision INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  occurred_at TEXT NOT NULL,
  actor_json TEXT NOT NULL,
  payload_json TEXT NOT NULL
);

CREATE TABLE outbox (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id),
  destination TEXT NOT NULL,
  state TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  next_attempt_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX idx_events_experiment_sequence
  ON events(experiment_id, sequence);

CREATE INDEX idx_nodes_experiment
  ON experiment_nodes(experiment_id);

CREATE INDEX idx_runs_node
  ON runs(node_id);

CREATE INDEX idx_attempts_run
  ON run_attempts(run_id);
