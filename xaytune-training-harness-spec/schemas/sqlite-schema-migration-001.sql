-- Migration 001 — core aggregates.
--
-- The implementation of this file is `xaytune/storage/migrations/001_core_aggregates.sql`.
-- The shipped sequence is:
--
--     001  core aggregates                      PR-004
--     002  events, outbox, runtime_operations    PR-005
--     003  actions                               PR-006a
--
-- 001 and 002 are both required before Phase 2, and 003 with them: ADR-013
-- cancellation needs a durable Action to hold the intent while the operation
-- carries the effect, and handle.cancel() is public API from Phase 2 onwards.
--
-- Still to come after 003, each in its own migration, each gated on the ADR
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
--                               table in 002 is the controller's domain event
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

