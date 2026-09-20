-- Migration 001 — INITIAL SUBSET, NOT THE TARGET SCHEMA.
--
-- This file covers only the aggregates Phase 1-2 needs: experiments, nodes,
-- edges, runs, attempts, events and the outbox. It is deliberately not the
-- schema the spec as a whole calls for.
--
-- Still to come, each in its own migration, and each gated on the ADR that
-- defines its shape:
--
--   actions, decisions          09-agent-planner-policy-budget.md
--   incidents                   08-resilience-and-recovery.md
--   evaluation_runs             ADR-015
--   evaluation_attempts         ADR-015
--   evaluations (results)       ADR-007, ADR-015 cache key
--   checkpoints                 ADR-009, ADR-012 (carries the DataCursor)
--   artifacts                   ADR-006
--   runtime_operations          ADR-013
--   budget_ledger               09-agent-planner-policy-budget.md
--   controller_leases           ADR-004
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
  revision INTEGER NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

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
