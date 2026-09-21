-- Migration 002 — the event log, the outbox, and the runtime operation journal.
--
-- Implemented as `xaytune/storage/migrations/002_events_outbox_operations.sql`.
-- Split from 001 because an operation record is not meaningful until intent can
-- be committed atomically with its events (ADR-005 §4), and 001 had already
-- shipped -- a shipped migration is never edited.
--
-- `caused_by_action_id` is deliberately NOT here. ADR-005 §5 requires an effect
-- to carry the Action that caused it, and SQLite cannot attach a foreign key to
-- an existing column without rebuilding the table, so the column and its
-- REFERENCES arrive together in 003.
--
-- `target_kind` is deliberately unconstrained. Freezing the vocabulary in a
-- CHECK would mean rebuilding this table for each new workload kind, and
-- ADR-015 §2 already names data preparation and reward-model scoring as likely
-- next ones. The database enforces structural shape; the domain type and the
-- repository validate the vocabulary.

PRAGMA foreign_keys = ON;

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
  target_kind TEXT NOT NULL,
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
