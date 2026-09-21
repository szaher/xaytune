-- Migration 002 — the event log, the outbox, and the runtime operation journal.
--
-- Completes the spec's migration 001. PR-004 shipped only the core aggregates,
-- because an operation record is not meaningful until intent can be committed
-- atomically with its events, and 001 has since been applied -- a shipped
-- migration is never edited, so the rest arrives here.
--
-- The sequence, which the spec now matches:
--
--     001  core aggregates                      PR-004
--     002  events, outbox, runtime_operations    PR-005   (this file)
--     003  actions                               PR-006a
--
-- All three are required before Phase 2: ADR-013 cancellation needs a durable
-- Action to hold the intent while the operation carries the effect, and
-- handle.cancel() is public API from Phase 2 onwards.

-- ADR-005 §3: written in the same transaction as the transition it describes.
-- There is no code path that writes one without the other.
CREATE TABLE events (
  -- Total order within the database. AUTOINCREMENT rather than plain ROWID so
  -- a sequence is never reused after a delete: consumers track a cursor by
  -- sequence, and a reused value would silently replay a different event.
  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT NOT NULL UNIQUE,
  experiment_id TEXT NOT NULL,

  aggregate_type TEXT NOT NULL,
  aggregate_id TEXT NOT NULL,
  -- The revision this transition PRODUCED, not the one it started from. Pairs
  -- each event with exactly one aggregate state, which is what makes the
  -- ADR-005 §10.2 invariant checkable.
  aggregate_revision INTEGER NOT NULL CHECK (aggregate_revision >= 0),

  event_type TEXT NOT NULL,
  schema_version TEXT NOT NULL,

  occurred_at TEXT NOT NULL,
  actor_json TEXT NOT NULL,
  payload_json TEXT NOT NULL
);

-- Reading one experiment's history in order, and the §10.2 revision check.
CREATE INDEX idx_events_experiment_sequence
  ON events(experiment_id, sequence);

-- NOT unique. ADR-005 §10.2 requires that an aggregate's revision equals the
-- revision of its latest *state-transition* event -- not that only one event
-- may carry a given revision. Observational events are recorded against the
-- state they were observed at and do not advance it, so one RunAttempt
-- revision legitimately carries many:
--
--     RunAttempt revision 7   MetricObserved, MetricObserved,
--                             CheckpointCommitted, Heartbeat, ...
--
-- ADR-014 emits thousands of those per attempt. A unique index here would make
-- the second one impossible, which would force a revision bump per metric and
-- turn provenance into state churn.
CREATE INDEX idx_events_aggregate_revision
  ON events(aggregate_type, aggregate_id, aggregate_revision);

-- ADR-005 §10.6: the outbox PUBLISHES events. It never submits or cancels
-- workloads. Delivery is at-least-once, which is correct for publishing a fact
-- and catastrophic for starting a GPU job -- a redelivery would be a second
-- workload. External effects go through runtime_operations below, which is
-- get-or-create rather than at-least-once.
CREATE TABLE outbox (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id),
  destination TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('pending', 'sending', 'sent', 'failed')),
  attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
  next_attempt_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- The delivery loop's query. Partial, so it stays small as delivered rows
-- accumulate.
CREATE INDEX idx_outbox_pending
  ON outbox(state, next_attempt_at)
  WHERE state IN ('pending', 'sending');

-- ADR-013: intent is committed with the attempt before any runtime call.
--
-- The target is typed rather than a foreign key, because evaluation attempts
-- use this same journal (ADR-015 §4) and live in a different table. The
-- repository enforces that target_id exists in the table named by target_kind;
-- SQLite cannot, across heterogeneous targets. That cost buys a journal that
-- does not need a schema migration for each new workload kind -- ADR-015 §2
-- already names data preparation and reward-model scoring as the likely next
-- ones.
--
-- Operation transition history is appended to the events table above, not to a
-- separate journal-transition table.
CREATE TABLE runtime_operations (
  id TEXT PRIMARY KEY NOT NULL,
  -- Deliberately NOT constrained to a fixed set of kinds. SQLite cannot alter a
  -- CHECK in place, so freezing the vocabulary here would mean rebuilding this
  -- table for each new workload type -- and ADR-015 §2 already names data
  -- preparation and reward-model scoring as the likely next ones. The database
  -- enforces structural shape; the domain type and _require_target() validate
  -- the vocabulary, which is the same division used for Action types.
  target_kind TEXT NOT NULL,
  target_id TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('submit', 'cancel')),
  -- Canonical hash of the FULL external request, not the ExecutionFingerprint.
  -- "Is this literally the same side-effect request?" is a different question
  -- from "are these executions equivalent?" (ADR-013 §2).
  request_digest TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('intended', 'sent', 'confirmed', 'failed')),
  runtime_ref_json TEXT,
  revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX idx_runtime_operations_target
  ON runtime_operations(target_kind, target_id);

-- Reconciliation's query after a restart: which effects may exist without a
-- known outcome? An unresolved operation means consult lookup_operation(),
-- never re-issue blindly.
CREATE INDEX idx_runtime_operations_unresolved
  ON runtime_operations(state, updated_at)
  WHERE state IN ('intended', 'sent');

-- ADR-005 §5's caused_by_action_id is NOT here. Adding the column now would
-- mean adding it without REFERENCES actions(id), since that table arrives in
-- 003 -- and SQLite cannot attach a foreign key to an existing column
-- afterwards without rebuilding the table. Migration 003 adds the column and
-- its foreign key together, so the relationship gets real referential
-- integrity instead of permanently relying on an application check.
