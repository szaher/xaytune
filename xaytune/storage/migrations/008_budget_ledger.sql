-- Migration 008 — the budget ledger (PR-016).
--
-- An append-only journal of what an experiment's runs and attempts reserve,
-- commit, consume and release against its budget (spec 09 §10). Balances are
-- derived from the entries and never stored, so there is no counter to drift
-- from the history that produced it. Each entry is written in the same
-- transaction as the state change that causes it.

CREATE TABLE budget_ledger (
  id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  dimension TEXT NOT NULL
    CHECK (dimension IN ('runs', 'parallel_runs', 'failures', 'wall_time_seconds')),
  kind TEXT NOT NULL CHECK (kind IN ('reserve', 'commit', 'consume', 'release')),
  -- A positive decimal, as text so no binary float rounds it. Nothing is
  -- recorded for nothing: an entry of zero would claim a consequence that
  -- did not happen.
  amount TEXT NOT NULL CHECK (CAST(amount AS REAL) > 0),
  subject_kind TEXT NOT NULL
    CHECK (subject_kind IN ('run', 'training-attempt', 'evaluation-attempt')),
  subject_id TEXT NOT NULL,
  created_at TEXT NOT NULL
);

-- One entry of each kind per subject and dimension: settling a run or an
-- attempt twice -- a restarted controller, say -- writes nothing the second
-- time, and cannot count it twice.
CREATE UNIQUE INDEX idx_budget_ledger_entry
  ON budget_ledger(subject_kind, subject_id, dimension, kind);

CREATE INDEX idx_budget_ledger_experiment ON budget_ledger(experiment_id);

-- A ledger that could be edited would be a counter with extra steps.
CREATE TRIGGER budget_ledger_entries_are_immutable
BEFORE UPDATE ON budget_ledger
BEGIN
  SELECT RAISE(ABORT, 'budget ledger entries are append-only');
END;

CREATE TRIGGER budget_ledger_entries_are_permanent
BEFORE DELETE ON budget_ledger
BEGIN
  SELECT RAISE(ABORT, 'budget ledger entries are append-only');
END;
