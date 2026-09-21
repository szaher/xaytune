-- Migration 002 — minimal Action substrate (PR-006a).
--
-- Required before Phase 2, not after it. ADR-013 models cancellation as
--
--     Action            durable desired intent
--     RuntimeOperation  durable external effect
--
-- and `handle.cancel()` is public API from Phase 2 onwards, so the Action
-- aggregate is band B work even though the PolicyEngine that later governs
-- actions is not (see 15-implementation-plan.md, PR-006a).
--
-- This migration deliberately contains only the substrate. Approval rules,
-- budget authorization and the mutating action types arrive in Phase 4 and will
-- add columns or tables of their own.

PRAGMA foreign_keys = ON;

CREATE TABLE actions (
  id TEXT PRIMARY KEY NOT NULL,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),

  -- PR-006a ships exactly these three. Phase 4 widens the CHECK rather than
  -- dropping it, so an unrecognised type cannot be written in the meantime.
  type TEXT NOT NULL
    CHECK (type IN ('cancel-attempt', 'cancel-run', 'cancel-experiment')),

  -- 04-state-machines.md section 5. APPROVAL_PENDING and APPROVED are reachable
  -- but unused until a PolicyEngine exists: a controller-owned cancellation goes
  -- VALIDATED -> EXECUTING, and is never marked approved by nobody.
  status TEXT NOT NULL
    CHECK (status IN (
      'proposed', 'validating', 'validated',
      'approval_pending', 'approved',
      'executing', 'succeeded', 'failed', 'rejected'
    )),

  -- ActionTarget: which aggregate this action acts on.
  target_kind TEXT NOT NULL
    CHECK (target_kind IN ('experiment', 'node', 'run', 'run-attempt')),
  target_id TEXT NOT NULL,

  proposed_by_json TEXT NOT NULL,      -- Actor
  reason TEXT NOT NULL,
  payload_json TEXT NOT NULL,

  -- Set only once a PolicyEngine exists; NULL means no policy applied, which is
  -- distinguishable from "a policy applied and allowed it".
  policy_decision_id TEXT,

  revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX idx_actions_experiment
  ON actions(experiment_id, created_at);

CREATE INDEX idx_actions_target
  ON actions(target_kind, target_id);

CREATE INDEX idx_actions_unresolved
  ON actions(status, updated_at)
  WHERE status NOT IN ('succeeded', 'failed', 'rejected');

-- The linkage ADR-013 requires: an action's intent and the external effects it
-- caused. Populated in the SAME transaction that creates the operation, so a
-- crash can never leave an operation whose cause is unrecorded, or an action
-- claiming an effect that was never requested (ADR-005 section 5).
ALTER TABLE runtime_operations
  ADD COLUMN caused_by_action_id TEXT REFERENCES actions(id);

CREATE INDEX idx_runtime_operations_action
  ON runtime_operations(caused_by_action_id);
