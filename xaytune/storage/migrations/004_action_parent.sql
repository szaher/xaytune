-- Migration 004 — an action's parent, for the cancellation saga (ADR-013 §6).
--
-- Cancelling an experiment is one intent carried out as many: a
-- cancel-experiment Action, and a cancel-attempt Action for each live attempt,
-- each causing its own runtime operation. The children have to point at the
-- parent durably, because the parent settles only when they have, and a
-- restarted controller must be able to find them from it.
--
-- A column with REFERENCES rather than a field in payload_json: an integrity
-- rule the database can check is not left to application code, the same reason
-- caused_by_action_id got a real foreign key in 003. ADD COLUMN can carry
-- REFERENCES when the column defaults to NULL, which it does -- a top-level
-- action has no parent.

ALTER TABLE actions ADD COLUMN parent_action_id TEXT REFERENCES actions(id);

-- The saga's question: which children does this action have, and are they done?
CREATE INDEX idx_actions_parent
  ON actions(parent_action_id)
  WHERE parent_action_id IS NOT NULL;
