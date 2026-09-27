-- Migration 009 — policy decisions (PR-023).
--
-- Whether policy permitted a proposed action, which engine and rules said so,
-- and the exact snapshot it judged. Written in the same transaction as the
-- action it governs and the transition it causes, so an action never rests on
-- a verdict that is not on record, nor a verdict on an action that is not.
--
-- The snapshot lives in payload_json with the decision: it includes the
-- runtime's declared capabilities, which exist nowhere else durable.

CREATE TABLE policy_decisions (
  id TEXT PRIMARY KEY,
  -- One decision per action. Approval approves this decision's proposal; it
  -- never adds a second one.
  action_id TEXT NOT NULL UNIQUE REFERENCES actions(id),
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  engine_name TEXT NOT NULL,
  engine_version TEXT NOT NULL,
  verdict TEXT NOT NULL CHECK (verdict IN ('allow', 'deny', 'require_approval')),
  -- policy_input_identity_v1 of the snapshot, hashed. No timestamps in it.
  input_fingerprint TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX idx_policy_decisions_experiment ON policy_decisions(experiment_id, created_at);

-- A decision names its action's experiment, not another one.
CREATE TRIGGER policy_decisions_belong_to_their_action
BEFORE INSERT ON policy_decisions
FOR EACH ROW
WHEN NOT EXISTS (
  SELECT 1 FROM actions AS action
  WHERE action.id = NEW.action_id AND action.experiment_id = NEW.experiment_id
)
BEGIN
  SELECT RAISE(ABORT, 'policy decision names an experiment its action does not belong to');
END;

-- A decision is a historical fact about a snapshot. A changed state is a new
-- proposal, not an edit of the verdict on the old one.
CREATE TRIGGER policy_decisions_are_immutable
BEFORE UPDATE ON policy_decisions
BEGIN
  SELECT RAISE(ABORT, 'policy decisions are append-only');
END;

CREATE TRIGGER policy_decisions_are_permanent
BEFORE DELETE ON policy_decisions
BEGIN
  SELECT RAISE(ABORT, 'policy decisions are append-only');
END;
