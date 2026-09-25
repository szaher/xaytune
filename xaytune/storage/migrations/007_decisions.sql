-- Migration 007 — decisions (PR-015).
--
-- What became of an evaluated candidate, and why. A decision is written in the
-- same transaction that applies it -- the node's transition, and the
-- experiment's when the decision ends it -- so the record never holds a
-- decision that was not applied, nor a candidate moved with no decision behind
-- it.
--
-- Shape, as in 001: identity, foreign keys and whatever is queried or
-- constrained are columns; the rest is payload_json.

CREATE TABLE decisions (
  id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  node_id TEXT NOT NULL REFERENCES experiment_nodes(id),
  -- Which of the node's evaluation rounds it decides (ADR-015 §5).
  evaluation_cycle INTEGER NOT NULL CHECK (evaluation_cycle >= 1),
  outcome TEXT NOT NULL,
  -- The objective and the exact results decided on, hashed. A restarted
  -- controller that decides a cycle again is recognised by this; a decision
  -- on different inputs for the same cycle is refused.
  input_fingerprint TEXT NOT NULL,
  engine_name TEXT NOT NULL,
  engine_version TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

-- One decision per node per cycle. A second would be a second answer to one
-- question, and nothing could say which the candidate's state reflects.
CREATE UNIQUE INDEX idx_decisions_cycle ON decisions(node_id, evaluation_cycle);

-- A decision names its candidate's experiment, not another one.
CREATE TRIGGER decisions_belong_to_their_node
BEFORE INSERT ON decisions
FOR EACH ROW
WHEN NOT EXISTS (
  SELECT 1 FROM experiment_nodes AS node
  WHERE node.id = NEW.node_id AND node.experiment_id = NEW.experiment_id
)
BEGIN
  SELECT RAISE(ABORT, 'decision names an experiment its node does not belong to');
END;

-- A decision is a historical fact. Deciding again is a new cycle, not an edit.
CREATE TRIGGER decisions_are_immutable
BEFORE UPDATE ON decisions
BEGIN
  SELECT RAISE(ABORT, 'decisions are immutable');
END;
