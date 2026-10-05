-- PR-032: every invocation of an agent model by a planner, from intent to outcome.
--
-- A row is written INTENDED before the model is asked, so a crash after the
-- call cannot erase that it may have happened; the answer is recorded
-- (ANSWERED) before anything is derived from it. What was asked is
-- immutable; only the outcome moves, forward. A planning round -- the
-- experiment, the bound planner's spec fingerprint, the planning context's
-- fingerprint -- has numbered attempts, and an attempt still INTENDED when the
-- next one begins is closed OUTCOME_UNKNOWN.
--
-- Never stored: an adapter's or SDK's exception, its message, cause, context
-- or traceback. failure_json is a classification xaytune wrote.
CREATE TABLE agent_invocations (
  id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  planner_spec_fingerprint TEXT NOT NULL,
  context_fingerprint TEXT NOT NULL,
  attempt INTEGER NOT NULL CHECK (attempt >= 1),
  request_fingerprint TEXT NOT NULL,
  intent_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN (
    'intended', 'answered', 'completed', 'refused', 'failed', 'outcome_unknown'
  )),
  revision INTEGER NOT NULL CHECK (revision >= 0),
  response_json TEXT,
  failure_json TEXT,
  proposal_json TEXT,
  proposal_fingerprint TEXT,
  created_at TEXT NOT NULL,
  settled_at TEXT,
  UNIQUE (experiment_id, planner_spec_fingerprint, context_fingerprint, attempt),
  CHECK ((status IN ('refused', 'failed')) = (failure_json IS NOT NULL)),
  CHECK ((status IN ('completed', 'refused', 'failed', 'outcome_unknown')) = (settled_at IS NOT NULL)),
  CHECK ((proposal_json IS NULL) = (proposal_fingerprint IS NULL)),
  CHECK (proposal_json IS NULL OR status = 'completed'),
  -- An answered or completed invocation carries its answer; an intended one,
  -- or one whose outcome is unknown, has none. Refused or failed: either.
  CHECK (status NOT IN ('answered', 'completed') OR response_json IS NOT NULL),
  CHECK (status NOT IN ('intended', 'outcome_unknown') OR response_json IS NULL)
);
CREATE INDEX idx_agent_invocations_experiment ON agent_invocations(experiment_id, created_at, id);

CREATE TRIGGER agent_invocations_new_are_intended
BEFORE INSERT ON agent_invocations
WHEN NEW.status != 'intended' OR NEW.revision != 0
BEGIN SELECT RAISE(ABORT, 'an agent invocation is recorded INTENDED at revision 0'); END;

CREATE TRIGGER agent_invocations_attempts_are_consecutive
BEFORE INSERT ON agent_invocations
WHEN NEW.attempt != 1 + COALESCE((
  SELECT MAX(attempt) FROM agent_invocations
  WHERE experiment_id = NEW.experiment_id
    AND planner_spec_fingerprint = NEW.planner_spec_fingerprint
    AND context_fingerprint = NEW.context_fingerprint
), 0)
BEGIN SELECT RAISE(ABORT, 'agent invocation attempts are numbered consecutively per round'); END;

CREATE TRIGGER agent_invocations_one_open_attempt
BEFORE INSERT ON agent_invocations
WHEN EXISTS (
  SELECT 1 FROM agent_invocations
  WHERE experiment_id = NEW.experiment_id
    AND planner_spec_fingerprint = NEW.planner_spec_fingerprint
    AND context_fingerprint = NEW.context_fingerprint
    AND status IN ('intended', 'answered', 'completed', 'refused')
)
BEGIN SELECT RAISE(ABORT, 'a new attempt follows only a failed or unknown one'); END;

CREATE TRIGGER agent_invocations_immutable_intent
BEFORE UPDATE ON agent_invocations
WHEN NEW.id != OLD.id OR NEW.experiment_id != OLD.experiment_id
  OR NEW.planner_spec_fingerprint != OLD.planner_spec_fingerprint
  OR NEW.context_fingerprint != OLD.context_fingerprint OR NEW.attempt != OLD.attempt
  OR NEW.request_fingerprint != OLD.request_fingerprint OR NEW.intent_json != OLD.intent_json
  OR NEW.created_at != OLD.created_at
BEGIN SELECT RAISE(ABORT, 'what an agent invocation asked is immutable'); END;

CREATE TRIGGER agent_invocations_forward_only
BEFORE UPDATE ON agent_invocations
WHEN NEW.revision != OLD.revision + 1 OR NOT (
  (OLD.status = 'intended'
    AND NEW.status IN ('answered', 'refused', 'failed', 'outcome_unknown'))
  -- Never answered -> failed: a recorded answer is a known outcome, and
  -- failed would let the round ask the model again.
  OR (OLD.status = 'answered' AND NEW.status IN ('completed', 'refused'))
)
BEGIN SELECT RAISE(ABORT, 'an agent invocation moves forward along its edges only'); END;

CREATE TRIGGER agent_invocations_answer_is_kept
BEFORE UPDATE ON agent_invocations
WHEN OLD.response_json IS NOT NULL AND NEW.response_json IS NOT OLD.response_json
BEGIN SELECT RAISE(ABORT, 'a recorded answer is never changed'); END;

CREATE TRIGGER agent_invocations_permanent
BEFORE DELETE ON agent_invocations
BEGIN SELECT RAISE(ABORT, 'agent invocations are permanent'); END;
