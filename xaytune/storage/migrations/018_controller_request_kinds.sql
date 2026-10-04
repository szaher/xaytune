-- PR-029 (ADR-004 §1, §3): every mutation a client makes is a mailbox request
-- the daemon carries out, so the mailbox gains one explicit kind per
-- mutation -- cancel, propose-action, approve-action, reject-action -- never
-- a generic command. Each moves PENDING → COMPLETED | FAILED, like attach:
-- it carries the identity of what it records or resolves, so carrying it out
-- again finds the first attempt's work rather than repeating it.
--
-- SQLite cannot alter a CHECK in place, so the table is rebuilt: same
-- columns, same rows, same indexes and the same triggers, with the new kinds
-- and their edges. Nothing references controller_requests, and its own
-- triggers are dropped first so the copy and the drop fire none of them.
DROP TRIGGER controller_requests_new_are_pending;
DROP TRIGGER controller_requests_immutable_request;
DROP TRIGGER controller_requests_forward_only;
DROP TRIGGER controller_requests_accepted_admits;
DROP TRIGGER controller_requests_permanent;

CREATE TABLE controller_requests_018 (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN (
    'submit', 'attach', 'cancel', 'propose-action', 'approve-action', 'reject-action'
  )),
  state TEXT NOT NULL CHECK (state IN ('pending', 'accepted', 'completed', 'failed')),
  revision INTEGER NOT NULL CHECK (revision >= 0),
  experiment_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  payload_digest TEXT NOT NULL,
  error_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  CHECK ((state = 'failed') = (error_json IS NOT NULL))
);
INSERT INTO controller_requests_018 (
  id, kind, state, revision, experiment_id, payload_json, payload_digest, error_json,
  created_at, updated_at
)
SELECT id, kind, state, revision, experiment_id, payload_json, payload_digest, error_json,
  created_at, updated_at
FROM controller_requests;
DROP TABLE controller_requests;
ALTER TABLE controller_requests_018 RENAME TO controller_requests;

CREATE INDEX idx_controller_requests_unfinished
  ON controller_requests(state, created_at, id);
CREATE INDEX idx_controller_requests_experiment
  ON controller_requests(experiment_id);

CREATE TRIGGER controller_requests_new_are_pending
BEFORE INSERT ON controller_requests
WHEN NEW.state != 'pending' OR NEW.revision != 0
BEGIN SELECT RAISE(ABORT, 'a controller request is recorded PENDING at revision 0'); END;

CREATE TRIGGER controller_requests_immutable_request
BEFORE UPDATE ON controller_requests
WHEN NEW.id != OLD.id OR NEW.kind != OLD.kind OR NEW.experiment_id != OLD.experiment_id
  OR NEW.payload_json != OLD.payload_json OR NEW.payload_digest != OLD.payload_digest
  OR NEW.created_at != OLD.created_at
BEGIN SELECT RAISE(ABORT, 'what a controller request asks is immutable'); END;

CREATE TRIGGER controller_requests_forward_only
BEFORE UPDATE ON controller_requests
WHEN NEW.revision != OLD.revision + 1 OR NOT (
  (OLD.kind = 'submit' AND OLD.state = 'pending' AND NEW.state IN ('accepted', 'failed'))
  OR (OLD.kind = 'submit' AND OLD.state = 'accepted' AND NEW.state = 'completed')
  OR (OLD.kind != 'submit' AND OLD.state = 'pending' AND NEW.state IN ('completed', 'failed'))
)
BEGIN SELECT RAISE(ABORT, 'controller request state moves forward along its kind''s edges'); END;

CREATE TRIGGER controller_requests_accepted_admits
BEFORE UPDATE ON controller_requests
WHEN NEW.state = 'accepted'
  AND NOT EXISTS (SELECT 1 FROM experiments e WHERE e.id = NEW.experiment_id)
BEGIN SELECT RAISE(ABORT, 'an ACCEPTED submission names the experiment it admitted'); END;

CREATE TRIGGER controller_requests_permanent
BEFORE DELETE ON controller_requests
BEGIN SELECT RAISE(ABORT, 'controller requests are permanent'); END;

-- Where the daemon's controller last came to rest on an experiment: nothing
-- running it could advance, observed at the experiment's event `sequence`.
-- A client's ExperimentHandle.wait() reads it, because the record alone
-- cannot tell a resting experiment from one between two steps -- a trained
-- run whose evaluation cycle is about to begin looks settled for a moment.
-- At rest means this row's sequence is still the experiment's latest event.
-- `escalation_json` says why the controller stopped short, if it did
-- (ReconciliationEscalatedError, ControllerNotRunningError). One row per
-- experiment, replaced as the controller comes to rest again.
CREATE TABLE controller_rests (
  experiment_id TEXT PRIMARY KEY REFERENCES experiments(id),
  controller_id TEXT NOT NULL,
  sequence INTEGER NOT NULL CHECK (sequence >= 0),
  escalation_json TEXT,
  recorded_at TEXT NOT NULL
);
