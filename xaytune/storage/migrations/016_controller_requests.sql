-- PR-027 (ADR-004 §2-§4): the local daemon's durable request mailbox.
-- A client commits a request; the daemon polls for it. The id is the
-- client's idempotency key, and what a request asks is immutable: only its
-- state moves, forward, along the edges its kind allows. An ACCEPTED
-- submission commits with the experiment it admits, so it must name one
-- that exists.
CREATE TABLE controller_requests (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('submit', 'attach')),
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
  OR (OLD.kind = 'attach' AND OLD.state = 'pending' AND NEW.state IN ('completed', 'failed'))
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
