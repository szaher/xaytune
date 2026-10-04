-- PR-028 (ADR-004 §8): durable controller ownership. One lease per database,
-- held by the daemon that may write controller state now. The epoch is the
-- fencing generation: it moves only forward, by one, on each new owner, so
-- an old generation can never become current again. The row is never
-- deleted -- a released lease expires in place, keeping its epoch.
CREATE TABLE controller_leases (
  singleton_key INTEGER PRIMARY KEY CHECK (singleton_key = 1),
  controller_id TEXT NOT NULL,
  epoch INTEGER NOT NULL CHECK (epoch >= 1),
  heartbeat_at TEXT NOT NULL,
  lease_expires_at TEXT NOT NULL
);

CREATE TRIGGER controller_leases_start_at_epoch_one
BEFORE INSERT ON controller_leases
WHEN NEW.epoch != 1
BEGIN SELECT RAISE(ABORT, 'the first controller lease is epoch 1'); END;

CREATE TRIGGER controller_leases_epoch_forward
BEFORE UPDATE ON controller_leases
WHEN NOT (
  (NEW.epoch = OLD.epoch AND NEW.controller_id = OLD.controller_id)
  OR NEW.epoch = OLD.epoch + 1
)
BEGIN SELECT RAISE(ABORT, 'a controller lease keeps its epoch or moves to the next one'); END;

CREATE TRIGGER controller_leases_permanent
BEFORE DELETE ON controller_leases
BEGIN SELECT RAISE(ABORT, 'the controller lease is never deleted'); END;
