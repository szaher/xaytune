-- Migration 005 — the durable telemetry cursor (ADR-014 §4, PR-012a).
--
-- telemetry_generation arrived in 001; this adds the sequence beside it, so the
-- pair is the StreamCursor a controller resumes watch() from after a restart:
-- "the last position whose consequences are recorded".
--
-- Advanced only in the same commit as the transition or artifact the event
-- caused. A cursor written on its own could run ahead of the effect it
-- describes -- a crash between the two would skip an event whose effect was
-- never recorded -- and one written behind it would merely replay work that is
-- idempotent. So the cursor never claims more than the record holds.
--
-- -1 means nothing has been recorded yet; sequences themselves start at 0.

ALTER TABLE run_attempts
  ADD COLUMN telemetry_sequence INTEGER NOT NULL DEFAULT -1 CHECK (telemetry_sequence >= -1);
