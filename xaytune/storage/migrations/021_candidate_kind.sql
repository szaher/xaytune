-- ADR-018 D10 (ADR-019 M6, step 2): a node can hold a candidate of any kind.
--
-- A node's candidate lives in its payload, as a CandidateSpecSnapshot. At
-- spec_version "1" it is a CandidateSpec (every node before this migration,
-- and every training node after it, byte for byte). At "2" it is an
-- ExperimentCandidate envelope, which is how a node holds a harness.
--
-- This column is what the kind is queried by: training-only work (compiling,
-- planning, branching) can tell a harness node from a training one without
-- parsing every payload. It is added, not rebuilt into the table: every
-- existing node is a training node, which is the default, and nothing
-- references experiment_nodes through it.
--
-- Integrity is layered, and each layer does one job.
--
-- These triggers, on every write path and not only the repository's, keep the
-- stored columns consistent with what the payload declares:
--   - the snapshot's spec_version is exactly the text "1" or "2", and its
--     shape is that version's ("1" holds candidate and no envelope, "2" holds
--     an envelope naming its payload's kind, and no candidate). Anything else -- a missing, unknown or
--     non-text version, or a mixed shape -- is refused, on insert and on
--     update, rather than read as training;
--   - on insert, candidate_kind is the kind that version declares ("1" is
--     training; "2" is its envelope's payload.kind), and
--     candidate_fingerprint is the fingerprint the payload declares;
--   - a node's candidate is immutable (Rule 5), so its kind and its identity
--     never change: an update may not change either column, nor give the
--     payload another declared kind or another declared fingerprint.
-- They compare declared fields, not bytes: a status update rewrites the
-- payload, and re-serializes a historical candidate with fields its schema has
-- since gained, which is the same candidate. A payload written before PR-007's
-- rename declares its fingerprint as training_fingerprint; ExperimentNode
-- still reads that alias, so the triggers do too.
--
-- What SQL cannot do is compute a fingerprint: it checks that the columns
-- agree with the fingerprint the payload declares, not that the declaration
-- is the candidate's actual scientific identity. That is the repository's
-- job, on every read (decode_node_row): it parses the snapshot, recomputes
-- the candidate's identities, and refuses a row whose columns are not among
-- them -- which also catches a row written around the triggers.
--
-- Release order: this migration ships only in a release AFTER the first one
-- with the downgrade guard (ADR-019 M6 step 1; 1.0.0a2). A build with the
-- guard and without 021 refuses a database 021 migrated, outright. A build
-- older than the guard (1.0.0a1) would open it, and a version 2 harness
-- snapshot is a payload it cannot read; the guard-bearing release coming
-- first is what keeps every supported deployment from meeting one.
--
-- Within that order: a build without this migration ignores the column and
-- still writes training nodes correctly (the default and the triggers agree),
-- and no public path writes a harness node yet (ADR-019 M6 step 5 adds
-- submission).
ALTER TABLE experiment_nodes
  ADD COLUMN candidate_kind TEXT NOT NULL DEFAULT 'training'
  CHECK (candidate_kind IN ('training', 'agent-harness'));

-- A snapshot version this schema knows, in that version's shape. Unknown is
-- refused here, never defaulted: the kind triggers below derive a kind only
-- from a snapshot of a known version in its shape (anything else derives
-- none, and is left to these two), so each refusal has one cause and says
-- which.
CREATE TRIGGER experiment_node_candidate_snapshot_is_a_known_version
BEFORE INSERT ON experiment_nodes
FOR EACH ROW
WHEN NOT COALESCE(
  json_type(NEW.payload_json, '$.candidate.spec_version') = 'text' AND (
    (json_extract(NEW.payload_json, '$.candidate.spec_version') = '1'
      AND json_type(NEW.payload_json, '$.candidate.candidate') = 'object'
      AND json_type(NEW.payload_json, '$.candidate.envelope') IS NULL)
    OR (json_extract(NEW.payload_json, '$.candidate.spec_version') = '2'
      AND json_type(NEW.payload_json, '$.candidate.envelope.payload.kind') = 'text'
      AND json_type(NEW.payload_json, '$.candidate.candidate') IS NULL)
  ),
  0
)
BEGIN
  SELECT RAISE(ABORT, 'an experiment node''s candidate snapshot is not a version this schema knows (spec_version "1" or "2", in its shape)');
END;

CREATE TRIGGER experiment_node_payload_keeps_a_known_version
BEFORE UPDATE OF payload_json ON experiment_nodes
FOR EACH ROW
WHEN NOT COALESCE(
  json_type(NEW.payload_json, '$.candidate.spec_version') = 'text' AND (
    (json_extract(NEW.payload_json, '$.candidate.spec_version') = '1'
      AND json_type(NEW.payload_json, '$.candidate.candidate') = 'object'
      AND json_type(NEW.payload_json, '$.candidate.envelope') IS NULL)
    OR (json_extract(NEW.payload_json, '$.candidate.spec_version') = '2'
      AND json_type(NEW.payload_json, '$.candidate.envelope.payload.kind') = 'text'
      AND json_type(NEW.payload_json, '$.candidate.candidate') IS NULL)
  ),
  0
)
BEGIN
  SELECT RAISE(ABORT, 'an experiment node''s candidate snapshot is not a version this schema knows (spec_version "1" or "2", in its shape)');
END;

CREATE TRIGGER experiment_node_candidate_kind_is_its_payloads
BEFORE INSERT ON experiment_nodes
FOR EACH ROW
WHEN NEW.candidate_kind IS NOT COALESCE((
  CASE WHEN json_type(NEW.payload_json, '$.candidate.spec_version') = 'text' AND (
    (json_extract(NEW.payload_json, '$.candidate.spec_version') = '1'
      AND json_type(NEW.payload_json, '$.candidate.candidate') = 'object'
      AND json_type(NEW.payload_json, '$.candidate.envelope') IS NULL)
    OR (json_extract(NEW.payload_json, '$.candidate.spec_version') = '2'
      AND json_type(NEW.payload_json, '$.candidate.envelope.payload.kind') = 'text'
      AND json_type(NEW.payload_json, '$.candidate.candidate') IS NULL)
  ) THEN
    CASE json_extract(NEW.payload_json, '$.candidate.spec_version')
      WHEN '1' THEN 'training'
      WHEN '2' THEN json_extract(NEW.payload_json, '$.candidate.envelope.payload.kind')
    END
  END
), NEW.candidate_kind)
BEGIN
  SELECT RAISE(ABORT, 'experiment_nodes.candidate_kind is not the kind its candidate holds');
END;

CREATE TRIGGER experiment_node_candidate_fingerprint_is_its_payloads
BEFORE INSERT ON experiment_nodes
FOR EACH ROW
WHEN NEW.candidate_fingerprint IS NOT COALESCE(
    json_extract(NEW.payload_json, '$.candidate_fingerprint'),
    json_extract(NEW.payload_json, '$.training_fingerprint')
  )
BEGIN
  SELECT RAISE(ABORT, 'experiment_nodes.candidate_fingerprint is not its payload''s');
END;

CREATE TRIGGER experiment_node_candidate_kind_is_permanent
BEFORE UPDATE OF candidate_kind ON experiment_nodes
FOR EACH ROW
WHEN NEW.candidate_kind IS NOT OLD.candidate_kind
BEGIN
  SELECT RAISE(ABORT, 'experiment_nodes.candidate_kind never changes');
END;

CREATE TRIGGER experiment_node_candidate_identity_is_permanent
BEFORE UPDATE OF candidate_fingerprint ON experiment_nodes
FOR EACH ROW
WHEN NEW.candidate_fingerprint IS NOT OLD.candidate_fingerprint
BEGIN
  SELECT RAISE(ABORT, 'experiment_nodes.candidate_fingerprint never changes');
END;

CREATE TRIGGER experiment_node_payload_keeps_its_candidate
BEFORE UPDATE OF payload_json ON experiment_nodes
FOR EACH ROW
WHEN COALESCE(
    json_extract(NEW.payload_json, '$.candidate_fingerprint'),
    json_extract(NEW.payload_json, '$.training_fingerprint')
  ) IS NOT OLD.candidate_fingerprint
  OR OLD.candidate_kind IS NOT COALESCE((
    CASE WHEN json_type(NEW.payload_json, '$.candidate.spec_version') = 'text' AND (
      (json_extract(NEW.payload_json, '$.candidate.spec_version') = '1'
        AND json_type(NEW.payload_json, '$.candidate.candidate') = 'object'
        AND json_type(NEW.payload_json, '$.candidate.envelope') IS NULL)
      OR (json_extract(NEW.payload_json, '$.candidate.spec_version') = '2'
        AND json_type(NEW.payload_json, '$.candidate.envelope.payload.kind') = 'text'
        AND json_type(NEW.payload_json, '$.candidate.candidate') IS NULL)
    ) THEN
      CASE json_extract(NEW.payload_json, '$.candidate.spec_version')
        WHEN '1' THEN 'training'
        WHEN '2' THEN json_extract(NEW.payload_json, '$.candidate.envelope.payload.kind')
      END
    END
  ), OLD.candidate_kind)
BEGIN
  SELECT RAISE(ABORT, 'an experiment node''s payload cannot change its candidate''s kind or identity');
END;
