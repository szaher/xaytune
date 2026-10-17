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
-- A trigger keeps the column honest: it must be the kind the payload holds,
-- and it never changes. A node's candidate is immutable (Rule 5); so is its
-- kind.
--
-- Compatibility: a build without this migration ignores the column, and no
-- public path writes a harness node yet (ADR-019 M6 step 5 adds submission).
-- A build that has the downgrade guard (ADR-019 M6 step 1) and not this
-- migration refuses the database outright.
ALTER TABLE experiment_nodes
  ADD COLUMN candidate_kind TEXT NOT NULL DEFAULT 'training'
  CHECK (candidate_kind IN ('training', 'agent-harness'));

CREATE TRIGGER experiment_node_candidate_kind_is_its_payloads
BEFORE INSERT ON experiment_nodes
FOR EACH ROW
WHEN NEW.candidate_kind IS NOT (
  CASE json_extract(NEW.payload_json, '$.candidate.spec_version')
    WHEN '2' THEN json_extract(NEW.payload_json, '$.candidate.envelope.payload.kind')
    ELSE 'training'
  END
)
BEGIN
  SELECT RAISE(ABORT, 'experiment_nodes.candidate_kind is not the kind its candidate holds');
END;

CREATE TRIGGER experiment_node_candidate_kind_is_permanent
BEFORE UPDATE OF candidate_kind ON experiment_nodes
FOR EACH ROW
WHEN NEW.candidate_kind IS NOT OLD.candidate_kind
BEGIN
  SELECT RAISE(ABORT, 'experiment_nodes.candidate_kind never changes');
END;
