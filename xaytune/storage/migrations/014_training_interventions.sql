-- PR-021: numerical recovery and TrainingIntervention lineage (ADR-011).
--
--   numerical_recovery_action_bindings   ChangeLearningRate Action -> one fresh plan revision
--   training_interventions               the scientific decision an authorized Action produced
--   intervention_applications            each confirmed effect, ordered by event sequence
--
-- All three are append-only. The OOM recovery_action_bindings v1alpha1 table is
-- untouched apart from a guard that one plan revision binds one Action family.

CREATE TABLE numerical_recovery_action_bindings (
  action_id TEXT PRIMARY KEY REFERENCES actions(id),
  episode_id TEXT NOT NULL REFERENCES recovery_episodes(id),
  plan_id TEXT NOT NULL UNIQUE REFERENCES recovery_plans(id),
  plan_sequence INTEGER NOT NULL CHECK (plan_sequence >= 1),
  proposal_fingerprint TEXT NOT NULL,
  input_fingerprint TEXT NOT NULL,
  source_execution_state_fingerprint TEXT NOT NULL,
  trigger_incident_id TEXT NOT NULL REFERENCES incidents(id),
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX idx_numerical_recovery_action_bindings_episode
  ON numerical_recovery_action_bindings(episode_id, plan_sequence);

CREATE TRIGGER numerical_recovery_action_binding_authority
BEFORE INSERT ON numerical_recovery_action_bindings
WHEN NOT EXISTS (
  SELECT 1 FROM recovery_episodes e
  JOIN recovery_effective_plans p ON p.episode_id = e.id
  JOIN recovery_episode_closure c ON c.id = e.id
  JOIN actions a ON a.id = NEW.action_id
  JOIN recovery_episode_incidents trigger_member
    ON trigger_member.episode_id = e.id AND trigger_member.incident_id = NEW.trigger_incident_id
    AND trigger_member.disposition = 'ACCEPTED_FOR_DECISION'
  WHERE e.id = NEW.episode_id AND p.id = NEW.plan_id
    AND p.sequence = NEW.plan_sequence AND c.successor_exists = 0
    AND p.accepted_through_sequence = (
      SELECT MAX(m.membership_sequence) FROM recovery_episode_incidents m
      WHERE m.episode_id = e.id AND m.disposition = 'ACCEPTED_FOR_DECISION'
    )
    AND NOT EXISTS (
      SELECT 1 FROM incidents i WHERE i.target_kind = e.target_kind
        AND i.target_id = e.target_id AND NOT EXISTS (
          SELECT 1 FROM recovery_episode_incidents m WHERE m.incident_id = i.id
        )
    )
    AND e.target_kind = 'training-attempt'
    AND a.experiment_id = e.experiment_id AND a.target_kind = 'run'
    AND a.target_id = e.run_id AND a.type = 'change-learning-rate'
)
BEGIN SELECT RAISE(ABORT, 'numerical Action binding lacks current plan or Action ownership'); END;

CREATE TRIGGER numerical_recovery_binding_excludes_oom
BEFORE INSERT ON numerical_recovery_action_bindings
WHEN EXISTS (SELECT 1 FROM recovery_action_bindings b WHERE b.plan_id = NEW.plan_id)
BEGIN SELECT RAISE(ABORT, 'recovery plan revision already binds an OOM Action'); END;

CREATE TRIGGER oom_recovery_binding_excludes_numerical
BEFORE INSERT ON recovery_action_bindings
WHEN EXISTS (SELECT 1 FROM numerical_recovery_action_bindings b WHERE b.plan_id = NEW.plan_id)
BEGIN SELECT RAISE(ABORT, 'recovery plan revision already binds a numerical Action'); END;

CREATE TRIGGER numerical_recovery_action_bindings_immutable
BEFORE UPDATE ON numerical_recovery_action_bindings
BEGIN SELECT RAISE(ABORT, 'numerical recovery Action bindings are append-only'); END;
CREATE TRIGGER numerical_recovery_action_bindings_permanent
BEFORE DELETE ON numerical_recovery_action_bindings
BEGIN SELECT RAISE(ABORT, 'numerical recovery Action bindings are append-only'); END;

-- The decision. One per Action: the Action is the governance, this is its
-- durable scientific outcome. No status column -- effect lives in applications.
CREATE TABLE training_interventions (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id),
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  action_id TEXT NOT NULL UNIQUE REFERENCES actions(id),
  origin TEXT NOT NULL
    CHECK (origin IN ('scheduled', 'reactive-agent', 'reactive-human', 'reactive-policy')),
  trigger_type TEXT NOT NULL
    CHECK (trigger_type IN (
      'step', 'optimizer-step', 'tokens', 'metric', 'incident', 'manual', 'policy'
    )),
  trigger_incident_id TEXT REFERENCES incidents(id),
  replay_policy TEXT NOT NULL
    CHECK (replay_policy IN ('reapply-after-rollback', 'apply-once', 'rearm-trigger')),
  schedule_ref TEXT,
  derived_from TEXT REFERENCES training_interventions(id),
  mutation_type TEXT NOT NULL CHECK (mutation_type IN ('learning-rate')),
  -- Sequence of the TrainingInterventionRecorded event committed with this row.
  recorded_event_sequence INTEGER NOT NULL UNIQUE CHECK (recorded_event_sequence >= 1),
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  -- ADR-011: only a reconstructible, non-monotone trigger can be re-armed.
  CHECK (replay_policy <> 'rearm-trigger' OR trigger_type = 'metric'),
  CHECK ((origin = 'scheduled') = (schedule_ref IS NOT NULL)),
  CHECK (trigger_type = 'incident' OR trigger_incident_id IS NULL)
);
CREATE INDEX idx_training_interventions_run
  ON training_interventions(run_id, recorded_event_sequence);

-- Only an authorized scientific-intervention Action on this same active Run
-- may produce an intervention: VALIDATED + ALLOW, or APPROVED + REQUIRE_APPROVAL.
CREATE TRIGGER training_intervention_authority
BEFORE INSERT ON training_interventions
WHEN NOT EXISTS (
  SELECT 1 FROM actions a
  JOIN runs r ON r.id = NEW.run_id
  JOIN policy_decisions d ON d.action_id = a.id
  WHERE a.id = NEW.action_id
    AND a.type = 'change-learning-rate' AND NEW.mutation_type = 'learning-rate'
    AND a.target_kind = 'run' AND a.target_id = NEW.run_id
    AND a.experiment_id = r.experiment_id AND NEW.experiment_id = r.experiment_id
    AND r.status = 'active'
    AND (
      (a.status = 'validated' AND d.verdict = 'allow')
      OR (a.status = 'approved' AND d.verdict = 'require_approval')
    )
)
BEGIN SELECT RAISE(ABORT, 'intervention lacks an authorized scientific Action on its Run'); END;

CREATE TRIGGER training_intervention_event
BEFORE INSERT ON training_interventions
WHEN NOT EXISTS (
  SELECT 1 FROM events ev
  WHERE ev.sequence = NEW.recorded_event_sequence
    AND ev.event_type = 'TrainingInterventionRecorded'
    AND ev.aggregate_type = 'Run' AND ev.aggregate_id = NEW.run_id
    AND json_extract(ev.payload_json, '$.intervention_id') = NEW.id
)
BEGIN SELECT RAISE(ABORT, 'intervention lacks its recorded event'); END;

-- Policy origin exists only as a numerical recovery decision in v1, and a
-- numerically bound Action yields only that decision, while its plan is fresh.
CREATE TRIGGER training_intervention_policy_origin
BEFORE INSERT ON training_interventions
WHEN (
  NEW.origin = 'reactive-policy'
  OR EXISTS (SELECT 1 FROM numerical_recovery_action_bindings b WHERE b.action_id = NEW.action_id)
) AND NOT EXISTS (
  SELECT 1 FROM numerical_recovery_action_bindings b
  JOIN recovery_episodes e ON e.id = b.episode_id
  JOIN recovery_effective_plans p ON p.id = b.plan_id AND p.episode_id = e.id
  JOIN recovery_episode_closure c ON c.id = e.id
  WHERE b.action_id = NEW.action_id
    AND NEW.origin = 'reactive-policy' AND NEW.trigger_type = 'incident'
    AND NEW.trigger_incident_id = b.trigger_incident_id
    AND NEW.replay_policy = json_extract(b.payload_json, '$.proposal.replay_policy')
    AND e.run_id = NEW.run_id AND c.successor_exists = 0
    AND p.accepted_through_sequence = (
      SELECT MAX(m.membership_sequence) FROM recovery_episode_incidents m
      WHERE m.episode_id = e.id AND m.disposition = 'ACCEPTED_FOR_DECISION'
    )
    AND NOT EXISTS (
      SELECT 1 FROM incidents i WHERE i.target_kind = e.target_kind
        AND i.target_id = e.target_id AND NOT EXISTS (
          SELECT 1 FROM recovery_episode_incidents m WHERE m.incident_id = i.id
        )
    )
)
BEGIN SELECT RAISE(ABORT, 'policy-origin intervention lacks a fresh numerical decision'); END;

CREATE TRIGGER training_interventions_immutable
BEFORE UPDATE ON training_interventions
BEGIN SELECT RAISE(ABORT, 'training interventions are append-only'); END;
CREATE TRIGGER training_interventions_permanent
BEFORE DELETE ON training_interventions
BEGIN SELECT RAISE(ABORT, 'training interventions are append-only'); END;

-- Each confirmed effect. event_sequence is the InterventionApplied event's
-- sequence: the canonical order. TrainingPosition is payload, never an order.
CREATE TABLE intervention_applications (
  id TEXT PRIMARY KEY,
  intervention_id TEXT NOT NULL REFERENCES training_interventions(id),
  attempt_id TEXT NOT NULL REFERENCES run_attempts(id),
  run_id TEXT NOT NULL REFERENCES runs(id),
  event_sequence INTEGER NOT NULL UNIQUE CHECK (event_sequence >= 1),
  checkpoint_id TEXT REFERENCES checkpoints(id),
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX idx_intervention_applications_run
  ON intervention_applications(run_id, event_sequence);
CREATE INDEX idx_intervention_applications_intervention
  ON intervention_applications(intervention_id, event_sequence);

CREATE TRIGGER intervention_application_ownership
BEFORE INSERT ON intervention_applications
WHEN NOT EXISTS (
  SELECT 1 FROM training_interventions t
  JOIN run_attempts a ON a.id = NEW.attempt_id
  JOIN events ev ON ev.sequence = NEW.event_sequence
  WHERE t.id = NEW.intervention_id
    AND t.run_id = NEW.run_id AND a.run_id = NEW.run_id
    AND NEW.event_sequence > t.recorded_event_sequence
    AND ev.event_type = 'InterventionApplied'
    AND ev.aggregate_type = 'RunAttempt' AND ev.aggregate_id = NEW.attempt_id
    AND json_extract(ev.payload_json, '$.application_id') = NEW.id
    -- The ancestor is the applying attempt's own restore checkpoint, not any
    -- checkpoint of the run: a caller cannot name an unrelated one.
    AND NEW.checkpoint_id IS json_extract(a.payload_json, '$.checkpoint_ref.id')
    AND json_extract(NEW.payload_json, '$.checkpoint_ancestor.id')
      IS json_extract(a.payload_json, '$.checkpoint_ref.id')
    AND (NEW.checkpoint_id IS NULL OR EXISTS (
      SELECT 1 FROM checkpoints ck WHERE ck.id = NEW.checkpoint_id AND ck.run_id = NEW.run_id
    ))
)
BEGIN SELECT RAISE(ABORT, 'application lacks its intervention, attempt, run, event or ancestor'); END;

CREATE TRIGGER intervention_applications_immutable
BEFORE UPDATE ON intervention_applications
BEGIN SELECT RAISE(ABORT, 'intervention applications are append-only'); END;
CREATE TRIGGER intervention_applications_permanent
BEFORE DELETE ON intervention_applications
BEGIN SELECT RAISE(ABORT, 'intervention applications are append-only'); END;
