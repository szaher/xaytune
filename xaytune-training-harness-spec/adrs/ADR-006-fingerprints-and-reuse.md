# ADR-006 — Identity is split across training, execution, evaluation, and checkpoint compatibility

## Status
Proposed — `TrainingSpecFingerprint` is replaced by two fingerprints in ADR-011.

A single training fingerprint cannot describe a run whose training semantics changed
partway through. ADR-011 splits it into `CandidateFingerprint` (what was declared,
including any pre-registered schedule) and `RunRealizationFingerprint` (what actually
happened, including reactive interventions), and adds the reuse mode this ADR is
missing: "do we already have *any* artifact from this candidate?"

## Decision

Use:

- CandidateFingerprint         (ADR-011; replaces TrainingSpecFingerprint)
- RunRealizationFingerprint    (ADR-011)
- ExecutionFingerprint
- EvaluationFingerprint
- CheckpointCompatibilityKey

Do not use one hash for all purposes.

Identical training fingerprints do not automatically suppress reruns.

Reuse is governed by explicit `ReusePolicy` and seed/replicate semantics.

## Rationale

Changing an evaluator must not retrain a model. Training is stochastic. Checkpoint compatibility has different constraints from experiment identity.
