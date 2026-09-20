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

Identical candidate fingerprints do not automatically suppress reruns.

Reuse is governed by explicit `ReusePolicy` and seed/replicate semantics, and asks four
distinct questions (ADR-011):

| Question | Match on |
|---|---|
| Has this hypothesis been explored? | `CandidateFingerprint` |
| Do we have *any* artifact from this candidate? | `CandidateFingerprint`, any terminal realization |
| Do we have *this exact* trajectory's artifact? | `RunRealizationFingerprint` |
| Has this artifact been scored by this evaluator? | artifact digest + `EvaluationFingerprint` |

Seed belongs to the realization, not the candidate: two replicates differing only by
seed are the same candidate run twice.

## Rationale

Changing an evaluator must not retrain a model. Training is stochastic. Checkpoint compatibility has different constraints from experiment identity.
