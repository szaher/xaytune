# ADR-006 — Identity is split across training, execution, evaluation, and checkpoint compatibility

## Status
Proposed

## Decision

Use:

- TrainingSpecFingerprint
- ExecutionFingerprint
- EvaluationFingerprint
- CheckpointCompatibilityKey

Do not use one hash for all purposes.

Identical training fingerprints do not automatically suppress reruns.

Reuse is governed by explicit `ReusePolicy` and seed/replicate semantics.

## Rationale

Changing an evaluator must not retrain a model. Training is stochastic. Checkpoint compatibility has different constraints from experiment identity.
