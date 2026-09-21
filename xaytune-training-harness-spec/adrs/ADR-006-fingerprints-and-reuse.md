# ADR-006 — Identity is split across training, execution, evaluation, and checkpoint compatibility

## Status
Accepted — 2026-09-21, for the identity model. **Reuse policy is deferred to
ADR-017 (`Proposed`)** and still gates band G.

Split because half-accepted was unusable as a gate: Phase 2 implements the
fingerprints from this ADR while the planner's reuse decisions genuinely are
undecided. Those are separable, so they are now separate documents rather than
one document in two states. The identity half is additionally superseded in
detail by ADR-011, which replaces `TrainingSpecFingerprint` with the layered
model below.

`TrainingSpecFingerprint` is replaced by the layered identity model in ADR-011
(`CandidateFingerprint`, `RunHistoryFingerprint`, `ArtifactLineageFingerprint`,
`ExecutionFingerprint`,
`EvaluationFingerprint`, `CheckpointCompatibilityKey`). **The reuse-policy half
of this ADR is still open** and gates band G — planner reuse decisions.

A single training fingerprint cannot describe a run whose training semantics changed
partway through. ADR-011 splits it into `CandidateFingerprint` (what was declared,
including any pre-registered schedule) and two run-level identities —
`RunHistoryFingerprint` (everything that happened, including rolled-back work) and
`ArtifactLineageFingerprint` (only the trajectory the artifact descends from) — and
adds the reuse mode this ADR is missing: "do we already have *any* artifact from this candidate?"

## Decision

Use:

- CandidateFingerprint         (ADR-011; replaces TrainingSpecFingerprint)
- RunHistoryFingerprint        (ADR-011; audit)
- ArtifactLineageFingerprint   (ADR-011; trajectory reuse)
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
| Do we have the artifact from this exact training trajectory? | `ArtifactLineageFingerprint` |
| What did this run actually do, rollbacks included? | `RunHistoryFingerprint` |
| Has this artifact been scored by this evaluator? | artifact digest + `EvaluationFingerprint` |

Seed belongs to the realization, not the candidate: two replicates differing only by
seed are the same candidate run twice.

## Canonical encoding

A fingerprint is only as stable as the bytes it is computed over. Canonical *values* are
not canonical *encoding*, and three traps follow from that:

- Two equal mappings can differ in insertion order, so a naive `model_dump_json()` may
  emit their keys in different orders.
- Python's built-in `hash()` is randomized per process, so it cannot appear anywhere in a
  persisted fingerprint.
- Python equality is untyped: `True == 1` and `1 == 1.0`, so `{"x": True}` compares equal
  to `{"x": 1}` while their JSON forms — `{"x":true}` and `{"x":1}` — should fingerprint
  differently.

Every fingerprint in this ADR is therefore computed by a canonical **typed** encoder:

```text
normalize -> sort mapping keys -> encode values with their type distinguished
          -> UTF-8 bytes -> stable digest
```

never from `hash()`, Python equality, or an unsorted JSON dump. The same encoder produces
the `request_digest` in ADR-013, so a resubmitted request compares equal to itself across
processes.

## Rationale

Changing an evaluator must not retrain a model. Training is stochastic. Checkpoint compatibility has different constraints from experiment identity.
