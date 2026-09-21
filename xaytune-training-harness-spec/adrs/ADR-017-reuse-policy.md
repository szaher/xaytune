# ADR-017 — Reuse policy

## Status
Proposed — 2026-09-21.

Split out of ADR-006, whose identity model is now `Accepted` while this half is
genuinely undecided. Gates band G — the planner's reuse decisions. It does not
gate band B, because nothing here changes a persisted column: a reuse policy is
a query over fingerprints that already exist.

## Context

ADR-006 established the fingerprints and ADR-011 layered them. Neither says when
a planner is *allowed* to substitute an existing artifact for work it was about
to schedule, and that question does not have an obvious answer.

Keeping it inside ADR-006 made that ADR permanently half-accepted, which is
unusable when status is being used as an implementation gate: Phase 2 implements
the identity model while band G waits on the reuse rules, and one status cannot
describe both.

The questions that are actually open:

- **Is a matching fingerprint sufficient, or only necessary?** Training is
  stochastic. Two runs of one candidate at one seed can still differ through
  nondeterministic kernels, and ADR-012 is explicit that `EXACT` data resume does
  not imply bitwise-identical numerics.
- **What does a replicate request mean when a match exists?** Asking for a third
  seed must never be satisfied by returning the first, for the same reason a
  stochastic evaluation must not memoize a sample (ADR-015 §3). The failure is
  silent and statistical.
- **Does reuse cross experiments?** An artifact from another experiment may be
  identical by fingerprint and unacceptable by governance, provenance or budget
  attribution.
- **Does a rolled-back trajectory count?** ADR-011 splits `RunHistoryFingerprint`
  from `ArtifactLineageFingerprint` precisely because those answer differently.
  Reuse asks the lineage question; audit asks the history one.
- **How does reuse interact with budget?** A reused artifact consumed GPU-hours
  that some ledger already paid for. Whether the reusing experiment is charged
  is a policy decision, not an accounting detail.
- **When is a match invalidated?** A dataset revision that moves, a container
  digest that no longer resolves, an evaluator whose provider changed underneath
  a fixed model name.

## Decision

**Not yet decided.** This ADR exists to hold the question, not to answer it
prematurely, and to stop ADR-006 carrying two statuses at once.

What is already settled and constrains any answer:

- The fingerprints themselves — ADR-006 for the identity model and the canonical
  typed encoder, ADR-011 for the four layers and the history/lineage split.
- In-flight runs are never reuse candidates; fingerprints are provisional until
  terminal (ADR-011).
- Evaluation reuse **is** decided, in ADR-015 §3, and keyed on evaluator
  determinism. This ADR covers training artifacts only, and should not
  contradict it: a `STOCHASTIC` evaluator's result is a sample, and by the same
  argument a replicate of a stochastic training run is a sample.

## Consequences

- Band G cannot start until this is accepted. Nothing earlier is blocked.
- Until then, the planner must schedule work rather than reuse it. Doing the work
  twice is wasteful; silently substituting a different trajectory is wrong, and
  the second is not recoverable from the provenance record.
