# ADR-017 — Reuse policy

## Status
Accepted — 2026-10-01

Proposed on 2026-09-21, when it was split out of ADR-006 so that ADR-006's
identity model could be `Accepted` while this half was still undecided. It gates
band G, the planner. Accepted with the conservative v1 decision below: training
artifact reuse is off. That answers every open question by refusing to
substitute, which is safe without a reuse implementation. A real reuse policy
can come later, under its own ADR.

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

**v1: training artifact reuse is disabled.** The planner and controller schedule
the work they are asked to do. They never substitute an existing training
artifact for it.

1. **Fingerprints are necessary evidence, never sufficient authority.** A
   matching `CandidateFingerprint`, `ExecutionFingerprint` or
   `ArtifactLineageFingerprint` does not authorize skipping execution. Training
   is stochastic: ADR-012's `EXACT` resume is a data-position guarantee, not
   bitwise numerics.
2. **No substitution.** No planner, including PR-024's `RuleBasedPlanner`, and
   no controller path may satisfy a requested Run with an existing artifact.
   Every requested Run executes as new work.
3. **Replicates always execute.** A request for another seed, or another
   replicate at the same seed, is a request for a new sample. Satisfying it
   from an existing run is the silent statistical failure that ADR-015 §3 rules
   out for stochastic evaluation.
4. **No cross-experiment training reuse.** An artifact from another experiment
   is never a candidate. Matching fingerprints say nothing about the other
   experiment's governance, provenance or budget attribution.
5. **Rolled-back trajectories are never reused as retained results.** ADR-011's
   history/lineage split stands. A trajectory a rollback abandoned is audit
   history, not a result.
6. **Evaluation reuse stays governed by ADR-015 §3.** That rule is keyed on
   evaluator determinism. This ADR neither widens nor narrows it.
7. **Matches may be shown, never acted on silently.** Fingerprints and
   provenance may expose a possible match for a person or tool to inspect. A
   match cannot skip, shorten or replace execution.
8. **Future reuse needs an explicit `ReusePolicy`.** Enabling any training
   reuse requires a new or superseding ADR: a versioned, durable `ReusePolicy`
   (ADR-016), opt-in per experiment, plus a separate implementation. That ADR
   must answer budget attribution and match invalidation (dataset revisions,
   container digests, evaluator providers). v1 does not answer them because v1
   never reuses.

Already settled before this decision, and unchanged by it:

- The fingerprints themselves: ADR-006 defines the identity model and the
  canonical typed encoder; ADR-011 defines the four layers and the
  history/lineage split.
- In-flight runs are never reuse candidates: their fingerprints stay
  provisional until the run is terminal (ADR-011).

## Consequences

- Band G is unblocked. PR-024 starts with no reuse decisions to make: a planner
  proposes work, and accepted work runs.
- Budget accounting stays simple. Every executed run is charged to the
  experiment that ran it, and no ledger has to attribute borrowed GPU-hours.
- Duplicate work costs more compute. That is the accepted price: doing work
  twice is wasteful, but silently substituting a different trajectory is wrong,
  and the provenance record cannot recover from it.
- Reproducibility evidence is preserved. Two runs with matching fingerprints
  stay two observations, so their differences remain measurable.
