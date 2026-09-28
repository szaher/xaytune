# ADR-018 — First-class agent harness candidates

## Status

Proposed — 2026-09-28.

Gates the future H02–H12 harness track. It does not gate or reorder PR-019–021
recovery or PR-024–026 planner/branching/adaptive MVP work. This is a decision
proposal for architectural review, not acceptance or implementation authority.

## Context

Today's `CandidateSpec` describes a model-training scientific proposition:
model, data, training, optional reward, environment and schedule. Harness-only
experiments instead compare how a fixed model is operated: prompts, tools,
context, memory, delegation, middleware, routing and stopping. Treating these
as training requires artificial fields and encourages generic planners to
depend on training-specific semantics.

Agent training changes weights; agent harness optimization changes the system
around model calls. Both should use the same durable experiment control plane.
See [chapter 23](../23-agent-harness-optimization.md) for the conceptual contracts
and [chapter 15](../15-implementation-plan.md#harness-optimization-track-planned)
for the separate roadmap track.

## Decision proposed

Introduce a **versioned candidate envelope with a discriminated payload union**
for future `TRAINING` and `AGENT_HARNESS` kinds. Preserve existing `CandidateSpec`
as the training API and readable representation. Add a separate, typed immutable
harness specification and scientific fingerprint; compile through a sibling
harness compiler into the existing resolved-plan/runtime architecture. No
candidate class, persistence schema or runtime changes occur in this spec PR.

The following answers are proposals while this ADR remains `Proposed`:

| Review question | Proposed answer / unresolved boundary |
|---|---|
| 1. Envelope or union? | Both: versioned envelope for durable interchange, payload discriminated by candidate kind. Avoid an untyped payload or training-specific optional-field bag. Concrete type names and wire shape remain open for H02. |
| 2. Existing CandidateSpec compatibility? | Keep imports, constructors, readers and training fingerprints; interpret historical unwrapped records as training at a future compatibility boundary. Do not relabel/recompute stored v1/v2 hashes or force an immediate rename. Migration details and new API exposure require a separate implementation review. |
| 3. Model in HarnessFingerprint or sibling subject? | Include a resolved model binding/configuration in harness scientific identity for harness-only work, and retain a separate model subject reference. Pin fallback/subagent models too. A future reusable harness template may omit the binding, but is not an executable candidate identity. Provider revision evidence requirements remain open. |
| 4. Science versus execution? | Behavior-defining prompts, tools, context, memory, delegation, middleware, routing, stopping and explicitly scientific behavior/environment contracts are science. Machine/runtime/container packaging and behavior-preserving compiler/provider client glue are execution. Record execution differences and enforce comparability policy. No behavior change may be hidden as an operational override. |
| 5. Implementation/code identity? | Versioned artifact references plus content digests and declared contracts. Behavior-defining code enters harness identity; execution glue enters execution identity. Live objects, closures and moving repository paths are excluded. Adapter-equivalence evidence remains open for H03. |
| 6. Benchmark/task pinning? | Keep the candidate's optional BehaviorEnvironmentContract separate from BenchmarkSuiteRef. Pin tool/action/observation/reset semantics and behavior-defining agent-visible retrieval/environment state in their explicit scientific role. Ordered samples, held-out inputs, sampling assignments and scoring/evaluator/judge configuration belong to evaluation/comparison. Same harness + different benchmark normally preserves HarnessFingerprint and changes EvaluationFingerprint/comparison cohort; a scientifically declared or behavior-defining task/environment input may also enter harness identity with its role recorded. Resolve and verify immutable manifests/digests before comparison. Non-snapshot admission remains open for H04; no concrete H02 wire schema is decided. |
| 7. Durable trajectories? | Typed, versioned manifests reference immutable event chunks and output artifacts with digests, producer/causal links and completeness/redaction metadata. Bounded summaries/references in core rows; no unlimited raw contexts. Concrete chunk format, retention and replay minimums are open for H05. |
| 8. Stochastic replicates? | New harness Runs produce independent trajectories of the same candidate with seed/replicate/reset evidence; retries are attempts, not samples. New EvaluationRuns sample stochastic judges on a fixed trajectory separately. Never reuse a sample to fulfill a replicate request. Sampling/aggregation rules remain open for H06. |
| 9. Independent objectives? | Preserve success, quality, cost, tokens and latency as separate provenance-rich metrics, with units/uncertainty/missingness. Providers may record scalarization, constraints or Pareto decisions without discarding the vector. Evolution of the current single-primary Objective contract remains open for H06. |
| 10. Prompt-controlled permissions? | Prompts and CandidateProposal cannot grant capabilities. Actual permission grants, connector/tool authorization and operational changes use typed ActionProposal/ActionSpec through PolicyEngine/approval, constrained by sandbox, connector trust and budgets. Candidate validation/budget checks and generic branching governance are separate; PR-025/H07 must settle that governance before harness mutation execution. A future BranchExperiment Action is an option, not a settled decision. Refuse an unmet requirement rather than silently change the claimed candidate. |
| 11. New node or new run? | A comparative scientific harness CandidateProposal creates a new immutable node through the candidate/branching path, subject to validation, budget and the governance PR-025/H07 defines. Another realization is a new run; an infrastructure retry is a new attempt; post-hoc rescoring is a new evaluation run. Declared dynamic behavior stays in trajectory. In-run harness interventions are deferred; existing ADR-011 training rules remain unchanged. |
| 12. Joint model+harness composition? | Keep component identities and role-tagged bindings distinct. Later joint identity must bind the training scientific fingerprint, actual trained-artifact digest/lineage, harness configuration and resolved model binding. Never concatenate bare hashes or use one hash for science, execution and output lineage. Exact template/binding projection and scheduling are open; no Cartesian-product implementation is designed now. |

All new projections must be domain-separated and versioned, following the
canonical typed encoding principles of ADR-006. A harness scientific hash,
execution hash, evaluation hash, trajectory history and artifact lineage remain
separate. Evaluation identity does not replace the evaluated subject digest.
Existing training projection/reuse semantics are not broadened by this ADR.

For later joint mode, the conceptual binding is:

```text
training candidate fingerprint
  + selected trained artifact digest and causal lineage
  + harness template/configuration identity
  + explicit model-role binding → resolved harness candidate identity
  → versioned joint scientific identity (future contract)
```

Two trained outputs of one stochastic training candidate must remain
distinguishable as harness inputs. The output artifact's lineage is not the
training hypothesis. A joint proposal may have unresolved model bindings,
but it cannot claim a resolved harness identity before those inputs exist.

## Alternatives considered

- **Put AgentHarnessSpec inside CandidateSpec:** rejected for the proposed
  direction; it couples harness-only work to mandatory training semantics and
  creates ambiguous scientific projections.
- **Rename CandidateSpec immediately:** rejected; it breaks a working API and
  durable readers before there is a reviewed migration contract.
- **Use arbitrary plugin dictionaries as candidates:** rejected; scientific
  identity, typed mutations and capability validation would be undefined.
- **Create a separate harness controller:** rejected; it duplicates experiments,
  policy, budgets, evaluations and provenance.
- **Put harness semantics inside RuntimeBackend:** rejected; runtimes execute
  plans, while harness adapters translate scientific behavior above runtime.

## Consequences and open questions

The proposed seam lets generic planner work avoid assuming all candidates
train weights, without requiring any Phase 5/6 implementation changes now.
Training-only, harness-only and later joint optimization share Experiment,
Node, Run, Attempt, graph, evaluation, decision, Action/Policy, budget, provenance,
artifacts and runtime reconciliation.

Acceptance would establish the direction, not settle the implementation details
marked open in the table. Those details need review before their H-track step,
including workload/state/telemetry versioning, model-provider reproducibility,
trajectory retention, replicate statistics, multi-objective contract evolution
and candidate-proposal governance before harness mutation execution.
ADR-017 still governs reuse policy; this ADR does not silently accept it.

The first MVP limits mutations to prompt, context policy and tool descriptions/
configuration within authorized tools on a pinned suite. Middleware, delegation,
persistent memory and joint optimization follow later. Named harness adapters
and search strategies are future plugins, not new dependencies in this PR.
