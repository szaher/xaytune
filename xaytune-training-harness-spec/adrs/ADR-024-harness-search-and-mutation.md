# ADR-024 — Harness search and mutation: providers, proposals and governance

## Status

Accepted — 2026-10-15. H07, Track C. It gates the H07 implementation PR.

Proposed the same day in [PR #85](https://github.com/szaher/xaytune/pull/85).
After the architecture review, the project owner approved P1–P13, decided
Q1–Q4 (§"Review questions") and approved the explicit
`autonomous_search: authorized` requirement (§P7). The review asked for
three corrections first, all made before acceptance:

- the enabled set only narrows: `child.enabled ⊆ parent.enabled ∩
  authorized_tools` (§P3, §P7.2);
- one provider-safe information policy covers the whole search context,
  including summaries, rankings and comparisons (§P4, §P6);
- the local loop runs under a declared spending envelope (§P12a).

Acceptance is architectural. The H07 implementation PR must still
demonstrate §P12's acceptance criteria and their negative tests. This
document does not verify those runtime guarantees.

It builds on several accepted ADRs:

- ADR-018: harness candidates, the `ExperimentCandidate` envelope and `AgentHarnessSpec`.
- ADR-019: compilation and execution, with `authorized_tools`.
- ADR-020: pinned tasks.
- ADR-021: trajectories.
- ADR-022: evaluation, provider feedback and comparison.

It also builds on the generic planner, search and branching contracts of
PR-024, PR-025 and PR-034.

What it does not change:

- No existing contract, fingerprint, migration or controller path.
- The training `CandidateProposal`, its identity and `SearchProvider` (PR-034).
- Branching for training candidates.

Chapter 23 §10–§11 asked H07 to settle four things before harness mutation runs:

- candidate-proposal governance;
- the mutation and search plugin boundaries;
- optimizer capabilities;
- what a provider may see.

This ADR settles them. Durable execution of generated harness candidates
follows the durable harness path (ADR-019 M1–M6, Track B). Until then, the
first implementation runs the whole loop locally against verified evidence and
refuses to branch a harness proposal into the control plane, explicitly
(§P12).

## Context

What exists:

- **The planner path.** `Planner → CandidateProposal | ActionProposal`. A
  `CandidateProposal` carries a full training `CandidateSpec`, its parents, a
  hypothesis, a reason, a free-form `mutation` rationale, evidence refs and
  `ProposalProvenance`. Branching (PR-025, `materialize_candidate_proposal`)
  checks, in one write transaction:
  - the recorded planner;
  - a fresh planning-context fingerprint;
  - candidate identity, lineage and evidence ownership;
  - duplicates;
  - the budget;
  - an explicit actor.
- **Search (PR-034).** A `SearchProvider` suggests training proposals through
  the planner path. It holds no durable state, replays its history from the
  record, and proves trials by their branch origin.
- **Harness candidates (H02, H03).**
  - An `AgentHarnessSpec` has the sections model, instructions, tools, context
    and stopping, identified by `harness_fingerprint` (v1).
  - `ExperimentCandidate` wraps either a training candidate or a harness
    candidate.
  - Harness candidates compile to `AgentExecutionSpec` and run in-process.
  - Execution enforces `authorized_tools`. A prompt cannot grant a tool.
- **Evaluation (H06).** `provider_feedback(report)` is the only view of an
  evaluation a provider may receive. It carries run values after the
  provider filter, plus per-task *execution* measurements. It never carries a
  per-task reference-derived score, a `successful-tasks` slice or any
  restricted content. `compare()` gives per-metric verdicts and a dominance
  relation, never a scalar.

What does not exist:

- **Harness nodes in the control plane.** `ExperimentNode.candidate` is a
  training snapshot. ADR-019 M1–M6 (durable harness candidates and execution)
  is Track B work.
- **A way to propose a harness candidate.** `CandidateProposal.candidate` is a
  `CandidateSpec`.

## Decisions

### P1. Search and mutation are planners; Xaytune owns everything else

- Harness search and mutation sit behind the existing planner contract. A
  bound `HarnessPlanner` (Xaytune-owned, like PR-034's `SearchPlanner`) is a
  `Planner`. It orchestrates two plugin roles:
  - a **`HarnessSearchProvider`**, which chooses *where* to search next;
  - a **`HarnessMutationProvider`**, which proposes *what* to change.
- Neither provider:
  - creates a node, run or attempt;
  - writes the store or calls a runtime;
  - mints an id or a timestamp;
  - edits a candidate.
- A provider's output is data that Xaytune validates, applies, deduplicates
  and governs. There is one controller, one experiment graph, one decision
  engine and one execution path for every proposal.

### P2. A distinct proposal type: `HarnessCandidateProposal`

- `Proposal` becomes
  `CandidateProposal | HarnessCandidateProposal | ActionProposal`.
  `CandidateProposal` (training), its identity v1 and every planner that
  returns it are unchanged.
- A `HarnessCandidateProposal` (`kind = "harness-candidate"`) carries:

  ```text
  harness:              AgentHarnessSpec          the full child, never a diff alone
  harness_fingerprint:  str                       checked against harness (as CandidateProposal does)
  parent_ids:           (ExperimentNodeId, ...)   ≥ 1; v1 patches have exactly one base
  patch:                HarnessPatch              typed (§P3); applied to the base, it must give `harness`
  hypothesis, reason:   str
  evidence_refs:        (EvidenceRef, ...)        decision | evaluation-result, as today
  provenance:           ProposalProvenance        the bound HarnessPlanner, as today
  search:               ProviderProvenance        the bound search provider (§P8)
  mutation:             ProviderProvenance        the bound mutation provider (§P8)
  feedback:             (FeedbackRef, ...)        every ProviderFeedback the providers received (§P6)
  ```

- Identity: `harness_candidate_proposal_identity_v1`. It covers every field
  above. The proposal's identity is not the candidate's: the candidate is
  still identified by `harness_fingerprint` alone.
- **Rejected alternative:** widening `CandidateProposal.candidate` to the
  `ExperimentCandidate` envelope. It would change the identity of every
  recorded training proposal (PR-025 branch origins carry
  `proposal_fingerprint`). It would also put an untyped `mutation`
  dictionary where a harness patch must be typed.

### P3. `HarnessPatch`: a closed, typed set of operations on a pinned base

```text
HarnessPatch
  schema_version     "xaytune.harness-patch/v1"
  base_fingerprint   the parent's harness_fingerprint
  operations         one or more, applied in order; one per surface field at most
  rationale          str (not identity)
```

The v1 operations are the first MVP's surfaces (chapter 23 §12): prompts,
context policy, and tool descriptions or configuration within an already
authorized tool set.

| Operation | Surface | Changes | Refused when |
|---|---|---|---|
| `set-system-instructions` | `instructions` | `instructions.system` | — |
| `set-task-template` | `instructions` | `instructions.task_template` (syntax, content, variables) | the variables are not the base's (the task binding fixes them) |
| `set-tool-description` | `tools.descriptions` | one defined tool's `description` | the tool is not defined in the base |
| `set-enabled-tools` | `tools.enabled` | the enabled subset, narrowed or reordered | a tool is not *enabled* in the base (defined but disabled counts as not enabled), or is outside the experiment's authorized tools (§P7.2) |
| `set-context-policy` | `context` | any of `max_context_tokens`, `selection`, `overflow`, `max_tool_result_tokens` | the result exceeds the experiment's ceilings (§P7) |
| `set-stopping-limits` | `stopping` | any of `max_turns`, `max_total_tokens`, `max_tool_calls` | the result exceeds the experiment's ceilings (§P7) |

- **Not in v1:**
  - `model` and generation settings, because a different pinned model is a
    different claim and changes the cost basis;
  - adding a tool *definition*, input schema or contract, because that is a
    new capability, not configuration;
  - enabling a tool the base has disabled, even one it defines and the
    experiment authorizes. Enabling is a grant, and a grant goes through
    explicit authorization (§P7);
  - `terminate_on`;
  - memory, delegation, middleware and routing, which `AgentHarnessSpec` v1
    does not have.

  Each needs its own operation, version and review.
- **Applying a patch** is one pure core function,
  `apply_patch(base, patch) -> AgentHarnessSpec`. It refuses, with every
  reason:
  - a base whose fingerprint is not `base_fingerprint` (stale base);
  - an unknown operation or field;
  - a no-op patch (the child's fingerprint equals the base's);
  - a result that fails `AgentHarnessSpec` validation.

  Content is inline and bounded, or a `ContentRef` with a digest (ADR-018).
  A patch never carries a live object, a mutable path or a moving reference.
- A patch is the declared change. The child harness is what is identified,
  evaluated and executed. Branching re-applies the patch to the *recorded*
  parent and requires the result to equal the proposed harness byte for
  byte, so a proposal cannot carry a different harness than its patch
  describes (§P7, H7-F2).

### P4. The two provider contracts (plugin ABI, ADR-008)

Both providers are versioned plugin kinds with descriptors. They are bound
from specs in the experiment (`PlannerSpec.config`), and their configuration
is identity, as for planners and search providers (ADR-016).

```text
HarnessSearchProvider                                   (kind "harness-search", api v1)
  descriptor: PluginDescriptor + HarnessSearchCapabilities
  select(context: HarnessSearchContext) -> SearchChoice | NoChoice
      SearchChoice = base node id + why (selection rationale) + the objective view it used
      NoChoice     = a typed reason (exhausted, nothing comparable, waiting for evaluation)

HarnessMutationProvider                                 (kind "harness-mutation", api v1)
  descriptor: PluginDescriptor + HarnessMutationCapabilities
  mutate(request: MutationRequest, count: int = 1) -> tuple[HarnessPatch, ...]
```

- **`HarnessSearchContext`** is built by Xaytune from the record, like
  `PlanningContext`. It contains:
  - the objective vector;
  - each harness node's id, harness fingerprint, parents and status;
  - each node's `ProviderFeedback` (§P6), and nothing else derived from its
    evaluations;
  - **provider-safe comparisons** between nodes: H06's `compare()` applied
    to the `ProviderFeedback` run values, never a comparison or ranking
    computed from a released report (§P6);
  - the spending envelope and what remains of it (§P12a);
  - the identity of the search history.

  It is ordered and fingerprinted (`harness_search_context_identity_v1`).
  The search context carries no released `HarnessEvaluationReport`,
  released summary or recorded comparison. A released report under
  `per-task` disclosure holds single-task scores, and a summary, rank or
  verdict computed from it would hand the provider what
  `provider_feedback()` withheld (H7-F12).
- **`MutationRequest`** contains:
  - the base harness and its fingerprint;
  - the allowed surfaces (the intersection in §P5);
  - the base's provider feedback, and its ancestors' (§P6);
  - the fingerprints of every harness the experiment already has (for the
    provider's own deduplication);
  - the seed;
  - `count`.

  It is fingerprinted (`mutation_request_identity_v1`).
- **`count` is 1** in v1, as in PR-034. Parallel proposals need admission and
  scheduling semantics that do not exist yet.
- **`HarnessPlanner.plan(context)`** runs these steps:
  1. builds the search context;
  2. `select()`;
  3. builds the mutation request for the chosen base;
  4. `mutate()`;
  5. for each patch: `apply_patch` → `verify_harness_proposal` (§P7) → dedupe;
  6. returns at most one `HarnessCandidateProposal`, or none, with the reason.

  A patch that `apply_patch` or `verify_harness_proposal` refuses is not
  proposed. The refusal is returned in the planner's result for the
  record. The planner asks again at most `max_retries` times (bound in its
  spec), then stops with no proposal.

### P5. Capabilities: declared, bound, and enforced on every patch

```text
HarnessMutationCapabilities
  surfaces       the operations it may produce (subset of §P3)
  determinism    deterministic | seeded | stochastic
  model          None | the AgentModel identity it calls (stochastic providers; PR-030/032)
  feedback       none | run-values | run-values+task-execution   (what it reads, §P6)

HarnessSearchCapabilities
  objectives     single | vector (Pareto over ADR-022 V10 relations)
  determinism    deterministic | seeded
  needs          evaluated-nodes | comparisons
```

- The experiment declares `allowed_surfaces` in the harness planner spec. A
  patch may use only the intersection of the provider's declared surfaces
  and the experiment's allowed surfaces. Binding refuses a provider whose
  declared surfaces do not intersect the allowed surfaces. `apply_patch`
  refuses any operation outside the intersection, whatever the provider
  declared (H7-F3).
- Capabilities describe technical support, never authorization (chapter 23
  §11). Authorization is §P7's.
- A **stochastic** provider:
  - binds its model identity into its spec (PR-031 rules);
  - goes through `invoke_agent_model`, with validated output and a closed
    schema;
  - records every call in the `AgentInvocation` journal (PR-032).

  A restarted planner replays the answered round instead of calling the
  model again, so a crash between the call and the proposal never asks
  twice.

### P6. What a provider may see: the H06 provider view, nothing more

- Providers receive evaluation evidence only as `ProviderFeedback`
  (`provider_feedback()`, ADR-022 V5 and G2c):
  - released run values after the provider filter;
  - per-task **execution** measurements.
- They never receive:
  - restricted content;
  - held-out references;
  - judge exchanges;
  - per-task reference-derived scores;
  - `successful-tasks` slices;
  - a reference-derived run value over one task.
- **No per-task score grant in v1.** ADR-022 left granting a provider
  per-task reference-derived scores to H07's contract. H07 v1 grants none. A
  provider that needs per-task scores (reflection over failures) waits for a
  reviewed grant (§P13).
- **No trajectory content in v1.** A trajectory may contain prompts, tool
  results and restricted-tier content (ADR-021). A provider that reads
  trajectories needs a released-tier trajectory projection and a decision
  of its own (§P13).
- **One policy for everything a provider receives.** The rule above covers
  the `MutationRequest` *and* the whole `HarnessSearchContext`: values,
  summaries, rankings, comparison verdicts, dominance relations, and the
  `NoChoice` and selection rationales Xaytune writes back into a later
  context. Each is computed only from `ProviderFeedback`. A metric
  `provider_feedback()` withheld (a single-task reference-derived value, any
  `successful-tasks` slice) does not exist for the provider. A comparison on
  it is absent with the reason `withheld-from-provider`. It is not a
  verdict, a tie or a rank, because a verdict on a single-task score is
  that score's sign. The rule holds under both disclosure modes
  (`aggregate` and `per-task`). The released report's disclosure governs
  what people see, never what a provider sees.
- Every feedback a provider received is named in the proposal as a
  `FeedbackRef`: the evaluation fingerprint, the evidence digest and the
  evaluation result id. A reviewer can see exactly what a proposal was
  allowed to learn from.
- **Feedback from incomparable evaluations is never mixed silently.** The
  search context groups feedback by evaluation fingerprint (ADR-022 C3).
  A provider declaring `objectives: vector` receives only provider-safe
  comparisons that `comparability()` passed.

### P7. Governance: the generic branching contract, plus four harness checks

This settles chapter 23's gate: *"PR-025/H07 must settle candidate-proposal
governance before harness mutation execution."*

- **A harness proposal is not an Action**, and it is not an input to
  `PolicyEngine`, consistent with chapter 23 §11 and PR-025. It is governed
  by the generic branching contract, unchanged:
  - the recorded planner spec;
  - a fresh context fingerprint;
  - candidate identity (`harness_fingerprint` recomputed);
  - lineage and evidence ownership;
  - duplicates;
  - a budget precheck;
  - an explicit actor.
- It is also governed by four harness checks. They are one pure function,
  `verify_harness_proposal(proposal, parent_harness, policy, existing)`,
  which runs in the planner *and again* inside the branching transaction
  against the recorded parent:
  1. **Patch integrity.** `apply_patch(recorded parent harness, patch)` equals
     `proposal.harness`, and `patch.base_fingerprint` is the recorded
     parent's.
  2. **No capability escalation.** The child's enabled tools satisfy

     ```text
     child.tools.enabled ⊆ parent.tools.enabled ∩ authorized_tools
     ```

     against the *recorded* parent and the experiment's `authorized_tools`
     (ADR-019). A patch can narrow or reorder the enabled set and nothing
     else. Being defined in the parent is not enough: a tool the parent
     defines but has disabled stays disabled in every descendant, even if
     the experiment authorizes it. Over a search the enabled set can only
     shrink, starting from the human-authored root. Prompt text grants
     nothing, because only `tools.enabled` is checked and execution enforces
     it again (ADR-019).
  3. **Ceilings.** Context and stopping limits stay within the experiment's
     declared ceilings (`HarnessSearchPolicy.ceilings`). Per-task limits are
     scientific; experiment ceilings are policy. The stricter limit governs
     (chapter 23 §11).
  4. **Surfaces.** Every operation is in the allowed intersection (§P5).
- **New authorization stays an Action.** A tool grant, connector
  authorization or permission change is an `ActionProposal → ActionSpec →
  PolicyEngine/approval`. So is enabling a tool the parent disabled. A
  harness planner never returns one in v1.
- **No per-candidate human approval in v1 (Q3, decided), on one condition:
  the experiment explicitly authorizes autonomous search.**
  `HarnessSearchPolicy` carries `autonomous_search: authorized`, and it has
  no default. Binding a `HarnessPlanner` (and starting the local loop)
  refuses a policy without it. It also refuses a policy that does not fix:
  - **the permissions:** `authorized_tools`, the allowed surfaces and the
    ceilings, bound into the policy's fingerprint;
  - **the budgets:** the spending envelope (§P12a).

  Changing any of them is a new policy, and a new policy is a new search.
  Within it, the checks above make every generated candidate a narrowing
  or reconfiguration of an already authorized harness. A `BranchExperiment`
  approval Action stays deferred (§P13).

### P8. Provenance: who proposed what, from what, under which configuration

```text
ProviderProvenance
  provider, name, plugin_version, api_version          descriptor
  spec_kind, spec_version, spec_identity_version, spec_fingerprint   bound configuration
  capabilities_fingerprint
  engine                                               exact library releases, as PR-034
  seed
  agent_invocation_ids                                 stochastic providers only (§P5)
```

- The durable branch origin (PR-025's `CandidateBranchOrigin`) of a harness
  node records:
  - the proposal fingerprint;
  - `ProposalProvenance`;
  - both `ProviderProvenance`s;
  - the typed patch and its fingerprint;
  - the base fingerprint;
  - the `FeedbackRef`s;
  - the evidence refs.

  This is everything needed to answer "why does this harness exist" from
  the record. Optimizer state, if a provider keeps any, is a
  content-addressed artifact reference in its provenance (chapter 23 §10),
  never process memory.

### P9. Restart and determinism

- Providers hold no durable state. The harness planner rebuilds its context
  from the record before every round. Deterministic and seeded providers
  then give the same proposal for the same record (PR-034's rule), and a
  proposal that was never branched is proposed again, identically.
  Branching it twice gives the same node, because duplicates are checked by
  fingerprint.
- Seeded providers derive every random draw from (bound seed, round index,
  base fingerprint). A search's history is the sequence of harness nodes
  whose branch origin names this planner as bound. A node made by hand or by
  another planner is never claimed as this search's trial (PR-034's proof
  rule).
- A provider whose engine or version changed refuses a record it does not
  reproduce. It never searches around it.

### P10. Duplicates and staleness

- **Duplicates.** A child whose `harness_fingerprint` matches any node of the
  experiment is never proposed. The planner asks the mutation provider
  again, within the retry bound. Branching re-checks inside the
  transaction.
- **Staleness.** A patch on a base that is not the recorded parent's
  fingerprint is refused (`apply_patch`). A proposal planned from a context
  that changed since is refused at branching (`StaleProposalError`), as
  today.

### P11. Objectives and selection

- Selection reads the objective vector, each node's `ProviderFeedback` and
  the provider-safe comparisons (§P6). A search provider never scalarizes the vector unless its
  bound configuration declares a scalarization. The weights are then part
  of its spec fingerprint, and the scalarization is named in every
  selection rationale.
- A missing measurement is absent with its reason, never zero.
  `not-comparable` is never treated as a tie or a loss.
- Selection does not decide a candidate's fate. Decisions stay with the
  decision engine (PR-015). Stopping rules stay with the experiment's
  objective and budget.

### P12. The bounded first implementation (the H07 implementation PR)

**In:**

- **Core contracts** in `xaytune/core/domain/harness_search.py`:
  - `HarnessPatch` and its operations;
  - `apply_patch`;
  - `HarnessCandidateProposal`;
  - `ProviderProvenance`, `FeedbackRef`;
  - `HarnessSearchContext`, `MutationRequest`;
  - every identity function;
  - `verify_harness_proposal`;
  - `HarnessSearchPolicy`: the autonomous-search authorization, allowed
    surfaces, ceilings, retry bound and spending envelope (§P7, §P12a);
  - `provider_comparisons`, H06's `compare()` restricted to
    `ProviderFeedback` (§P6).
- **Plugin roles** in `xaytune/harness/search.py`, with descriptors,
  capabilities and binding through the plugin ABI:
  - the two protocols;
  - `HarnessPlanner`.
- **Two reference providers**, deterministic, no dependency, no model:
  - `EnumeratedMutationProvider`: a typed, declared list of alternatives per
    allowed surface (for example, three system prompts as content refs, two
    context selections, two `max_turns` values). It proposes the first
    alternative not yet tried on the base, in declared order, seeded for
    tie-breaks.
  - `BestFirstSearchProvider`: the best `ProviderFeedback` run value under
    the objective's primary metric, with ADR-022 comparability required. A
    node whose primary value was withheld is not ranked on it. For a
    vector objective, Pareto-nondominated nodes, ties broken by node id.
- **A local loop**, `run_local_harness_search(...)`, for H07 before Track B.
  It runs rounds until the policy or the spending envelope (§P12a) stops it.
  Each round:
  1. compile and run the candidate (H03);
  2. assemble and verify evidence (H06 V3);
  3. evaluate (`run_harness_evaluation`);
  4. give `provider_feedback` to the planner;
  5. plan;
  6. `verify_harness_proposal`;
  7. take the child as the next round's candidate.

  It writes a local search report: every proposal, refusal, provenance
  record, feedback reference and spending entry. It does **not** touch the
  control plane, and it claims no durable orchestration (Q4). A crashed
  local search is re-run from its report, not resumed by a controller.
- **Branching refuses harness proposals.**
  `materialize_candidate_proposal` refuses a `HarnessCandidateProposal` with
  a typed `BranchRefusedError` ("harness candidates are not durable yet:
  ADR-019 M1–M6"), so nothing half-supports them.

**Acceptance criteria (each with negative tests):**

1. **Patches.**
   - `apply_patch` handles every v1 operation.
   - It refuses: stale base, unknown operation, undefined tool, no-op,
     invalid result, an operation outside the allowed surfaces.
   - A patched harness's fingerprint changes exactly when its identity does.
2. **Governance.** `verify_harness_proposal` refuses each of:
   - a harness that is not the patch applied to the parent (H7-F2);
   - a tool outside `authorized_tools` (H7-F1);
   - **a tool the parent defines but has disabled**, even when the
     experiment authorizes it. This is a regression test for the
     `child.enabled ⊆ parent.enabled ∩ authorized_tools` rule, run against
     both `apply_patch` and `verify_harness_proposal` (H7-F1);
   - a ceiling exceeded;
   - a surface outside the intersection;
   - a duplicate.

   A prompt that *says* it grants a tool changes nothing that is enforced.
   Binding refuses a policy without `autonomous_search: authorized`, fixed
   permissions or a spending envelope.
3. **Information boundary.**
   - The `MutationRequest` and `HarnessSearchContext` types carry no
     reference, restricted content, per-task reference-derived score,
     slice, released report, released summary or recorded comparison. A
     schema test pins their fields.
   - A provider spy receives only `ProviderFeedback` and values computed
     from it. A test with a judge evaluation shows the restricted report is
     never reachable from the request or the context.
   - **Both disclosure modes.** The same tests run with the evaluation under
     `aggregate` and under `per-task` disclosure, for:
     - a single-task evaluation (its `task_success` and `quality`);
     - a `successful-tasks` slice over several tasks.

     In each case, neither withheld value appears in the search context,
     in any form: as a value, a summary, a rank, a best-first choice, a
     Pareto relation, a comparison verdict or a rationale. A comparison
     on such a value is absent with `withheld-from-provider`. Two contexts
     that differ only in a withheld value have the same fingerprint, and
     a deterministic provider makes the same choice from both.
4. **Determinism and restart.**
   - The same record and bound specs give the same proposal and the same
     proposal fingerprint.
   - A re-run of the local loop from its report reproduces every round.
   - A changed engine version refuses.
5. **Provenance.** Every proposal names both providers as bound, the patch,
   the base and every `FeedbackRef`. The identity changes when any of them
   does.
6. **Selection.**
   - Best-first never compares across evaluation fingerprints.
   - Pareto selection matches ADR-022 V10 relations.
   - `not-comparable` is never a tie.
7. **Local validation.** Two allowed surfaces (system prompt, `max_turns`)
   over the pinned support suite (H04) with a scripted model, under a
   spending envelope. Three rounds produce two proposals, each evaluated,
   verified and recorded. The report explains:
   - why each child exists;
   - what each evaluation released to the providers;
   - what each round spent.
8. **Compatibility.**
   - `CandidateProposal`, its identity, `SearchProvider` and branching for
     training are byte-identical. Pinned from `main`.
   - Branching a harness proposal is refused with the typed reason.
   - No migration.
9. **Spending (§P12a).**
   - A scripted model replies at maximum length on every call, and the
     envelope fits exactly *k* task attempts' reservations. The search
     runs *k* attempts and stops with `spending-limit` before attempt
     *k + 1* starts. Every recorded spend is at most its limit.
   - The same holds for each other unit: a judge call, a stochastic
     provider call, a candidate evaluation and a round.
   - Retries count. An attempt abandoned with its spend unaccounted keeps
     its full reservation charged.
   - The search refuses to start:
     - with no token limit for a unit some work is charged in;
     - with a cost limit;
     - with a ceiling or generation setting that leaves an attempt
       unbounded;
     - with a model call whose worst case cannot be reserved.
   - A measured spend above its reservation stops the search with an
     error. It is never settled silently.

### P12a. The local search's spending envelope

Bounded rounds do not bound spending. One round can run every task of a
suite, retry each task, call a judge for each attempt and call a stochastic
provider. So the local loop runs under an explicit envelope in
`HarnessSearchPolicy`, and every limit in it is enforced *before* the work
that would cross it starts.

```text
LocalSearchEnvelope
  max_rounds                   rounds of §P12's loop
  max_candidate_evaluations    harness evaluations, across the search
  max_task_attempts            task attempts, retries included, across every evaluation
  max_model_calls              agent, judge and provider model calls, per role
  token_limits                 {token unit: limit}; one limit per unit, units never added together
  cost                         refused in v1
```

- **Units.** The harness runner counts tokens with its bound counter
  (`tokens(<counter>)`, the H06 metric unit). A judge or model adapter
  reports its provider's own tokens. These are different units, so each
  needs its own limit, and they are never summed. Work charged in a unit
  with no declared limit is refused before the search starts.
- **Cost is refused.** v1 records no cost: H06's `cost` metric is absent,
  `not-recorded`, and PR-016 refuses cost budgets for the same reason. A
  declared cost limit refuses the search (`cost-not-metered`) rather than
  pretending to bound it. External provider spending is bounded through
  token and call limits, and the report says so.
- **Reserve before start.** Every unit of work has a worst-case charge that
  Xaytune computes before starting it:
  - **A task attempt** reserves `max_total_tokens + max_context_tokens +
    max_output_tokens`, taken from the candidate's stopping, context and
    generation limits. The runner checks the running total after each
    reply, so the last reply can overshoot by at most one call. §P7.3's
    ceilings fix these limits for the whole search. A harness or ceiling
    that leaves any of them unset is refused at start, because its attempt
    cannot be bounded.
  - **A judge or provider model call** reserves its declared prompt bound
    plus its `max_output_tokens`. A call without both is refused at start.

  A unit starts only if, for every limit it touches, spent plus its
  reservation is within the limit. Otherwise the search stops with
  `spending-limit`, naming the limit, and the report records the unit that
  was not started.
- **Settlement.** After the unit, the reservation becomes the measured
  spend:
  - unaccounted spend (lost evidence, an unfinished call, a reply without
    token counts) keeps the full reservation charged;
  - a measured spend above its reservation is an error that stops the
    search. That would mean the bound is wrong, and the search does not
    continue on a wrong bound.

  v1 runs one unit at a time (`count = 1`), so at most one reservation is
  open.
- **Precedence.** The envelope and §P7.3's ceilings are the experiment's
  policy, and per-task limits are the task's. The stricter one governs.
  Exhausting the envelope ends the search. It never fails or cancels a
  recorded evaluation, and the report keeps every completed round.

### P13. Deferred, each with what it waits for

| Deferred | Waits for |
|---|---|
| Branching harness proposals into the control plane | Durable harness candidates and execution, ADR-019 M1–M6 (Track B); §P7's checks move into the branching transaction unchanged |
| LLM or reflection-based mutation (GEPA/MIPRO-style) | §P5's stochastic path is decided; also needs a reviewed per-task feedback grant if it reads failures |
| Granting providers per-task reference-derived scores | Its own decision under ADR-022 V5: who grants, how it is recorded and audited |
| Trajectory-reading providers | A released-tier trajectory projection (ADR-021) and its access decision |
| Optuna, evolutionary and external optimizers | Adapters below the boundary, as Ray Tune is for PR-034 |
| `count > 1` and parallel proposals | Admission and scheduling semantics (shared with PR-034) |
| Human approval of generated candidates (`BranchExperiment` Action) | A need beyond Q3's condition (an experiment authorizing autonomous search within fixed permissions and budgets) |
| Enabling a tool the parent disabled, or adding one | An explicit authorization Action (`ActionProposal → PolicyEngine`) whose grant the next policy binds |
| Cost limits | Metered cost: a recorded `cost` metric with its unit and source, and PR-016's cost budget |
| Model and generation, tool definitions, memory, delegation, middleware and routing surfaces | Each its own operation version; some need `AgentHarnessSpec` v2 |
| Joint training and harness search | ADR-018's joint-mode identity |
| Studio optimization views (UX-04) | This ADR's records: proposals, refusals, provenance and feedback refs, read through the Studio API once durable |

## Review questions (decided 2026-10-15)

| Question | Decision |
|---|---|
| **Q1.** A distinct `HarnessCandidateProposal`, or widen `CandidateProposal` to the envelope? | **Distinct** `HarnessCandidateProposal` (§P2) |
| **Q2.** The v1 surfaces: prompts, context policy, tool descriptions, enabled-set narrowing, stopping limits? | **Approved**, with strict enabled-tool narrowing (§P7.2) and experiment ceilings (§P7.3) |
| **Q3.** Human approval of generated candidates? | **No per-candidate approval in v1**, provided the experiment explicitly authorizes autonomous search within fixed permissions and budgets (§P7, §P12a) |
| **Q4.** The local loop before Track B, or wait for M1–M6? | **The local loop first**, with no claim of durable orchestration (§P12) |

## Failure scenarios

- **H7-F1. Capability escalation by prompt or patch.** There are three
  cases:
  - a patch enables a tool outside authorization;
  - a patch re-enables a tool the parent defines but disabled;
  - a system prompt claims a permission.

  §P7.2 (`child.enabled ⊆ parent.enabled ∩ authorized_tools`) refuses the
  first two, and so does `apply_patch`. The third changes nothing that is
  enforced, and execution enforces `authorized_tools` again (ADR-019).
- **H7-F2. Patch smuggling.** A proposal's harness differs from its patch
  applied to the recorded parent. This also covers a patch built against a
  parent edited after planning. §P7.1, checked again in the branching
  transaction.
- **H7-F3. Surface creep.** A provider declares prompts but emits
  `set-enabled-tools`. `apply_patch` refuses outside the intersection.
- **H7-F4. Reference leakage.** A provider receives a held-out reference or a
  per-task score, and the search optimizes to the test set. §P6: the types
  cannot carry them, and a spy test proves what is passed.
- **H7-F5. Incomparable feedback.** Selection compares a node evaluated on
  another suite revision or judge. Summaries are grouped by evaluation
  fingerprint, and best-first requires comparability (§P6, §P11).
- **H7-F6. A restart re-asks a model.** A stochastic provider is called
  again after a crash. The `AgentInvocation` journal replays the answered
  round (§P5).
- **H7-F7. A non-reproducing history.** An engine upgrade changes the next
  suggestion. The engine is bound, and replay refuses (§P9).
- **H7-F8. Duplicate exploration.** Two rounds propose the same harness under
  different patches. Duplicates are checked by harness fingerprint, in the
  planner and at branching (§P10).
- **H7-F9. A hidden scalarization.** A search ranks by an undeclared
  weighting. Scalarization must be bound in the spec, and it is named in
  every rationale (§P11).
- **H7-F10. Spending overrun by search.** Bounded rounds still allow
  unbounded tokens and provider spending: every task, every retry, every
  judge call and every provider call. In the control plane, the budget
  precheck at branching is unchanged. In the local loop, §P12a's envelope
  reserves each unit's worst case before it starts. A unit that would cross
  a limit is never started, and a limit that cannot be enforced (cost, an
  unbounded attempt, an unlimited unit) refuses the search before it
  starts.
- **H7-F11. Half-supported durability.** A harness proposal reaches
  `materialize_candidate_proposal` before M1–M6 and is written as a training
  node. Refused with a typed reason (§P12).
- **H7-F12. Disclosure bypass through the search context.**
  `provider_feedback()` withholds a single-task score or a
  `successful-tasks` slice. But under `per-task` disclosure the released
  report still holds it, and a summary, rank or comparison verdict computed
  from that report would reveal it. §P4 and §P6 build the whole search
  context from `ProviderFeedback`. A comparison on a withheld value is
  absent (`withheld-from-provider`), and contexts that differ only in a
  withheld value have the same fingerprint.

## Consequences

- Harness optimization reuses the planner, branching, budget, evaluation and
  decision paths. The only new governance is four harness checks in one
  pure function.
- Providers are swappable at the plugin boundary: random, grid, Pareto, LLM
  reflection, external optimizers. The information boundary and the
  governance do not depend on which one runs.
- The first implementation is useful without durable harness execution. It
  produces verified, explained proposals and evaluations locally. The step
  to the control plane is one refusal that becomes a check.
- What a search learned from is always recorded. That is what an
  optimization audit needs.

## Alternatives considered

- **One combined "optimizer" plugin** that selects, mutates and applies.
  Simpler to write, but the plugin would then own application and
  validation. Splitting selection from mutation lets a Pareto selector work
  with any mutation source, and keeps `apply_patch` in the core.
- **Patches as JSON Patch over the spec.** Untyped. It cannot be checked
  against surfaces or ceilings, and it can address fields that are identity
  without being configuration.
- **Governing candidate creation through `PolicyEngine`.** Policy governs
  Actions, and a candidate is not one (chapter 23 §11). Folding candidates
  into Actions would make every planner proposal an approval workflow. Q3
  decided against it for v1. The experiment authorizes autonomous search
  once, within fixed permissions and budgets.
- **Waiting for Track B before any H07 code.** It would leave the provider
  contracts unexercised until M1–M6 lands, and the contracts are what
  M1–M6 must carry.
