# ADR-022 — Harness evaluation, scoring and comparison

## Status

Accepted — 2026-10-12. Proposed 2026-10-12 with H06, and revised twice
after architecture reviews of
[PR #79](https://github.com/szaher/xaytune/pull/79) at `ff0d2ab` and
`b46b541`.

The project owner approved decisions **V1–V13** at `1a22f2c`, including
the seven decisions recorded in §"Decisions on the review's questions".
This status records architectural acceptance only. The H06 implementation
is a separate PR, reviewed on its own, and adds nothing to this ADR's
decisions.

**Implementation safeguard noted at acceptance** (no decision changed): an
evidence entry for an executed task lists at least one attempt.
`verify_run_evidence` refuses an empty attempt list, which is never a
verified execution (V3, G3g).

**Disclosure clarification at implementation review** (2026-10-13, review
of [PR #81](https://github.com/szaher/xaytune/pull/81); no measurement,
fingerprint or scoring rule changed): under `aggregate`, V5 also withholds
every `successful-tasks` slice, at any sample count, because next to the
released per-task execution measurements a slice's mean can identify which
tasks succeeded. A provider never receives such a slice, nor a
reference-derived run value over a single task, under either disclosure.

**Revision 2** (after the architecture review of
[PR #79](https://github.com/szaher/xaytune/pull/79) at `ff0d2ab`). The
review approved V1–V13's direction and decided the seven review questions
(§"Decisions on the review's questions"). Its three findings change:
- **V3:** two evidence entry states. An invalid claim of a complete outcome
  is refused; a legitimate execution with no attested result is valid
  evidence, with its outcome absent and its spend kept. Attempts are read
  from the store, so none can be omitted;
- **V5:** outputs are restricted, reference-derived measurements or
  execution measurements. A disclosure policy governs the measurements, no
  provider receives per-task reference-derived scores by default, and
  sending a reference to a judge needs an explicit `ReferenceEgressGrant`;
- **V10:** verdicts are an exploratory decision rule, not a statistical
  claim. A provisional direction needs R ≥ 3.

**Revision 3** (after the review of revision 2 at `b46b541`, which approved
V1–V13 and the seven decisions and found one S2 and two S3 issues):
- **V3:** one operation, `verify_run_evidence`, independently rechecks the
  assignment, attribution, complete attempt membership (listed from the
  store) and trajectory integrity. Assembly and every evaluator call it,
  so a self-consistent but incomplete record is refused (G3f);
- **V2:** only an attested `complete` trajectory contributes a scored
  outcome. Verified `partial` attempts contribute missingness and spend;
- **G3e:** `post-commit-evidence-lost` is `no-attested-result` only when
  its prefix verifies. Otherwise it is refused under G3d.

Builds on:
- ADR-007 and ADR-015 (accepted): evaluation is an independent workload with
  its own `EvaluationRun`/`EvaluationAttempt` lifecycle, and reuse is keyed
  on the evaluator's declared determinism;
- ADR-017 (accepted): fingerprints are evidence, never authority, and
  replicates always execute;
- ADR-018 (accepted), whose questions 8 and 9 left replicate aggregation and
  the evolution of the single-primary `Objective` open for H06;
- ADR-019 (E1–E10 accepted): H03's compiler, `AgentExecutionSpec`,
  `HarnessRunResult` and in-process runner;
- ADR-020 (accepted), whose B7 leaves comparability conditions C3 and C4 to
  H06 and whose B9 leaves the wiring of resolved tasks into execution to H06;
- ADR-021 (accepted): a sealed, verified trajectory is the evidence an
  evaluation measures.

## Context

H03 runs a harness on one task. H04 pins which tasks, and proves two runs
saw the same ones. H05 records what each run did, as sealed, verifiable
evidence. Nothing yet says **how good** a run was, or whether one harness
configuration did better than another.

Chapter 23 §9 sets the requirements:
- trajectories and outputs are measured through the existing independent
  evaluator and `EvaluationRun`/`EvaluationAttempt` lifecycle. Execution
  produces the subject; evaluation measures it;
- outcome, cost, tokens, time, interaction and context are separate metric
  families, each with units, missingness, uncertainty and an aggregation
  method;
- zero is never a substitute for unavailable data;
- execution replicates, retries and judging replicates are different things;
- quality, success, cost, tokens and latency stay independent. The
  single-primary `Objective` and threshold engine stay unchanged.

Chapter 23 §13 and chapter 15 gate H06 on **replicate aggregation and the
objective contract**. Reuse stays governed by ADR-017 and ADR-015 §3.

**What exists, and what it cannot yet do:**

| Existing | Fits | Gap H06 must close |
|---|---|---|
| `EvaluationSpec` + `evaluation_identity_v1` | One evaluator, its name, bound version and `config`; the dataset; slices. The evaluator's typed config is the established place for what it measures (`NativeEvaluationConfig`) | No benchmark suite, sampling or scoring rules for agent tasks |
| `Evaluator` / `ResolvableEvaluator` | `supports` → `resolve` → `supports`, then a deterministic `prepare` | No evaluator measures an agent run |
| `EvaluationRun` / `EvaluationAttempt` / `EvaluationResult` | One subject, one spec, one seed and replicate per run; results attributed to their run; `result_provenance_problems` | A harness candidate cannot be persisted, so these can't be repository rows yet (ADR-019 M1–M6 is not accepted) |
| `MetricResult` | Value, sample count, slice, evaluator, seed, standard error | `value` is required and finite, so it cannot say "not measured"; it has no unit field |
| `ThresholdDecisionEngine` | Point estimate against a target | Refuses a metric reported more than once in a cycle, so R replicates are undecidable by design |
| `HarnessRunResult` | Termination, answer, turns, tool calls (refused, failed), context truncations, prompt/output/total tokens and the `token_counter` rule | No cost, no provider usage, no run wall time |
| Trajectory events (ADR-021) | Per-call prompt and output tokens; `latency_ms` on model calls, optional and advisory; every attempt instance, including retries | No tool latency, no end-to-end elapsed time |
| H04 | `agent_visible`, `resolve_reference`, `verify_suite_content`, `assign_tasks`, `verify_assignment`, `benchmark_cohort`, `assignment_mismatches` | Tasks are not wired into `AgentExecutionSpec` |

**One distinction governs every decision below.** A score is a measurement
of evidence under a declared rule. It is not a fact about the harness. Two
scores say something about two harnesses only when C1–C4 all hold. Even
then, a v1 difference is a provisional direction for deciding what to try
next, never a statistical claim that one harness is better.

## The boundary H06 stops at

- **No durable harness path.** A harness candidate can't be persisted or
  submitted to a controller until ADR-019's M1–M6 migration is accepted and
  implemented. So H06 uses the evaluation **domain contracts** (spec, run,
  attempt, result, evaluator protocol, state machines, provenance checks)
  through a local, in-process evaluation driver, as H03 did for execution.
  It writes no repository rows and adds no migration.
- **No decision engine change.** `Objective`, `DecisionContext`,
  `ThresholdDecisionEngine` and `AdaptiveThresholdDecisionEngine` are
  unchanged. A comparison report is evidence, not a `Decision`, and moves
  no node.
- **No search or mutation.** Choosing the next harness from the evidence is
  H07's. H06 fixes only what a mutation provider may never see (V5).
- **No new trajectory schema.** ADR-021's event union is unchanged. A metric
  the events don't record is reported as absent, not derived from
  something else.
- **No real adapters.** Provider-reported usage, cost and real latency
  arrive with H08–H11. The contracts here are ready for them; v1 reports
  them absent.

## Decisions

### V1. The pipeline: execution produces the subject, evaluation measures it

```text
ResolvedBenchmarkSuite ─verify_suite_content(source)→ ok
TaskSampling + seed ─assign_tasks→ TaskAssignment ─verify_assignment→ ok
for each assigned task, in assignment order:
  bind_task(task) + harness ─HarnessCompiler.compile→ AgentExecutionSpec (task-bound, recording on)
  ─ScriptedHarnessRunner.run→ HarnessRunResult + sealed trajectory (ADR-021)
assemble_evidence(...) ─verify_run_evidence→ HarnessRunEvidence          one per execution replicate
                                                  (subject: agent_run_evidence)
EvaluationSpec(harness-outcome | harness-judge) ─supports→resolve→supports→ recorded
EvaluationRun(subject, seed, replicate) → EvaluationAttempt → prepare → execute (verify_run_evidence first)
  → EvaluationResult (run-level MetricResults) + HarnessEvaluationReport (per-task observations)
summarize(results of one candidate) → MetricSummary per metric               (V8)
compare(A, B, ComparisonSpec) → HarnessComparisonReport                     (V9, V10)
```

- Harness execution and evaluation are separate workloads. The evaluator
  never runs the harness, and the harness never runs or sees the evaluator.
- An evaluation starts only on sealed evidence. Nothing evaluation reads or
  writes can flow back into an execution that has already sealed.
- One step's output is verified by the next step, never trusted (ADR-020
  B4, ADR-021 T5).

### V2. Wiring H04 tasks into H03 execution

A resolved task is carried into `AgentExecutionSpec` through existing
fields, so H03's identities don't move (ADR-020 B9; the same technique
ADR-021 T9 used for recording):

- `task` (`HarnessTaskInput`) gets the H04 task's id and
  `agent_visible(task).variables`.
- `config["task"]` holds a task binding
  (`xaytune.harness-task-binding/v1alpha1`): the suite ref (id, revision,
  `suite_fingerprint`), the task's id, revision and `task_fingerprint`, its
  position in the manifest, and the assignment's cohort.
- The compiler checks the binding against the resolved task and refuses,
  listing every reason:
  - an id or variables that differ from the task's agent-visible inputs;
  - a task fingerprint or position that disagrees with the suite;
  - **a task with attachments.** `HarnessTaskInput` has no place for them.
    Admitting them needs a new, versioned `AgentExecutionSpec` input, which
    is deferred.
- The binding is execution context. It enters the execution fingerprint and
  request digest, because `config` does. It never enters
  `HarnessFingerprint` (ADR-020 B8).
- A spec without `config["task"]` is H03's, unchanged and byte-identical.
- The trajectory's attribution (ADR-021 T3) records the task's revision,
  `task_fingerprint` and the assignment cohort, slots it already has.
- **Evaluation needs recording.** A task-bound run executed for evaluation
  must record its trajectory. A run with no sealed, verifiable trajectory
  can't be part of a subject. Only an attested, sealed `complete`
  trajectory contributes a **scored outcome**. Verified `partial` attempts
  contribute **missingness and spend** evidence only (V3).
- `verify_suite_content` runs before anything executes, and
  `verify_assignment` runs before execution and again when evidence is
  assembled. A stored assignment is a claim until it's verified.

### V3. The subject: `HarnessRunEvidence`, one per execution replicate

An `EvaluationRun` measures one subject. For a harness, the subject is
**one execution replicate**: one candidate, run at one seed and replicate
over one verified assignment. It is a new artifact,
`HarnessRunEvidence` (`xaytune.harness-run-evidence/v1alpha1`), with a new
artifact kind, `agent_run_evidence`.

It records:
- the suite ref, the sampling policy and the verified `TaskAssignment`;
- the harness fingerprint, seed and replicate;
- the execution conditions (V9's C4 facets);
- one entry per assigned task, in assignment order:
  - the task id, revision and `task_fingerprint`;
  - **every attempt instance** of that task's request digest, with its
    sealed completeness and, for a partial one, its abandonment reason.
    Attempts other than the outcome are retries. They count toward spend
    (V6), never toward the outcome;
  - **the entry's state**, exactly one of the two below.

**Two entry states, and the line between them:**

| State | Means | Requires | Observations |
|---|---|---|---|
| `attested` | One attempt published a result this attempt can attest | That attempt is the **outcome attempt**: sealed `complete`, passes `verify_trajectory` and, on the host that assembles it, `verify_publication`. The entry records its `trajectory_fingerprint`, `result_digest` and termination | Scored from the attested result (V6) |
| `no-attested-result` | The task was executed, and no attempt can attest a result | Every attempt is sealed `partial` and passes `validate_manifest` and `verify_trajectory` over its recorded prefix. This includes `post-commit-evidence-lost`, whose `unattested_result` is never read | Outcome observations absent (`no-attested-result`). Spend still counted from the recorded events |

The difference is between a **claim** and an **execution**:
- **A false or unverifiable claim of a complete outcome is refused.** It is
  never downgraded to `no-attested-result`, because a failed verification
  says the record is wrong, not that the task failed.
- **A legitimate execution with no attested result is valid evidence.**
  It's a fact about that run, with its spend, and it is never dropped.
  Dropping it would bias every aggregate toward the tasks that finished.

Its digest hashes an explicit identity projection,
`run_evidence_identity_v1` (`type="harness-run-evidence"`). That digest is
the `ArtifactRef.digest` of the subject, and so the subject half of every
reuse key (V11).

**One operation verifies evidence, wherever it is consumed:
`verify_run_evidence(evidence, suite, source, store)`.** A digest proves
only that a record is self-consistent: an edited record whose digest was
recomputed is self-consistent too. So no consumer trusts the record's own
contents. `verify_run_evidence` rebuilds every fact the record states from
independent sources, and refuses, listing every reason:

- **The suite and assignment:**
  - the suite's bytes, by `verify_suite_content`;
  - the assignment, by `verify_assignment` against the suite, the sampling
    policy and the seed;
  - one entry per assigned task, in assignment order, none missing or
    repeated;
  - at least one attempt per entry. An empty attempt list is refused,
    never read as a verified execution (G3g).
- **Complete attempt membership, read from the store, not the record:**
  - it lists every attempt instance in the store whose attribution names
    each task's request digest. The entry's attempt set must **equal** that
    list;
  - an attempt the record leaves out is refused, so a dropped retry can't
    understate spend;
  - an attempt it adds that isn't in the store, or that names another
    request digest, is refused;
  - an attempt that is still open is refused. Reconciliation (ADR-021 T4)
    runs first.
- **Attribution:** every attempt's attribution matches the entry: the task
  fingerprint, cohort, seed, replicate and harness fingerprint.
- **Trajectory integrity:** every attempt, complete or partial, passes
  `validate_manifest` and `verify_trajectory`. A failure is an integrity
  incident, never a missing observation (G3d).
- **The entry state:**
  - `attested` needs its outcome attempt sealed `complete`, and on the host
    where it was published, `verify_publication`. Elsewhere, ADR-021 T5
    applies: the sealed `complete` status stands, and no new ownership
    claim is made;
  - `no-attested-result` is refused when any attempt is sealed `complete`
    and verifies. A result can't be hidden by declaring it absent.
- **The digest:** the record's digest equals `run_evidence_identity_v1` of
  its contents.

**Who calls it:**
- **assembly**, as its last step: assembly builds a record from the store
  and then verifies it, so nothing is published that wouldn't verify;
- **every evaluator, at the start of execution, before it measures
  anything.** Not in `prepare()`, which must stay deterministic and read
  nothing outside its arguments. Each `EvaluationRun` verifies the subject
  itself and lists the store afresh. It never relies on assembly, or on
  another run, having verified the same digest.

**Its limit.** Membership is checked against the store the verifier reads.
An attempt whose whole directory was deleted can't be listed, so local
evidence can't prove its absence. Proving that needs the durable path's
attempt ordinals, where a missing ordinal is a gap (V13).

Partial trajectories are never scored. They are read only to account for
missingness and spend.

### V4. Two evaluators, on the existing protocol

Both implement `Evaluator`, and `ResolvableEvaluator` where something needs
pinning. Each is one `EvaluationSpec`, so outcome and quality are separate
`EvaluationRun`s of one subject in one cycle, each with its own determinism
(the rule `EvaluationSpec` already states).

| Evaluator | Measures | Determinism | Reads |
|---|---|---|---|
| `harness-outcome` | Task success by the declared rule, and every resource, interaction and context metric (V6) | `DETERMINISTIC` | The evidence, each outcome attempt's attested `result.json`, the trajectory events (no blobs), and each task's reference through `resolve_reference` |
| `harness-judge` | Quality, by a pinned judge model against a pinned rubric | `STOCHASTIC` | The task's agent-visible inputs, the attested answer, the rubric, and the reference only when its config declares `reference_access="reference"` and a matching `ReferenceEgressGrant` exists (V5) |

**Configuration.**
- `EvaluationSpec.evaluator.config` holds a typed, versioned config:
  `xaytune.harness-evaluation/v1alpha1`, with a variant for each evaluator.
  `EvaluationSpec.dataset` is `None`.
- **The outcome config:**
  - the benchmark suite ref and the sampling policy;
  - the metric catalog version (V6);
  - the success rule: `exact-match`, with a declared normalization:
    `none`, `strip` or `casefold-strip`;
  - the partial-evidence policy, whose only v1 value is `refuse`;
  - the timing environment label (V9), or `None`.
- **The judge config:** everything the outcome config pins, plus:
  - the judge model binding, pinned like a harness model binding (ADR-018
    D4);
  - the prompt and rubric, as digest-pinned content;
  - the integer score scale;
  - the generation settings;
  - `reference_access`: `none` or `reference`.
- `supports()` refuses a config that isn't exactly its typed model's
  canonical form, with every field explicit (the rule ADR-018 D7 set for
  harnesses). Two configs that mean the same thing therefore can't have
  two fingerprints.
- `resolve()` pins any suite alias to a `BenchmarkSuiteRef` and the judge
  model's revision. `supports()` runs again on the result.
- `supports()` also refuses `exact-match` on a suite where any task has no
  reference.

**The judge's answer is validated.** It is asked through
`invoke_agent_model` with a closed output schema: one integer on the
declared scale. An invalid answer or a failed call is the per-task absence
`judge-failed`. It is never a default score.

**Execution is local in H06.** `prepare()` returns an
`EvaluationExecutionSpec` naming a worker entrypoint, as `NativeEvaluator`
does. A local driver mints the run and attempt and walks them through
`EVALUATION_RUN_MACHINE` and `EVALUATION_ATTEMPT_MACHINE`. It executes the
prepared spec in-process, builds the `EvaluationResult` and refuses it if
`result_provenance_problems` reports anything. Durable submission through
the runtime waits for the durable harness path, because the
`EvaluationRun`'s node must exist in the repository.

### V5. Independence: held-out references never reach the harness

- **Nothing on the execution side holds a reference.** No
  `AgentExecutionSpec`, task binding, toolbox, trajectory or published
  result contains a reference's bytes or locator. The task binding carries
  `task_fingerprint`, which hashes the reference digest; it does not carry
  the digest. Only the evaluator calls `resolve_reference` (ADR-020 B2).
- **Evaluation is read-only on evidence.** The evaluator opens the
  trajectory store and evidence for reading only, and writes nothing to
  them.
- **Scores are derived from references.** `task_success` and `quality`
  are computed against held-out content, so "nothing derived from a
  reference leaves" can't be the rule. Every evaluation output instead
  belongs to exactly one of three classes:

| Class | Contents | Where it may go |
|---|---|---|
| **Restricted** | Reference bytes and locators; judge requests and replies; any free-form text that depends on a reference, such as a rationale or a diff against the reference | Only the restricted tier ADR-021 T8 defines, as its own artifact. Never into a summary, a metric, a comparison report or a provider's input |
| **Reference-derived measurements** | Numeric scores computed against a reference: per-task `task_success` and `quality`, and their run-level and summary values | Released only as the disclosure policy below allows |
| **Execution measurements** | Metrics computed from the evidence alone: tokens, turns, tool calls, context, latency, spend, retries | Unrestricted, per task and aggregated. They reveal nothing about a reference |

- **The disclosure policy.** `disclosure` is declared with the evaluation,
  recorded in its report, and kept outside `EvaluationFingerprint`,
  because it governs who sees a measurement, not what is measured:
  - **`aggregate`, the default:** run-level and summary values are
    released. Per-task reference-derived scores stay restricted, and so
    does a run-level value over a single task, since it equals that task's
    score. So does every `successful-tasks` slice, at any sample count:
    with three tasks of 1, 3 and 9 turns, a slice mean of 2 turns over two
    tasks says the first two succeeded;
  - **`per-task`:** per-task scores and `successful-tasks` slices are
    also released, to people and reports.
- **No provider receives reference-derived feedback by default.** A
  mutation or search provider never receives restricted content. It
  receives **no** per-task reference-derived score, whatever the
  disclosure, unless H07's own reviewed contract grants it explicitly. A
  per-task held-out score is feedback a provider can optimize against,
  even though it's numeric.
- **The disclosure policy limits leakage; it doesn't eliminate it.** A
  provider that is shown aggregates over many evaluations of one suite
  can still fit the held-out set. Bounding that, for example with
  evaluation budgets or a fresh suite for confirmation, is H07's.
- **Sending a reference to a judge is data egress, and needs a grant.**
  `reference_access="reference"` declares that the judge's measurement uses
  the reference, and enters `EvaluationFingerprint`. It does not permit
  sending the reference anywhere. That needs a `ReferenceEgressGrant`:
  - it names the suite ref and the judge's pinned model binding (provider
    and revision), and records who granted it, as an `Actor`, and when;
  - the host supplies it at evaluation time. `prepare()` refuses a judge
    with `reference_access="reference"` and no grant matching that suite
    and binding, and a grant for another provider or model doesn't match;
  - it is recorded with the run's report, outside `EvaluationFingerprint`,
    like the disclosure policy;
  - it is required for every judge, local or hosted. The protocol has no
    trustworthy "this never leaves the host" declaration yet;
  - issuing grants through `PolicyEngine` and approval is deferred to the
    durable path. H06 requires the grant and records it; it adds no new
    policy system.
- **A judge without the reference** (`reference_access="none"`) sees only
  the task's agent-visible inputs, the attested answer and the rubric.
- An answer equal to the reference is scored normally, and the reference
  artifact stays unreachable (ADR-021 F20).

### V6. Metrics: a versioned catalog, with units and declared absences

The catalog is `xaytune.harness-metrics/v1`. Its version is part of the
outcome config, so it is part of `EvaluationFingerprint`. A metric's
definition can't change without changing the fingerprint.

**Per task, each metric is an observation: a value, or an absence with a
reason.** Never a placeholder value.

| Family | Metric | Unit | Source | Absent when |
|---|---|---|---|---|
| Outcome | `task_success` | fraction (per task: 0 or 1) | success rule on the attested answer (below) | `no-attested-result`, `unscored-cancelled` |
| Outcome | `quality` | `rubric:<rubric digest>` points | `harness-judge` | `judge-failed`, `no-attested-result`, `unscored-cancelled` |
| Tokens | `prompt_tokens`, `output_tokens`, `total_tokens` | `tokens(<token_counter>)` | `HarnessRunResult` | `no-attested-result` |
| Tokens | `cached_tokens`, `reasoning_tokens` | provider tokens | provider usage | always in v1: `not-recorded` |
| Cost | `cost` | a currency, with a rate-card version and accounting scope | provider usage × a pinned rate card | always in v1: `not-recorded` |
| Time | `model_latency_ms` | ms; the runner's monotonic clock around each model call, summed | `model-call-finished.latency_ms` | any model call has no `latency_ms`: `not-recorded` |
| Time | `elapsed_ms` | ms, end to end | — | always in v1: `not-recorded` |
| Interaction | `turns`, `tool_calls`, `failed_tool_calls`, `refused_tool_calls` | count | `HarnessRunResult` | `no-attested-result` |
| Interaction | `retries` | count of attempt instances minus one | the evidence | never |
| Context | `context_truncations` | count | `HarnessRunResult` | `no-attested-result` |
| Context | `peak_prompt_tokens` | `tokens(<token_counter>)` | the maximum `prompt_tokens` over the model calls | `no-attested-result` |
| Context | `token_budget_utilization` | fraction | `total_tokens` / the harness's `max_total_tokens` | no token budget: `not-applicable` |
| Spend | `spend_total_tokens` | `tokens(<token_counter>)` | the outcome attempt plus every retry's recorded calls | a started call with no finish: `spend-unaccounted` |

- **The token counter is part of the unit.** H03 counts tokens by a named
  rule, not a provider tokenizer, so two counters are two units and are
  never compared (C3).
- **Cost is not estimated from counted tokens.** A cost computed from a
  rule-counted token total would look like a provider bill and wouldn't
  be one. Cost needs provider-reported usage and a pinned rate card
  (currency, rate source and version, accounting scope). The catalog
  defines the contract. v1 reports cost absent.
- **Spend includes failed attempts and retries**, as chapter 23 §9 requires.
  When a partial attempt has a started call with no finish, the spend
  metric is absent. The report records the known lower bound, never as a
  `MetricResult`.
- **Success follows the termination reason:**
  - `final-answer` or `no-tool-call`: the success rule decides;
  - `max-turns`, `token-budget`, `tool-call-budget`, `context-overflow` or
    `model-error`: 0. They are attested outcomes of the harness's own
    stopping rules or model calls (chapter 23 §11: a budget-forced
    termination is not task success);
  - `cancelled`: absent (`unscored-cancelled`). It's not an outcome of the
    candidate;
  - no attested result: absent (`no-attested-result`).
- **Successful-task and all-task values are both reported.** A metric's
  `successful-tasks` slice averages over the tasks that succeeded only.
  The unsliced metric averages over every task.

**Run-level aggregation, inside the evaluator.**
- A metric's run value is the mean over the assigned tasks, in assignment
  order, with `math.fsum`. `sample_count` is the number of tasks.
- **If any task's observation is absent, the run has no `MetricResult` for
  that metric.** The report records how many tasks were absent and why,
  and keeps every observed per-task value, under V5's disclosure policy.
  No imputation, no denominator shrunk to the tasks that happened to be
  observed.
- The `successful-tasks` slice is absent when no task succeeded.

**No schema change.**
- `MetricResult` is unchanged. Its `metadata` repeats the unit and catalog
  version for readers. Equal units are guaranteed by an equal
  `EvaluationFingerprint`, not by metadata.
- `EvaluationResult` is unchanged. A run in which every metric is absent
  produces no `EvaluationResult`, only a report, and the run fails with that
  reason.
- `HarnessEvaluationReport` (`xaytune.harness-evaluation-report/v1alpha1`)
  is attached through `EvaluationResult.artifacts`, with the existing
  `evaluation_report` kind. It holds the per-task observations, the absence
  counts and their reasons, and the spend lower bound.

### V7. Three sources of variation, kept apart

| Source | What creates one | Record | Counts as a sample |
|---|---|---|---|
| **Execution replicate** | A new run of the same candidate at a new seed or replicate | A new `HarnessRunEvidence` | Yes, of the harness's behavior |
| **Retry** | A new attempt instance for the same request digest, after an attempt failed | An attempt entry in the evidence | **Never.** It counts toward spend only |
| **Judging replicate** | A new `EvaluationRun` of the same subject and judge spec, at a new seed or replicate | A new `EvaluationResult` | Yes, of the judge's variation only |

- **Candidates are matched by seed.** Each execution replicate of each
  candidate runs at a seed from one declared seed list. ADR-020's
  `assign_tasks` depends only on the suite, policy and seed, so matched
  seeds get identical assignments (C2).
- **Judging replicates nest inside execution replicates**, and must be
  balanced: every execution replicate has the same number J of judging
  replicates, or summary refuses.
- **A replicate request is never satisfied from an existing sample** (ADR-015
  §3, ADR-017 decision 3). Not for execution, and not for a stochastic judge.
- `HarnessRunResult.seed` is a request to the harness. It doesn't claim
  the provider is deterministic.

### V8. Replicate aggregation and uncertainty

`summarize` is a pure function over one candidate's results for one
evaluation fingerprint. Its method is versioned: `replicate-mean/v1`.

For each metric:
- **Per execution replicate r**, the value `v_r` is the run value (V6). For
  a judged metric, `v_r` is the mean of its J judging runs.
- **The estimate** is the mean of `v_r` over the R execution replicates,
  in seed order, with `math.fsum`.
- **Execution uncertainty** is the standard error `sd(v_r) / sqrt(R)`, with
  the sample standard deviation. With R = 1 it is **unknown** (`None`),
  never 0.
- **Judge uncertainty**, for judged metrics with J ≥ 2, is the pooled
  within-replicate standard deviation of the judging runs, reported
  separately. The two variance sources are never added into one number.
- `R`, `J` and the tasks per replicate are recorded.
- **Balanced design required.** If any execution replicate has no
  `MetricResult` for the metric, or the replicates' task counts differ, the
  summary for that metric is absent, with the reason. A summary over
  whichever replicates happened to finish would quietly change what is
  being estimated.
- v1 reports no confidence interval. A standard error is descriptive. An
  interval would claim a distribution that v1 doesn't establish.

### V9. Comparability: C3, C4, and the comparison key

ADR-020 B7 requires all of C1–C4. H06 supplies C3 and C4, and the single
check that combines all four.

**C1, same benchmark.** Both candidates' evidence has the same
`benchmark_cohort`, and it equals the cohort of the evaluation config's
suite and sampling.

**C2, matched assignments.**
- Both candidates have exactly the same set of seeds, one execution
  replicate per seed.
- `verify_assignment` passes for every replicate of both.
- `assignment_mismatches` is empty for each seed's pair.
- **Unpaired designs are refused in v1.** ADR-020 left H06 the choice of
  refusing or declaring an unpaired design with statistics that account
  for it. v1 refuses.

**C3, same scoring and units.**
- For each evaluator used, both candidates' runs have the same
  `EvaluationFingerprint`. That covers the suite, the sampling, the rules,
  the catalog version (and so every unit) and every judge pin.
- The same determinism declaration and the same summary method.

**C4, compatible execution conditions, declared and never inferred.**
- An execution fingerprint can't serve: it covers the harness fingerprint,
  task, seed and replicate, so two candidates always differ.
- Instead, the evidence records **execution facets**: the runner's name and
  version, the compiler's name and version, the token counter, the
  entrypoint, container, dependencies and resources, the recording spec,
  and the declared timing-environment label.
- v1 has one constraint, `exact-facets/v1`: every facet is equal across
  every run of both candidates. Within one candidate, too: replicates under
  different runner versions are not one sample.
- **Time is comparable only under a declared timing environment.** Time
  metrics are compared only when both candidates' evidence carries the
  same declared timing-environment label. Otherwise each time metric is
  `not-comparable`, and the other metrics are unaffected.

**One check.** `comparability(a, b)` returns every failed condition, not the
first. A per-metric check adds that a metric absent on either side is
`not-comparable` for that metric.

**The comparison key** (`type="harness-comparison"`, version 1) is built
only when C1–C4 all hold. It hashes:
- the benchmark cohort;
- the evaluation fingerprints;
- the execution facets;
- the seed list;
- the summary method;
- the comparison rule.

Nothing else may say two harness results are comparable.

Comparing a candidate with itself is allowed. An A/A comparison is how the
noise floor is measured.

### V10. Comparison, multi-objective vectors and the existing `Objective`

**A verdict here is an exploratory decision rule, not a statistical
claim.** It says which way the evidence currently points, under a declared
threshold, for choosing what to try next. It is **not**:
- statistical significance;
- a confidence level or coverage guarantee;
- proof that one harness is better.

With few replicates the standard error is itself a noisy estimate, and
comparing many candidates multiplies the chance that some difference
crosses the threshold by luck. v1 controls neither. Formal inference and
multiple-comparison control are deferred (V13). No field, verdict or
report text may call a v1 verdict "significant", "confident" or "proven".

**A `ComparisonSpec` declares what is compared, and how.**
- **An objective vector:** an ordered list of metrics, each with a
  direction and a minimum effect. Directions are declared, never inferred
  from a metric's name.
- **A rule, `paired-difference/v1`:**
  - for each seed r, `d_r = v_A,r − v_B,r`, oriented so that positive
    favors A under the metric's direction;
  - the mean `d̄` and its standard error `se`;
  - the verdict, by `k` (declared, default 2). `k` is a decision threshold,
    not a quantile of any distribution:

| Condition | Verdict |
|---|---|
| C1–C4 fail, or the metric is absent on either side | `not-comparable` |
| R < 3 | `inconclusive`: too few replicates for even a provisional direction |
| `d̄ > max(min_effect, k·se)` | `favors-a` (provisional) |
| `−d̄ > max(min_effect, k·se)` | `favors-b` (provisional) |
| otherwise | `inconclusive` |

- `task_success` also reports wins, losses and ties, paired by seed and
  task, as counts. Which task was won is a per-task reference-derived
  score, and V5's disclosure policy governs it.

**The vector's relation**, under the same rule and the same caveat:
- `favors-a`: some metric `favors-a`, none `favors-b`, none
  `not-comparable`;
- `favors-b`: the mirror;
- `trade-off`: each side is favored on some metric;
- `inconclusive`: no metric favors either side;
- `not-comparable`: the pair fails C1–C4, or any vector metric is
  `not-comparable`.

**The report.** `HarnessComparisonReport`
(`xaytune.harness-comparison/v1alpha1`) is deterministic and fingerprinted.
It records:
- both candidates' evidence digests and evaluation result ids;
- the comparability verdict, with every reason;
- per metric, both summaries, the paired difference, its verdict, and the
  absences;
- the vector relation;
- `claim: "exploratory"`, the rule's name, R, `k` and each minimum effect,
  so a reader sees what the verdicts are and aren't.

It never scalarizes. A weighted score belongs to a provider that declares
weights and normalization (H07), and the raw vector is kept beside it.

**The existing `Objective` is unchanged.** `Objective` keeps its single
primary metric and constraints. `ThresholdDecisionEngine` keeps refusing a
metric reported more than once in a cycle. With R > 1 a harness cycle is
therefore undecidable by that engine, which is the safe answer. A test pins
it. Connecting harness evidence to decisions, through a replicate-aware
engine or an aggregate result type, is H07's or the durable path's, with its
own ADR.

### V11. Reuse

- **Evaluation reuse follows ADR-015 §3**, keyed on the subject digest (the
  evidence digest) and the `EvaluationFingerprint`:
  - `harness-outcome` is `DETERMINISTIC`, so a terminal result may be reused
    for the identical subject and fingerprint in the same experiment;
  - `harness-judge` is `STOCHASTIC`, so a sample is never reused
    implicitly.
- **Execution is never reused** (ADR-017). Every requested execution
  replicate runs.
- No cross-experiment reuse. Partial evidence is never scored or reused.
- **What invalidates a match:** a new evaluator version, a new catalog
  version, a re-pinned judge model or a suite that fails
  `verify_suite_content`. The first three change the fingerprint; the
  last refuses evaluation.
- **H06 implements no lookup.** Every requested evaluation runs. The rule
  above is what a later lookup must enforce, as ADR-015's
  `EvaluationReusePolicy` does.

### V12. Compatibility

**Unchanged, and pinned by tests:**
- `EvaluationSpec` and `evaluation_identity_v1`, and with them
  `NativeEvaluator` and `LMEvalEvaluator` fingerprints;
- `MetricResult`, `EvaluationResult`, `EvaluationRun` and
  `EvaluationAttempt`;
- `Objective`, `DecisionContext` and both threshold engines;
- H02–H05 identities, `HarnessRunResult` v1alpha1, the trajectory schema,
  and the request digest of every spec without `config["task"]`.

**No database migration.** The last migration stays 019.

**New:**
- the task binding;
- `HarnessRunEvidence`, `verify_run_evidence` and the `agent_run_evidence`
  artifact kind;
- the two evaluators and their config;
- the disclosure policy and `ReferenceEgressGrant`;
- the metric catalog;
- `HarnessEvaluationReport`;
- `summarize`, `comparability` and `compare`, and the comparison report.

### V13. Deferred

Each needs its own review:
- the durable path: persisting harness candidates, evaluation runs and
  results, submission through the runtime, and budget-ledger settlement of
  spend;
- connecting harness evidence to the decision engine;
- search, mutation and the feedback providers receive (H07);
- provider-reported usage, cost with rate cards, cached and reasoning
  tokens, tool latency and end-to-end elapsed time (H08–H11, with a
  trajectory schema version);
- task attachments in `AgentExecutionSpec`;
- unpaired comparison designs;
- formal statistical inference: confidence intervals, significance tests
  and multiple-comparison control across many candidates;
- bounding held-out overfitting through repeated evaluation (H07);
- issuing `ReferenceEgressGrant`s through `PolicyEngine` and approval;
- proving that no attempt's whole directory was deleted, through the
  durable path's attempt ordinals;
- the evaluation reuse lookup;
- scoring partial evidence;
- success rules beyond `exact-match`, such as programmatic checkers or
  tests run in a sandbox.

## Decisions on the review's questions

Decided in the architecture review of PR #79 at `ff0d2ab`:

| Question | Decision |
|---|---|
| What is a budget-stopped or `model-error` task? | **Accepted:** a failure (0), when there is an attested result. External cancellation stays unscored |
| A metric observed on only some tasks? | **Accepted:** no run value, with no imputation and no shrunken denominator. The observed per-task values are kept separately in the report |
| Unmatched assignments? | **Accepted:** refused in v1, not pooled |
| Cost from counted tokens? | **Accepted:** not estimated. Absent until provider usage and a pinned rate card exist |
| Durable evaluation? | **Accepted:** local and in-process on the existing evaluation domain contracts, with no second engine, until the durable harness path exists |
| Tasks with attachments? | **Accepted:** refused until a versioned spec input exists. Inputs are never silently omitted |
| When is a difference a claim? | **Revised:** never a statistical claim in v1. `paired-difference/v1` gives a provisional direction only with R ≥ 3, as an exploratory decision rule with a configurable k = 2 (V10) |

## Failure scenarios

The implementation must handle each scenario as stated:

| # | Scenario | Required outcome |
|---|---|---|
| G1 | A task's reference is reachable from the execution side | Impossible by construction. Tests search every spec, binding, trajectory file, published result and summary report for the reference bytes and locator |
| G2 | A judge prompt includes the reference | Stored only in the restricted report. The summary report, metrics and comparison report carry no reference content |
| G2a | A judge has `reference_access="reference"` and no `ReferenceEgressGrant`, or a grant for another provider, model or suite | `prepare()` refuses. Nothing is sent |
| G2b | Disclosure is `aggregate` | Per-task `task_success` and `quality` stay restricted, and so does a run-level value over one task. Per-task execution measurements are released |
| G2c | A provider's input is assembled from evaluation output | It contains no restricted content and no per-task reference-derived score, whatever the disclosure |
| G3 | Every attempt of a task is sealed `partial` and verifies | A valid `no-attested-result` entry. Its outcome is absent, so the run has no `task_success`; its spend is counted |
| G3a | An entry claims `attested`, but its outcome attempt is `partial`, or fails `verify_trajectory` or `verify_publication` | Assembly refuses. The entry is never downgraded to `no-attested-result` |
| G3b | An entry claims `no-attested-result`, but one attempt is sealed `complete` and verifies | Assembly refuses: a result can't be hidden |
| G3c | An attempt of the task is left out, or is still open | Assembly refuses, naming the attempt |
| G3d | A partial attempt fails `verify_trajectory` | Assembly refuses. A legitimate failure needs verifiable evidence too |
| G3e | The only attempt is `post-commit-evidence-lost`, and its recorded prefix passes `validate_manifest` and `verify_trajectory` | `no-attested-result`. Its `unattested_result` is never read or scored |
| G3e′ | The only attempt is `post-commit-evidence-lost`, and its prefix fails verification | Refused under G3d. Evidence loss is not an authorized integrity exception |
| G3g | An entry lists no attempts, in either state | `verify_run_evidence` refuses it. An empty attempt list is not an execution |
| G3f | After assembly, an evidence record is edited to leave out a retry, and its digest recomputed | `verify_run_evidence` refuses it: the store lists an attempt the record omits. The evaluator measures nothing |
| G4 | A sealed chunk is edited after assembly | `verify_run_evidence`, run by the evaluator, refuses the subject, naming the trajectory |
| G5 | An evidence record claims another task fingerprint, cohort or seed than its trajectories | Assembly refuses, listing every reason |
| G6 | Cost, or cached tokens, are requested | Absent (`not-recorded`). No `MetricResult`, no zero |
| G7 | One model call has no `latency_ms` | `model_latency_ms` is absent for that task, so there is no run value |
| G8 | A retry's partial trajectory has a call with no finish | `spend_total_tokens` is absent (`spend-unaccounted`). The report records the lower bound |
| G9 | Candidate B has no replicate at one of A's seeds | `not-comparable` (C2), naming the seed |
| G10 | The two candidates were scored with different evaluator versions, catalogs or judge pins | `not-comparable` (C3) |
| G11 | The runner version or token counter differs, between candidates or within one | `not-comparable` (C4), naming the facet |
| G12 | Time metrics without a shared declared timing environment | Each time metric `not-comparable`. The others are decided normally |
| G13 | R = 1 | Uncertainty `None`. Every verdict `inconclusive` |
| G13a | R = 2, with a difference far above `k·se` and the minimum effect | Every verdict `inconclusive` |
| G13b | R ≥ 3 and the difference exceeds `max(min_effect, k·se)` | `favors-a` or `favors-b`, provisional. The report says `claim: "exploratory"` and names the rule, R and `k` |
| G14 | A judge answer is outside the scale or doesn't parse | Absent (`judge-failed`) for that task. No default score |
| G15 | A second judging replicate is requested for a subject already judged | A new `EvaluationRun` executes. The first sample is never returned |
| G16 | A harness cycle's R results reach `ThresholdDecisionEngine` | `UndecidableError`: the metric is reported R times |
| G17 | A non-canonical evaluator config (a default omitted, an extra key) | `supports()` refuses it |
| G18 | A run where every task was `cancelled` | No `EvaluationResult`. The run fails with that reason |
| G19 | An A/A comparison | Comparable. Verdicts are typically `inconclusive` and measure the noise floor |
| G20 | A task with attachments is bound | The compiler refuses it |

## Acceptance criteria for the implementation PR

Each criterion needs positive tests, and the **negative tests listed**.

1. **Task binding.** A bound spec carries the task's agent-visible variables
   and pin. A spec without a binding keeps H03's request digest (pinned).
   Negative: different variables, a wrong fingerprint or position, a task
   with attachments.
2. **Evidence.** Assembly records every attempt, in assignment order, and
   its digest is stable. Both entry states are built and scored:
   - positive: an `attested` entry; a valid `no-attested-result` entry
     whose outcome is absent and whose spend is counted (G3); a verified
     `post-commit-evidence-lost` prefix (G3e);
   - negative: an entry with no attempts (G3g); an invalid claim of a
     complete outcome (G3a); a hidden
     result (G3b); an omitted or open attempt (G3c); an unverifiable
     partial attempt (G3d); an unverifiable `post-commit-evidence-lost`
     prefix (G3e′); G5; a missing, repeated or reordered task; an edited
     chunk (G4);
   - **evaluator side:** `verify_run_evidence` is the one operation used by
     assembly and by every evaluator. Records altered after assembly, with
     their digests recomputed, are refused before anything is measured:
     a retry left out (G3f), an attempt added, an entry changed from
     `attested` to `no-attested-result`, and another seed or cohort. A
     test checks that each evaluator calls it on every run, including a
     second run of an already-verified subject.
3. **Evaluators.** Both pass `supports` → `resolve` → `supports` and prepare
   deterministically. Negative: G17; `exact-match` on a task without a
   reference; an unpinned judge model.
4. **Independence and disclosure.** G1, G2, G2a, G2b and G2c. The
   evaluator opens evidence read-only. A grant is recorded with the report
   and is not in `EvaluationFingerprint`; `reference_access` is.
5. **Metrics.** Every catalog metric, with its unit and source, on known
   evidence. Every absence reason. Negative: G6, G7, G8, G14, G18. No test
   may find a zero standing in for an absence.
6. **Termination mapping.** Each H03 termination reason maps as V6 states.
7. **Variance sources.** A retry changes spend and never the outcome. A
   judging replicate is a new run (G15). Unbalanced judging is refused.
8. **Summary.** Known values give the expected estimate and standard error.
   R = 1 gives `None` (G13). An unbalanced design gives an absent summary.
9. **Comparability.** Every C1–C4 failure is reported, all at once.
   Negative: G9, G10, G11, G12. Positive: G19.
10. **Comparison.** Each verdict and each vector relation, on constructed
    values, including G13a and G13b. No scalarization. No report field or
    text says "significant", "confidence" or "proven".
11. **Compatibility.** Every pin in V12, including G16. The last migration
    is still 019.
12. **Validation with two configurations.** A small pinned static suite,
    two harness configurations that differ in one behavior field, matched
    seeds and at least three execution replicates, recorded trajectories,
    and a comparison report that explains success, resource use and
    comparability. With the scripted model this demonstrates the pipeline,
    not real-world quality, and the report says so. A second report on a
    deliberately mismatched pair is `not-comparable`, with its reasons.

## Consequences

- A harness candidate's runs can be measured, summarized with honest
  uncertainty and compared, with every condition for comparability checked
  and every missing value shown as missing.
- Existing evaluation and decision contracts don't change, so nothing
  already recorded moves.
- Comparisons stay local and in-process until the durable path exists, and
  cost and real latency stay absent until real adapters report them. The
  first comparisons will mostly show success, tokens and interaction.
- Conservative v1 rules (balanced designs only, no unpaired comparisons, no
  intervals, R ≥ 3 for any direction) will report `inconclusive` or
  `not-comparable` more often than a looser design would. A direction they
  do report is a reason to try something next, not proof of improvement. That is deliberate: a missed claim can be
  revisited; a false one becomes a decision.

## Alternatives considered

- **A new evaluation engine for agents:** rejected. The evaluator protocol,
  the run and attempt lifecycle and the result provenance checks already
  express everything needed; only the subject and the metrics are new.
- **One `EvaluationRun` per task trajectory:** rejected. A cycle would hold
  one result per task, so the per-task values would need an aggregation
  outside any run, with no provenance, and the threshold engine would
  refuse every metric. One run per execution replicate keeps per-task
  values in the report and run values in the result.
- **A new evaluation identity projection with a benchmark field:** rejected
  for v1. The evaluator's typed config is where the existing evaluators put
  what they measure, and `evaluation_identity_v1` already covers it. Every
  existing fingerprint stays put.
- **Adding `unit` and an optional value to `MetricResult`:** rejected for
  v1. It's a persisted schema with a migration. Units are fixed by the
  fingerprint, and absence lives in the report.
- **Zero for unrecorded cost, tokens or latency:** rejected (chapter 23 §9).
- **Estimating cost from counted tokens and list prices:** rejected. The
  counter isn't the provider's tokenizer, so the number would look like a
  bill and not be one.
- **Pooling unmatched assignments:** rejected. It mixes task difficulty into
  the candidate difference (ADR-020 B7).
- **Inferring execution compatibility from fingerprints:** rejected. An
  execution fingerprint always differs between candidates, and an equal one
  still wouldn't say which differences matter (chapter 23 §4).
- **Trusting an evidence record's own digest:** rejected. A digest proves
  self-consistency only, and an edited record can be re-hashed. The
  evaluator re-derives membership, attribution and integrity from the
  store and the suite.
- **Downgrading a failed verification to `no-attested-result`:** rejected.
  A record that doesn't verify is wrong, not a failed task. Scoring it as a
  failure would turn tampering or a bug into a measurement.
- **Dropping tasks without an attested result:** rejected. The remaining
  tasks would be the ones that finished, and their spend would vanish.
- **No reference-derived output outside the restricted tier:** rejected.
  Every score is reference-derived, so nothing could be measured. The
  boundary is drawn at raw references and free text instead, with
  disclosure governing the numbers.
- **`reference_access` as permission to send the reference:** rejected.
  What a measurement uses and where data may go are different questions,
  answered by different people.
- **Calling a `k·se` threshold significance:** rejected. With few
  replicates the standard error is itself noisy, and many comparisons
  inflate false directions. v1 claims a direction, not a probability.
- **A single scalar score:** rejected. It discards the vector and hides the
  trade-off that a multi-objective comparison exists to show.
- **Replacing `Objective` with a multi-objective contract now:** rejected.
  Nothing yet decides harness candidates, and chapter 23 §9 keeps the
  current engine unchanged until H07 or later.
