# ADR-020 — Pinned benchmark and task contracts

## Status

Accepted — 2026-10-11. Proposed 2026-10-11 with H04.

The project owner approved decisions **B1–B9**, including B7's comparison
conditions C1–C4, as implemented in
[PR #76](https://github.com/szaher/xaytune/pull/76) at head `d1eb95e`. That
head includes review 1's fixes: `verify_assignment`, and the split between
structural and byte verification.

**Still deferred, and not decided here:**
- trajectories (H05);
- wiring resolved tasks into execution, scoring and `EvaluationFingerprint`
  (H06), with C3 and C4;
- persistence of resolved manifests and assignments.

Each needs its own review.

Builds on ADR-018 (accepted), which kept a candidate's behavior apart from the
benchmark that measures it, and ADR-019 (E1–E10 accepted), whose
`HarnessTaskInput` is a placeholder for the task contract decided here.

## Context

H03 can run a harness on one task. Nothing yet says **which** tasks, or proves
that two runs saw the same ones. A comparison between harness candidates is
meaningful only if both ran on the same pinned inputs, and were sampled and
scored the same way. Chapter 23 §7 lists the requirements:

- tasks have stable namespaced ids **and** a resolved revision and content
  digest, because an id alone proves nothing;
- a suite manifest pins ordered task membership and input digests;
- requested and resolved specs are both retained, mutable aliases are pinned
  before comparison, and an alias that moves later changes nothing already
  pinned;
- held-out scoring inputs stay out of the agent's reach;
- a suite change starts a new comparison cohort;
- dynamic services need snapshot or replay evidence, or an explicit weaker
  declaration. ADR-018 question 6 left non-snapshot admission open for H04.

The identity roles are fixed already (ADR-018, chapter 23 §4). Benchmark
membership, sampling, held-out inputs and scoring belong to **evaluation and
comparison identity**. Changing the benchmark does not change
`HarnessFingerprint`.

## Decisions

**B1. `TaskSpec` (`xaytune.task/v1alpha1`) is a pinned, self-describing
task.** It has:

- **`id`:** a lowercase namespaced name, such as `support/refund-001`.
- **`revision`:** a pinned revision. Moving aliases are refused, as they are
  for model bindings (ADR-018 D4).
- **`inputs`:** what the agent may see. That is string `variables` for its
  task template, plus named `attachments` as content (inline, or a
  `ContentRef` pinned by digest).
- **`output`:** the output contract. v1 supports a single text answer.
- **`reference`:** an optional held-out reference, as a `ContentRef` only.
  It is never inline, so a task record never carries the answer itself.
- **`metadata`:** annotations, never identity.

`task_fingerprint` hashes the explicit projection `task_identity_v1`
(`type="task"`). It covers the id, the revision, the variables, attachment
content **by digest** under its name, the output contract and the reference
digest. Metadata and locators are excluded. Every field is required, as in
ADR-018 D7.

**B2. Held-out references never reach the agent.** `agent_visible(task)`
returns only `inputs`. Anything that hands a task to an agent goes through it.
A reference is resolved only by the explicit evaluation-role function
`resolve_reference`, never by `resolve_task_inputs`. H04 verifies reference
bytes but does not score them; scoring is H06.

**B3. `BenchmarkSuiteManifest` (`xaytune.benchmark-suite/v1alpha1`) pins an
ordered membership.** It has:

- the suite `id` and a pinned `revision`;
- the `provider` (name and version);
- `members`: an ordered list of `(task id, task revision, task_fingerprint)`.
  Ids are unique, and **order is identity**, because samplers and reports
  index by position;
- `reproducibility` (B6), and an environment snapshot when it is `snapshot`;
- `reset`, the reset semantics (B6);
- `metadata`.

`suite_fingerprint` hashes `suite_identity_v1` (`type="benchmark-suite"`).
That covers the id, revision, provider, ordered members, reproducibility,
snapshot digest and reset semantics. Metadata is excluded.

**B4. Resolution is a recorded step, verified end to end.**

- **Two kinds of reference.** A `BenchmarkSuiteRequest(suite_id, revision)`
  is what was asked for, and its `revision` may be an alias. A
  `BenchmarkSuiteRef(suite_id, revision, suite_fingerprint)` is what it
  resolved to: a pin.
- **What the resolver checks.** `resolve_benchmark_suite(request, source)`
  asks a host-supplied `BenchmarkSource` for the manifest, then every member
  task, then every reachable attachment. It refuses, listing every reason:
  - a manifest whose id, revision or fingerprint differs from what was asked
    or pinned;
  - an alias left unresolved;
  - a member whose fetched task has a different id, revision or fingerprint;
  - attachment bytes that don't match their digest, aren't `text/*` or
    aren't UTF-8.
- **The result.** It is a `ResolvedBenchmarkSuite`, keeping both the request
  and the ref. Its validators recompute every fingerprint and the membership
  order whenever it is built or parsed.
- **Two levels of verification.** A `ResolvedBenchmarkSuite` carries
  fingerprints and digests, not bytes, so the two checks are separate:
  - **Structural, on load.** Parsing a resolved record proves it is
    internally consistent: ref, manifest fingerprint, and every task's
    position and fingerprint. It cannot prove that the content those digests
    name is still held or unchanged.
  - **Bytes, against a source.** Resolution checks every attachment,
    reference and snapshot once. `verify_suite_content(suite, source)`
    repeats the check for a record that was stored, sent between processes
    or kept. Whatever executes or evaluates on a loaded suite (H06) must
    repeat it first. H04 adds no persistence.
- **Re-resolving a pin.** Resolving a `BenchmarkSuiteRef` again must produce
  the same `suite_fingerprint`, or it is refused. An upstream alias that has
  moved cannot change a pinned suite.
- **No source fetches anything itself.** H04 ships only an in-memory
  `MappingBenchmarkSource`.

**B5. Sampling is part of evaluation identity. Assignments are run
provenance.** `TaskSampling` is one of:

- `AllTasks`, in manifest order;
- `FirstTasks(n)`;
- `SeededSample(n)`.

`assign_tasks(suite, sampling, seed)` is deterministic and uses no Python
`random`. A seeded sample ranks members by `sha256(seed, suite fingerprint,
task id)` and keeps the first *n*, in manifest order. The result is stable
across processes, platforms and Python versions. The policy is part of the
comparison cohort (B7). The concrete assignment and its seed are recorded per
run, and never enter `HarnessFingerprint`.

**B6. Reproducibility is declared, and v1 admits only what it can prove.**
`reproducibility` is one of:

- **`static`:** the tasks' pinned inputs are everything the agent can
  observe;
- **`snapshot`:** the environment's state is a pinned `ContentRef`,
  verified like any other content.

**Reset semantics are recorded, not implied.** `reset` is required, and v1
has one value: `per-task`. Every task starts from the pinned inputs, or from
the snapshot, so no task sees another's effects. A later mode, such as state
shared across a suite, is therefore a visible change of suite identity and
cohort, never a silent one.

Dynamic services without a snapshot are **refused in v1**. That includes a
declared-weaker mode. Admitting them needs replay evidence (H05) and a
comparison rule that keeps weaker cohorts apart, and neither exists yet. This
closes ADR-018 question 6 for v1, and leaves a versioned way to open it later.

**B7. A comparison needs every condition. H04 supplies two of them and
decides none.** Two results are comparable only if **all** of these hold:

| # | Condition | Carried by | Owner |
|---|---|---|---|
| C1 | Same benchmark: suite, members and order, reproducibility, environment snapshot, reset semantics, and sampling policy | `benchmark_cohort(ref, sampling)` | H04 |
| C2 | Matched task assignments: the same replicate seed and the same tasks, compared task by task | `TaskAssignment` + `assignment_mismatches` | H04 |
| C3 | Same scoring rules and measurement units | `EvaluationFingerprint` | H06 |
| C4 | Compatible execution conditions, declared, not inferred from a fingerprint (chapter 23 §4) | Execution identity + a declared comparability constraint | H06 / H07 |

- **C1 is necessary, not sufficient.** `benchmark_cohort` is a digest of the
  suite fingerprint and the sampling policy (`type="benchmark-cohort"`). The
  seed is excluded, so independent replicates share a cohort. Equal cohorts
  say only that C1 holds. No H04 function, name or document treats it as
  "comparable".
- **C2: matched assignments.**
  - `assign_tasks` depends on the suite, the policy and the seed, never on
    the candidate. So candidates run under the **same replicate seeds**
    receive identical assignments.
  - `record_assignment` records each run's assignment as run provenance:
    cohort, seed and ordered task ids, each id once.
  - **A recorded assignment is a claim until it is verified.**
    `verify_assignment(suite, sampling, assignment)` recomputes the cohort
    and the selected tasks. It refuses, listing every reason:
    - a cohort from another suite or policy;
    - ids that aren't members, or that repeat;
    - ids out of manifest order;
    - any set another seed would assign.
    For `AllTasks` and `FirstTasks` the seed selects nothing, so it's only
    the replicate label.
  - `assignment_mismatches(a, b)` reports every way two assignments differ.
    It compares claims, and two fabricated records can agree, so callers
    verify **both** records first.
  - By default, comparisons are **paired by replicate seed and compared task
    by task**. A comparison of unmatched assignments is not silently pooled.
    H06 must either refuse it or apply a declared unpaired design whose
    statistics account for the sampling difference. H04 reports the mismatch;
    it does not choose the statistics.
- **C3 and C4 belong to H06/H07.** Until a comparison key combines C1–C4,
  nothing may claim that two harness results are comparable. H04 wires
  nothing into `evaluation_identity_v1`.

**B8. Benchmarks are not harness identity.** Nothing in H04 touches
`AgentHarnessSpec` or `harness_identity_v1`, so the same harness on a
different suite keeps its `HarnessFingerprint`, and a test pins that. A
behavior-defining environment bound into **scientific** identity (chapter 23
`BehaviorEnvironmentContract`) needs a new harness schema version. It is
deferred, and not decided here.

**B9. Integration is deferred.** H04 changes no existing module.

- `AgentExecutionSpec` keeps H03's `HarnessTaskInput`, so no H03 identity
  moves.
- Carrying a resolved task (its fingerprint and verified inputs) into the
  spec is a versioned spec change. It belongs with the evaluation execution
  path (H06), together with persistence of resolved manifests. H05
  trajectories will reference task fingerprints.

## Consequences

- A benchmark is now a pinned, verifiable object. Two runs whose verified
  assignments match can be shown to have seen identical inputs. Comparability
  itself waits for H06's scoring and execution conditions.
- Held-out references are kept out of agent inputs by construction, not by
  convention.
- Nothing runs, scores or persists a benchmark yet. Non-snapshot environments
  are refused rather than admitted with weaker guarantees.

## Alternatives considered

- **Task ids without digests:** rejected. Chapter 23 §7 says an id does not
  prove that the contents are unchanged.
- **Unordered membership:** rejected. Positional sampling and reports would
  differ between two "equal" suites.
- **A single "comparison cohort" key that implies comparability:** rejected
  after review. A shared suite and policy are necessary, not sufficient:
  assignments can differ, and scoring, units and execution conditions are
  separate conditions (B7).
- **Python `random` for sampling:** rejected. Its algorithm isn't a stable
  contract across versions, and the assignment must reproduce forever.
- **Inline reference answers:** rejected. They would be serialized wherever
  the task goes, including into agent-facing payloads.
- **Admitting dynamic services under a weaker declaration now:** rejected for
  v1. Without replay evidence and a cohort rule, weaker and exact results
  would mix.
- **Binding the suite into `HarnessFingerprint`:** rejected (ADR-018). The
  same harness on another benchmark is the same hypothesis.
