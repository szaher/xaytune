# 24 — Studio experience: journeys, information architecture and wireframes

Status: **Accepted** with [ADR-023](adrs/ADR-023-studio-read-api-and-web-client.md),
2026-10-14 (review of #82 at `bb79049`). This chapter is the design input for UX-02 (React client) and
UX-03 (evaluation views). ADR-023 decides the API and the technical
boundaries. This chapter decides what a person sees and in what order.

Studio V1 is **read-only and built on real data**. Every screen below is
backed by an ADR-023 §S6 endpoint over the durable control-plane store. No
screen uses mocked experiments. A screen whose data is not durable yet
(ADR-023 §S12) says so in the screen rather than faking it.

---

## 1. Who uses Studio, and for what

| Person | Wants to know | Today they |
|---|---|---|
| **Experimenter**: runs training or harness experiments | Is my experiment running? What did it try, what won, and why? | Run `xaytune status` / `watch` / `results` and read JSON |
| **Reviewer**: checks someone else's result before it is used | Is the winner's evidence sound: comparable runs, a decision engine and inputs recorded, no failures hidden? | Read the database or the code |
| **Operator**: keeps the daemon and runtimes healthy | What failed, what recovered, what is waiting for approval, how much budget is left? | Read events and incidents in the CLI |

Studio V1 serves all three without Python or the CLI. It changes nothing.
Actions (approve, cancel, propose) come with UX-04 through the daemon mailbox.

## 2. User journeys

Each journey lists its steps, then the screen and endpoint behind each step.

### J1. "What is running right now?" (operator, experimenter)

1. Open Studio → **Overview**: active experiments, anything waiting for
   approval, recent failures and incidents, and whether a daemon holds the
   lease.
   - `GET /meta`, `GET /experiments?status=active`
2. Click an active experiment → **Experiment / Overview** tab, live while it
   is not at rest.
   - `GET /experiments/{id}`, polling with the event-sequence header
3. See its next stage ("training", "evaluation", "decision", "action-approval",
   …) as the controller reports it, not inferred.

### J2. "What did my experiment try, and what won?" (experimenter)

1. **Experiments** → filter and sort the explorer → open one.
2. **Candidates** tab: the candidate graph. Each node shows its status,
   decision outcome and primary objective value. The best node is marked.
   Edges show who created the child (`created_by`: planner, agent or
   human) and, for a branched node, its recorded `branch_origin` mutation.
   - `GET /experiments/{id}/graph`
3. Click a node → **side panel**: hypothesis, reason, what changed from its
   parent, its runs and evaluation results. Open it → **Candidate detail**.
   - `GET /nodes/{id}`
4. Select two nodes → **Compare**.
   - `GET /compare?left=&right=`

### J3. "Why did this candidate win (or lose)?" (reviewer)

1. **Candidate detail** → **Decision** section: outcome, the engine and its
   version, the reason, and the exact evaluation results it read, each
   linked.
   - `GET /nodes/{id}`, `GET /experiments/{id}/decisions`
2. Follow a result → **Evaluation run**: evaluator, fingerprint, subject,
   attempts, metrics with sample counts and uncertainty. An absent value is
   shown as absent with its reason, never as `0`.
   - `GET /evaluation-runs/{id}`
3. **Compare** against the parent or a sibling: how the two relate in the
   lineage, what differs in the candidate spec, and the metrics side by side.
   For each metric: was it measured under the same evaluation on both
   sides? A difference is shown only when it was (ADR-023 §S6).

### J4. "What went wrong with this run?" (operator)

1. **Experiment / Activity** tab: the event journal, filterable by aggregate
   and event type. Incidents and recovery episodes are grouped.
   - `GET /experiments/{id}/events`, `GET /experiments/{id}/incidents`
2. Click a run → **Run detail**: attempts as a timeline (queued → running →
   preempted → recovering → succeeded), with checkpoints, incidents,
   interventions and directives placed on it.
   - `GET /runs/{id}`, `GET /attempts/{id}`
3. See the recovery plan that was chosen, the action it proposed and its
   policy decision.
   - `GET /experiments/{id}/actions`

### J5. "Can I trust this harness comparison?" (reviewer; UX-03, after the durable harness evaluation record exists)

1. **Comparisons**: two harness candidates' summaries, C1–C4 comparability
   (every failure listed), per-metric verdicts (favors-a / favors-b /
   inconclusive / not-comparable) and the vector relation. No single score
   and no "significant".
2. Withheld values (ADR-022 V5 `aggregate`) are shown as **withheld by
   aggregate disclosure**, never hidden and never zero.
3. **Trajectories**: open one task's verified, released-tier trajectory, with
   its completeness and replay level.

### J6. "How much budget is left, and what is waiting for me?" (experimenter, operator)

1. **Experiment / Budget**: each limited dimension's limit, reserved,
   committed, consumed and remaining, and the ledger.
2. **Overview → Waiting for approval**: actions in `approval_pending`, with
   the policy decision that required approval. Approving them is UX-04;
   until then the screen shows the CLI command to run.

## 3. Information architecture

```text
Studio
├── Overview                         J1, J6
├── Experiments                      J2
│   └── Experiment  /experiments/:id
│       ├── Overview                 status, objective, best candidate, next stage, budget summary
│       ├── Candidates               candidate graph  (+ side panel)
│       ├── Runs                     every run and evaluation run of the experiment, as a table
│       ├── Decisions                decisions, in order
│       ├── Actions                  governed actions + policy decisions
│       ├── Activity                 event journal, incidents, recovery
│       ├── Budget                   ledger and balances
│       └── Configuration            the recorded specs: objective, planner, compiler, runtime, evaluation, resilience
├── Candidate      /experiments/:id/candidates/:nodeId     J2, J3
├── Run            /runs/:runId                             J4
├── Evaluation run /evaluation-runs/:id                     J3
├── Compare        /compare?left=&right=                    J2, J3, J5
├── Monitoring     (cross-experiment failures, incidents; charts after the metric store, ADR-023 §S12)
├── Trajectories   (UX-03, after the durable harness record)
└── Optimization   (UX-04, after H07)
```

**Navigation rules**

- The left rail holds the top-level sections. Monitoring, Trajectories and
  Optimization are present from V1 but show what they are waiting for
  (§6), so the product's shape is visible and nothing looks broken.
- Breadcrumbs always show Experiment → Candidate → Run, so you can go back
  up the lineage from anywhere.
- Every id is a link and is copyable. Fingerprints show abbreviated, with
  the full value on hover and copy.
- **The URL is the state**: tab, filters, sort, selected node, compare pair
  and event cursor. Reload or restart, and you see the same thing.

**Status language.** Statuses are shown exactly as recorded (`deciding`,
`approval_pending`), each with a fixed color and icon, and are never renamed.
Colors are consistent across aggregates: running = blue, succeeded or
completed = green, failed = red, cancelled or rejected = grey, waiting =
amber. An unknown status (a newer server) renders as itself in neutral.

## 4. Wireframes

### 4.1 Overview

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│ ◆ Xaytune Studio                       state.db · migration 019 · daemon ● live  │
├────────────┬─────────────────────────────────────────────────────────────────────┤
│ Overview   │  Active experiments (3)                                  view all → │
│ Experiments│  ┌──────────────────────┬───────────┬─────────────┬──────────────┐  │
│ Compare    │  │ Name                 │ Status    │ Next stage  │ Updated      │  │
│ Monitoring │  │ sft-lora-sweep       │ ● active  │ evaluation  │ 12 s ago     │  │
│ Trajectories│ │ dpo-ablation         │ ● active  │ approval    │ 3 min ago    │  │
│ Optimization│ │ qwen-recovery-test   │ ● paused  │ —           │ 1 h ago      │  │
│            │  └──────────────────────┴───────────┴─────────────┴──────────────┘  │
│            │                                                                     │
│            │  Waiting for you (1)                 Recent failures (2)           │
│            │  ┌────────────────────────────────┐  ┌──────────────────────────┐  │
│            │  │ ◐ reduce-lr on run r-7f…        │  │ ✕ attempt a-91… OOM      │  │
│            │  │   policy: require-approval      │  │   recovered → attempt 2  │  │
│            │  │   $ xaytune approve act-3c…     │  │ ✕ eval e-22… failed      │  │
│            │  └────────────────────────────────┘  └──────────────────────────┘  │
└────────────┴─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Experiment Explorer

```text
┌ Experiments ─────────────────────────────────────────────────────────────────────┐
│ [Search name/id…]  Status: [all ▾]  Objective: [any ▾]          123 experiments  │
├──────────────────────┬───────────┬──────────────────┬──────┬───────┬─────────────┤
│ Name ▲               │ Status    │ Objective        │ Cand.│ Best  │ Created     │
├──────────────────────┼───────────┼──────────────────┼──────┼───────┼─────────────┤
│ sft-lora-sweep       │ ● active  │ max eval/acc ≥.8 │ 7 ▪▪▪│ n-4a… │ 2026-10-14  │
│ dpo-ablation         │ ● active  │ min eval/loss    │ 3 ▪▪ │ —     │ 2026-10-13  │
│ baseline-qwen        │ ✓ success │ max eval/acc     │ 2 ▪  │ n-19… │ 2026-10-10  │
│ …                                                                               │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                        [← newer]   [older →]    │
└──────────────────────────────────────────────────────────────────────────────────┘
 Cand. = candidate count with a mini bar of node statuses (completed / rejected / active / failed).
 Paging is keyset ("older →"), not page numbers: the list is live.
```

### 4.3 Candidate Graph (Experiment → Candidates)

```text
┌ sft-lora-sweep ─ ● active ─ next: evaluation ─────────────────────────────────────┐
│ Overview  [Candidates]  Runs  Decisions  Actions  Activity  Budget  Configuration │
├───────────────────────────────────────────────────────┬──────────────────────────┤
│                                                       │ n-4a… ★ best             │
│      ┌──────────────┐                                 │ ✓ completed · cycle 1    │
│      │ n-01 root    │                                 │ Hypothesis               │
│      │ ✓ completed  │                                 │  "rank 16 recovers the   │
│      │ acc 0.71     │                                 │   lost accuracy"         │
│      └──────┬───────┘                                 │ Changed from n-01        │
│      planner│        planner                          │  lora.r   8 → 16         │
│     ┌───────┴──────┐ ┌──────────────┐                 │ Decision                 │
│     │ n-4a… ★      │ │ n-7c…        │                 │  completed · threshold   │
│     │ ✓ completed  │ │ ✕ rejected   │                 │  engine v1 · acc .82≥.80 │
│     │ acc 0.82     │ │ acc 0.69     │                 │ Runs (1)  Evals (1)      │
│     └──────────────┘ └──────┬───────┘                 │ [Open candidate →]       │
│                       human │                         │ [Compare with parent]    │
│                      ┌──────┴───────┐                 │                          │
│                      │ n-9e…        │                 │                          │
│                      │ ● evaluating │                 │                          │
│                      └──────────────┘                 │                          │
│ [Fit] [Layout: top-down ▾] [Color: status ▾]  ⇧-click two nodes to compare        │
└───────────────────────────────────────────────────────┴──────────────────────────┘
 Node label: short id, status, primary objective value (or "—, not evaluated").
 Edge label: the child's creator (`created_by`); hovering shows its `branch_origin` mutation. Layout: ELK layered; a node with two parents shows both edges.
```

### 4.4 Candidate detail

```text
┌ sft-lora-sweep › n-4a… ★ ────────────────────────────────────────────────────────┐
│ ✓ completed · cycle 1 · fingerprint sha256:9c1e…  ⧉ · created by planner         │
├──────────────────────────────────────────────────────────────────────────────────┤
│ Lineage   n-01 ─► n-4a…                       Candidate spec   [diff vs parent ▾] │
│                                                 training.kind   sft              │
│ Runs                                            lora.r          8 → 16           │
│  r-7f…  seed 11  ✓ succeeded  2 attempts  →     optimizer.lr    2e-4             │
│                                                                                  │
│ Evaluations                                                                      │
│  e-22…  lm-eval hellaswag  ✓  acc 0.82 ±0.01 (n=10042)  acc_norm 0.85  →          │
│                                                                                  │
│ Decision  completed · threshold v1 · "acc 0.82 ≥ target 0.80" · reads e-22… →    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 4.5 Run detail

```text
┌ Run r-7f… · node n-4a… · seed 11 (inherited) · ✓ succeeded ──────────────────────┐
│ Attempts                                                                         │
│  #1 ├─queued─┼─starting─┼──────running──────┼✕ preempted                          │
│     │        │          │  ◆ckpt step 400   │  ⚠ incident: OOM                    │
│  #2                                  ├─starting─┼───recovering───┼──running──┼✓  │
│                                      │ resumed from ckpt step 400   ◆ckpt 800    │
│  ─────────────────────────────────────────────────────────────────────── time ─► │
├──────────────────────────────────────────────────────────────────────────────────┤
│ Attempt #2   runtime local · workload w-55…   execution fp sha256:3b0a… ⧉        │
│  Overrides   micro_batch_size 8 → 4 (recovery plan p-12…, action act-3c…)        │
│  Checkpoints step 400 (restored) · step 800 (committed, resume: exact)           │
│  Incidents   —                                                                   │
│  Metrics     Not recorded yet. Training metric series need the metric store      │
│              (ADR-023 §S12). Status, checkpoints and incidents are complete.     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 4.6 Evaluation Comparison

```text
┌ Compare  n-4a… ★  vs  n-7c… ─────────────────────────────────────────────────────┐
│ Lineage   distinct candidates · nearest common ancestor n-01                     │
│           (lineage only: it does not make the measurements comparable)           │
├─────────────────────────┬────────────────────┬────────────────────┬──────────────┤
│ Metric (evaluator·slice)│ n-4a…              │ n-7c…              │ a − b        │
├─────────────────────────┼────────────────────┼────────────────────┼──────────────┤
│ acc (lm-eval · —)       │ 0.82 ±0.01 n=10k   │ 0.69 ±0.01 n=10k   │ +0.13        │
│                         │ eval 9c1e…         │ eval 9c1e…         │ ✓ same eval  │
│ acc_norm (lm-eval · —)  │ 0.85               │ 0.72               │ +0.13        │
│                         │ eval 9c1e…         │ eval 9c1e…         │ ✓ same eval  │
│ eval/loss (native · —)  │ 1.21  eval 3b0a…   │ 1.34  eval 7d2f…   │ ⚠ not        │
│                         │                    │                    │ comparable:  │
│                         │                    │                    │ different    │
│                         │                    │                    │ evaluations  │
│ task_success (h · —)    │ 0.71 ratio         │ — not measured     │ ⚠ one side   │
├─────────────────────────┴────────────────────┴────────────────────┴──────────────┤
│ Candidate spec difference          lora.r  16 | 8                                │
│ Decisions                          completed | rejected (acc 0.69 < 0.80)        │
└──────────────────────────────────────────────────────────────────────────────────┘
 Each value shows the evaluation fingerprint (abbreviated, full on hover) and unit
 it was measured under. A difference appears only when both sides share an
 evaluation fingerprint and unit; otherwise the cell says why, and the two values
 are never subtracted, even visually (no bar or sparkline across them).
 A difference is never labelled "better"; the objective's direction is shown in the
 header.
 Harness candidates (UX-03): this table becomes ADR-022's comparison, read from
 H06's records: C1–C4 with every failure, per-metric verdict favors-a / favors-b /
 inconclusive / not-comparable, the vector relation, and "withheld by aggregate
 disclosure" cells. The Studio API does not reimplement C1–C4.
```

## 5. Interaction and presentation rules

- **Absent is never zero.** One component renders every missing value as
  `—` with its reason on hover (`not-applicable`, `not-observed`, withheld by
  disclosure, not evaluated). It is tested never to print `0` for an absent
  value.
- **No editorialising.** Studio shows recorded outcomes, deltas and
  verdicts. It does not label something "better", "significant" or
  "regressed" unless a recorded decision or an ADR-022 verdict says so.
- **Free text is text.** Hypotheses, reasons, planner output and metadata
  render as plain text, never HTML or Markdown (ADR-023 §S10).
- **Live but calm.** A view polls while its experiment is not at rest, and
  shows "updated N s ago". New events append without moving what you are
  reading.
- **Accessible.** Keyboard navigation for tables and the graph (arrow keys
  move between nodes), status conveyed by icon and text as well as color, and
  WCAG AA contrast.
- **Scale targets for V1.** 1,000 experiments in the explorer (paged),
  500 nodes in a graph (virtualized rendering, ELK layout in a worker), and
  10,000 events per experiment (paged).

## 6. What each V1 section shows before its backend exists

| Section | V1 shows | Fills in when |
|---|---|---|
| Monitoring | Cross-experiment failures, incidents, recovery episodes and approvals | Charts after the metric-point store (ADR-023 §S12) |
| Comparisons (harness) | Training candidate comparison | ADR-022 views after the durable harness evaluation record |
| Trajectories | An explanation and a link to the ADR | The durable harness attempt records its trajectory |
| Optimization | Objective, planner spec and the planner's invocation history (model, status, proposal fingerprint) | H07 search/mutation history and UX-04 controls |

## 7. Delivery

| PR | Delivers | Depends on |
|---|---|---|
| This PR | ADR-023 (proposed) and this chapter | — |
| UX-01 | `xaytune/api/` read-only `/api/v1`, OpenAPI document, `list_experiments`, end-to-end real-data test, Gradio tests in CI | ADR-023 accepted |
| UX-02 | `web/` React client: shell, Overview, Explorer, Candidate Graph, Candidate, Run, Compare; packaging; `web` CI job; Playwright smoke test on a real state database | UX-01's OpenAPI |
| UX-03 | Evaluation views: ADR-022 comparison, disclosure rendering, trajectory inspector | UX-02 and the durable harness evaluation record |
| UX-04 | Controls through the daemon mailbox; optimization history | H07 governance |
| UX-05 | Authentication, multi-user, remote bind, Kubernetes views | Its own ADR |
