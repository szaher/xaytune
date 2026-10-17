# ADR-023 — Studio: a read-only API over the durable store, and a web client

## Status

Accepted — 2026-10-14. Proposed the same day with
[chapter 24](../24-studio-experience.md) (user journeys, information
architecture and wireframes), and accepted after the architecture review of
[PR #82](https://github.com/szaher/xaytune/pull/82) at `bb79049`. The project
owner decided Q1–Q4 (§"Decisions on the review questions") and asked for one
clarification, recorded below and in §S6.

**Clarification at acceptance: lineage is not measurement comparability.**
`ExperimentGraph.compare(...).is_comparable` says only that two nodes are
distinct scientific candidates. It says nothing about whether their
evaluation results can be compared. `/compare` therefore reports lineage
(`lineage.distinct_candidates`) apart from a per-metric judgement. A metric pair
is comparable only when both sides were measured under the same
`evaluation_fingerprint` and the same recorded unit. Otherwise it is marked
not comparable, with a reason, and no difference is given. Harness
comparisons keep ADR-022's C1–C4 and are not reimplemented here. The
OpenAPI drift check also compares types, formats, nullability, references
and request and response schemas, not only field names (§S5).

This ADR gates UX-01 (the read-only API), UX-02 (the React client) and UX-03
(evaluation views). UX-00 (#80, `b6df566`) is a prerequisite and is merged.
The UX track was approved on 2026-10-13 (a FastAPI service in `xaytune/api/`, a
React + TypeScript + Vite client in `web/`, Gradio kept as an optional legacy
Studio) and prioritised on 2026-10-14 to run alongside H07.

Nothing here changes a domain contract, a fingerprint, a state machine, a
migration or the controller. The API reads only what the control plane
already records, through the stores that already read it.

## Context

Xaytune's durable record is the control-plane SQLite database (ADR-005) that
the daemon owns (ADR-004). It holds experiments, the candidate graph,
training runs and attempts, evaluation runs, attempts and results, decisions,
governed actions and policy decisions, the budget ledger, incidents, recovery
episodes and plans, checkpoints, interventions, agent invocations, controller
requests and the domain event journal. Today you read it through the CLI
(`xaytune status|watch|events|results|actions`) or from Python.

The existing Studio (`xaytune/studio/`, Gradio) is not built on that
record. Its `JobManager` is a second, informal control plane with its own
persistence. The spec has treated it as prior art to supersede, never to
extend, since the September architecture review. UX-00 removed its worst
exposure, but it still has no authentication and cannot show an experiment
the control plane ran.

Four facts about the current record shape what Studio V1 can show:

1. **Training experiments are fully durable.** A daemon-run experiment
   records its lineage (`experiment_edges`, `branch_origin`), every run and
   attempt, native `EvaluationResult`s, `Decision`s and the event journal,
   and all of it survives restarts.
2. **Harness candidates are not in the control plane yet.** H03 shipped a
   compiler and an in-process runner. Its durable path (ADR-019 M1–M6) is
   deferred, so no harness experiment, node or run is in the database.
3. **H06 harness evaluations are not durable yet.** `run_harness_evaluation`
   walks the evaluation state machines in memory and writes its released
   `report.json` (and an owner-only `restricted/` tier) under an output
   directory. Nothing links that report to the control plane (ADR-022 V13
   defers the durable path). H05 trajectories are likewise file stores.
4. **Training-metric time series are not durable.** Workers emit
   `MetricObserved` telemetry into the runtime's per-workload `events.jsonl`
   (ADR-014). The control plane consumes lifecycle, checkpoint, incident and
   evaluation events from it and drops metric observations, so there is no
   recorded loss curve to chart.

A Studio built today on real data can therefore show training experiments
end to end. Harness evaluations, trajectories and metric charts each need
backend work first (§S12). This ADR sets the contract so that work slots in
without changing the API's shape.

## Decisions

### S1. Studio V1 is read-only

- No V1 endpoint changes anything. The API serves `GET` and `HEAD`. Every
  other method returns `405`, enforced by one middleware rather than by each
  route's absence.
- Controls (cancel, approve, reject, propose, submit) are UX-04 and later.
  When they come, they are requests through `LocalDaemonControllerHost`'s
  mailbox (ADR-004 §9), the only write path the CLI uses. The API never gets
  its own write path and never uses `studio.JobManager`.
- Optimization controls wait for H07 to settle candidate-proposal governance.

### S2. One process, separate from the daemon, stateless

- `xaytune studio-web --state state.db` (name: review question Q1) runs a
  FastAPI application under uvicorn. It serves `/api/v1/...`; from UX-02 on,
  it also serves the built web client at `/`.
- **It is not the daemon.** It takes no lease and no lock and writes nothing.
  It works when no daemon is running (history stays readable) and alongside
  one (WAL readers do not block the writer, ADR-005 §8).
- **It holds no state of its own**: no cache that outlives a request, no
  session and no job table. Everything it returns is read from the durable
  store on each request. That is why a restart of the API or the browser
  loses nothing, which is the validation criterion in §Acceptance.

### S3. Read-only database access, never a migration

- The API opens the state database with the SQLite URI `mode=ro` and sets
  `PRAGMA query_only = ON`. It does not call `migrate()`. A database that does
  not exist, or that is not a Xaytune state database, is refused at startup.
- **Schema check.** At startup and on every request, the API reads the latest
  applied migration from `schema_migrations` and compares it with the
  migrations this build ships.
  - An older database is refused with a fixed message: run the daemon or any
    CLI command of this version once to migrate it.
  - A newer database is refused: this build cannot read it.
  - Either way, the API serves `/api/v1/meta` with the reason and `503`
    everywhere else. It never guesses at a shape it does not know.
- **One snapshot per response.** Each request runs all its reads inside
  `read_snapshot()`, so a response never mixes two commits (for example, a node
  whose status disagrees with its latest event).
- A connection per request, closed at the end. SQLite connections are not
  shared across threads, and opening a read connection is cheap next to the
  queries.

### S4. Reuse the existing stores; no SQL in the API

- Every endpoint composes the existing read surfaces: `AggregateStore`,
  `ExperimentGraph`, `EventJournal`, `ActionStore`, `PolicyDecisionStore`,
  `BudgetLedgerStore` / `budget_status`, `IncidentStore`,
  `RecoveryEpisodeStore`, `RecoveryPlanStore`, `CheckpointRecordStore`,
  `TrainingInterventionStore`, `InterventionApplicationStore`,
  `AgentInvocationStore` and `ControllerRequestStore`, reached through
  `ControlPlaneRepository`'s read properties.
- `xaytune/api/` contains no SQL. A query the stores lack is added to the
  store, beside its siblings, with its own storage test. UX-01 needs three
  small reads:
  - `AggregateStore.list_experiments(status, before, limit)`, a keyset page
    of experiments;
  - `EventJournal.latest_sequence()`, the journal's latest sequence across
    experiments, for the §S7 header;
  - a migration-ledger read that does not create the ledger, because
    `applied_versions()` does.

  Per-status node counts come from `nodes_for_experiment`.
- The API never re-derives a domain answer. "Is this experiment at rest" is
  `experiment_result`'s, "can these two candidates be compared" is
  `ExperimentGraph.compare`'s, and "how much budget is left" is
  `budget_status`'s. The client displays those answers; it does not recompute
  them.

### S5. A versioned wire contract, separate from the domain models

- Routes live under `/api/v1`. Every response body carries
  `"schema": "xaytune.studio-api/v1"`.
- Response models are their own Pydantic classes in `xaytune/api/v1/`, mapped
  from the domain models, never the domain models themselves. A domain
  refactor (a renamed field, a retired one) must not silently change the
  wire. A mapping test pins every domain field as either mapped or
  deliberately omitted, so a new domain field fails a test until someone
  decides.
- **Recorded documents.** Nested specs and payloads are passed through as
  JSON objects exactly as recorded, and typed on the wire as objects. This
  covers the objective, candidate, evaluation, planner, compiler and runtime
  specs, decision evidence, action payloads, incident evidence, recovery
  plans and interventions. Their shape is the domain's own versioned schema
  (for example `CandidateSpecSnapshot.spec_version`), not this API's. The
  client renders them generically as a key/value tree, plus purpose-built
  summaries for the fields a view needs (rank, learning rate, metric name).
  Re-modelling every spec in the API would duplicate the domain and drift
  from it.
- **Actors** appear as `{type, id}`. Actor `metadata` is never returned,
  for the same reason policy never sees it (PR-023): it is free-form, and
  nothing in a view depends on it.
- **Within v1, changes are additive**: new fields, new endpoints, new enum
  values. A client tolerates unknown fields and enum values, and shows an
  unknown status as itself.
  Removing or renaming anything is v2.
- The OpenAPI document is generated from the app and committed
  (`xaytune/api/v1/openapi.json`). A test fails when the app and the committed
  document differ in anything a client is generated from: paths, methods,
  parameters, request and response schemas, and every schema's types,
  formats, nullability, references, enums, constants, defaults and required
  fields. Only presentational text (titles, descriptions, summaries,
  examples) is ignored, because it moves with the FastAPI and Pydantic
  versions an install resolves. From UX-02 on, the client's TypeScript types are generated
  from that file and CI fails on drift. The contract has one source.
- Identifiers, statuses and fingerprints are passed through verbatim.
  Timestamps are RFC 3339 UTC. Numbers are JSON numbers, and an absent value
  is `null` with its recorded reason where one exists. It is never `0` (the
  ADR-022 V6 rule, applied everywhere).

### S6. V1 resources

| Endpoint | Returns | Built from |
|---|---|---|
| `GET /api/v1/meta` | API and Xaytune versions; the state database's migration and readability; the latest event sequence; the daemon lease (holder, expiry, live or not) | `schema_migrations`, `EventJournal`, `ControllerLeaseStore.current()` |
| `GET /api/v1/experiments?status=&before=&limit=` | Experiment summaries, newest first: id, name, status, primary objective, best node, node counts by status, created/updated | `list_experiments`, `nodes_for_experiment` |
| `GET /api/v1/experiments/{id}` | The experiment: objective, budget and its status, planner, compiler, runtime, evaluation and resilience specs as recorded, best node, node counts, at-rest state and next stage | `load_experiment`, `budget_status`, `experiment_result` |
| `GET /api/v1/experiments/{id}/graph` | Every node (id, status, parents, hypothesis, reason, branch origin, fingerprint, evaluation cycle, latest decision outcome) and every edge | `nodes_for_experiment`, `ExperimentGraph`, `decisions_for_node` |
| `GET /api/v1/experiments/{id}/events?after=&limit=` | The event journal, by sequence | `events_for_experiment_after` |
| `GET /api/v1/experiments/{id}/actions` | Governed actions with their policy decision and children | `ActionStore`, `PolicyDecisionStore` |
| `GET /api/v1/experiments/{id}/decisions` | Every decision: outcome, reason, engine, evidence, evaluation results it read | `decisions_for_node` over the nodes |
| `GET /api/v1/experiments/{id}/budget` | Ledger entries and the derived status | `BudgetLedgerStore`, `budget_status` |
| `GET /api/v1/experiments/{id}/incidents` | Incidents, recovery episodes and their effective plans | `IncidentStore`, `RecoveryEpisodeStore`, `RecoveryPlanStore` |
| `GET /api/v1/experiments/{id}/agent-invocations` | Planner invocation summaries: model identity, status, round, proposal fingerprint, timings. No prompt or response bodies in V1 | `AgentInvocationStore` |
| `GET /api/v1/nodes/{id}` | The node and its candidate snapshot, lineage paths, runs, evaluation runs with results, and decisions | `load_node`, `ExperimentGraph.lineage_paths`, `runs_for_node`, `evaluation_runs_for_node`, `evaluation_results_for_node`, `decisions_for_node` |
| `GET /api/v1/compare?left=&right=` | Two candidates: their lineage (structural), candidate spec difference, and each metric side by side with the evaluation fingerprint and unit it was measured under, judged comparable or not (below) | `ExperimentGraph.compare`, `evaluation_results_for_node` |
| `GET /api/v1/runs/{id}` | The run with its attempts, checkpoints, interventions and their applications | `load_run`, `attempts_for_run`, `CheckpointRecordStore`, intervention stores |
| `GET /api/v1/attempts/{id}` | One attempt with its incidents, checkpoints, directives and runtime reference | `load_attempt`, `IncidentStore.for_attempt`, `CheckpointRecordStore` |
| `GET /api/v1/evaluation-runs/{id}` | The evaluation run with its spec, subject, attempts and result | `load_evaluation_run`, `evaluation_attempts_for_run`, `evaluation_result_for_run` |

- **Comparison: lineage apart from measurement.**
  - `lineage` is structural: `distinct_candidates` is
    `ExperimentGraph.compare`'s `is_comparable` (two scientific candidates,
    not two records of one fingerprint), with the common and nearest common
    ancestors. It never implies the measurements can be compared.
  - Measurements are paired by (evaluator, metric, slice). Each side carries
    its value, result id, `evaluation_fingerprint` and recorded unit (`null`
    if the evaluator records none).
  - A pair is `comparable` only when both sides share an evaluation
    fingerprint (evaluator, version and configuration, dataset and slices,
    ADR-015) and a unit. Only then is `difference` (left − right) given.
    Otherwise `reasons` names the mismatch (measured on one side only,
    different evaluations, different units), and both values are still shown,
    each with what it was measured under.
  - When a candidate has results under several evaluations, a pair measured
    under a shared fingerprint is preferred over a newer unshared one.
  - This is a recorded-identity check, not inference. It never labels a
    difference significant or better. Harness comparisons (ADR-022 C1–C4,
    verdicts and uncertainty) are H06's and are shown by UX-03 from H06's
    own records.
- **Pagination.** Lists are keyset-paged and never offset-paged, because
  offsets skip or repeat rows under a live writer. Experiments page by
  `(created_at, id)` with `before`. Events page by `sequence` with `after`.
  The default `limit` is 100 and the maximum is 500; anything larger is a
  `422`, never silently clamped. Every other list in V1 is bounded by one
  experiment or one node and is returned whole.
- **Not in V1:** file contents behind an `ArtifactRef` (checkpoint files,
  logs, reports), controller requests beyond what `meta` summarizes, and
  prompt or response bodies of agent invocations. The API returns artifact
  references (kind, URI, digest) as data and serves no file. Serving
  artifacts is a separate decision, because it is how a read API leaks a
  filesystem.

### S7. Freshness without a push channel

- V1 has no WebSocket or SSE. The client polls, and the API makes polling
  cheap and exact:
  - every response carries `X-Xaytune-Event-Sequence`, the journal's latest
    sequence inside the response's snapshot;
  - `ETag` on every `GET`, so an unchanged resource returns `304`;
  - the events endpoint's `after` cursor, so a live view fetches only what is
    new.
- An open experiment polls every 2 s while it is not at rest and stops when
  it is. Push (SSE over the same journal cursor) is a later, additive change.

### S8. Errors never echo

- Errors are `application/problem+json` with a fixed `type`, `title` and
  `detail` per error kind: not-found (`404`), invalid parameter (`422`),
  unreadable store (`503`), method not allowed (`405`).
- `detail` names the parameter or the kind of record, never a value read from
  the store, a stack trace, a path outside what the caller sent, or the text
  of an underlying exception. This is the same rule as UX-00 and the H-series:
  an error never echoes a server's body or sensitive input.
- An unexpected exception is logged server-side with a correlation id. The
  response carries only that id.

### S9. Disclosure: Studio reads the released tier only

- The API returns what the control plane records about evaluation:
  `EvaluationResult.metrics`, the result's artifact references and its
  provenance. For H06 results, those are the **released** values. ADR-022 V5
  already decides what is withheld before a result is written: the
  `aggregate` disclosure withholds single-task reference-derived run values
  and every `successful-tasks` slice.
- The API never reads or serves an ADR-022 restricted report, a held-out
  reference, judge exchanges, or H05 `restricted`- or `secret`-tier
  trajectory blobs. When UX-03 shows harness evaluations, a withheld value is
  rendered as **withheld, with the disclosure that withheld it**. It is never
  hidden, and it is never zero.
- An owner view of the restricted tier would need authentication and an
  owner identity. That is UX-05 at the earliest, under its own decision.

### S10. Exposure: loopback only until authentication exists

- The API binds `127.0.0.1` by default. A non-loopback `--host` is **refused**
  in V1. It would serve the whole experiment record without authentication.
  This is stricter than UX-00's warning for Gradio because nothing here needs
  a remote bind, and UX-05 brings authentication with it.
- A `Host` header allow-list (`127.0.0.1`, `localhost`, `[::1]`, with the
  bound port) defends against DNS rebinding: a remote page that resolves its
  own name to 127.0.0.1 must not be able to read the API from a browser.
- There is no CORS. The client is served from the same origin. The Vite dev
  server proxies `/api` instead of the API allowing another origin.
- Every response sets `Content-Security-Policy: default-src 'self'`,
  `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` and
  `Cache-Control: no-store`. The client renders every recorded string as text
  and never as HTML or Markdown, because hypotheses, reasons and planner
  output are free text an agent may have written.

### S11. The web client (UX-02, decided here so UX-01 serves it)

- React + TypeScript + Vite in `web/`, using established components rather
  than bespoke ones:
  - TanStack Router (typed routes; URL is the state);
  - TanStack Query (polling, ETag-aware fetches, cache);
  - TanStack Table (experiment explorer, runs, events);
  - React Flow (`@xyflow/react`) with ELK layout for the candidate graph;
  - Radix UI primitives styled with Tailwind;
  - Apache ECharts or Recharts for metric charts, chosen in UX-02 when there is
    a series to chart (§S12).
- Every view is addressable by URL, including selected node, compare pair,
  tab and filters, so a link reproduces what someone saw. Reloads and
  restarts are lossless because the URL plus the store is the whole state.
- **Packaging** (decided in UX-02, constrained here): `pip install
  xaytune[studio-web]` must serve the UI without Node installed. The release
  build compiles `web/` and ships the bundle inside the wheel. The repository
  does not commit build output.
- CI gains a `web` job: typecheck, lint, unit tests (Vitest), build, and the
  OpenAPI drift check. A browser smoke test (Playwright) against a real
  state database comes with UX-02.

### S12. What V1 cannot show yet, and what unblocks it

These are not UX work. Each needs a backend PR, reviewed on its own, before
the view that needs it. They are listed here so the views are designed
against the right future shape. The API's resource layout already has a
place for each.

| Gap | Needed by | Recommended unblock | Alternative |
|---|---|---|---|
| **Training metric series** are not persisted (`MetricObserved` stays in runtime `events.jsonl`). *Done in Track B's migration 020: points are written beside the incident path, idempotent on the stream position, without moving the telemetry cursor* | Monitoring charts (UX-02/03) | A metric-point store: migration 020, written by the existing telemetry ingestion path inside the attempt's transaction, keyed by attempt, stream generation, sequence, name and step; read by `GET /api/v1/attempts/{id}/metrics`. Downsampling at read time | The API tails the runtime's `events.jsonl` via the attempt's `runtime_ref`. Rejected as the main path: it couples the API to one runtime's file layout, does not work for Ray or KubeRay, and reads transport rather than the record |
| **Harness evaluations** are not durable (ADR-022 V13) | UX-03 | The durable harness evaluation record: the daemon runs `harness-outcome`/`harness-judge` through the existing durable evaluation lifecycle (migration 006), so `EvaluationResult` (released values only) and the released `report.json` reference are recorded like any other result. Studio then needs no new path | Rejected (Q2): an API index over `report.json` files in a directory. It would be a second source of truth with no lineage link or durable identity |
| **Harness candidates** are not in the control plane (ADR-019 M1–M6 deferred) | Harness lineage in the candidate graph | The durable harness path. The graph view already renders any node whose candidate envelope it can summarise | None. Studio does not invent harness lineage |
| **Trajectories** are file stores (ADR-021) | UX-03 trajectory inspector | Trajectory references recorded with the durable harness attempt. The API serves verified, **released-tier** events only, after `verify_trajectory`, with replay level and completeness shown | Rejected (Q2): reading a trajectory store directly by path, for the same reason |

**Order (decided at acceptance).** Three tracks run in parallel:

- **Track A, Studio** (starts now): UX-01 API → UX-02 React client → UX-03
  harness evaluation views. UX-02 starts with training-experiment views on the
  current contract.
- **Track B, durable platform**:
  - the metric-point store (migration 020), independent of everything else,
    for loss and telemetry charts;
  - durable harness candidates and execution (ADR-019 M1–M6), which
    establishes harness nodes, runs and attempts;
  - durable H06 evaluations and trajectory references, planned with their
    harness-execution prerequisites, never as a temporary file index.
- **Track C, H07**: mutation/search architecture and provider contracts. It
  does not wait for the frontend. Execution integration follows the approved
  governance and the durable harness work.

UX-03 follows the durable evaluation and trajectory work.

**Consequence for the first milestone.** The first user-visible milestone (a
real experiment inspected in Studio with its lineage and evaluation results,
surviving restarts) is reachable with training experiments today: native
evaluation results, decisions and lineage are durable. Harness evaluation in
Studio needs the durable harness evaluation record first.

### S13. The legacy Gradio Studio

- `xaytune studio` (Gradio) is unchanged and stays an optional, legacy entry
  point under the `studio` extra. Its job submission is not part of Studio V1.
- Its tests run in CI from UX-01 on, in a dedicated `studio` job that installs
  the extra. They have only ever run locally. UX-01 fixes the six stale
  `TestPollJob` tests first (failing on `main` since `bf913d5`), so the job
  starts green rather than with known failures.
- Retiring the Gradio Studio is a later decision. It is not made until the
  web client covers the training-launch flow through the daemon mailbox
  (UX-04).

## Decisions on the review questions

Decided by the project owner at acceptance (review of #82 at `bb79049`).

- **Q1. Command name: `xaytune studio-web`.** `xaytune studio` stays the
  legacy Gradio Studio. `xaytune ui` may be added later as a convenience
  alias.
- **Q2. Backend gaps: durable backend work before UX-03.** No alternative,
  directory-indexed source of truth (`--evaluation-root` is rejected). The
  order is §S12's three tracks.
- **Q3. Network exposure: non-loopback binds are refused** until
  authenticated remote access exists (UX-05).
- **Q4. Dependencies: `xaytune[studio-web]`** with FastAPI and Uvicorn, and
  a dedicated CI job for the Gradio Studio. The Python-only installation
  stays lightweight: neither is a core dependency.

## Failure scenarios

- **S-F1. The API writes.** A route or dependency calls `migrate()`, a
  `ControlPlaneRepository` write, or opens a writable connection. Prevented by
  `mode=ro` plus `query_only`, a method-filter middleware, and a test that
  runs every route against a database whose file is `chmod 0444` and checks
  it is byte-identical afterwards.
- **S-F2. A torn read.** A node's status disagrees with its events because
  two reads straddled a commit. Prevented by one `read_snapshot()` per
  request. Tested with a writer committing between the reads of a
  multi-store endpoint.
- **S-F3. Schema skew.** A new daemon migrates the database while an old API
  is running. The per-request schema check returns `503` with the reason
  instead of misreading rows.
- **S-F4. DNS rebinding.** A web page reads `http://127.0.0.1:PORT` through
  a rebinding domain. The `Host` allow-list refuses it.
- **S-F5. Stored XSS.** A planner writes markup into `hypothesis`. The client
  renders text only, and CSP blocks inline script even if a renderer
  regresses.
- **S-F6. Restricted leak.** A harness evaluation's restricted report or a
  withheld slice reaches a response. The API reads only control-plane
  records, which hold released values by ADR-022 V5. Tests assert that no
  route ever opens a `restricted/` path.
- **S-F7. A silent zero.** An absent metric renders as `0`. The wire carries
  `null` plus a reason, and the client has one component for absent values,
  with a test that it never prints `0`.
- **S-F8. Unbounded responses.** An experiment with many events. Events are
  paged; the per-experiment lists are bounded by the experiment and measured
  in the acceptance fixture.
- **S-F9. Wire drift.** A domain field rename changes a response. The mapping
  test and the committed OpenAPI document both fail.
- **S-F10. A misleading comparison.** Two candidates measured under
  different datasets or scoring configurations are shown side by side with a
  delta, as if comparable. Prevented by the per-metric judgement in §S6: no
  difference without a shared evaluation fingerprint and unit, and a reason
  otherwise. Tested with the same metric name under two fingerprints.

## Acceptance criteria for UX-01

1. Every §S6 endpoint, with response models, a committed OpenAPI document and
   a type-aware drift test. A test shows that changing a response field's
   type, nullability, format or reference fails it.
2. A read-only connection, the schema check (older, newer, not a Xaytune
   database, missing file) and S-F1's byte-identical test, run over every
   route.
3. One snapshot per request (S-F2), keyset pagination with refused oversize
   limits, `ETag` and `304`, and the event-sequence header.
4. Errors per §S8, with tests that a not-found or invalid id is not echoed
   back.
5. Exposure per §S10: non-loopback refused, `Host` allow-list, no CORS,
   security headers, and `405` on every non-read method.
6. The three §S4 reads, each with its storage test. No other new query and
   no SQL under `xaytune/api/`.
7. **Real data, not mocks.** An end-to-end test drives the spec 18 adaptive
   scenario through a controller host into a file database: training with a
   CUDA OOM and a recovered attempt, evaluation, a decision, a planner
   branch and the child's training. It then starts the API on that file and
   reads the graph, both nodes, the runs with their incidents and
   checkpoints, the evaluations and the decisions. It restarts the API and
   reads them again, identically, with no host running.
8. The Gradio tests run in a CI `studio` job, with the stale `TestPollJob`
   tests fixed.
9. `/compare` reports lineage apart from measurement comparability (§S6),
   with a regression test for one metric name under two evaluation
   fingerprints.

## Consequences

- Studio becomes a view of the control plane rather than a second one. What
  you see in the browser is what `xaytune status` and `ExperimentHandle`
  see, from the same rows.
- The API and the client can each be reviewed on their own. The OpenAPI
  document is the boundary between them.
- Four views wait on backend work that is now named (§S12) rather than
  discovered during UI implementation.
- One more process for a local user to run. The daemon and the API stay
  separate deliberately: a UI crash or restart must never touch a lease.

## Alternatives considered

- **Serve the API from the daemon.** One process to run, but a UI bug
  would then run with the lease and the write connection. The daemon's
  surface is the mailbox, and that should stay its only one.
- **Extend the Gradio Studio.** It is built on `JobManager`, it exposes every
  callback as an endpoint, and it cannot render a graph or a table at the
  scale of an experiment. That is the reason for the UX track.
- **GraphQL.** Flexible, but V1 has a dozen fixed views over a few resource
  shapes. REST with OpenAPI gives typed clients, caching with `ETag`, and a
  smaller attack surface.
- **Return domain models directly.** Faster to write, but every domain
  refactor would change the wire (§S5).
- **A Studio-side cache or database.** It would be a second source of truth.
  The control-plane store is already indexed for these reads.
