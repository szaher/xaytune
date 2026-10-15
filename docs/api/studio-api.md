# Studio API

A read-only HTTP API over the control-plane state database (UX-01, ADR-023). It is what the Studio web client is built on, and it shows exactly what `xaytune status` and `ExperimentHandle` show, read from the same rows.

```bash
pip install "xaytune[studio-web]"
xaytune studio-web --state state.db          # http://127.0.0.1:8765/api/v1/meta
```

`--state` defaults to `$XAYTUNE_STATE`. The same process serves the Studio web client at `/` (see [Training Studio](../studio.md#studio-web)); everything below is the API it reads. The API can run with or without a daemon: it reads history whether or not anything is running.

## What it guarantees

- **Read-only.** The database is opened `mode=ro` with `query_only`, and is never migrated. Only `GET` and `HEAD` are answered; every other method is `405`.
- **Stateless.** No cache, session or job table. Restarting the API, or the browser, loses nothing.
- **One snapshot per response.** A response never mixes two commits.
- **Version-checked.** A database at an older migration (run the daemon or any control command of this version once to migrate it) or a newer one gets `503` with the reason; `/api/v1/meta` still answers.
- **Loopback only.** There is no authentication yet, so `--host` must be a loopback address. Requests whose `Host` is not `127.0.0.1`, `localhost` or `[::1]` are refused (`421`), which blocks DNS rebinding. There is no CORS.
- **Errors never echo** an id, a parameter value, a path or an exception message. They are `application/problem+json`; an unexpected error carries only a correlation id that is also in the server log.

## Endpoints

All under `/api/v1`; every body has `"schema": "xaytune.studio-api/v1"`.

| Endpoint | Returns |
|---|---|
| `GET /meta` | Versions, whether the database is readable and at which migration, the latest event sequence, and the controller lease |
| `GET /experiments?status=&before=&limit=` | Experiments, newest first. Page with `before=<next_before>` |
| `GET /experiments/{id}` | The experiment, its recorded specs, budget status, `quiescent` and `next_stage` |
| `GET /experiments/{id}/graph` | Candidate nodes with their latest decision, and lineage edges |
| `GET /experiments/{id}/events?after=&limit=` | The event journal by sequence. Read on with `after=<next_after>` |
| `GET /experiments/{id}/actions` | Governed actions with their policy decisions |
| `GET /experiments/{id}/decisions` | Decisions, in order |
| `GET /experiments/{id}/budget` | Limits, balances and ledger entries (amounts are exact decimal strings) |
| `GET /experiments/{id}/incidents` | Incidents, and recovery episodes with their effective plans |
| `GET /experiments/{id}/agent-invocations` | Planner model calls, summarised (no prompt or response bodies) |
| `GET /nodes/{id}` | A candidate: its spec, lineage paths, runs, evaluation runs with results, and decisions |
| `GET /compare?left=&right=` | Two candidates: their lineage (structural only), spec differences, and each metric side by side with the evaluation fingerprint and unit it was measured under. A pair is `comparable`, with a `difference`, only when both sides share an evaluation fingerprint and unit; otherwise `reasons` says why and there is no difference |
| `GET /runs/{id}` | A run with its attempts, checkpoints and interventions |
| `GET /attempts/{id}` | An attempt with its incidents, checkpoints and directives |
| `GET /attempts/{id}/metrics?name=&max_points=` | A training attempt's recorded metric series (`training.loss`, `training.learning_rate`, `resource.*`, `custom.*`, ...), one per name, unit, worker and rank, in stream order (a name reported in two units is two series). Each point has its optimizer step (`null` when the event had none, never 0) and stream position. `name` (repeatable) keeps only those series. `max_points` (4–5000, default 1000) downsamples at read time, keeping each bucket's lowest and highest value and the first and last point; `downsampled` says when it did |
| `GET /evaluation-runs/{id}` | An evaluation run with its attempts and result |

`limit` defaults to 100; above 500 is refused, not clamped.

**Freshness.** Every `200` has an `ETag` (send it back as `If-None-Match` for a `304`) and `X-Xaytune-Event-Sequence`, the journal's latest sequence in the response's snapshot. Poll; there is no push channel yet.

**Recorded documents.** Specs and payloads (objective, candidate, evaluation spec, action payloads, incident evidence, ...) are returned as recorded, in the domain's own versioned shape. Actors are always `{type, id}`: their metadata is never returned, including inside recorded payloads.

## The contract

The OpenAPI document is committed at `xaytune/api/v1/openapi.json` and served at `/api/v1/openapi.json`. Within v1 changes are additive only. After changing a route or a model, regenerate it and review the diff:

```bash
python -m xaytune.api.v1.openapi
```

## What it cannot show yet

Harness evaluations (H06), harness candidates and trajectories are not in the control-plane store yet; ADR-023 §S12 names the backend work each needs.

Training-metric series are recorded since migration 020: the controller writes every scalar a training attempt's telemetry reports into the metric-point store as it ingests the stream, idempotently on the event's stream position, so a replayed stream never duplicates a point. Points recorded before 020, and metrics from evaluation attempts, are not in it.

::: xaytune.api.create_app
::: xaytune.api.StudioStore
