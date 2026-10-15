# Studio API

A read-only HTTP API over the control-plane state database (UX-01, ADR-023). It is what the Studio web client is built on, and it shows exactly what `xaytune status` and `ExperimentHandle` show, read from the same rows.

```bash
pip install "xaytune[studio-web]"
xaytune studio-web --state state.db          # http://127.0.0.1:8765/api/v1/meta
```

`--state` defaults to `$XAYTUNE_STATE`. The API can run with or without a daemon: it reads history whether or not anything is running.

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

Training-metric series (loss curves), harness evaluations (H06), harness candidates and trajectories are not in the control-plane store yet; ADR-023 §S12 names the backend work each needs.

::: xaytune.api.create_app
::: xaytune.api.StudioStore
