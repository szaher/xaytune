# Xaytune Studio web client

The React + TypeScript client for Xaytune Studio (UX-02, ADR-023). It is a
read-only view of the control-plane record, served by `xaytune studio-web`
from the API's own origin.

```bash
npm ci
npm run build                      # writes ../xaytune/api/static/
xaytune studio-web --state state.db   # http://127.0.0.1:8765/
```

## Developing

```bash
xaytune studio-web --state state.db   # the API, on 8765
npm run dev                           # the client on 5173, /api proxied to 8765
```

| Command | What it does |
|---|---|
| `npm run api:generate` | Regenerate `src/api/schema.d.ts` from `xaytune/api/v1/openapi.json` |
| `npm run api:check` | Fail if `schema.d.ts` is stale (CI runs this) |
| `npm run typecheck` / `npm run lint` | TypeScript and Biome |
| `npm test` | Vitest unit tests |
| `npm run e2e` | Playwright browser tests on a real recorded experiment |

The browser tests record a real experiment with
`python -m tests.studio_record` (the spec 18 adaptive scenario, through a real
controller host) and start `xaytune studio-web` over it themselves. Set
`XAYTUNE_PYTHON` to a Python with this repository installed (for example
`../.venv/bin/python`), and run `npx playwright install chromium` once.

## Rules the client keeps

- **The URL is the state.** Tab, selection, compare pair, filters and cursors
  are in the URL, so a reload or a Studio restart shows the same view.
- **Absent is never zero.** Every missing value renders through `Absent`
  with its reason.
- **Recorded text is text.** Hypotheses, reasons and documents are never
  rendered as HTML or Markdown.
- **Only comparable values are subtracted.** The comparison shows a
  difference only when the API says both sides share an evaluation and unit.
- **No mocked records.** Unit tests use small typed values for presentation
  rules; everything end to end runs against a recorded experiment.
