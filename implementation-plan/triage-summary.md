# Triage Summary

Last updated: 2026-06-03 13:30

## Sources

No `gap-analysis/` directory existed. All findings come from:

1. **Session audit** — manual code review of core modules (trainer, recipes, eval, data, config, export, models, studio, CLI, alignment losses).
2. **Runtime bug discovery** — 10 bugs found and fixed inline during session.
3. **Deep scan agent** — 26 additional issues found across remaining modules.

## Dedupe Decisions

| Merged Into | Merged From | Rationale |
|-------------|-------------|-----------|
| BUG-009 | BUG-025 (`preferences.py` split without shuffle) | Same bug pattern — ordered split without shuffle. Fix in `loader.py` covers the general case; `preferences.py` needs the same fix. |
| BUG-006 | BUG-020 (eval_callback dummy metrics) | Same root cause — non-loss metrics get empty lists. Fix in `evaluate.py` pattern applies to `eval_callback.py`. |

## Consolidated Themes

| Theme | Item Count | Severity Range |
|-------|-----------|---------------|
| Training loop correctness | 3 | Critical–High |
| Numerical stability (alignment losses) | 2 | Critical–Low |
| Checkpoint/device portability | 3 | High–Medium |
| Eval pipeline completeness | 3 | High–Medium |
| Config validation gaps | 4 | High–Medium |
| Logging backend robustness | 4 | High–Low |
| Export pipeline correctness | 3 | High–Medium |
| Data pipeline edge cases | 4 | Medium–Low |
| Studio/CLI surface bugs | 3 | Medium |
| Dependency import safety | 2 | High–Medium |

## Confirmed vs Unconfirmed

All items confirmed by reading source code. No speculative items.

## Blockers

- **No GPU locally** — cannot run integration tests that require CUDA. Verification must happen on the OpenShift AI cluster or any CUDA-capable environment.
- **No `pydantic`/`torch` installed locally** — `pytest` cannot import xaytune modules on this machine. Syntax validation via `ast.parse()` is the only local check available.
