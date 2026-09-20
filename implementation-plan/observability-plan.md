# Observability Plan

Last updated: 2026-06-03 13:30

> **Status: historical plan, with instrumentation guidance worth keeping.**
>
> Several entries below are written as future work ("Add …") for instrumentation
> that has since shipped, and the per-epic FIXED markers are the 2026-06-03
> audit's, not current. Keep this file for how it reasons about metrics, logs
> and warnings in a library rather than a service. **Do not use it to decide
> what instrumentation is missing** — check the code, or `backlog.md` for the
> six live tasks.

## Overview

xaytune is a library, not a service. Observability means: correct metrics reported to users, clear warnings/errors, and logging backends that don't crash.

## Per-Epic Observability

### EPIC-1: Training Loop (FIXED)
- **Metrics:** `state.metrics["loss"]` now reports undivided loss. `state.global_step` now counts optimizer steps.
- **Logs:** No change needed.
- **Alerts:** N/A.

### EPIC-2: Alignment Stability
- **Metrics:** Add `torch.isnan(loss).any()` check after ORPO/SimPO loss computation. If NaN detected, log warning with input stats before returning.
- **Logs:** Warn on NaN loss values during alignment training.

### EPIC-3: Eval Pipeline
- **Metrics:** `eval_token_accuracy` now reported correctly in `state.metrics`.
- **Logs:** No change.

### EPIC-4: Checkpoints
- **Logs:** Log device mismatch info when `map_location` is used (debug level).
- **Metrics:** None.

### EPIC-5: Config Validation
- **Logs:** Validation errors produce structured messages: which field, what's wrong, what's expected.

### EPIC-6: Logging Robustness
- **Logs:** Backend failures logged to stderr with `warnings.warn()`.
- **Metrics:** Count of suppressed backend errors per backend (internal counter, not a user metric).
- **Alerts:** After 10 consecutive failures, warning: "Backend X disabled due to repeated failures."

### EPIC-7: Export
- **Logs:** GGUF tool-not-found error includes install instructions.
- **Logs:** Hub push warns about missing tokenizer.

### EPIC-8: Data Pipeline
- **Logs:** `format_text()` warns on unknown keys (once per unique key set).

## SLO/SLA

Not applicable — xaytune is a client-side library, not a hosted service. No SLOs or alerting thresholds apply.
