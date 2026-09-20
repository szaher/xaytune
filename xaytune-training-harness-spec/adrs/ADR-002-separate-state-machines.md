# ADR-002 — Separate aggregate state machines

## Status
Ratified by merged implementation — 2026-09-21.

Implemented in `xaytune/core/state/machines.py` and under test on `main`. The
transition tables in `04-state-machines.md` are verified equal to that module;
where the two disagree, the code is authoritative.

## Decision

Experiment, ExperimentNode, Run, RunAttempt, Action, and Incident each have independent state machines.

Experiment state remains coarse (`ACTIVE`, `PAUSED`, terminal states).

Node/RunAttempt states contain execution/evaluation detail.

## Rationale

Concurrent branches make a single experiment state such as `EVALUATING` invalid.

## Consequences

- more domain types
- cleaner concurrency
- simpler reconciliation
- targeted transition validation
