# ADR-002 — Separate aggregate state machines

## Status
Proposed

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
