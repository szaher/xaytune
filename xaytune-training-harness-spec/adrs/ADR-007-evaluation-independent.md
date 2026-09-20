# ADR-007 — Evaluation is independent from trainer compilers

## Status
Proposed

## Decision

Trainer compilers do not expose `evaluate()`.

Evaluation has its own spec, evaluators, execution, metrics, and fingerprint.

## Rationale

Evaluation may use different runtimes, datasets, judges, or policies and must not be coupled to the training implementation.
