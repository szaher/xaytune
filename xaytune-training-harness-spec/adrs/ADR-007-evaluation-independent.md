# ADR-007 — Evaluation is independent from trainer compilers

## Status
Accepted — 2026-09-21. Extended by ADR-015, which adds the durable execution
lifecycle this ADR left out.

Accepted rather than proposed because ADR-015 depends on it: evaluation having
its own Run/Attempt model is only coherent if evaluation is independent of
trainers in the first place. An accepted ADR resting on a proposed one is not a
gate anyone can use.

## Decision

Trainer compilers do not expose `evaluate()`.

Evaluation has its own spec, evaluators, execution, metrics, and fingerprint.

## Rationale

Evaluation may use different runtimes, datasets, judges, or policies and must not be coupled to the training implementation.
