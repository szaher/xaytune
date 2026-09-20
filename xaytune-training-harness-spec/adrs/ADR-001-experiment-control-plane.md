# ADR-001 — Xaytune is an experiment control plane

## Status
Proposed

## Context

Xaytune needs to support local, Ray, Training Hub, and future remote execution. A design where TrainerBackend both prepares and executes training conflicts with RuntimeBackend ownership.

## Decision

Xaytune controls experiments and compiles training intent.

Trainer integrations implement:

```text
TrainingSpec → TrainingExecutionSpec
```

Runtime integrations execute:

```text
ResolvedExecutionPlan → RuntimeRef
```

Trainer compilers do not submit remote workloads.

Runtime backends do not interpret scientific training intent beyond execution requirements.

## Consequences

Positive:

- clean remote boundary
- serializable execution plans
- controller restart support
- easier platform integration
- easier testing

Negative:

- requires worker entrypoints/artifact contracts
- existing native trainer must be wrapped

## Rejected alternatives

- TrainerBackend.train() for remote execution
- RuntimeBackend owning recipe translation
