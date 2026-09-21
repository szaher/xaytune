# ADR-001 — Xaytune is an experiment control plane

## Status
Accepted — 2026-09-21.

Every accepted ADR from ADR-011 onwards assumes the experiment control plane this
ADR proposes, and the repository is being built to it. "Proposed but ratified in
effect" was not a status a coding agent could act on — status is used as a gate,
so a premise the whole package depends on has to read as settled.

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
