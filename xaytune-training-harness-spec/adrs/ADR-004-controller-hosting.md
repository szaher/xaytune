# ADR-004 — Controller hosting is explicit and durable

## Status
Proposed

## Decision

Introduce `ControllerHost`.

Implementations:

- EmbeddedControllerHost
- LocalDaemonControllerHost
- RemoteControllerHost (future)

Primary API is `submit() -> ExperimentHandle`.

`run()` is synchronous convenience.

## Rationale

The client process cannot be assumed to live as long as remote training.

## Consequences

- controller reconciliation required
- runtime operations must be idempotent
- experiment state must be durable
