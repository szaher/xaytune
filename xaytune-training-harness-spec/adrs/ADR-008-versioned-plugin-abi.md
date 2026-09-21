# ADR-008 — Plugin interfaces and capabilities are versioned

## Status
Proposed

## Decision

Every plugin provides a `PluginDescriptor`.

Capabilities use a versioned parameterized schema.

Unknown major plugin API versions fail closed with actionable errors.

## Rationale

Trainer/runtime APIs evolve quickly. Boolean support flags and unversioned entry points are insufficient.
