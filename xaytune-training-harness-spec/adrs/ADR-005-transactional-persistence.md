# ADR-005 — State transitions and events commit atomically

## Status
Proposed

## Decision

For the local implementation, SQLite is authoritative.

Each transition atomically:

1. validates expected revision
2. updates aggregate state
3. increments revision
4. inserts event
5. inserts outbox record

## Rationale

Independent StateStore/EventStore writes can diverge after crashes.

## Consequences

- repository is more opinionated
- simple crash consistency
- external sinks become eventually consistent through outbox
