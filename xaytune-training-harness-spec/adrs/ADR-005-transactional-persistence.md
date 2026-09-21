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

- must be accepted before band B starts, including PR-004's SQLite repository
  and persistent schema; transaction boundaries and revision semantics must not
  be deferred until PR-005
- repository is more opinionated
- simple crash consistency
- external sinks become eventually consistent through outbox
