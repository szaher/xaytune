# Coding Agent Prompt — SQLite Repository and Events

Implement:

- SQLite repository
- revision-based optimistic concurrency
- atomic aggregate transition + event insert + outbox insert
- experiment/node/run/attempt persistence
- ADR-013 `RuntimeOperation` journal in migration 001, separate from the outbox
- operation ID, attempt ID, submit/cancel type, canonical request digest, state,
  runtime reference and revision persistence
- atomic attempt + INTENDED submit operation + events/outbox creation
- journal APIs: create, get by operation ID, list by attempt, list unresolved,
  and revision-checked transition; indexes for these lookups
- event sequence
- crash consistency tests

Use `schemas/sqlite-schema-migration-001.sql` as a starting point, adapting as
needed. It is migration 001 only — an initial subset, not the target schema.
Its header lists the tables still to come and the ADR that defines each. Do not
read the absence of a table there as a decision that it is not needed.

Requirements:

- one transaction per state transition
- no separate StateStore/EventStore writes
- stale revision raises ConcurrentModificationError
- outbox delivery is not required for transaction success
- migrations are versioned
- repository contains no ML runtime imports
- PR-005 includes the journal; PR-009 LocalRuntime cannot start without it
- operation transitions: INTENDED → SENT / CONFIRMED / FAILED;
  SENT → CONFIRMED / FAILED; terminal states cannot transition
- unresolved outcomes remain INTENDED/SENT for reconciliation, never become
  FAILED merely because a response was lost
- same operation ID and request returns the existing record; changed attempt,
  type or digest raises `IdempotencyConflict`
- record operation transitions in the domain event log atomically with state
- persist submit/cancel intent before any runtime call; outbox consumers must
  never submit or cancel workloads

Test:

1. successful transition
2. invalid state transition
3. stale revision
4. transaction rollback
5. event and state commit together
6. outbox row created
7. restart and reload
8. failure between attempt and operation insert rolls back both, events and outbox
9. committed operation intent and request digest survive database reopen
10. duplicate operation creation is idempotent; a conflicting request is rejected
11. operation transitions reject illegal edges and stale revisions atomically
12. lookup by operation ID/attempt and unresolved-state queries return durable records
13. cancellation intent survives restart independently of observed attempt status
