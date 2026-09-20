# Coding Agent Prompt — SQLite Repository and Events

Implement:

- SQLite repository
- revision-based optimistic concurrency
- atomic aggregate transition + event insert + outbox insert
- experiment/node/run/attempt persistence
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

Test:

1. successful transition
2. invalid state transition
3. stale revision
4. transaction rollback
5. event and state commit together
6. outbox row created
7. restart and reload
