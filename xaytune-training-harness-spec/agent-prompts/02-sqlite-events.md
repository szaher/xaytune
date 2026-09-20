# Coding Agent Prompt — SQLite Repository and Events

Implement:

- SQLite repository
- revision-based optimistic concurrency
- atomic aggregate transition + event insert + outbox insert
- experiment/node/run/attempt persistence
- event sequence
- crash consistency tests

Use `schemas/sqlite-schema.sql` as a starting point, adapting as needed.

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
