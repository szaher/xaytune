# Coding Agent Prompt — SQLite Repository and Events

Implement:

- SQLite repository
- revision-based optimistic concurrency
- atomic aggregate transition + event insert + outbox insert
- experiment/node/run/attempt persistence
- ADR-013 `RuntimeOperation` journal in migration 002, separate from the outbox
- operation ID, typed target (`training-attempt` | `evaluation-attempt` — never a
  `RunAttemptId` column, because evaluation attempts use this same journal),
  submit/cancel type, canonical request digest, state, runtime reference and
  revision persistence
- atomic attempt + INTENDED submit operation + events/outbox creation (ADR-005 §4)
- `BEGIN IMMEDIATE` for every write transaction, WAL, `synchronous=FULL`,
  explicit busy timeout (ADR-005 §8)
- **correctness must not assume a single writer.** Multiple processes may
  contend; SQLite serializes write transactions and revision CAS protects
  aggregate concurrency, and that pairing is the guarantee this PR must
  establish on its own. Do not require, and do not wait for, ADR-004's
  controller lease — that lease is about two controllers making duplicate
  *control decisions*, which is a different problem and a later band
- journal APIs: create, get by operation ID, list by target, list unresolved,
  and revision-checked transition; indexes for these lookups
- event sequence
- crash consistency tests

Use `schemas/sqlite-schema-migration-001.sql` and `-002.sql` as a starting
point, adapting as needed. The shipped sequence is:

```text
001  core aggregates                      PR-004
002  events, outbox, runtime_operations    PR-005
003  actions                               PR-006a
```

001's header lists the tables still to come and the ADR that defines each. Do
not read the absence of a table there as a decision that it is not needed.
`-003.sql` adds `actions` and belongs to PR-006a, not this PR; all three must
land before Phase 2.

**ADR-005 is the contract this PR implements.** Read it in full before starting:
§3 through §10 define every transaction boundary and repository invariant below,
and §11 is the required test list.

Requirements:

- one transaction per state transition
- no separate StateStore/EventStore writes
- stale revision raises ConcurrentModificationError
- outbox delivery is not required for transaction success
- migrations are versioned
- repository contains no ML runtime imports
- `request_digest` is the canonical hash of the full external request
  (operation type + `ResolvedExecutionPlan`), never `ExecutionFingerprint`:
  the question is "the same side-effect request?", not "equivalent executions?"
- the repository enforces that `target_id` exists in the table named by
  `target_kind`; SQLite cannot, since the targets live in different tables
- PR-005 includes the journal; PR-009 LocalRuntime cannot start without it
- operation transitions: INTENDED → SENT / CONFIRMED / FAILED;
  SENT → CONFIRMED / FAILED; terminal states cannot transition
- unresolved outcomes remain INTENDED/SENT for reconciliation, never become
  FAILED merely because a response was lost
- same operation ID and request returns the existing record; changed target,
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
12. lookup by operation ID/target and unresolved-state queries return durable records
13. cancellation intent survives restart independently of observed attempt status
14. an operation targeting an evaluation attempt persists and reloads exactly as a
    training one does
15. a minimal `Action` (PR-006a) commits atomically with the cancellation
    operations it causes, for an evaluation-attempt target as well as a training one
16. a cancel that loses the race to natural completion records
    `SUCCEEDED`/`SUPERSEDED` and leaves the attempt `SUCCEEDED`
17. two separate processes writing concurrently: one wins, one retries, no lost
    update, with no external lease involved
