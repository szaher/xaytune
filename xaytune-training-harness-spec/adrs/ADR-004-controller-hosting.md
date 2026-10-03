# ADR-004 — Controller hosting is explicit and durable

## Status
Accepted — 2026-10-03.

Gates band H (spec 15 Phase 7): PR-027 `LocalDaemonControllerHost`, PR-028
host leases and whole-controller startup reconciliation, PR-029 CLI
submit/attach/watch. Expanded before acceptance with the decisions PR-027
implements, for the reason ADR-005 was: the original text named the hosts
but not the boundary between a client and a durable controller, which is
what the band builds.

## Context

The client process cannot be assumed to live as long as remote training. An
`EmbeddedControllerHost` dies with the process that created it; everything it
records is durable (ADR-005) and adoptable by a later `attach()` (ADR-013),
but nothing drives an experiment while no process hosts it.

## Decision

### 1. Hosts

Introduce `ControllerHost`. Implementations:

- `EmbeddedControllerHost` -- in-process; tests, notebooks, local development.
- `LocalDaemonControllerHost` -- a persistent local process owning one state
  database (PR-027).
- `RemoteControllerHost` -- future.

Primary API is `submit() -> ExperimentHandle`; `run()` is synchronous
convenience. Every experiment records the host that admitted it as a
`ControllerHostRef` (`embedded`, `local_daemon` or `remote`). For the daemon
the reference's id is the daemon *instance* -- provenance for the process that
admitted the experiment, not a controller identity or a lease (§7).

### 2. Local IPC: the state database is the mailbox

The local daemon's command channel is the SQLite state database itself. A
client commits a durable request row (`controller_requests`); the daemon polls
for it. No Unix socket, HTTP server, gRPC server, authentication protocol or
network listener is introduced.

The filesystem permissions protecting the database are the local security
boundary. A process that can write the control-plane database already holds
highly privileged access; the daemon does not pretend to add authentication
on top of it.

Unix-socket or localhost-HTTP transports may be added later, as transports
over the same request semantics. gRPC is not needed for the local daemon.

### 3. Requests

A request has a client-generated id, which is the idempotency identity of the
handoff, a kind, a state, a revision, the experiment it concerns, a canonical
payload and its digest, and an error when it failed. Repeating an id with the
same kind, experiment and payload is the same request; with anything else it is
an idempotency conflict.

Kinds in v1: `submit` (the payload is the `ExperimentSpec`; the client
pre-mints the `ExperimentId`) and `attach` (names an existing experiment).
This is not a generic command bus. A later mutating operation exposed through
the daemon becomes an explicit request kind; a client never runs controller
logic itself (§6).

States:

```text
PENDING     durable; no control-plane work admitted
ACCEPTED    submit only: the initial control-plane intent exists and the
            request is linked to it
COMPLETED   the handoff reached the point at which submit()/attach() returns;
            not that the experiment has finished
FAILED      a definitive failure before anything was admitted: invalid
            payload, unavailable implementation, unsupported candidate
```

```text
submit   PENDING → ACCEPTED → COMPLETED
         PENDING → FAILED
attach   PENDING → COMPLETED
         PENDING → FAILED
```

There is no durable `PROCESSING` state: one daemon consumes the database, and a
daemon that died holding `PROCESSING` would need a lease or timeout to release
it, which is §7's PR-028 work. A failure after `ACCEPTED` is never rewritten as
`FAILED` for want of an outcome; the runtime-operation journal and ADR-013
reconciliation own it.

### 4. Admission is one transaction

Initial submission is admitted atomically. After validation, binding and
compilation, one write transaction records:

```text
request ACCEPTED (daemon-admitted submissions)
experiment recorded and ACTIVE
root node recorded and ACTIVE
run 1 recorded and ACTIVE, with its max_runs reservation
attempt 1 with its parallel-run reservation
its submit RuntimeOperation, INTENDED
```

and only then is the runtime called, outside the transaction (ADR-013). A crash
leaves either nothing admitted (`PENDING`) or an `INTENDED` operation that
reconciliation resolves; never a request that is half-admitted, and never a
second experiment for one request. The embedded host uses the same admission:
its initial submission gains the durable restart boundary PR-026 gave branch
realization.

A budget with nothing left for the first run admits the experiment and ends it
`BUDGET_EXHAUSTED` in the same transaction, with no run and no effect.

### 5. Singleton ownership on one machine

A daemon holds an exclusive, non-blocking POSIX advisory lock
(`fcntl.flock(LOCK_EX | LOCK_NB)`) on `<resolved state database path>.lock`
for its lifetime. The kernel lock is the authority: the file's content (pid,
instance id, start time, database) is diagnostic only. There is no stale-lock
cleanup and no PID-existence check -- a process that dies releases the lock,
and a PID alone never establishes ownership (ADR-013 AC-10). On a platform
without `fcntl` the daemon refuses to start; it never falls back to an unsafe
lock.

The flock prevents two daemons on one machine driving one database. It is not
distributed ownership.

### 6. The process boundary

The daemon is a foreground process (`python -m xaytune.daemon --state …
--config module:factory`), supervised by whatever the operator uses; it does
not daemonize itself. Its implementations -- compilers, runtimes, evaluators,
planners, decision engine, policy, checkpoint manager, recovery configuration
-- come from an explicit configuration factory, never from the embedded host's
silent defaults: the decision engine is not yet recorded with an experiment,
so the configuration is what an operator must keep consistent.

The daemon owns its controller and is the only component that calls runtime
effects for the requests it admitted. Client code writes requests and reads
durable state; it never instantiates a controller against the daemon's
database to attach, wait or cancel.

The daemon recovers **unfinished mailbox requests** at startup (`PENDING`, and
`ACCEPTED` through the existing attach/reconciliation path, never creating a
second experiment, run or attempt). A `COMPLETED` request's experiment is not
adopted again by restarting the daemon; an explicit `attach` request does that
until PR-028.

### 7. Controlled shutdown

On SIGTERM or SIGINT the daemon stops dequeuing requests, stops starting
controller work, cancels its observers, closes its runtimes and its database
connection, and releases the lock last. External workloads are not cancelled:
the runtime owns them. Nothing synthetic (`CANCELLED`, `FAILED`, success) is
written; uncertain operations stay in the journal for reconciliation. A client
may still commit a `PENDING` request during shutdown; it waits for the next
daemon.

### 8. What PR-028 adds

Durable controller identity and leases (`controller_id`, `heartbeat_at`,
`lease_expires_at`, takeover rules), the startup sweep over every active
experiment, deadline and budget re-evaluation after downtime, and whole-
controller control-loop reconstruction. PR-027 adds none of these, and no
partial lease.

## Rationale

The client process cannot be assumed to live as long as remote training.

SQLite polling is chosen over a socket or HTTP API for the local v1 daemon
because the request is then as durable as the state it changes: a client that
dies after committing it has handed it off, a daemon that dies mid-handoff
finds it again, and there is no second protocol to secure or keep consistent
with the record. Polling latency is acceptable for a local host.

## Consequences

- controller reconciliation required
- runtime operations must be idempotent
- experiment state must be durable
- a request's handoff is durable and idempotent by its client-generated id
- initial submission is atomic for every host
- one daemon per state database per machine, by kernel lock
