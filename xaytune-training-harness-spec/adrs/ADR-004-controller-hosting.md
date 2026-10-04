# ADR-004 — Controller hosting is explicit and durable

## Status
Accepted — 2026-10-03. §8 settled for PR-028 on 2026-10-04: the lease model,
its fences, and the startup sweep. §3's mutation kinds and §9, the caller-side
host, settled for PR-029 on 2026-10-04.

Gates band H (spec 15 Phase 7): PR-027 the local daemon process, PR-028
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
- `LocalDaemonControllerHost` -- the caller-side host for a local daemon:
  `submit()` and `attach()` return an `ExperimentHandle` answered from the
  durable record, with every mutation sent as a mailbox request (PR-029).
- `RemoteControllerHost` -- future.

The local daemon has two halves, and only the caller's is a `ControllerHost`:

```text
LocalDaemonControllerServer   the persistent process owning one state
                              database; serve() drives the controller (PR-027)
DaemonClient                  the v1 mailbox API: writes requests, reads the
                              record; returns requests, not handles (PR-027)
LocalDaemonControllerHost     ControllerHost over DaemonClient (PR-029)
```

The server is not a `ControllerHost` -- it has no caller to return a handle
to -- and `DaemonClient` is deliberately not one either: it returns requests.
`LocalDaemonControllerHost` is the host, because PR-029 made every handle
operation answerable without a controller in the caller's process, which §6
forbids: reads from the record, mutations as request kinds (§3), `wait()`
from the controller's recorded rest (§9).

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

Kinds: `submit` (the payload is the `ExperimentSpec`; the client pre-mints
the `ExperimentId`) and `attach` (names an existing experiment), from PR-027;
and from PR-029 one kind per mutation a handle or host exposes:

```text
cancel           {action_id, reason}            the cancel-experiment Action's id
propose-action   {action_id, type, target,      the Action as action_from_spec
                  payload, reason, proposed_by}  builds it: the typed envelope
approve-action   {action_id, approver, reason}  an action awaiting approval,
reject-action    {action_id, approver, reason}  under the experiment it is of
```

This is not a generic command bus: every mutating operation exposed through
the daemon is an explicit kind with exactly its payload, and a client never
runs controller logic itself (§6). Each mutation kind carries the identity of
what it records or resolves -- an action id the client mints, or the action
it resolves, approved or rejected idempotently by the same human and reason --
so carrying it out again, after a crash between its intent and its effects or
a retried send, finds the first attempt's record and finishes it. That is why
they need no `ACCEPTED`: a submission's admission is what names its
experiment, but these name theirs from the start. A request is carried out
through the Action it names or not at all: a cancellation request whose
Action is not recorded while another cancellation of the experiment is in
flight is refused (`CancellationInFlightError`), never answered with the
other one -- so a request cannot mean one saga now and another after a
restart. (Without an id, as an embedded caller asks, a second cancellation
still joins the one in flight.)

The daemon carries each out in order: **record the intent** -- the only step
that can refuse it, and one that calls no runtime -- then **attach** the
experiment, so it observes the effects and owns the experiment from then on
(§8), then **carry the intent on**, then `COMPLETED`. Once the intent is
recorded nothing makes the request `FAILED`: an attach or an effect that
fails leaves it `PENDING`, and an unfinished request is carried out again
first thing after a restart, finding its intent by its identity.

States:

```text
PENDING     durable; no control-plane work admitted
ACCEPTED    submit only: the initial control-plane intent exists and the
            request is linked to it
COMPLETED   the handoff reached the point at which submit()/attach() returns;
            not that the experiment has finished
FAILED      a definitive refusal of the request before anything was
            admitted: invalid or non-canonical payload, unavailable
            implementation, unsupported candidate; an unexpected error
            leaves the request where it was, to be retried
```

```text
submit   PENDING → ACCEPTED → COMPLETED
         PENDING → FAILED
others   PENDING → COMPLETED
         PENDING → FAILED
```

A mutation request is `FAILED` only for a refusal by the step that records
its intent, which writes nothing -- not even ownership, since the attach comes
after it: no such experiment or action, a payload that is not a well-formed
actor or action, an action of another experiment, an id recorded for something
else, another cancellation in flight, a cancellation proposed as a governed action,
a type or spec nothing accepts, an approval by a non-human, of an action not
awaiting one, or one a human already resolved otherwise. What follows a
recorded intent -- a cancellation's runtime effects, the recovery an approval
releases -- is never caught as a refusal: it leaves the request `PENDING`, to
be carried out again by its identity. SQLite cannot alter the kind `CHECK`, so
migration 018 rebuilds `controller_requests` with every row, index and trigger
it had.

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

and only then is the runtime called, outside the transaction (ADR-013). The
request is accepted only for the spec it carries: the admitting caller passes
the canonical digest of the spec the aggregates were derived from, and the
transaction refuses it unless it equals the request's `payload_digest`. A crash
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
second experiment, run or attempt). Since PR-028 it then sweeps every
nonterminal experiment it is responsible for (§8); in PR-027 a `COMPLETED`
request's experiment was adopted again only by an explicit `attach` request.

### 7. Controlled shutdown

On SIGTERM or SIGINT the daemon stops dequeuing requests, stops starting
controller work, cancels its observers, closes its runtimes and its database
connection, and releases the lock last. External workloads are not cancelled:
the runtime owns them. Nothing synthetic (`CANCELLED`, `FAILED`, success) is
written; uncertain operations stay in the journal for reconciliation. A client
may still commit a `PENDING` request during shutdown; it waits for the next
daemon.

### 8. Durable ownership: the controller lease (PR-028)

**Two layers, both mandatory.** The flock (§5) keeps a second daemon on the
same machine off the database, and the kernel releases it when a process dies.
The **controller lease** is ownership in the record: it carries the fencing
epoch, excludes embedded hosts, and decides takeover. Neither replaces the
other. The state database must be on a local filesystem: SQLite over NFS, SMB
or any shared or distributed filesystem is unsupported, and nothing tries to
detect it.

**The lease** is one row per database (`controller_leases`, migration 017):
`controller_id` (the daemon process's instance id), `epoch`, `heartbeat_at`,
`lease_expires_at`. It is never deleted.

- *Epoch.* Starts at 1 and increments on **every new acquisition** -- after a
  clean release as after a crash -- never on renewal. An earlier generation,
  even under a reused controller id, never becomes current again.
- *Acquire.* One transaction. No row: insert epoch 1. An expired row: a
  compare-and-set on the old `(controller_id, epoch)` to the next epoch. A
  live row is never taken: the new daemon waits for it to expire, and the wait
  is interruptible by SIGTERM/SIGINT. There is no hostname, PID, container or
  process-existence shortcut; durable expiry is the only authority.
- *Renew.* Every TTL/3: a compare-and-set on `(controller_id, epoch)` of a
  lease still live. An expired lease is not revived; zero rows changed is
  `LeaseLostError`.
- *Release.* On controlled shutdown, after controller work has stopped and
  before the flock: `lease_expires_at = now`, only if `(controller_id, epoch)`
  still match. A crash writes nothing; the successor waits out the TTL.
- *Timing.* One setting, `DaemonConfig.lease_ttl_seconds` (finite, positive,
  default 30 s); renewal is always TTL/3. No command-line flag.

`Experiment.controller_host` stays *provenance* -- which host admitted it --
and is never rewritten by a takeover; the lease is *current* ownership.

**Fencing.** Every `ControlPlaneRepository` write transaction proves ownership
inside the transaction, after `BEGIN IMMEDIATE` has taken SQLite's write lock
and before its first mutation, so a write serializes wholly before or wholly
after a takeover and never across one. The repository has a single
transaction helper, and an architecture test keeps it the only one.

```text
daemon controller   controller_id == mine AND epoch == mine AND unexpired
                    else LeaseLostError, nothing written
embedded host       no unexpired lease exists
                    else ControllerLeaseHeldError, nothing written
DaemonClient        unfenced: committing a mailbox request is a client's
                    write, not controller state
```

The embedded check runs on **every write**, not when the host opens: a host
opened before the daemon took the lease loses write authority from then on.
Reads stay open to it -- status, events, actions, results. Lease writes check
ownership themselves; migrations run before ownership is taken.

**Runtime effects are not fenced.** Every external mutation has a durable
`RuntimeOperation` first (ADR-005 §5), so a process whose lease expired can
only re-issue an effect recorded while it still owned the database, under the
same operation id: get-or-create (ADR-013), not a second workload. Its
confirmation of that effect is fenced, and the new owner reconciles the
operation. There is no second, runtime-level fencing token.

**Lease loss is fatal.** A `LeaseLostError` from anywhere -- renewal, a
mailbox request, an observer, recovery, evaluation, planning, decision,
cancellation, budget settlement -- is not a retry, an escalation or an
experiment failure: the process no longer has authority over the database.
The fence reports it to the server before raising, so a caller further up
cannot swallow it. The daemon stops dequeuing, cancels observers, stops
renewing, closes its controller and runtimes, records no workload outcome,
leaves the lease alone, releases the flock and exits with status 5.

**Startup.** flock → migrate → acquire (or wait for) the lease → the fenced
controller → unfinished mailbox requests (§6) → the sweep → the mailbox and
heartbeat loop. A crash anywhere in it is safe: the next epoch repeats it from
the record.

**The sweep.** Every nonterminal experiment the daemon is responsible for:
`controller_host.kind == "local_daemon"`, or a `COMPLETED` request of any
kind but `submit` for it -- an attach, or since PR-029 a cancellation,
proposal or approval, each of which attached it once its intent was
recorded: durable adoption, with
no adoption table. An experiment an embedded host
admitted and no daemon attached is not swept: that host may still be driving
it. Each is reconciled through `attach()` and ADR-013; there is no
startup-specific recovery of runs, attempts, operations or planning. Sweeping
twice repeats nothing.

**Paused experiments are swept, and stay paused.** A workload already running
still needs an owner, so reconciliation may look up, adopt and settle existing
work, carry on a requested cancellation and settle the ledger. It starts
nothing new: no evaluation cycle, no first attempt for an evaluation run, no
planned or realized candidate, no recovery successor (recovery already asks
for review when paused). Automatic evaluation progression requires `ACTIVE`.

**Budget after downtime.** No budget dimension measures time:
`max_wall_time_seconds`, `max_gpu_hours`, `max_tokens` and `max_cost` are
refused at submission, and the downtime is charged nowhere. The sweep repairs
accounting only (`settle_budget`) and moves no experiment: the state a restart
finds is not an effect. Exhaustion is decided where the controller is about to
create one -- a run, an evaluation cycle, an evaluation attempt, a recovery
successor -- by the same code before a restart as after it, so a restart alone
never changes an outcome. An experiment resting with a used-up quota and
nothing about to run (failure-handling, or planning with no planner) stays
`ACTIVE`, as it would have; a `PAUSED` experiment is settled and stays
paused.

Distributed or remote controller ownership remains future work.

### 9. The caller-side host (PR-029)

`LocalDaemonControllerHost` runs no controller. `submit()` and `attach()`
send their request and return an `ExperimentHandle` once it is `COMPLETED` --
handed off, not finished; `handle()` returns one with no request, for an
experiment the daemon already owns. The handle is the embedded host's, over a
different host:

```text
status(), actions(), events()   reads of the record
cancel(), propose()             cancel and propose-action requests
wait()                          polling of the record, until the daemon's
                                controller is at rest on the experiment
host.approve_action(),          approve-action and reject-action requests
host.reject_action()
```

A request the daemon refuses raises `ControllerRequestFailedError`, carrying
the name and message of what the daemon raised. A caller can exit or be killed
at any point: a sent request is the daemon's, and waiting again, from any
process, reads the same record. Retrying a request whose answer was lost means
sending the same request id, which reuses the experiment or action id it
minted: `submit`, `attach`, `cancel`, `approve_action`, `reject_action` and
`propose` -- on the host and on the handle -- all take one. A daemon-backed
`cancel()` while another cancellation is in flight joins it, as the embedded
one does, without a second one -- but truthfully: its own request is refused
and records nothing, and joining is a separate `attach` request, after which
the daemon owns the experiment and its reconciliation has carried the other
cancellation on. It returns only if the experiment is then cancelled or a
cancellation is still in flight; otherwise -- the other cancellation ended
without cancelling it, as on a retry of the refused request -- the refusal is
raised.

**Rest.** The embedded `wait()` waits for the controller's own tasks; a client
has none to wait for, and the record alone cannot tell rest from the instant
between two steps -- a training run has succeeded and its evaluation cycle is
about to begin, and every run is terminal with no operation unsettled. So the
daemon records, per experiment, when its controller came to rest on it: after
each request it carries out and each experiment it sweeps, it waits for the
controller's `wait()`, and in the same event-loop step records a
`controller_rests` row (migration 018) stamped with the experiment's latest
event sequence -- with the error, if `wait()` raised one. A rest is skipped
while a request for the experiment is being carried out. The client's
`wait()` returns once, in one read snapshot, the rest's sequence is still the
experiment's latest event, no request for it is unfinished and the result is
quiescent -- or once the experiment is terminal with nothing unsettled -- and
raises the recorded error otherwise. The rest is a fenced controller write
like any other.

**The CLI** (`xaytune submit`, `attach`, `status`, `watch`, `events`,
`results`, `actions`, `cancel`, `approve`, `reject`) is a thin layer over this
host: it has no database or runtime code of its own. Each mutating command
prints its request id before sending, and takes `--request-id` to retry.
Pause and resume are not exposed: they need their own durable design.

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
- one controller per state database, by a durable epoch-fenced lease; an
  embedded host cannot write beside a live daemon
- a restarted daemon reconciles everything it owns from the record
- every mutation through the daemon is an explicit request kind, idempotent by
  the identity it carries; a client never runs controller logic or calls a
  runtime
- a client's `wait()` is answered by the controller's recorded rest, not by
  inferring rest from the record
