# State Machines

The previous single lifecycle is replaced by separate aggregate state machines.

> **These tables are normative and mirror `xaytune/core/state/machines.py`.**
> That module is the implementation of this chapter, not a separate design. If
> the two disagree, the code is authoritative and this chapter is a bug. The
> guiding rule, from ADR-002: **every non-terminal state of a long-lived
> aggregate can reach `FAILED` and `CANCELLED`.** Anything that can be waited
> on can be abandoned, and a state machine that omits those edges forces the
> controller to either invent an illegal transition or leave the aggregate
> stuck. The one deliberate exception is noted under Experiment.

## 1. Experiment

```text
CREATED
  ↓
ACTIVE ↔ PAUSED
  ├── SUCCEEDED
  ├── FAILED
  ├── CANCELLED
  └── BUDGET_EXHAUSTED
```

Valid transitions:

| From | To |
|---|---|
| CREATED | ACTIVE |
| CREATED | CANCELLED |
| ACTIVE | PAUSED |
| ACTIVE | SUCCEEDED |
| ACTIVE | FAILED |
| ACTIVE | CANCELLED |
| ACTIVE | BUDGET_EXHAUSTED |
| PAUSED | ACTIVE |
| PAUSED | CANCELLED |
| PAUSED | FAILED |

Experiment state remains `ACTIVE` while individual nodes are training/evaluating/deciding concurrently.

**Experiment is the deliberate exception to the reach-FAILED rule: there is no
`CREATED → FAILED`.** A `CREATED` experiment owns no workload and has consumed
no budget, so there is nothing to have failed; if activation itself fails, the
experiment was never activated and `CANCELLED` is the honest terminal state.
Adding the edge would make "this experiment failed" ambiguous between "the
science failed" and "we could not start it", which are different facts for a
planner reading history.

## 2. ExperimentNode

```text
CREATED ──→ PLANNED ──→ READY ──→ ACTIVE ──→ EVALUATING ──→ DECIDING
                                                                ├── COMPLETED
                                                                ├── REJECTED
                                                                └── ACTIVE

every state above except the terminals also ──→ FAILED
                                            └─→ CANCELLED
```

Valid transitions:

| From | To |
|---|---|
| CREATED | PLANNED, CANCELLED, FAILED |
| PLANNED | READY, CANCELLED, FAILED |
| READY | ACTIVE, CANCELLED, FAILED |
| ACTIVE | EVALUATING, CANCELLED, FAILED |
| EVALUATING | DECIDING, CANCELLED, FAILED |
| DECIDING | ACTIVE, COMPLETED, REJECTED, CANCELLED, FAILED |
| COMPLETED, REJECTED, CANCELLED, FAILED | *(terminal)* |

`ACTIVE` may include one or more runs/replicates.

`DECIDING → ACTIVE` is the loop that makes a node a node rather than a run: a
decision can send the same candidate back for more training.

**`CANCELLED` and `REJECTED` are different terminal states and must not be
merged.** `REJECTED` is a scientific verdict — the node was evaluated and the
decision went against it. `CANCELLED` is an operational stop — a human or a
budget ended it before any verdict. A planner mining history has to tell "we
tried this and it was worse" from "we never found out".

## 3. Run

```text
CREATED ──→ ACTIVE ──→ SUCCEEDED
   │           ├─────→ FAILED
   ├───────────┴─────→ CANCELLED
   └─────────────────→ FAILED
```

Valid transitions:

| From | To |
|---|---|
| CREATED | ACTIVE, CANCELLED, FAILED |
| ACTIVE | SUCCEEDED, FAILED, CANCELLED |
| SUCCEEDED, FAILED, CANCELLED | *(terminal)* |

Run is intentionally coarse. Infrastructure detail belongs to RunAttempt.
`CREATED → FAILED` exists because a Run can fail before its first attempt is
ever submitted — for example when compilation or admission fails.

## 4. RunAttempt

```text
CREATED ──→ QUEUED ──→ STARTING ──→ RUNNING ──→ SUCCEEDED
                                       ├──→ CHECKPOINTING ──┐
                                       └──→ RECOVERING ─────┤
                                       ┌────────────────────┘
                                       └──→ RUNNING

every non-terminal state also ──→ FAILED, CANCELLED
QUEUED onwards also          ──→ PREEMPTED
```

Valid transitions:

| From | To |
|---|---|
| CREATED | QUEUED, CANCELLED, FAILED |
| QUEUED | STARTING, CANCELLED, FAILED, PREEMPTED |
| STARTING | RUNNING, CANCELLED, FAILED, PREEMPTED |
| RUNNING | CHECKPOINTING, RECOVERING, SUCCEEDED, CANCELLED, FAILED, PREEMPTED |
| CHECKPOINTING | RUNNING, RECOVERING, CANCELLED, FAILED, PREEMPTED |
| RECOVERING | RUNNING, CANCELLED, FAILED, PREEMPTED |
| SUCCEEDED, FAILED, PREEMPTED, CANCELLED | *(terminal)* |

A failed/preempted attempt may cause creation of another attempt under the same Run.

Three details that are easy to get wrong:

- **`PREEMPTED` starts at `QUEUED`, not `CREATED`.** A `CREATED` attempt is not
  yet known to any runtime, so there is nothing holding resources to reclaim.
  From `QUEUED` onwards the workload exists to the scheduler and can be evicted
  before it ever starts.
- **`CHECKPOINTING → FAILED` and `CHECKPOINTING → PREEMPTED` both exist.** A
  checkpoint write can fail on its own, and a node can be preempted mid-write.
  Omitting these is the common bug: it assumes checkpointing is atomic and
  instantaneous, which is exactly what ADR-009's layered checkpoint contract
  exists to handle.
- **`CHECKPOINTING → RECOVERING` exists.** A failed checkpoint write is
  recoverable without abandoning the attempt.

There is no `CANCELLING` state. Cancellation is intent plus an observed terminal
state, not a status the attempt occupies — see ADR-013, which also covers what
happens when a workload succeeds before a cancel is confirmed.

## 5. Action

```text
PROPOSED
  ↓
VALIDATING
  ├── REJECTED
  └── VALIDATED
         ↓
      APPROVAL_PENDING
         ├── REJECTED
         └── APPROVED
                 ↓
             EXECUTING
              ├── SUCCEEDED
              └── FAILED
```

For actions not requiring human approval:

```text
VALIDATED → APPROVED → EXECUTING
```

## 6. Incident

```text
DETECTED
  ↓
CLASSIFIED
  ↓
PLANNING_RECOVERY
  ├── RECOVERY_STARTED
  │     ├── RECOVERED
  │     └── RECOVERY_FAILED
  ├── ESCALATED
  └── UNRECOVERABLE
```

## 7. Decision

```text
CREATED
  ↓
POLICY_EVALUATED
  ├── REJECTED
  └── ACCEPTED
         ↓
      APPLIED
```

## 8. State transition API

No direct assignment:

```python
run.status = RunStatus.FAILED  # forbidden outside aggregate internals
```

Use a transition service:

```python
repository.transition_attempt(
    attempt_id,
    expected_revision=12,
    transition=AttemptFailed(
        incident_id=incident.id,
    ),
)
```

Transition code must:

1. validate current state
2. validate transition
3. mutate aggregate state
4. increment revision
5. append event
6. append outbox record
7. commit atomically

## 9. Reconciliation

After controller restart:

```text
load ACTIVE experiments
  ↓
load active nodes/runs/attempts
  ↓
query runtime status
  ↓
compare remote vs local state
  ↓
apply idempotent reconciliation transitions
  ↓
continue control loop
```

Runtime status is not blindly trusted to rewrite scientific state.

## 10. Concurrency

Phase 1 may enforce:

```python
max_active_nodes = 1
```

but schemas, storage, state machines, and budget ledger must support N concurrent nodes from day one.
