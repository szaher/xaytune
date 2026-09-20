# State Machines

The previous single lifecycle is replaced by separate aggregate state machines.

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

## 2. ExperimentNode

```text
CREATED
  ↓
PLANNED
  ↓
READY
  ↓
ACTIVE
  ↓
EVALUATING
  ↓
DECIDING
  ├── COMPLETED
  ├── REJECTED
  ├── FAILED
  └── ACTIVE
```

`ACTIVE` may include one or more runs/replicates.

## 3. Run

```text
CREATED
  ↓
ACTIVE
  ├── SUCCEEDED
  ├── FAILED
  └── CANCELLED
```

Run is intentionally coarse. Infrastructure detail belongs to RunAttempt.

## 4. RunAttempt

```text
CREATED
  ↓
QUEUED
  ↓
STARTING
  ↓
RUNNING
  ├── CHECKPOINTING → RUNNING
  ├── RECOVERING → RUNNING
  ├── SUCCEEDED
  ├── FAILED
  ├── PREEMPTED
  └── CANCELLED
```

A failed/preempted attempt may cause creation of another attempt under the same Run.

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
