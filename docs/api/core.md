# Control-Plane Core

`xaytune.core` holds the domain types for the experiment control plane: typed identifiers, the aggregates (`Experiment`, `ExperimentNode`, `Run`, `RunAttempt`), their state machines, and the value objects that reference things outside the control plane.

It is deliberately free of ML and runtime dependencies — no torch, transformers, peft, trl, ray, torchft, kubernetes, mlflow or wandb — so a controller host or a client can import it without the training stack installed.

```python
import xaytune.core as core  # no torch required
```

This is the foundation the rest of the control plane builds on: persistence (`xaytune.storage`), the trainer compilers (`xaytune.compilation`), the local runtime (`xaytune.runtimes`), the evaluation contract (`xaytune.evaluation`) and the embedded controller (`xaytune.experiment`) all depend on it, and it depends on none of them. For how those fit together, see [Control-plane concepts](../control-plane/concepts.md).

## Identifiers

Every aggregate has its own `str` subclass carrying a stable prefix, so passing a `RunId` where an `ExperimentId` is expected fails both type checking and runtime validation.

```python
from xaytune.core import ExperimentId, RunId

experiment_id = ExperimentId.generate()   # 'exp_01M2ZERGPP6TE4620TXNPZWF9V'
RunId.generate()                          # 'run_...'

ExperimentId.validate(RunId.generate())   # raises InvalidIdError
```

The body after the prefix is ULID-shaped: 10 Crockford base32 characters of millisecond timestamp followed by 16 of randomness. Lexicographic order is therefore creation order, so event streams and listings can sort on the id alone. Generation is monotonic within a process, so two ids minted in the same millisecond still sort in the order they were created.

| Type | Prefix |
|------|--------|
| `ExperimentId` | `exp_` |
| `ExperimentNodeId` | `node_` |
| `RunId` | `run_` |
| `RunAttemptId` | `attempt_` |
| `ActionId` | `act_` |
| `IncidentId` | `inc_` |
| `RecoveryPlanId` | `recovery_` |
| `EvaluationId` | `eval_` |
| `EvaluationRunId` | `evalrun_` |
| `EvaluationAttemptId` | `evalattempt_` |
| `ArtifactId` | `artifact_` |
| `CheckpointId` | `ckpt_` |
| `DecisionId` | `decision_` |
| `EventId` | `event_` |
| `OperationId` | `op_` |

`created_at_ms` recovers the embedded timestamp:

```python
experiment_id.created_at_ms   # 1789909549782
```

## Aggregates

`Experiment` owns an optimization objective. `ExperimentNode` is one scientific candidate within it — a hypothesis, not an infrastructure attempt. `Run` is a logical execution of a candidate; a node may own several when seeds or replicates are wanted. `RunAttempt` is one infrastructure attempt at a run.

```python
from xaytune.core import (
    ControllerHostRef, Experiment, ExperimentId, Objective, ObjectiveMetric,
)

experiment = Experiment(
    id=ExperimentId.generate(),
    name="support-sft",
    objective=Objective(
        primary=ObjectiveMetric(name="task_success", direction="maximize"),
        target=0.85,
    ),
    controller_host=ControllerHostRef(kind="embedded"),
)
```

All aggregates are frozen, and frozen means frozen all the way down. Pydantic's `frozen=True` only blocks attribute assignment, so container fields would otherwise stay mutable and keep a reference to whatever the caller passed in:

```python
snapshot.payload["optimizer"]["lr"] = 7   # blocked: TypeError
source_dict["optimizer"]["lr"] = 7        # does not reach the snapshot
```

Mappings become `FrozenDict` and sequences become tuples, recursively. `FrozenDict` freezes **at construction**, so every instance is deeply immutable by class invariant rather than by how it happened to be built, and its backing store is a `MappingProxyType` — there is no mutable dictionary to reach through. The conversion copies, which severs the caller's reference. Use `thaw()` when you need a mutable copy to build on.

Freezing also enforces a canonical value contract, because a record that cannot round-trip cannot be fingerprinted:

| Accepted | Rejected |
|---|---|
| `null`, `bool`, `int`, finite `float`, `str` | non-string mapping keys |
| mappings → `FrozenDict` | sets — no stable iteration order, so fingerprints would vary by process |
| sequences → `tuple` | `NaN` / infinity — no JSON representation |
| | `bytes` and arbitrary objects |

Violations raise `InvalidDomainValueError`, which is also a `ValueError`, so Pydantic reports it as a validation error.

!!! warning "Canonical values are not canonical encoding"

    The value contract makes records *representable*; it does not make their serialization *canonical*. Two equal `FrozenDict`s can differ in insertion order, so `model_dump_json()` may emit their keys in different orders, and Python's built-in `hash()` is randomized per process.

    Equality has Python semantics too, not JSON semantics: `True == 1` and `1 == 1.0`, so `FrozenDict({"x": True})` compares equal to `FrozenDict({"x": 1})` even though their canonical JSON forms — `{"x":true}` and `{"x":1}` — should fingerprint differently.

    Fingerprints must therefore be computed from a canonical *typed* encoder — sorted keys, deterministic number and string encoding, UTF-8 bytes, then a stable digest — never from `hash()`, Python equality, or a naive JSON dump.

The four aggregates — `Experiment`, `ExperimentNode`, `Run`, `RunAttempt` — go further and **refuse updates entirely**:

```python
experiment.model_copy(update={"status": ExperimentStatus.SUCCEEDED})
# TypeError: Experiment cannot be updated through model_copy ...
```

Re-validating an update checks the *schema*. It cannot check that the transition is legal, that the revision moved, or that the timestamps are consistent — so an update would produce an object that is valid to Pydantic and impossible in the domain, such as a `SUCCEEDED` attempt with no `started_at` and revision 0. Value objects like `DatasetRef` keep a validated `model_copy`; aggregates change only through transition methods.

A field with no transition method yet cannot be changed at all. That is deliberate: the next person needs an explicit named operation rather than a generic escape hatch.

Status changes go through `with_status()`, which validates the transition and returns a new instance with the revision bumped:

```python
from xaytune.core import ExperimentStatus

active = experiment.with_status(ExperimentStatus.ACTIVE)

active.status      # ExperimentStatus.ACTIVE
active.revision    # 1
experiment.status  # ExperimentStatus.CREATED — the original is untouched
```

An illegal transition raises `InvalidTransitionError` rather than silently succeeding:

```python
experiment.with_status(ExperimentStatus.SUCCEEDED)
# InvalidTransitionError: Experiment cannot transition from 'created' to 'succeeded'
```

`revision` supports optimistic concurrency. Appending the durable event and committing atomically is the persistence layer's job, which arrives with the SQLite repository.

## Scientific versus operational lineage

This split is the reason the graph stays meaningful instead of degenerating into an infrastructure log.

Lineage has four levels:

| Level | Represents |
|---|---|
| `ExperimentNode` | An alternative scientific candidate |
| `TrainingIntervention` | A scientific change to a continuing model trajectory |
| `ExecutionOverride` | An operational adjustment preserving declared training intent |
| `RunAttempt` | One infrastructure execution attempt |

An **operational** event — worker restart, preemption, checkpoint restore — creates a new `RunAttempt` under the same `Run`, and never branches the graph.

**Comparability** decides between a node and an intervention:

> A change creates a new `ExperimentNode` when the changed configuration is an alternative candidate you may want to compare independently against the current one.
>
> A change is a `TrainingIntervention` when it only makes scientific sense as a continuation of the existing model trajectory.

The kind of parameter does not decide this — experimental intent does, so the same technical change can be either:

```text
At checkpoint 20k, branch A keeps LR 2e-5 and branch B uses 1e-5,
to see which is better.                    → two nodes

At step 20k the run destabilises, so LR is
lowered and training continues.            → one intervention
```

Between the operational and scientific levels sits `ExecutionOverride`: a policy-approved operational adjustment that preserves the declared training intent, recording what it claims to hold.

```python
from xaytune.core import ExecutionOverride

ExecutionOverride(
    id="ovr-1",
    kind="micro_batch_resize",
    reason="CUDA OOM at step 400",
    values={"micro_batch_size": 2, "gradient_accumulation": 4},
    preserves=["effective_batch_size"],
)
```

The `kind` vocabulary is closed. Learning rate, optimizer, LoRA rank, data, scheduler, reward and model revision changes are not override kinds, and the type rejects them. Depending on experimental intent they are either a `TrainingIntervention` or a new `ExperimentNode`.

`TrainingIntervention` itself is not in this package yet — it arrives with the controller work. The rule is recorded here so the distinction is not rediscovered from the older binary model.

## Recovery decisions (PR-019)

`RecoveryPlan`, `RecoveryStrategy`, `Recoverability`, `RecoveryLimits` and
`RecoveryRequest` describe what recovery should happen. The coordinator records
one immutable plan per incident with a `RecoveryPlanned` event and outbox in one
transaction. It does not execute recovery, change an aggregate's status, create
an Action, submit a workload, or create an attempt or experiment node.

```python
from xaytune.core import RecoveryLimits, RecoveryRequest
from xaytune.resilience.recovery import RecoveryCoordinator

coordinator = RecoveryCoordinator(repository, checkpoint_manager)
request = RecoveryRequest(
    limits=RecoveryLimits(
        max_attempts_per_run=3,
        max_recoveries_per_experiment=10,
        max_same_incident_repeats=2,
    ),
    restore_context=consumer_restore_context,
)
plan = await coordinator.plan(incident_id, request)

# After restart, preserve existing plans and fill incident→plan gaps.
plans = await coordinator.reconcile(
    experiment_id, request_for_incident=lambda incident: request,
)
```

The consumer must supply its own `RestoreContext`; copying compatibility from an
available checkpoint would hide incompatibility. Selection considers committed
reports from the same logical run, through the incident's attempt, preceding the
incident. It orders checkpoints by optimizer step, producer attempt number, then
checkpoint ID, newest first. It validates bytes, producer/candidate/execution
provenance, codec layout, topology, dataset and ordering through
`CheckpointManager.validate_recorded`, without decoding or applying state.
This API requires the optional `CheckpointValidationCodec` extension, explicitly
declared by `descriptor.metadata["checkpoint_validation_api"]` equal to
`CHECKPOINT_VALIDATION_API_VERSION` (`xaytune.checkpoint-validation/v1alpha1`).
Both names are exported from `xaytune.checkpoints`. Method presence does not declare
support; absent/unsupported declarations fail closed. It adds no required method
to the v1alpha1 `CheckpointCodec` ABI. Legacy codecs retain save, restore and
recorded-restore support, but are ineligible for validation-only recovery.
The extension raises `CheckpointCompatibilityError` for unsupported layouts or
`CheckpointCorruptionError` for malformed capture. The manager also normalizes
`ValueError` (including Pydantic `ValidationError`) to corruption with its cause
preserved. Those failures and unavailable bytes (`OSError`) become checkpoint
ineligibility; unexpected programmer errors propagate without recording a plan.
Eligibility reasons distinguish `codec lacks validation capability`,
`checkpoint incompatible with intended consumer`, and
`checkpoint bytes/provenance missing or corrupt`. Missing capability is checked
before consumer compatibility and is not recorded as consumer incompatibility.
Generic resume requires `FULL + EXACT`, an optimizer-step boundary and a data
cursor. A corrupt or incompatible newest checkpoint can fall back to an older
eligible checkpoint. Reports from other runs and future training positions are
excluded. Execution must revalidate the selected checkpoint later.

Generic transient infrastructure failures select `RESUME` when eligible. Without
an eligible checkpoint they select `PAUSE_FOR_APPROVAL`, unless
`allow_retry_without_checkpoint=True` explicitly permits a fresh `RETRY`.
CUDA OOM and numerical incidents pause for their later specialised planners;
evaluation recovery and unknown diagnoses require review. Corrected data or
configuration is required for fatal diagnoses. Completed/cancelled work,
terminal experiments/candidates and superseded attempts cannot be recovered.

Limits count recorded `RETRY`/`RESUME` decisions as conservative recovery
reservations, not completed executions. Current-attempt reservations consume
future attempt slots; once a successor attempt exists, it represents older
reservations in the actual attempt count. Experiment recovery limits count all
such decisions. A failure pattern includes category, structured code location,
resource shape, numerical quantity and candidate fingerprint. It ignores delivery
IDs, counters and free-form detail. Repeating that pattern in the same run under
unchanged execution/override values refuses another identical recovery; the
repeat limit also bounds patterns across execution changes.
The repeat threshold counts earlier matching planned incidents; zero still
permits an initial response and refuses a repeated response.

Each plan preserves its planning snapshot, request and checkpoint eligibility
evidence. Recording rechecks durable inputs under the write lock, so concurrent
planners cannot bypass limits. Exact semantic recording retries return the
original ID/time; changed decisions for the same incident conflict. Coordinator
replay returns the recorded decision without reevaluating changed policy or
checkpoint files. Reconciliation resolves consumer configuration only for
missing plans. A pause plan describes a future approval requirement; it does not
create an approval request or grant execution authority. Action governance,
capability checks, execution budget checks, overrides and scientific interventions
belong to the later execution path. EmbeddedControllerHost does not yet invoke
the coordinator automatically.

## State machines

Each aggregate has its own lifecycle. There is deliberately no single experiment-wide status covering training and evaluation: with concurrent branches, an experiment-wide `EVALUATING` would be meaningless.

```python
from xaytune.core import EXPERIMENT_MACHINE, ExperimentStatus

EXPERIMENT_MACHINE.allowed_from(ExperimentStatus.ACTIVE)
# frozenset({PAUSED, SUCCEEDED, FAILED, CANCELLED, BUDGET_EXHAUSTED})

EXPERIMENT_MACHINE.is_terminal(ExperimentStatus.SUCCEEDED)   # True
EXPERIMENT_MACHINE.can(ExperimentStatus.CREATED, ExperimentStatus.ACTIVE)  # True
```

| Machine | States |
|---------|--------|
| `EXPERIMENT_MACHINE` | `CREATED`, `ACTIVE`, `PAUSED`, `SUCCEEDED`, `FAILED`, `CANCELLED`, `BUDGET_EXHAUSTED` |
| `NODE_MACHINE` | `CREATED`, `PLANNED`, `READY`, `ACTIVE`, `EVALUATING`, `DECIDING`, `COMPLETED`, `REJECTED`, `CANCELLED`, `FAILED` |
| `RUN_MACHINE` | `CREATED`, `ACTIVE`, `SUCCEEDED`, `FAILED`, `CANCELLED` |
| `ATTEMPT_MACHINE` | `CREATED`, `QUEUED`, `STARTING`, `RUNNING`, `CHECKPOINTING`, `RECOVERING`, `SUCCEEDED`, `FAILED`, `PREEMPTED`, `CANCELLED` |

Two rules run through the node, run and attempt tables. **Any non-terminal state can reach `FAILED`**, because anything unfinished can break — a checkpoint write, a recovery attempt, a node mid-evaluation. **Any non-terminal state can reach `CANCELLED`**, because an operator can stop work at any point.

`Experiment` is deliberately outside the first rule: it has no `CREATED → FAILED` edge, because terminalizing an experiment is a controller policy decision rather than a consequence of one workload breaking. An experiment whose nodes have all failed has not necessarily failed.

`REJECTED` and `CANCELLED` are deliberately different outcomes for a node: rejected is a judgement on the candidate's merit, reached only from `DECIDING`, while cancelled means the work stopped before that judgement could be made.

An attempt can be `PREEMPTED` from `QUEUED` onwards but not from `CREATED` — nothing has been submitted yet, so there are no resources to reclaim.

## Value objects

These are immutable descriptors, not handles: none carries a live model, dataset or runtime connection. That is what allows an execution plan to be serialized and handed to another process.

| Type | Purpose |
|------|---------|
| `Actor` | Who or what caused a change — `human`, `rule`, `llm_agent`, `search_provider`, `system` |
| `DatasetRef` | Immutable dataset identity, including transform/tokenizer/template fingerprints |
| `ModelRef` | Immutable model identity |
| `ArtifactRef` | A produced artifact and which attempt or evaluation produced it |
| `RuntimeRef` | A handle to submitted work; backend-neutral, assumes no Kubernetes identifiers |
| `ControllerHostRef` | Which controller host owns an experiment |
| `CheckpointRef` | A committed checkpoint |
| `ResourceUsage` | What an attempt consumed; every field optional |

Dataset fingerprints matter as much as the URI: the same source data processed with a different template or tokenizer is a different scientific input.

```python
from xaytune.core import DatasetRef

base = DatasetRef(uri="support-v4", revision="2026-01-01")
base != base.model_copy(update={"template_fingerprint": "sha256:tpl"})  # True
```

All of these reject unknown fields. Silently dropping a field would lose provenance rather than surface a schema mismatch.

## Errors

| Error | Raised when |
|-------|-------------|
| `XaytuneError` | Base class for every Xaytune error |
| `DomainError` | A domain rule was violated |
| `InvalidIdError` | An identifier is malformed or has the wrong prefix (also a `ValueError`, so Pydantic reports it as a validation error) |
| `InvalidTransitionError` | A state transition is not permitted; carries `aggregate`, `current`, `requested` |
| `ConcurrentModificationError` | An aggregate was modified since it was read |
