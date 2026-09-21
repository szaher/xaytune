# Observability and Provenance

## 1. Structured event-first design

Do not make MLflow/W&B the source of truth.

Xaytune emits structured events.

Sinks consume them.

## 2. Event classes

Initial high-value events:

```text
ExperimentCreated
ExperimentActivated
ExperimentPaused
ExperimentCompleted

NodeCreated
NodeActivated
NodeDecisionMade
NodeCompleted

RunCreated
RunAttemptCreated
RuntimeSubmitted
RuntimeStarted
RuntimeCompleted

EventGapDetected
TelemetryStreamGenerationAdvanced
ObservabilityDegraded

MetricObserved

IncidentDetected
IncidentClassified

RecoveryPlanned
RecoveryStarted
RecoverySucceeded
RecoveryFailed

CheckpointSaveStarted
CheckpointCommitted
CheckpointRestoreStarted
CheckpointRestored

ActionProposed
ActionValidated
ActionApproved
ActionRejected
ActionExecuted

BudgetReserved
BudgetCommitted
BudgetConsumed
BudgetReleased
BudgetExceeded

ArtifactCreated
```

Evaluation has its own lifecycle events, because after ADR-015 it has durable
`EvaluationRun` and `EvaluationAttempt` entities and a single
`EvaluationCompleted` cannot express a queued, preempted or retried evaluation:

```text
EvaluationRunCreated
EvaluationAttemptCreated
EvaluationRuntimeSubmitted
EvaluationStarted
EvaluationSucceeded
EvaluationFailed
EvaluationPreempted
EvaluationCancelled
```

These mirror the `EvaluationRun`/`EvaluationAttempt` transitions in ADR-015 —
one event per transition, with no `Checkpoint*` or `Recovery*` counterparts,
because evaluation has neither. They must exist before PR-005 defines the event
schema; adding a transition event afterwards is a migration.

## 3. Log context

Every log should include where applicable:

```text
experiment_id
node_id
run_id
attempt_id
stream_generation
evaluation_run_id
evaluation_attempt_id
action_id
incident_id
runtime_backend
compiler
```

## 4. MLflow mapping

Recommended:

```text
Xaytune Experiment
  → MLflow Experiment

ExperimentNode
  → parent/nested run

Run / RunAttempt
  → nested child run or tagged child run
```

Tags:

```text
xaytune.experiment_id
xaytune.node_id
xaytune.run_id
xaytune.attempt_id
xaytune.candidate_fingerprint
xaytune.execution_fingerprint
xaytune.compiler
xaytune.runtime
xaytune.recovery_count
```

## 5. Provenance bundle

Every promoted artifact should be able to export the following. Note the three
separate identity documents: `candidate-spec.json` is what was *declared*,
`run-realization.json` is what actually *happened* — the ordered intervention
applications that ADR-011 hashes into `RunHistoryFingerprint`, with
`ArtifactLineageFingerprint` covering only the trajectory the artifact descends
from — and
`execution-plan.json` is *how* it was run. A candidate spec alone no longer
represents the scientific result, because two runs of the same candidate can
have different realizations.

```text
provenance/
  experiment.json
  graph.json
  candidate-spec.json
  run-realization.json
  execution-plan.json
  runtime.json
  evaluations.json
  incidents.json
  recoveries.json
  actions.json
  budget.json
  software.json
  artifacts.json
```

### Observability-degraded intervals

`TelemetryStreamGenerationAdvanced` and `ObservabilityDegraded` exist so that
provenance can distinguish

```text
nothing happened in this interval
we were not watching during this interval
```

which is otherwise the same absence of events. A telemetry supervisor that dies
while its workload keeps running produces the second, and the record says so on
the **same** `RunAttempt` — the attempt is not re-minted to represent a
monitoring failure (ADR-014 §1a).

## 6. Resource usage

Normalize:

```python
class ResourceUsage(BaseModel):
    wall_seconds: float

    gpu_seconds: float | None
    gpu_count: int | None
    gpu_model: str | None

    cpu_seconds: float | None

    peak_gpu_memory_bytes: int | None
    peak_host_memory_bytes: int | None

    tokens_processed: int | None
    examples_processed: int | None
```

## 7. Training memory

Phase 1:

structured search by:

- model
- dataset
- algorithm
- adapter
- candidate fingerprint — "has this hypothesis been explored?"
- run realization fingerprint — "do we have *this exact* trajectory?"
- execution fingerprint — "was it run on this software/hardware?"
- result
- incident category

Later:

semantic retrieval plugin.

Avoid a vector DB dependency in core.

## 8. Explainability

CLI/API should answer:

```text
Why was this node created?
Why was this candidate promoted?
Why was this run retried?
Why did the controller reduce microbatch?
Why did the experiment stop?
Which policy rejected an action?
```

These answers come from recorded structured rationale and provenance, not reconstructed guesses.
