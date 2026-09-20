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
NodeEvaluationStarted
NodeDecisionMade
NodeCompleted

RunCreated
RunAttemptCreated
RuntimeSubmitted
RuntimeStarted
RuntimeCompleted

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
EvaluationCompleted
```

## 3. Log context

Every log should include where applicable:

```text
experiment_id
node_id
run_id
attempt_id
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
xaytune.training_fingerprint
xaytune.execution_fingerprint
xaytune.compiler
xaytune.runtime
xaytune.recovery_count
```

## 5. Provenance bundle

Every promoted artifact should be able to export:

```text
provenance/
  experiment.json
  graph.json
  training-spec.json
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
- training fingerprint
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
