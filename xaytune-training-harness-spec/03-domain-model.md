# Domain Model

## 1. Aggregate hierarchy

```text
Experiment
  ├── ExperimentNode*
  │     ├── Run*
  │     │     └── RunAttempt*
  │     ├── EvaluationResult*
  │     └── Decision*
  ├── Action*
  ├── Incident*
  ├── BudgetLedger
  └── Artifact*
```

## 2. Experiment

Represents the complete optimization objective.

```python
class Experiment(BaseModel):
    id: ExperimentId
    name: str
    objective: Objective
    policy_ref: str | None
    budget: BudgetSpec | None

    status: ExperimentStatus
    active_node_ids: list[ExperimentNodeId]
    best_node_id: ExperimentNodeId | None

    controller_host: ControllerHostRef

    created_at: datetime
    updated_at: datetime
    revision: int
    metadata: dict[str, Any]
```

`revision` is used for optimistic concurrency.

## 3. ExperimentNode

Represents a scientific candidate.

```python
class ExperimentNode(BaseModel):
    id: ExperimentNodeId
    experiment_id: ExperimentId

    parent_ids: list[ExperimentNodeId]

    hypothesis: str | None
    reason: str | None

    training_spec: TrainingSpecSnapshot
    training_fingerprint: str

    status: ExperimentNodeStatus

    run_ids: list[RunId]
    evaluation_ids: list[EvaluationId]
    decision_ids: list[DecisionId]

    created_by: Actor
    created_at: datetime
    updated_at: datetime
    revision: int
```

A node should not be created for worker replacement, preemption, or ordinary retry.

## 4. Run

Represents a logical execution of a scientific candidate.

A node can have multiple runs where repeated seeds/replicates are desired.

```python
class Run(BaseModel):
    id: RunId
    node_id: ExperimentNodeId
    experiment_id: ExperimentId

    seed: int | None
    replicate: int | None

    training_fingerprint: str
    execution_plan_ref: str | None

    attempt_ids: list[RunAttemptId]
    final_attempt_id: RunAttemptId | None

    status: RunStatus

    created_at: datetime
    updated_at: datetime
    revision: int
```

## 5. RunAttempt

Represents one infrastructure/runtime attempt.

```python
class RunAttempt(BaseModel):
    id: RunAttemptId
    run_id: RunId
    attempt_number: int

    status: RunAttemptStatus

    runtime_ref: RuntimeRef | None
    execution_fingerprint: str | None

    execution_overrides: list[ExecutionOverride]

    checkpoint_ref: CheckpointRef | None
    artifact_refs: list[ArtifactRef]

    incident_ids: list[IncidentId]

    resource_usage: ResourceUsage

    started_at: datetime | None
    ended_at: datetime | None
    revision: int
```

## 6. ExecutionOverride

An operational modification that preserves scientific training intent.

```python
class ExecutionOverride(BaseModel):
    id: str
    kind: ExecutionOverrideKind
    reason: str

    values: dict[str, Any]

    preserves: list[str]

    incident_id: IncidentId | None
    created_at: datetime
```

Initial allowed override kinds:

- micro batch resize
- gradient accumulation adjustment when preserving effective batch
- worker count adjustment within declared elastic range
- placement retry
- checkpoint restore
- timeout increase within policy
- runtime-native recovery mode

Changes to LR, optimizer, LoRA rank, data, scheduler, reward, algorithm, or model revision are **not** `ExecutionOverride`.

## 7. Objective

```python
class Objective(BaseModel):
    primary: ObjectiveMetric
    target: float | None
    constraints: list[MetricConstraint]
```

Example:

```python
Objective(
    primary=ObjectiveMetric(
        name="task_success",
        direction="maximize",
    ),
    target=0.85,
    constraints=[
        MetricConstraint(name="latency_ms", operator="<=", value=120),
    ],
)
```

## 8. Action

All mutations are represented as typed actions.

```python
class Action(BaseModel):
    id: ActionId
    experiment_id: ExperimentId

    type: str
    status: ActionStatus

    proposed_by: Actor
    reason: str

    target: ActionTarget

    payload: dict[str, Any]

    policy_decision_id: str | None

    created_at: datetime
    updated_at: datetime
    revision: int
```

## 9. Incident

```python
class Incident(BaseModel):
    id: IncidentId
    experiment_id: ExperimentId
    run_id: RunId
    attempt_id: RunAttemptId

    category: IncidentCategory
    severity: IncidentSeverity

    message: str
    evidence: IncidentEvidence

    recoverability: Recoverability
    classification_source: str

    created_at: datetime
```

## 10. Artifact

```python
class ArtifactRef(BaseModel):
    id: ArtifactId
    kind: ArtifactKind
    uri: str

    digest: str | None

    producer_attempt_id: RunAttemptId | None
    producer_evaluation_id: EvaluationId | None

    metadata: dict[str, Any]
```

Artifact kinds:

- model
- adapter
- checkpoint
- tokenizer
- metrics
- evaluation report
- dataset snapshot
- logs
- execution manifest
- provenance bundle

## 11. DatasetRef

Dataset identity must be immutable enough for provenance.

```python
class DatasetRef(BaseModel):
    uri: str
    revision: str | None
    split: str | None
    content_digest: str | None

    transform_fingerprint: str | None
    tokenizer_fingerprint: str | None
    template_fingerprint: str | None

    metadata: dict[str, Any]
```

## 12. Actor

```python
class Actor(BaseModel):
    type: Literal[
        "human",
        "rule",
        "llm_agent",
        "search_provider",
        "system",
    ]
    id: str
    metadata: dict[str, Any] = {}
```

## 13. IDs

Use UUIDv7 or ULID-like sortable IDs.

Recommended prefixes:

```text
exp_
node_
run_
attempt_
act_
inc_
eval_
artifact_
ckpt_
decision_
event_
```

IDs are stable and never recycled.
