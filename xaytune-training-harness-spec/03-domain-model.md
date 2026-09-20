# Domain Model

## 1. Aggregate hierarchy

```text
Experiment
  ├── ExperimentNode*
  │     ├── Run*
  │     │     ├── RunAttempt*
  │     │     ├── TrainingIntervention*
  │     │     │     └── InterventionApplication*
  │     │     └── RunRealization (projection)
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

    candidate: CandidateSpecSnapshot
    candidate_fingerprint: str

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

### CandidateSpec

A candidate is the whole scientific proposition, not just trainer hyperparameters
(ADR-011):

```python
class CandidateSpec(BaseModel):
    model: ModelSpec
    data: DataSpec
    training: TrainingSpec
    reward: RewardSpec | None
    environment: EnvironmentSpec | None
    schedule: TrainingSchedule | None    # pre-registered interventions
```

`EvaluationSpec` is deliberately **not** part of a candidate. Making a grader part of
candidate identity would mean changing a grader implies a retrain, which is the failure
ADR-006 exists to prevent. Evaluation attaches to the node, run or artifact.

The same grader can appear in both roles: used inside the training loop it is a reward
and belongs to `RewardSpec`; used to score the artifact it is an evaluator and
contributes only to `EvaluationFingerprint`. The role decides, not the object.

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

    candidate_fingerprint: str
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

## 6b. TrainingIntervention and InterventionApplication

A scientifically meaningful change to a run that is still going (ADR-011). Never an
execution mechanism: every intervention is the recorded outcome of an approved `Action`,
so there is one governance path and one audit trail.

```python
class TrainingIntervention(BaseModel):
    id: InterventionId
    run_id: RunId
    action_id: ActionId

    origin: InterventionOrigin           # scheduled | reactive-agent
                                         # | reactive-human | reactive-policy
    trigger: InterventionTrigger         # why it exists; required
    replay_policy: InterventionReplayPolicy   # explicit, immutable, never inferred

    schedule_ref: ScheduledInterventionId | None
    derived_from: InterventionId | None  # set when replaying a prior run

    mutation: TrainingMutation
    rationale: str
    evidence_refs: list[str]

    created_at: datetime
```

The **decision** above is distinct from each **application** below. A decision is made
once; it may take effect more than once, because a checkpoint restore can rewind past
the point where it was applied.

```python
class InterventionApplication(BaseModel):
    id: InterventionApplicationId
    intervention_id: InterventionId
    attempt_id: RunAttemptId

    event_sequence: int                  # assigned by the repository at commit
    position: TrainingPosition

    trigger_evaluation: TriggerEvaluation | None

    previous_value: Any
    applied_value: Any
    checkpoint_ancestor: CheckpointRef | None

    created_at: datetime
```

Restoring a checkpoint never deletes prior applications. After a restore the controller
re-applies only interventions whose position is ahead of the restored position *and*
whose replay policy calls for it. `REEVALUATE_TRIGGER` is valid only for conditions that
can become false again — pairing it with a step or token trigger would re-match on every
restore and double-apply a change already present in the restored optimizer state.

### TrainingPosition

```python
class TrainingPosition(BaseModel):
    global_step: int | None
    optimizer_step: int | None
    tokens_seen: int | None
    examples_seen: int | None
```

`TrainingPosition` says *where in training* something happened. It is not monotonic
across a run, because a restore moves it backwards, so it can never be the ordering key.
`event.sequence` orders history; `TrainingPosition` decides whether restored model state
predates an application.

## 6c. RunRealization

What actually produced an artifact, as against what the node intended to test.

```python
class RunRealization(BaseModel):
    run_id: RunId

    candidate_fingerprint: str
    realization_fingerprint: str

    initial_spec: CandidateSpecSnapshot

    scheduled_interventions: list[ScheduledInterventionRef]
    applied_interventions: list[InterventionApplicationRef]
    execution_overrides: list[ExecutionOverrideRef]

    final_effective_state: EffectiveTrainingState
```

`applied_interventions` holds **applications, not interventions**, and the
distinction is load-bearing. One intervention can be applied more than once —
apply, roll back to an earlier checkpoint, apply again — and ADR-011 hashes the
ordered `InterventionApplication` records into `RunRealizationFingerprint`. With
only intervention references these two histories are indistinguishable:

```text
intervention I, application A                          -> one trajectory
intervention I, application A, rollback, application B -> a different trajectory
```

They are different realizations of the same candidate and must not collide.
Each `InterventionApplication` records when it was applied, at which step and
checkpoint, and by which `Action`.

**`RunRealization` is a projection, never a second source of truth.** Every field is
derived from the durable event log, which stays authoritative. It must be recomputable
from that log and the recompute must be exercised in tests:

```python
assert repository.get_run_realization(run_id) == projector.rebuild_run_realization(
    repository.events_for_run(run_id)
)
```

A failure there is a provenance bug, not a caching bug. Materialising
`final_effective_state` as independently mutable state would reintroduce exactly the
divergence that atomic state-and-event commits exist to prevent.

`node.candidate` answers "what did we intend to test?"; `run.realization()` answers
"what actually trained this?"

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
intervention_
application_
```

IDs are stable and never recycled.
