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
  │     ├── EvaluationRun*
  │     │     ├── EvaluationAttempt*
  │     │     └── EvaluationResult
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
    evaluation_run_ids: list[EvaluationRunId]
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

An attempt exists because the *workload* was started, restarted or replaced —
never because observability failed. A telemetry stream that dies while the
workload keeps running advances `telemetry_generation` on the same attempt
instead of creating a new one; minting an attempt there would record an
execution retry that never happened, and corrupt retry counts, per-attempt
resource usage and incident attribution (ADR-014 §1a).

```python
class RunAttempt(BaseModel):
    id: RunAttemptId
    run_id: RunId
    attempt_number: int

    status: RunAttemptStatus

    runtime_ref: RuntimeRef | None
    execution_fingerprint: str | None

    telemetry_generation: int            # ADR-014; controller-owned

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
whose replay policy calls for it. `REARM_TRIGGER` is valid only for conditions that
can become false again — pairing it with a step or token trigger would re-match on every
restore and double-apply a change already present in the restored optimizer state. A
re-armed trigger that evaluates false at the restored position is **armed, not
discarded**: the condition stands and may fire again later in the run.

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
    history_fingerprint: str            # everything, rolled-back work included
    artifact_lineage_fingerprint: str   # only the trajectory behind the artifact

    initial_spec: CandidateSpecSnapshot

    scheduled_interventions: list[ScheduledInterventionRef]
    applied_interventions: list[InterventionApplicationRef]
    execution_overrides: list[ExecutionOverrideRef]

    final_effective_state: EffectiveTrainingState
```

`applied_interventions` holds **applications, not interventions**, and the
distinction is load-bearing. One intervention can be applied more than once —
apply, roll back to an earlier checkpoint, apply again — and ADR-011 hashes the
ordered `InterventionApplication` records into `RunHistoryFingerprint`. With
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
"what actually trained this?" — and it answers it twice, because those are two
questions. `history_fingerprint` covers everything the run did, including work
that was rolled back and discarded. `artifact_lineage_fingerprint` covers only
the trajectory the artifact descends from. A rollback changes the first and not
the second, which is what makes trajectory reuse possible at all (ADR-011).


## 5a. EvaluationRun and EvaluationAttempt

Evaluation is a workload, not a function call, so it has the same Run/Attempt
split as training (ADR-015):

```python
class EvaluationRun(AggregateModel):
    id: EvaluationRunId
    node_id: ExperimentNodeId

    spec: EvaluationSpec
    subject: ArtifactRef              # the checkpoint or model being evaluated
    fingerprint: str                  # EvaluationFingerprint

    seed: int | None                  # replicate identity lives here, never on
    replicate: int | None             # EvaluationSpec -- see below

    attempt_ids: list[EvaluationAttemptId]
    result: EvaluationResult | None   # set only when terminal and SUCCEEDED

    status: EvaluationRunStatus

class EvaluationAttempt(AggregateModel):
    id: EvaluationAttemptId
    evaluation_run_id: EvaluationRunId
    attempt_number: int

    runtime_ref: RuntimeRef | None
    telemetry_generation: int         # ADR-014; same rule as RunAttempt
    status: EvaluationAttemptStatus
```

`telemetry_generation` is present for the same reason it is on `RunAttempt`: an
evaluation streams through the same `RuntimeEventEnvelope`, so a supervisor that
dies over a live evaluation needs somewhere durable to allocate its next
generation (ADR-014 §1a). A lost stream never creates a new attempt here either.

The state machines are **not** copies of `Run`/`RunAttempt`. Evaluation produces
no checkpoints, so there is no `CHECKPOINTING` and nothing to recover into, so
no `RECOVERING`; a failed evaluation is retried as a new attempt. The tables are
in ADR-015.

`EvaluationSpec` is deliberately not part of `CandidateSpec` — see §4. An
evaluation attaches to a node, run or artifact, and contributes only to
`EvaluationFingerprint`.

`seed` and `replicate` sit on `EvaluationRun` for the same reason they sit on
`Run` and not on `CandidateSpec`: a seed on the spec would make two seeds of one
evaluation into two different evaluations, and would drag replicate identity
into `EvaluationFingerprint`. The seed participates in the `SEEDED` reuse
lookup key instead (ADR-015 §3).

`EvaluationResult` carries `evaluation_run_id`. With several runs over the same
subject — replicates of a stochastic evaluation — it is the only thing that
answers which execution produced a given sample.

**Invariant (reconciliation, not point-in-time).** A node in `EVALUATING`
resolves to exactly one of: a required `EvaluationRun` is non-terminal, so wait;
all required runs are terminal with results, so reconcile the node forward to
`DECIDING`; or neither, which raises `EvaluationStalled`. Without it a node sits
in `EVALUATING` forever when the evaluation process dies. Asserting it
point-in-time instead would fire on every successful evaluation, in the gap
between the run reaching `SUCCEEDED` and the node leaving `EVALUATING`
(ADR-015 §5).

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

    type: str                      # validated by the action registry, not a DB CHECK
    status: ActionStatus
    outcome: ActionOutcome | None  # set exactly when status is SUCCEEDED

    proposed_by: Actor
    reason: str

    target: ActionTarget

    payload: dict[str, Any]

    policy_decision_id: str | None

    created_at: datetime
    updated_at: datetime
    revision: int
```

### ActionTarget

```python
ActionTargetKind = Literal[
    "experiment",
    "node",
    "run",
    "training-attempt",
    "evaluation-run",
    "evaluation-attempt",
]

class ActionTarget(BaseModel):
    kind: ActionTargetKind
    id: str
```

The attempt kinds match `RuntimeOperationTarget` (ADR-013) deliberately. An
Action's target and the target of the operation it causes are the same subject,
and a separate `run-attempt` spelling would mean translating between two
contracts that are supposed to be shared.

`evaluation-attempt` is not deferrable. ADR-015 says an evaluation can be
cancelled and its AC-7 requires a cancelled evaluation to leave no executing
workload; ADR-005 §5 requires an `Action` to own that intent in the same
transaction as the cancel operation. Without this kind there is no legal Action
for an evaluation cancellation, so the invariant could not be satisfied.

`CancelAttempt` is therefore **workload-neutral** — it targets a training or an
evaluation attempt — rather than splitting into `CancelRunAttempt` and
`CancelEvaluationAttempt`. Cancelling is the same operation on the same kind of
subject; only the table differs.

### ActionOutcome

```python
class ActionOutcome(str, Enum):
    APPLIED = "applied"          # the change was made
    SUPERSEDED = "superseded"    # overtaken by events; nothing left to do
    NOOP = "noop"                # already in the requested state
```

Every member describes a way of *succeeding*, so the field pairs with
`SUCCEEDED` and with nothing else:

```text
status == SUCCEEDED     outcome MUST be non-null
status != SUCCEEDED     outcome MUST be null
```

```text
SUCCEEDED / APPLIED       SUCCEEDED / SUPERSEDED      SUCCEEDED / NOOP
FAILED    / null          REJECTED  / null
```

"Set when terminal" would have been wrong: `FAILED` and `REJECTED` are terminal
too, and no member of this enum describes either. Their reasons belong to the
transition event and the policy decision that produced them, and inventing an
outcome to duplicate them there would make the field mean two different things.

`status` says whether the action completed; `outcome` says how it turned out.
Collapsing them loses ADR-013 §5:

```text
cancel requested, workload finishes naturally first
    status  = SUCCEEDED
    outcome = SUPERSEDED
```

That action did exactly what it was asked, and the answer was that there was
nothing left to stop. Recording it `FAILED` would make a routine race look like
a defect; recording it plain `SUCCEEDED` would claim it cancelled something it
did not. The distinction generalises to every idempotent action — already at the
requested value is `NOOP`, not a successful change — which is worth having in a
control plane where an agent reads outcomes back.

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
