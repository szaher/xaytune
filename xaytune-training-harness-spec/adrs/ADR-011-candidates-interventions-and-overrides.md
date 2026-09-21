# ADR-011 — Scientific candidates, training interventions, and execution overrides

## Status
Accepted — 2026-09-20.

Supersedes the two-level lineage model in ADR-003 and extends the identity model in
ADR-006. It was accepted ahead of several earlier ADRs
because PR-005 cannot define its event schema without it.

The rollback question raised in the first draft is decided below rather than deferred:
it constrains event identity, checkpoint metadata, replay and realization fingerprinting,
so leaving it to implementation would have meant a migration.

## Context

ADR-003 splits every change into two categories: an operational one that produces a
`RunAttempt` or `ExecutionOverride`, and a scientific one that produces a new
`ExperimentNode`. That binary breaks down for changes applied to a run that is still
going.

A learning-rate drop at step 14,250, taken because the loss became unstable, is not
operational — it changes training semantics. But forking a node is the wrong record:
the optimizer state, the data position and the model weights all carry forward, so the
"before" is not a candidate anyone would ship, and the two sides are not comparable as
alternatives. Forcing it into a node turns the experiment graph into a change log; forcing
it into an `ExecutionOverride` makes that type's "preserves training intent" guarantee
false.

Continued pretraining, curriculum schedules, reward-coefficient ramps and staged
freeze/unfreeze all have the same shape. The gap is a category for a scientifically
meaningful change applied to a continuing model trajectory.

## Decision

In brief:

> A `TrainingIntervention` persists both the condition that justified it
> (`InterventionTrigger`) and an explicit `InterventionReplayPolicy`. Replay policy is
> never inferred from origin. `REARM_TRIGGER` is valid only when the persisted
> trigger is reconstructible from restored state *and* can become false again.
> `InterventionApplication` records each concrete application separately; rollback does
> not delete previous applications, and re-application produces a new application event.
> `RunHistoryFingerprint` hashes the ordered sequence of applications, not only
> decisions, and `ArtifactLineageFingerprint` hashes only the applications on the
> trajectory that actually produced the artifact.

### 1. Lineage has four levels, not two

| Level | Represents | Created by |
|---|---|---|
| `ExperimentNode` | An alternative scientific candidate | A hypothesis worth comparing independently |
| `TrainingIntervention` | A scientifically meaningful change to a continuing trajectory | An approved `Action` targeting a live `Run` |
| `ExecutionOverride` | An operational adjustment that preserves declared training intent | Recovery policy |
| `RunAttempt` | One infrastructure execution attempt | Retry, preemption, checkpoint restore |

### 2. Comparability decides node versus intervention

> A change creates a new `ExperimentNode` when the changed configuration is an
> alternative candidate you may want to compare independently against the current one.
>
> A change is a `TrainingIntervention` when it only makes scientific sense as a
> continuation of the existing model trajectory.

The kind of parameter does not decide this. Experimental intent does. The same
technical change is either, depending on what is being asked:

```text
At checkpoint 20k, branch A keeps LR 2e-5 and branch B uses 1e-5,
to see which is better.
    -> two nodes, because the point is the comparison

At step 20k the run destabilises, so LR is lowered and training continues.
    -> one run, one intervention, because there is no "branch A" to ship
```

### 3. An intervention is an Action outcome, never a mechanism

`Action` already carries the governance path — proposed, validated against policy,
charged to budget, held for human approval where required, then executed. Interventions
reuse it exactly. `TrainingIntervention` is the durable scientific record of an Action
that was approved and applied; it is never a second way to mutate training state.

This keeps one governance path and one audit trail, and it keeps Invariant E intact:
an LLM planner proposes a `ChangeLearningRate` action like any other, and policy decides.

### 4. Origin is recorded, not inferred

```python
class InterventionOrigin(str, Enum):
    SCHEDULED = "scheduled"              # declared in the candidate before training
    REACTIVE_AGENT = "reactive-agent"
    REACTIVE_HUMAN = "reactive-human"
    REACTIVE_POLICY = "reactive-policy"
```

A scheduled intervention carries `schedule_ref` pointing at the `ScheduledIntervention`
declared in the candidate. This distinction is scientific, not cosmetic: a pre-registered
training program and a researcher reacting to a result they have already seen are
different claims about a result, and a provenance system that cannot separate them hides
the thing a reviewer most needs.

### 5. Identity is layered

```text
CandidateFingerprint
    ModelSpec + DataSpec + TrainingSpec + RewardSpec + EnvironmentSpec
  + pre-registered intervention schedule

RunHistoryFingerprint
    CandidateFingerprint + seed/replicate + ordered applied interventions,
    including rolled-back work

ArtifactLineageFingerprint
    CandidateFingerprint + seed/replicate + causal checkpoint ancestry
    + applications on the retained trajectory only

ExecutionFingerprint
    software / runtime / hardware / topology + execution overrides

EvaluationFingerprint          (unchanged, ADR-007)
CheckpointCompatibilityKey     (unchanged, ADR-009)
```

A scheduled LR decay belongs to `CandidateFingerprint`, because it was declared before
training: two runs executing the same declared schedule remain replicates of one
candidate. A reactive LR drop was not declared, so it changes
both run fingerprints while leaving the node and candidate intact.

### 6. Evaluation is not part of candidate identity

`CandidateSpec` contains model, data, training, reward and environment. It does **not**
contain `EvaluationSpec`. Evaluation attaches to the node, the run or the artifact.

This preserves ADR-006 and ADR-007: changing a grader must never imply retraining.

**Reward versus evaluator.** The same grader object can appear in both roles, and agent
training makes this common. The role, not the object, decides:

```text
rollout -> grader -> reward -> gradient update
    the grader is a training input; it belongs to RewardSpec
    and contributes to CandidateFingerprint

trained artifact -> grader -> score
    the grader is an evaluator; it belongs to EvaluationSpec
    and contributes only to EvaluationFingerprint
```

### 7. Interventions across checkpoint rollback

> An intervention **decision** is durable and belongs to the `Run`. Each time it takes
> effect, Xaytune records a distinct **application**. Restoring a checkpoint never erases
> prior applications. After a restore the controller re-applies only interventions whose
> effective training position is ahead of the restored position *and* whose replay policy
> calls for it.

Three objects, deliberately separate:

```text
TrainingIntervention     the scientific/governance decision
                         "lower LR to 1e-5 when condition X is reached"

InterventionApplication  one concrete occurrence
                         "int_42 applied at optimizer step 20,000"

CheckpointRef            the recoverable training position
```

```python
class TrainingIntervention(BaseModel):
    id: InterventionId
    run_id: RunId
    action_id: ActionId

    origin: InterventionOrigin

    # Why this intervention exists. Required: every intervention records its cause.
    trigger: InterventionTrigger

    # Explicit. Never inferred from origin. Immutable once created.
    replay_policy: InterventionReplayPolicy

    schedule_ref: ScheduledInterventionId | None = None
    derived_from: InterventionId | None = None   # set when replaying a prior run

    mutation: TrainingMutation
    rationale: str
    evidence_refs: list[str] = []

    created_at: datetime


class InterventionApplication(BaseModel):
    id: InterventionApplicationId
    intervention_id: InterventionId
    attempt_id: RunAttemptId

    event_sequence: int                      # assigned by the repository at commit
    position: TrainingPosition

    trigger_evaluation: TriggerEvaluation | None

    previous_value: Any
    applied_value: Any
    checkpoint_ancestor: CheckpointRef | None

    created_at: datetime


class TriggerEvaluation(BaseModel):
    matched: bool
    observed_values: dict[str, Any]
    evaluated_at: datetime
```

`TriggerEvaluation` is what lets the system answer, months later, *why* an intervention
re-applied after a rollback — with the observed values it decided on, not a guess.

#### Triggers are a tagged union, not a loose dict

```python
class StepTrigger(BaseModel):
    type: Literal["step"]
    global_step: int

class OptimizerStepTrigger(BaseModel):
    type: Literal["optimizer-step"]
    optimizer_step: int

class TokenCountTrigger(BaseModel):
    type: Literal["tokens"]
    tokens_seen: int

class MetricTrigger(BaseModel):
    type: Literal["metric"]

    # Identity of the quantity, not just its display name. A re-armed trigger
    # must resolve to the same series months later.
    metric_ref: str
    metric_schema_version: str
    source: MetricSource                 # training stream | evaluator | derived
    aggregation: MetricAggregation       # last | mean | median | max | min

    operator: Literal[">", ">=", "<", "<=", "=="]
    threshold: float
    window: int | None = None
    consecutive: int | None = None
    evaluator_ref: str | None = None

class IncidentTrigger(BaseModel):
    type: Literal["incident"]
    incident_category: IncidentCategory
    incident_id: IncidentId | None = None

class ManualTrigger(BaseModel):
    type: Literal["manual"]
    actor: Actor

class PolicyTrigger(BaseModel):
    type: Literal["policy"]

    policy_rule_id: str
    policy_version: str
    policy_digest: str                   # canonical hash of the rule as evaluated


InterventionTrigger = (
    StepTrigger | OptimizerStepTrigger | TokenCountTrigger
    | MetricTrigger | IncidentTrigger | ManualTrigger | PolicyTrigger
)
```

#### Replay policy is explicit, required and immutable

```python
class InterventionReplayPolicy(str, Enum):
    REAPPLY_AFTER_ROLLBACK = "reapply-after-rollback"
    APPLY_ONCE = "apply-once"
    REARM_TRIGGER = "rearm-trigger"
```

`REARM_TRIGGER` was called `REEVALUATE_TRIGGER`, and the rename fixes a real
bug rather than a word. "Re-evaluate" implies a single test at restore time, so
a condition that is false at the restored position simply drops the intervention
— permanently:

```text
step 14,000   loss > threshold     -> LR reduction applied
failure       rollback to 12,000
step 12,000   loss < threshold     -> evaluates false
                                   -> intervention silently gone forever
```

The run then continues past 14,000 with nothing watching, even though the
condition that justified the intervention is exactly the kind that recurs. A
reactive trigger is a standing condition, not a one-shot test.

Re-arming is the correct semantics:

```text
restore to a position before the application
        ↓
reconstruct the trigger's evaluation state
        ↓
evaluate immediately
        ├── true  -> apply now
        └── false -> ARM it and keep observing
        ↓
it may fire again later in the run
```

The field has **no default**. An intervention created without choosing replay semantics
fails validation.

> **Invariant.** Replay behaviour is part of intervention semantics. It must be explicit,
> durable and immutable after the intervention is created, and must never be inferred
> from origin at replay time.

Position alone cannot decide re-application. A scheduled LR stage at step 20,000 is part
of the declared program and must fire again if the run rewinds past it. A human emergency
adjustment, taken once against a condition that may no longer hold, must not silently
become permanent automation because a worker died.

| Origin | Recommended policy | Reason |
|---|---|---|
| `SCHEDULED` | `REAPPLY_AFTER_ROLLBACK` | It was part of the declared training program |
| `REACTIVE_AGENT` | `REARM_TRIGGER` | The condition stands; keep watching for it |
| `REACTIVE_POLICY` | `REARM_TRIGGER` | Re-arm the policy condition against restored state |
| `REACTIVE_HUMAN` | `APPLY_ONCE` | Do not convert a one-off human judgement into automation |

These are **recommendations for the proposer, not implicit defaults**. Recording the
policy explicitly is what stops a later change of origin — say `REACTIVE_HUMAN` to
`REACTIVE_AGENT` — from quietly changing rollback behaviour.

#### `REARM_TRIGGER` requires a non-monotone trigger

A trigger is only re-evaluable if its condition can become **false** again.

Position, optimizer-step and token triggers are monotone: once `global_step >= 20000`
holds, it holds forever. Re-evaluating one after a restore always matches, so the
intervention re-applies even when its effect is already present in the restored optimizer
state — a silent double-application. The positional guard in `REAPPLY_AFTER_ROLLBACK`
exists precisely to handle those correctly.

| Trigger | Valid with `REARM_TRIGGER` | Why |
|---|---|---|
| `MetricTrigger` | yes | The metric can fall back below threshold, and rise again |
| `PolicyTrigger` | yes, if the rule is state-dependent | Re-evaluable against restored state |
| `StepTrigger`, `OptimizerStepTrigger`, `TokenCountTrigger` | **no** | Monotone — always re-matches, double-applies |
| `IncidentTrigger` | **no** | An incident is a past occurrence, not a standing condition |
| `ManualTrigger` | **no** | A human decision cannot be re-derived from state |

Rejected at Action validation time, not at replay time, so the failure surfaces when the
intervention is proposed rather than during a recovery.

#### Re-arming requires reconstructible trigger state

Arming a condition is more demanding than testing one, because the trigger must
be evaluable against a *restored* position rather than against whatever the
controller happens to hold in memory. A trigger recorded as

```python
metric="loss", threshold=0.5, window=100
```

cannot be reconstructed: "loss" does not say which series, from which source,
aggregated how, under which schema. Two months later it may not even name the
same quantity.

So a re-armable trigger persists its full evaluation contract — for metrics the
`metric_ref`, schema version, source and aggregation alongside the comparator,
window and `consecutive` count; for policy rules the `policy_version` and
`policy_digest` as well as the rule id. Without the digest a replay evaluates
whatever the rule means *now*, silently answering a different question than the
one that justified the original intervention, and the provenance record would
claim otherwise.

Where the state cannot be reconstructed, the intervention is not eligible for
`REARM_TRIGGER` and validation rejects it — the same rule as monotone triggers,
for the same reason: fail at proposal time, not during a recovery.

#### Rollback logic

```python
for intervention in run.interventions:
    if intervention.replay_policy is APPLY_ONCE:
        continue

    if intervention.replay_policy is REAPPLY_AFTER_ROLLBACK:
        if applied_after(intervention, restored_position):
            schedule_reapplication(intervention)

    elif intervention.replay_policy is REARM_TRIGGER:
        if not applied_after(intervention, restored_position):
            continue                      # its effect survives in restored state

        state = reconstruct_trigger_state(intervention.trigger, restored_position)
        evaluation = evaluate(intervention.trigger, state)

        if evaluation.matched:
            schedule_reapplication(intervention, evaluation)
        else:
            arm(intervention.trigger)     # keep observing; it may fire again
```

The `else` branch is the whole fix. Dropping the intervention there is what made
the old name wrong, and the positional guard on the first line is what stops a
re-arm from double-applying an effect that is already baked into the restored
optimizer state.

Both identifiers are load-bearing: `event.sequence` orders history, while
`TrainingPosition` decides whether the restored model state predates an application.

#### Deliberately not split: cause versus effective-from

There are two notions hiding in `trigger` — why an intervention exists, and when it
should take effect. They usually coincide, but not always: a loss spike at step 14,000
may justify a change that is applied at the next checkpoint boundary.

For v1 the pair `trigger` (why it exists) and `InterventionApplication.position` (where
it took effect) carries that distinction without a further abstraction. A separate
`effective_from` or `ApplicationCondition` is deferred until a case needs it. Recorded
here so it is a known deferral rather than an oversight.

#### Two histories, two fingerprints

An earlier version had one `RunRealizationFingerprint` hashing
`CandidateFingerprint + seed/replicate + ordered InterventionApplication[]`, and
used it to answer both "what happened during this run?" and "what trajectory
produced this artifact?". Those are different questions, and one hash cannot
answer both:

```text
checkpoint C
    ↓ intervention A applied
    ↓ 5,000 steps
  failure
    ↓ rollback to C          <- those 5,000 steps are discarded
    ↓ intervention A applied again
    ↓ train to completion
```

The first application and its 5,000 steps are part of the run's history and
**did not causally contribute to the artifact**. Including them in the identity
used for trajectory reuse means two runs that produced the same trajectory look
different because one of them had a bad afternoon. Excluding them from the audit
record means the rollback disappears. Both are wrong, which is the signal that
there are two identities.

```text
RunHistoryFingerprint
    all durable history: every attempt, every application, every rollback,
    every discarded span
    -> audit, debugging, "what did this run actually do?"

ArtifactLineageFingerprint
    CandidateFingerprint + seed/replicate
    + the causal checkpoint ancestry of the artifact
    + only the applications on the retained trajectory
    + effective execution lineage
    -> scientific trajectory reuse, "was this the same training path?"

ArtifactDigest
    the bytes
    -> byte-identical artifact identity
```

`RunRealizationFingerprint` is **retired as a single name**, because every use
of it had to mean one or the other. Call sites say which they want.

This also repairs a claim made elsewhere in this ADR: "the exact trajectory's
artifact" was never true of a fingerprint that included rolled-back work.

Two consequences carry over to both:

- They are **provisional until the run reaches a terminal state**. Reuse lookups
  must not match against an in-flight run.
- `event_sequence` is assigned by the repository inside the transaction, so an
  application record cannot be fully constructed by the caller beforehand.

And one is specific to `ArtifactLineageFingerprint`: computing it requires the
checkpoint ancestry, so a checkpoint must record the position and applications
it embodies. ADR-012 §4 already requires exactly that, which is what makes the
causal trajectory recoverable rather than inferred.

#### Reproduction is a distinct provenance claim

Replaying a reactive intervention as a predeclared schedule may reproduce approximately
the same parameter trajectory, but it is not a reproduction of the original scientific
process. The record must say so:

```text
original      origin = REACTIVE_HUMAN,  schedule_ref = None
reproduction  origin = SCHEDULED,       schedule_ref = reproduction-plan,
                                        derived_from = <original intervention id>
```

## Consequences

### A fingerprint is an identity, not a reproduction recipe

`ArtifactLineageFingerprint` answers "was this the same trajectory?" It does **not**
answer "can I reproduce this?" A reactive intervention was triggered by a stochastic
event — a loss spike at a particular step — so rerunning the candidate with the same
seed will not reproduce it.

Reproducing such a run means replaying its recorded interventions at their recorded
positions, which is a different operation from rerunning a candidate, and it produces a
run whose interventions are `SCHEDULED` rather than reactive. Documentation must not
let matching fingerprints imply reproducibility.

### Ordering is by event sequence, not training position

```python
class TrainingPosition(BaseModel):
    global_step: int | None
    optimizer_step: int | None
    tokens_seen: int | None
    examples_seen: int | None
```

The two answer different questions, and both are needed:

```text
event.sequence      in what order did Xaytune observe and apply things?
TrainingPosition    where in model training did this occur?
```

`TrainingPosition` cannot be the ordering key, because it is not monotonic across a run.
This is a perfectly valid history:

```text
step 20,000
step 25,000
restore -> step 22,000
step 23,000
```

`event.sequence` is authoritative for ordering, and for the canonical sequence of
`InterventionApplication` records that the run fingerprints hash.

## Reuse modes

ADR-006 described reuse as a single fingerprint match. There are four questions, and
they take different keys:

| Question | Match on |
|---|---|
| **Candidate reuse** — has this hypothesis been explored before? | `CandidateFingerprint` |
| **Artifact reuse** — give me any acceptable completed artifact from this candidate | `CandidateFingerprint`, any terminal realization |
| **Trajectory reuse** — do we have the artifact from this exact trajectory? | `ArtifactLineageFingerprint` |
| **Audit** — what did this run actually do? | `RunHistoryFingerprint` |
| **Evaluation reuse** — has this artifact been scored with this evaluator? | artifact digest + `EvaluationFingerprint` |
| **Resume** — can we restart from this checkpoint? | `CheckpointCompatibilityKey` |

Artifact reuse is the common case in practice and had no expressible answer before.

## Projection integrity is a test, not a convention

`RunRealization` is derived from the event log and the log stays authoritative (ADR-005).
That has to be enforced, not assumed:

```python
stored = repository.get_run_realization(run_id)
recomputed = projector.rebuild_run_realization(repository.events_for_run(run_id))

assert stored == recomputed
```

A failure here is a provenance bug, not a caching bug. PR-005 owns this test.

## Rejected alternatives

**Keep the binary split and force in-run changes into new nodes.** Turns the experiment
graph into a change log and makes "which candidates did we compare?" unanswerable.

**Widen `ExecutionOverride` to cover scientific changes.** Destroys the one guarantee
that type provides — that it preserves declared training intent — and with it the value
of the operational/scientific boundary.

**Give `TrainingIntervention` its own approval path.** Creates a second way to mutate
training state, which puts a hole in Invariant E and splits the audit trail.

**Put `EvaluationSpec` inside `CandidateSpec`.** Makes grader changes alter candidate
identity, implying a retrain. This is the exact failure ADR-006 exists to prevent.
