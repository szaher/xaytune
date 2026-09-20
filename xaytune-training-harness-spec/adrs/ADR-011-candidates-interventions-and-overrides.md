# ADR-011 — Scientific candidates, training interventions, and execution overrides

## Status
Accepted — 2026-09-20.

Supersedes the two-level lineage model in ADR-003 and extends the identity model in
ADR-006. ADR-001 through ADR-010 remain `Proposed`; this one was accepted ahead of them
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

RunRealizationFingerprint
    CandidateFingerprint + seed/replicate + ordered applied interventions

ExecutionFingerprint
    software / runtime / hardware / topology + execution overrides

EvaluationFingerprint          (unchanged, ADR-007)
CheckpointCompatibilityKey     (unchanged, ADR-009)
```

A scheduled LR decay belongs to `CandidateFingerprint`, because it was declared before
training: two runs executing the same declared schedule remain replicates of one
candidate. A reactive LR drop was not declared, so it changes
`RunRealizationFingerprint` while leaving the node and candidate intact.

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
    schedule_ref: ScheduledInterventionId | None
    derived_from: InterventionId | None      # set when replaying a prior run

    mutation: TrainingMutation
    trigger: InterventionTrigger | None      # required for REEVALUATE_TRIGGER
    replay_policy: InterventionReplayPolicy  # explicit, never inferred

    rationale: str
    evidence_refs: list[str]
    created_at: datetime


class InterventionApplication(BaseModel):
    id: InterventionApplicationId
    intervention_id: InterventionId
    attempt_id: RunAttemptId

    event_sequence: int                      # assigned by the repository at commit
    position: TrainingPosition

    previous_value: Any
    applied_value: Any
    checkpoint_ancestor: CheckpointRef | None

    created_at: datetime
```

#### Replay policy is explicit and required

```python
class InterventionReplayPolicy(str, Enum):
    REAPPLY_AFTER_ROLLBACK = "reapply-after-rollback"
    APPLY_ONCE = "apply-once"
    REEVALUATE_TRIGGER = "reevaluate-trigger"
```

Position alone cannot decide re-application. A scheduled LR stage at step 20,000 is part
of the declared program and must fire again if the run rewinds past it. A human emergency
adjustment taken once, in response to a condition that may no longer hold, must not
silently become a permanent schedule because a worker died.

| Intervention | Policy |
|---|---|
| Scheduled LR stage, declared curriculum transition | `REAPPLY_AFTER_ROLLBACK` |
| "At exactly token 10B, change the data mixture" | `REAPPLY_AFTER_ROLLBACK` |
| Reactive human or agent adjustment | `APPLY_ONCE` unless stated otherwise |
| "If loss exceeds T for 100 steps" | `REEVALUATE_TRIGGER` |

The field is **required, not derived from `origin`**. Inferring it would mean that
changing an intervention's origin silently changes its replay behaviour, and the table
above is guidance for the proposer rather than a rule the system applies behind their
back.

`REEVALUATE_TRIGGER` requires `trigger` to be set. The condition must itself be durable,
or there is nothing to re-evaluate after the restore. A validation rule rejects the
combination at Action time, not at replay time.

#### The realization fingerprint hashes applications, not decisions

```text
RunRealizationFingerprint =
    CandidateFingerprint + seed/replicate + ordered InterventionApplication[]
```

This is what distinguishes a run where an LR change applied once from a run where it
applied, rolled back, and applied again. Those are different realized trajectories even
though they share one intervention decision, and the artifacts differ.

Two consequences worth stating:

- The fingerprint is **provisional until the run reaches a terminal state**. Reuse
  lookups must not match against an in-flight run's realization.
- `event_sequence` is assigned by the repository inside the transaction, so an
  application record cannot be fully constructed by the caller beforehand.

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

`RunRealizationFingerprint` answers "was this the same trajectory?" It does **not**
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
`InterventionApplication` records that `RunRealizationFingerprint` hashes.

## Reuse modes

ADR-006 described reuse as a single fingerprint match. There are four questions, and
they take different keys:

| Question | Match on |
|---|---|
| **Candidate reuse** — has this hypothesis been explored before? | `CandidateFingerprint` |
| **Artifact reuse** — give me any acceptable completed artifact from this candidate | `CandidateFingerprint`, any terminal realization |
| **Realization reuse** — do we have the artifact from this exact trajectory? | `RunRealizationFingerprint` |
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
