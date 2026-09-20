# ADR-011 — Scientific candidates, training interventions, and execution overrides

## Status
Proposed

Supersedes the two-level lineage model in ADR-003 and extends the identity model in ADR-006.

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

### Reuse has three modes, not one

| Question | Match on |
|---|---|
| Has this hypothesis already been tested? | `CandidateFingerprint` |
| Do we already have *any* artifact from this candidate? | `CandidateFingerprint`, any realization |
| Do we have *this exact* trajectory's artifact? | `RunRealizationFingerprint` |
| Can we reuse an evaluation? | artifact digest + `EvaluationFingerprint` |
| Can we resume from this checkpoint? | `CheckpointCompatibilityKey` |

The middle row is the common case and has no answer under ADR-006 as written.

### RunRealization is a projection, never a second source of truth

```python
class RunRealization(BaseModel):
    run_id: RunId
    candidate_fingerprint: str
    realization_fingerprint: str
    initial_spec: CandidateSpecSnapshot
    scheduled_interventions: list[ScheduledInterventionRef]
    applied_interventions: list[TrainingInterventionRef]
    execution_overrides: list[ExecutionOverrideRef]
    final_effective_state: EffectiveTrainingState
```

Every field is derived from the durable event log, which stays authoritative (ADR-005).
`RunRealization` must be recomputable from that log, and the recompute must be exercised
in tests. Materialising `final_effective_state` as independently mutable state would
reintroduce precisely the divergence ADR-005 exists to prevent.

### Ordering is by event sequence, not training position

```python
class TrainingPosition(BaseModel):
    global_step: int | None
    optimizer_step: int | None
    tokens_seen: int | None
    examples_seen: int | None
```

`TrainingPosition` records *where* an intervention took effect. It cannot be the ordering
key, because it is not monotonic across a run: restoring from a checkpoint moves the
position backwards. `event.sequence` is authoritative for ordering and for the canonical
sequence that `RunRealizationFingerprint` hashes.

## Open question — interventions across checkpoint rollback

Not yet decided. It must be, before the recovery coordinator is built.

A run applies an intervention at step 10,000, reaches 14,000, loses a worker, and
restores from a checkpoint. Whether the intervention re-applies depends on the
checkpoint:

- restored from step 12,000 — the change is already in the restored optimizer state, so
  re-applying it would double-apply
- restored from step 8,000 — the change is not in that state, so it must re-apply when
  the run reaches 10,000 again

**Recommendation.** Separate the *decision* from its *applications*. A
`TrainingIntervention` records a decision, attached to the `Run`. Each time it takes
effect, an application event is appended, carrying its own `TrainingPosition` and
attempt id. On resume, the coordinator compares the restored position against each
intervention's position and re-applies only those ahead of it. The realization
fingerprint then hashes the ordered *applications*, so a run that double-applied an
intervention after a rollback is correctly distinguished from one that did not.

This needs sign-off because it constrains both the checkpoint metadata (the restored
position must be recoverable) and the event schema in PR-005.

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
