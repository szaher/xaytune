# MVP Reference Scenario

This is the end-to-end acceptance test for the first valuable Xaytune 2.0 release.

## 1. User intent

```text
Fine-tune Qwen3-8B on support dataset v4.

Objective:
task_success >= 0.82

Budget:
maximum 4 runs (one run per candidate in this scenario)

Recovery:
adapt to CUDA OOM while preserving effective batch when possible
```

## 2. Input

```yaml
apiVersion: xaytune.ai/v1alpha1
kind: Experiment

metadata:
  name: support-qwen

objective:
  primary:
    name: task_success
    direction: maximize
  target: 0.82

budget:
  maxRuns: 4

candidate:
  model:
    uri: Qwen/Qwen3-8B

  data:
    uri: ./data/support-v4.jsonl
    revision: sha256:example

  training:
    kind: sft

    adapter:
      type: lora
      rank: 16
      alpha: 32

    optimization:
      learningRate: 2e-5
      microBatchSize: 4
      gradientAccumulation: 8
      epochs: 2

evaluation:
  evaluator:
    name: support-task

resilience:
  cudaOOM:
    strategy: execution-override
    preserveEffectiveBatch: true

planner:
  kind: rule-based
  config:
    rules:
      - kind: increase-lora-rank   # node_A LoRA 16 → node_B LoRA 32
        factor: 2
        max_rank: 64

runtime:
  backend: local
```

`maxRuns` bounds **runs**, not candidates (PR-016): it is reserved when a run
is created. This scenario realizes each candidate once, so four runs allow at
most four candidates. A separate `max_candidates` budget can come later. The
planner is the bound `PlannerSpec` (PR-024): rule-based planning needs an
explicit, typed mutation rule, here LoRA-rank growth.

GPU-hour enforcement is deferred because Xaytune does not yet have
authoritative measured GPU consumption. The eventual implementation must
consume runtime-reported measured usage; it must not approximate usage as
requested GPUs × controller wall-clock time. (Spec 15: *Future budget work --
authoritative GPU-hour metering*.)

**Decision engine.** The PR-026 EmbeddedControllerHost is configured with
`AdaptiveThresholdDecisionEngine 1.0.0`, so a missed target is `BRANCH`
rather than `STOP_FAILED`. Persisting a `DecisionEngineSpec` is separate
ADR-016 follow-up work; until then a host attaching to this experiment must
be configured with the same engine.

## 3. Expected execution

### Step A — experiment creation

Creates:

```text
Experiment exp_1
ExperimentNode node_A
Run run_A1
RunAttempt attempt_A1_1
```

### Step B — compile

```text
SFT CandidateSpec
  ↓
TRLCompiler or NativeCompiler
  ↓
TrainingExecutionSpec
  ↓
LocalRuntime
```

### Step C — injected OOM

At step 300:

```text
CUDA OOM
```

Xaytune:

1. records incident
2. classifies recoverable with execution override
3. finds latest committed compatible checkpoint
4. **verifies the checkpoint is eligible for batch-size-changing resume** — taken at an
   optimizer-step boundary, carrying a batch-size-independent `DataCursor` (ADR-012).
   If it is not, the recovery is rejected rather than approximated.
5. calculates:
   - microbatch 4 → 2
   - grad accumulation 8 → 16
6. policy approves
7. budget approves
8. attempt_A1_1 ends failed/recoverable
9. creates attempt_A1_2
10. restores checkpoint, resuming at the next unconsumed sample
11. resumes, recording the achieved `ResumeGuarantee`

No new ExperimentNode.

Step 4 is what makes this scenario sound. Resuming on a batch index would replay 200
samples when the micro-batch halves, while reporting exact continuation.

### Step D — first evaluation

Node A result:

```text
task_success = 0.79
```

Decision:

```text
objective not met
budget remains
planner proposes LoRA rank 32
```

Since LoRA rank is scientific training intent:

```text
new node_B
parent = node_A
hypothesis = "Adapter capacity may be limiting task performance."
```

The decision engine decides (`BRANCH`); the experiment's recorded planner
proposes; branching (PR-025) admits node_B as `PLANNED` after its
candidate-governance checks. This is not PolicyEngine governance: a
`CandidateProposal` is not an Action.

### Step E — second candidate

Nobody intervenes between A's `BRANCH` and B's training. The embedded host
realizes node_B automatically -- only because its `branch_origin` proves the
experiment's own recorded planner proposed it; a `PLANNED` node planned any
other way is never run automatically.

Run B is a **new** run (new `RunId`, attempt, execution spec, submission) with
the **same seed** as Run A: for this comparative search, the seed of the
parent's single training run is inherited, so A and B differ by the intended
mutation rather than by chance. The seed stays a `Run` property; Run B records
`seed_origin = parent-run(Run A)`, which is provenance and enters no
fingerprint. Same seed never means reuse (ADR-017).

Before Run B is created, the recorded compiler re-checks that it supports the
branched candidate; `maxRuns` is reserved when Run B is created, not when B
was planned. If no run is left, the experiment ends `BUDGET_EXHAUSTED` and A's
`BRANCH` decision stands.

Node B trains successfully.

Evaluation:

```text
task_success = 0.83
```

### Step F — decision

Objective met.

Experiment:

```text
SUCCEEDED
best_node = node_B
```

## 4. Required output

### PR-026 acceptance result

What `ExperimentResult` returns today, and what the acceptance test asserts
from it:

```text
status       SUCCEEDED
next_stage   None
nodes        node_A and node_B, with their runs and evaluations
budget       settled: 2 runs consumed of 4, nothing outstanding
```

and from the durable record: `experiment.best_node_id == node_B`.

### Durable provenance assertions (PR-026)

The rest is asserted directly from the repository's records: node_B's
`branch_origin`, decisions, evaluation results, incidents, recovery episodes
and actions, checkpoints, execution overrides, the budget ledger, Run B's
`seed_origin`, and the compiler/runtime identities recorded with the
experiment.

### Future rich result / export projection

The target projection -- not part of PR-026, and not fields of
`ExperimentResult` today:

```python
ExperimentResult(
    best_artifact=...,
    best_node_id="node_B",
    objective_met=True,
    experiment_graph=...,
    evaluations=...,
    incidents=...,
    recovery_history=...,
    resource_usage=...,
    budget_status=...,
    provenance_bundle=...,
)
```

## 5. Required provenance assertions

Must be able to answer:

- Why does node B exist?
- Which node is its parent?
- Which spec field changed?
- Why did attempt A1_1 fail?
- Which checkpoint resumed A1_2?
- Which execution override was applied?
- Did effective batch stay constant?
- Which evaluator measured 0.83?
- Which dataset revision was used?
- Which compiler/runtime versions executed the training?
- How much budget was consumed?

For the **candidate branch** (candidate governance, PR-025 -- not PolicyEngine):

- Who/what proposed LoRA rank 32?
- Which `PlannerSpec` and configuration produced it?
- Which `PlanningContext` did it use?
- Which decision and evaluation evidence supported it?
- Which candidate-governance checks admitted it?
- Which node is its parent?

For the **OOM recovery Action** (Action/Policy governance):

- Which Action proposed the micro-batch resize?
- Which `PolicyDecision` allowed it?
- Which incident justified it?
- Which checkpoint was restored?

## 6. Demo value

This single scenario demonstrates the reason Xaytune exists:

```text
experiment semantics
+ resilience
+ adaptive recovery
+ reproducibility
+ branching
+ policy
+ budget
+ provenance
```
