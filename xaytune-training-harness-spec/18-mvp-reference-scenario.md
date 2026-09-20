# MVP Reference Scenario

This is the end-to-end acceptance test for the first valuable Xaytune 2.0 release.

## 1. User intent

```text
Fine-tune Qwen3-8B on support dataset v4.

Objective:
task_success >= 0.82

Budget:
maximum 4 scientific candidates
maximum 8 GPU-hours

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
  maxGpuHours: 8

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
  evaluators:
    - name: support-task

resilience:
  cudaOOM:
    strategy: execution-override
    preserveEffectiveBatch: true

planner:
  type: rule-based

runtime:
  backend: local
```

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

### Step E — second candidate

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
- Who/what proposed LoRA rank 32?
- Which policy allowed it?

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
