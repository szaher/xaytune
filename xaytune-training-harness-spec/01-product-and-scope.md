# Product and Scope

## 1. Product statement

Xaytune is an **agent-native experiment control plane for model post-training and model adaptation**.

It governs post-training and adaptation experiments across trainer implementations and execution runtimes: it decides what to run, what to do about the result, and keeps the record — it does not execute the training itself.

"Training harness" is a fair secondary description and appears in this package's name for historical reasons. It is not the primary positioning, because it suggests competing with the training runtime rather than directing it.

The boundary is **experiment topology, not GPU count**. Xaytune may direct very large distributed jobs; it is not the runtime responsible for executing a single months-long frontier pretraining run.

It does not replace the trainer or runtime.

The product surface should make this possible:

```python
experiment = xaytune.Experiment(
    objective=xaytune.Objective(
        maximize="task_success",
        target=0.85,
    ),
    training=xaytune.SFT(
        model="Qwen/Qwen3-8B",
        dataset="support-v4",
        adapter=xaytune.LoRA(rank=16),
    ),
    budget=xaytune.Budget(
        max_runs=8,
        max_gpu_hours=20,
    ),
    resilience=xaytune.ResiliencePolicy.adaptive(),
    planner=xaytune.RuleBasedPlanner(),
    runtime=xaytune.TrainingHubRuntime("torch-distributed"),
)

handle = experiment.submit()
result = handle.wait()
```

The same experiment should be able to execute through:

- local process / torchrun
- Ray Train
- Training Hub
- future platform runtimes

without changing the scientific training intent.

## 2. Xaytune owns

Xaytune owns:

- experiment definitions
- scientific candidate graph
- execution lineage
- experiment planning
- hypothesis and branch metadata
- immutable training specifications
- runtime-independent execution contracts
- trainer compiler adapters
- runtime adapters
- controller lifecycle
- controller reconciliation
- structured events
- policy enforcement
- typed actions
- budget accounting
- evaluation orchestration
- decisioning
- failure/incident classification
- adaptive recovery policy
- checkpoint metadata and compatibility
- provenance
- experiment memory
- capability resolution
- plugin contracts

## 3. Xaytune does not own

Do not turn Xaytune into:

- a Kubernetes operator
- a cluster scheduler
- a replacement for Kueue
- a replacement for Kubeflow Trainer
- a replacement for Training Hub
- a replacement for Ray Train
- a replacement for Ray Tune
- a replacement for TorchFT
- a replacement for TRL
- a replacement for torchtune
- a replacement for verl
- a replacement for MLflow/W&B
- a generic workflow engine
- a model registry
- a general-purpose agent shell

## 4. Design invariants

These are architectural invariants. PRs violating them require an ADR.

### Invariant A — compile vs execute

Trainer integrations compile a `CandidateSpec` into a `TrainingExecutionSpec`. The
whole candidate is passed, not just its `TrainingSpec`: a GRPO or agent compiler
needs the reward and environment to emit a runnable plan.

Runtime integrations execute `TrainingExecutionSpec`.

A trainer integration must not own remote execution.

### Invariant B — scientific, in-run and operational lineage

Lineage has three levels, not two (ADR-011).

Operational recovery creates a new `RunAttempt` or `ExecutionOverride`. An alternative
scientific candidate creates a new `ExperimentNode`. A scientifically meaningful change
to a run that is still going creates a `TrainingIntervention` on that run.

**Comparability decides between the last two:**

> A change creates a new `ExperimentNode` when the changed configuration is an
> alternative candidate you may want to compare independently against the current one.
>
> A change is a `TrainingIntervention` when it only makes scientific sense as a
> continuation of the existing model trajectory.

The kind of parameter does not decide this. Experimental intent does, so the same
technical change can be either.

| Change | Lineage |
|---|---|
| worker restart | same node, new run attempt |
| restore checkpoint | same node, new run attempt |
| pod eviction | same node, new run attempt |
| microbatch 4→2 while preserving effective batch | same node, execution override |
| declared LR schedule firing at step 20k | same run, scheduled intervention |
| LR lowered to stabilise a run that is destabilising | same run, reactive intervention |
| planned curriculum or data-mixture transition | same run, scheduled intervention |
| LR 2e-5 vs 1e-5 branched from a checkpoint to compare | new node |
| LoRA rank 16→32 | new node |
| dataset revision change | new node |
| reward definition change | new node |
| optimizer change | new node |

### Invariant C — controller is durable

The architecture must not assume the Python client stays alive.

`experiment.run()` may be convenience sugar, but the fundamental API is `submit() -> Handle`.

### Invariant D — events and state are consistent

Materialized state and durable event history must commit atomically in the local implementation.

### Invariant E — LLMs never bypass policy

An LLM may propose typed actions.

It does not directly mutate training state, run shell commands, submit workloads, or bypass budgets/policies.

### Invariant F — evaluation is independent

Evaluation is not a method on a trainer backend.

It is a separate subsystem with versioned evaluators and metric provenance.

### Invariant G — capabilities are versioned and parameterized

Avoid boolean-only capability contracts.

### Invariant H — core does not require ML runtimes

The control-plane core must be importable without PyTorch, Transformers, Ray, TRL, TorchFT, or Kubernetes dependencies.

## 5. Primary user personas

### ML engineer

Wants reproducible training runs, fault recovery, comparison, and clear lineage.

### ML platform engineer

Wants runtime-neutral intent, policy, resource accounting, and integration with Training Hub/Ray/Kubeflow.

### Research engineer

Wants adaptive experimentation, search providers, branching, hypotheses, and evaluation-driven iteration.

### Agent/coding agent

Wants a typed API for planning and changing experiments without uncontrolled infrastructure access.

## 6. Primary use cases

1. resilient SFT / LoRA training
2. adaptive recovery from OOM / NaN / process failures
3. evaluation-driven experiment branching
4. HPO/search with Ray Tune / Katib / Optuna while preserving experiment lineage
5. agent-planned model training
6. agent training through TRL/OpenEnv/verl backends
7. local-to-platform portability
8. reproducibility and provenance
9. training knowledge reuse
10. policy- and budget-constrained autonomous experimentation
