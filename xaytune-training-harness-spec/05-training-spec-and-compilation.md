# Training Specification and Compilation

## 1. Why this boundary exists

Trainer libraries and execution runtimes are different concerns.

Bad architecture:

```text
TrainerBackend.train()
RuntimeBackend.submit(PreparedTraining)
```

This creates two possible execution owners.

Correct architecture:

```text
TrainingSpec
   ↓
TrainerCompiler.compile()
   ↓
TrainingExecutionSpec
   ↓
RuntimeBackend.submit()
```

The trainer compiler translates intent.

The runtime executes.

## 2. TrainingSpec

`TrainingSpec` is scientific training intent.

It must be immutable after a node becomes active.

```python
class TrainingSpec(BaseModel):
    api_version: str = "xaytune.ai/v1alpha1"
    kind: TrainingKind

    model: ModelRef
    dataset: DatasetRef

    algorithm: AlgorithmSpec
    adapter: AdapterSpec | None

    optimization: OptimizationSpec
    precision: PrecisionSpec
    checkpoint: CheckpointIntent

    seed: int | None

    metadata: dict[str, Any]
```

Initial `TrainingKind`:

- SFT
- PRETRAIN
- DPO
- GRPO

## 3. TrainerCompiler

```python
class TrainerCompiler(Protocol):
    descriptor: PluginDescriptor

    def capabilities(self) -> CapabilityDocument: ...

    def supports(self, spec: TrainingSpec) -> SupportResult: ...

    def compile(
        self,
        spec: TrainingSpec,
        context: CompilationContext,
    ) -> TrainingExecutionSpec: ...
```

Compilation must be deterministic for the same inputs.

Compilation must not submit workloads.

## 4. TrainingExecutionSpec

This object crosses process/runtime boundaries.

Therefore:

- JSON serializable
- versioned
- no open Python objects
- no live tokenizer/model objects
- no closures
- no filesystem assumptions unless represented as artifact/input refs

```python
class TrainingExecutionSpec(BaseModel):
    api_version: str = "xaytune.execution/v1alpha1"

    compiler: CompilerIdentity

    entrypoint: EntrypointSpec
    arguments: list[str]
    config: dict[str, Any]

    environment: dict[str, str]

    dependencies: DependencySpec
    container: ContainerSpec | None

    inputs: list[ArtifactInput]
    outputs: list[ArtifactOutput]

    resources: ResourceRequirements

    checkpoint: CheckpointExecutionContract
    telemetry: TelemetryContract

    required_capabilities: CapabilityRequirements

    training_fingerprint: str
```

## 5. EntrypointSpec

Examples:

```python
EntrypointSpec(
    kind="python-module",
    value="xaytune_runtime.trl_worker",
)
```

or:

```python
EntrypointSpec(
    kind="command",
    value=["python", "/opt/xaytune/run.py"],
)
```

Do not allow arbitrary user shell strings to be constructed by an LLM.

## 6. Compiler examples

### TRLCompiler

Maps:

```text
SFT TrainingSpec → SFTConfig + SFTTrainer worker config
DPO TrainingSpec → DPOConfig + DPOTrainer worker config
GRPO TrainingSpec → GRPOConfig + GRPOTrainer worker config
```

### NativeCompiler

Wraps the existing Xaytune trainer.

The current trainer becomes an implementation detail of a compiler/worker package.

### TorchtuneCompiler

Future.

### VerlCompiler

Future, particularly for RL/GRPO.

## 7. Execution-independent fingerprints

Compilation receives fingerprints but must not conflate them.

### CandidateFingerprint

What was declared — the scientific identity of the candidate (ADR-011):

- model revision
- dataset revision
- tokenizer/template
- algorithm
- optimizer
- LR
- LR schedule
- effective batch
- adapter config
- reward config
- environment config
- pre-registered intervention schedule

**`seed` is deliberately not here.** Seed belongs to the realization, not the candidate:
two replicates differing only by seed are the same scientific candidate run twice, and
folding seed into candidate identity would make the replicate concept meaningless.

### RunRealizationFingerprint

What actually happened — the identity of one realized trajectory:

- `CandidateFingerprint`
- seed / replicate identity
- the ordered sequence of `InterventionApplication` records

A reactive intervention changes this and leaves `CandidateFingerprint` untouched. This
is what stops reuse confusing "the same candidate, run clean" with "the same candidate,
plus an LR drop at step 14,250".

Note it is **provisional until the run reaches a terminal state**, so reuse must not
match against an in-flight run. And it is an identity, not a reproduction recipe: a
reactive intervention was triggered by a stochastic event, so rerunning with the same
seed will not reproduce it.

### ExecutionFingerprint

Execution identity:

- compiler version
- framework versions
- code revision
- runtime
- GPU type
- world size
- topology
- distributed mode
- container digest

### EvaluationFingerprint

Separate.

### CheckpointCompatibilityKey

Separate.

## 8. Reuse policy

Identical `CandidateFingerprint` does **not** automatically mean “never run again.”

Reuse asks four different questions and they take different keys (ADR-011):

| Question | Match on |
|---|---|
| Has this hypothesis been explored? | `CandidateFingerprint` |
| Do we have *any* artifact from this candidate? | `CandidateFingerprint`, any terminal realization |
| Do we have *this exact* trajectory's artifact? | `RunRealizationFingerprint` |
| Has this artifact been scored by this evaluator? | artifact digest + `EvaluationFingerprint` |


Support:

```python
class ReusePolicy:
    mode: Literal[
        "never",
        "exact-result",
        "same-seed",
        "reuse-artifact",
        "warn-only",
    ]
```

Replicates must be explicit:

```python
replicate = 2
seed = 1234
```

## 9. Training semantic mutation

A mutation never edits a snapshot in place.

```python
new_spec = old_spec.with_changes(optimization__learning_rate=1e-5)
```

Mutation object:

```python
class TrainingMutation(BaseModel):
    path: str
    old_value: Any
    new_value: Any
    reason: str
```

Where the mutation lands depends on experimental intent, not on which field changed
(ADR-011):

```text
alternative candidate to compare against  -> new ExperimentNode, new CandidateSpec
change to a run that is still going       -> TrainingIntervention on that run
operational adjustment preserving intent  -> ExecutionOverride on the attempt
```

An intervention leaves the node's `CandidateSpec` and `CandidateFingerprint` untouched —
they record what was declared — and changes the run's `RunRealizationFingerprint`, which
records what actually happened.
