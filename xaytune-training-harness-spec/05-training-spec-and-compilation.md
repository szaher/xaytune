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

### TrainingSpecFingerprint

Scientific identity:

- model revision
- dataset revision
- tokenizer/template
- algorithm
- optimizer
- LR
- schedule
- effective batch
- adapter config
- seed
- reward config

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

Identical `TrainingSpecFingerprint` does **not** automatically mean “never run again.”

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

A `TrainingSpec` mutation always creates a new snapshot.

```python
new_spec = old_spec.with_changes(optimization__learning_rate=1e-5)
```

Mutation object:

```python
class TrainingSpecMutation(BaseModel):
    path: str
    old_value: Any
    new_value: Any
    reason: str
```

Any scientific mutation results in a new `ExperimentNode`.
