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
CandidateSpec
   ↓
TrainerCompiler.compile()
   ↓
TrainingExecutionSpec
   ↓
RuntimeBackend.submit()
```

(`02-architecture.md` shows the same seam with the node and capability
resolution around it. The input is the whole `CandidateSpec` — see §3.)

The trainer compiler translates intent.

The runtime executes.

## 2. TrainingSpec

`TrainingSpec` is scientific training intent.

It must be immutable after a node becomes active.

`TrainingSpec` is **one component of a `CandidateSpec`**, not the whole scientific
proposition. It carries the training program and nothing else:

```python
class TrainingSpec(BaseModel):
    api_version: str = "xaytune.ai/v1alpha1"
    kind: TrainingKind

    algorithm: AlgorithmSpec
    adapter: AdapterSpec | None

    optimization: OptimizationSpec
    precision: PrecisionSpec
    checkpoint: CheckpointIntent

    metadata: dict[str, Any]
```

Three fields that used to live here have moved, and the moves are the substance
of ADR-011:

| Field | Now lives on | Why |
|---|---|---|
| `model` | `CandidateSpec.model` (`ModelSpec`) | The model is part of the scientific proposition, not the training program. A GRPO candidate also has `reward` and `environment` at the same level |
| `dataset` | `CandidateSpec.data` (`DataSpec`) | Same |
| `seed` | `Run.seed` | **Seed belongs to the realization, not the candidate.** Two replicates differing only by seed are the same scientific candidate run twice; folding seed into candidate identity would make the replicate concept meaningless (ADR-006 §7) |

Ownership, in full:

```text
CandidateSpec                     Run
├── model:       ModelSpec        ├── seed
├── data:        DataSpec         └── replicate
├── training:    TrainingSpec
├── reward:      RewardSpec?
├── environment: EnvironmentSpec?
└── schedule:    TrainingSchedule?
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

    def supports(self, candidate: CandidateSpec) -> SupportResult: ...

    def compile(
        self,
        candidate: CandidateSpec,
        context: CompilationContext,
    ) -> TrainingExecutionSpec: ...
```

**The compiler takes the whole `CandidateSpec`, not just `TrainingSpec`.** A
compiler for SFT could work from optimizer hyperparameters alone, but one for
GRPO or agent training cannot: it needs the reward definition, the environment,
and any pre-registered schedule in order to emit a runnable plan. Passing only
the training program would force every RL compiler to reach around the
interface for the rest.

`supports()` takes the candidate for the same reason — whether a compiler can
handle a workload depends on its reward and environment, not only its
algorithm.

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

    candidate_fingerprint: str
```

The plan carries `candidate_fingerprint`, **not** a realization fingerprint.
At compile time no trajectory exists yet: the run has not started, no
intervention has been applied, and `RunRealizationFingerprint` is provisional
until the run reaches a terminal state (ADR-011). The execution plan can only
record which candidate it was compiled from.

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
