# Testing and Fault Injection

## 1. Test layers

### Unit

Must cover:

- state transitions
- graph rules
- fingerprints
- capability matching
- policy rules
- action validation
- budget arithmetic
- incident classification
- recovery planning
- checkpoint compatibility
- metric comparison

### Component

- SQLite repository
- outbox
- LocalRuntime
- NativeCompiler
- TRLCompiler
- checkpoint manager
- evaluator adapter

### Integration

- train → evaluate
- train → incident → recover
- train → evaluate → branch
- controller restart
- runtime submit idempotency
- MLflow event sink

### Distributed

- torchrun
- Ray
- TorchFT

### Platform

- Training Hub
- Kubeflow Trainer
- Kueue

## 2. Fault injection API

Create:

```text
tests/faults/
```

Base:

```python
class FaultInjector(Protocol):
    async def arm(self, context: FaultContext) -> None: ...

    async def disarm(self) -> None: ...
```

Initial injectors:

- RaiseCudaOOM
- ReturnNaNLoss
- KillWorker
- KillProcess
- SimulatePreemption
- FailCheckpointWrite
- CorruptCheckpointManifest
- FailObjectStore
- FailEvaluation
- CrashController

## 3. Required scenarios

### OOM recovery

Given:

- microbatch=4
- grad_accum=8
- checkpoint step=100

When:

- OOM at step 120

Then:

- incident=CUDA_OOM
- new RunAttempt
- microbatch=2
- grad_accum=16 if preserving effective batch
- same ExperimentNode
- checkpoint restored
- provenance includes ExecutionOverride

### LR recovery

Given:

- NaN incident

When policy chooses:

- LR 2e-5 → 1e-5

Then:

- new ExperimentNode (only for alternatives being compared)
- TrainingIntervention + InterventionApplication (for in-run scientific changes)

Resume correctness (ADR-012) needs its own tests, and the decisive one is cheap: run N
steps uninterrupted, then run the same configuration with an interruption and resume at
a **different micro-batch size**, and assert both consumed the same samples in the same
order. A batch-index cursor fails this immediately.
- old node remains immutable
- hypothesis/reason recorded

### Controller crash

Given remote/local-daemon execution active.

When controller process dies.

Then:

- runtime execution continues where supported
- daemon restarts
- active attempt is reconciled
- no duplicate submission
- no duplicate branch

### Duplicate action

Given same action operation ID executed twice.

Then:

- only one side effect
- same result returned or recognized as completed

## 4. Architecture boundary tests

Automate import scans.

Fail CI if:

```text
xaytune.core imports torch
xaytune.core imports transformers
xaytune.policy imports trl
xaytune.domain imports ray
xaytune.training_spec imports kubernetes
xaytune.experiment imports mlflow directly
```

## 5. Serialization tests

Every cross-runtime object must round-trip:

```python
obj == Type.model_validate_json(obj.model_dump_json())
```

Test with no plugin installed.

## 6. Migration tests

Fixtures from v0.6 behavior.

At minimum:

- current SFT config parses
- current finetune API works
- pipeline config works
- current callback registration compatibility
- current checkpoint can either load or fails with explicit migration guidance

## 7. Performance tests

Control-plane performance targets are modest but should be measured:

- 10k events append/read
- graph with 10k nodes
- 100 concurrent pending actions
- 1k budget ledger entries

Do not optimize prematurely, but avoid O(N²) graph scans.

## 8. Test markers

Recommended:

```text
unit
integration
gpu
distributed
ray
torchft
training_hub
slow
fault
```

CI default excludes expensive markers.

Nightly runs expensive suites.
