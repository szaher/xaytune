# Current Repository Refactor Map

This plan assumes the existing Xaytune v0.6 repository.

## 1. Keep and adapt

### `xaytune/config/`

Keep Pydantic/YAML parsing.

Refactor:

- separate training intent from runtime configuration
- add experiment schemas
- add capability validation as a separate phase

### `xaytune/data/`

Keep:

- formats
- tokenization
- dataset loading
- validation
- preparation

Add immutable `DatasetRef` and provenance fingerprints.

Do not make the core Experiment object depend directly on Hugging Face Dataset objects.

### `xaytune/models/`

Keep existing model loading for the native compiler/worker.

Do not expose loaded model objects through core contracts.

### `xaytune/recipes/`

Existing recipes become sources for `TrainingSpec` construction and/or NativeCompiler mapping.

Recommended migration:

```text
recipes/finetune.py
  ↓
TrainingSpec builder + compatibility wrapper

recipes/pretrain.py
  ↓
TrainingSpec builder + compatibility wrapper

recipes/align/*
  ↓
Native compiler implementation and/or adapter layer
```

Do not delete working algorithms during the control-plane refactor.

### `xaytune/trainer/`

Existing trainer becomes implementation behind:

```text
NativeCompiler
NativeWorker
```

Keep:

- loop
- callbacks
- checkpoint integration
- progress
- early stopping
- existing distributed support during migration

But do not let it define the control-plane API.

### `xaytune/eval/`

Keep metrics and lm-eval integration.

Wrap behind the new Evaluator contract.

Change metric return shape from plain floats to `MetricResult`.

### `xaytune/logging/`

Keep console / TensorBoard / W&B / MLflow.

Convert them to event sinks / observability adapters.

### `xaytune/pipeline.py`

Keep deterministic pipeline semantics.

Do not merge pipeline and adaptive Experiment.

Relationship:

```text
ExperimentNode
  can execute
    Pipeline
```

or:

```text
legacy Pipeline
  remains independently usable
```

### `xaytune/plugins.py`

Extend existing entry-point discovery rather than replace it.

Add version validation and new plugin groups.

### `xaytune/studio/`

Do not rewrite Studio during early architecture phases.

Later migrate job management to ExperimentHandle APIs.

## 2. New modules

Add incrementally:

```text
xaytune/core/
xaytune/experiment/
xaytune/compilation/
xaytune/runtimes/
xaytune/resilience/
xaytune/checkpoint/
xaytune/policy/
xaytune/budget/
xaytune/storage/
xaytune/agents/
xaytune/search/
xaytune/provenance/
```

## 3. Compatibility bridge

`xaytune.compatibility.legacy_api` should translate current APIs.

Example:

```python
def finetune(...):
    spec = build_legacy_sft_spec(...)

    experiment = Experiment(
        objective=LegacyTrainingCompletionObjective(),
        training=spec,
        planner=NoOpPlanner(),
        runtime=LocalRuntime(),
    )

    result = experiment.run()

    return result.to_legacy_train_state()
```

Only introduce this wrapper after the new local execution path is stable.

## 4. Do not do a package-wide move first

Avoid:

```text
PR: move every module to new directories
```

Prefer:

1. add new contracts
2. adapt one old path
3. add tests
4. route new API through it
5. migrate another path
6. delete obsolete internals only after parity

## 5. PR-009a observation boundary before NativeWorker

Keep `trainer/callbacks.py` and `logging/` implementations intact. CallbackManager
is an internal NativeWorker mechanism, never a field on TrainingExecutionSpec.
NativeWorker callbacks may emit the shared typed telemetry bodies. Scientific
changes require Action → TrainingIntervention and operational adaptation
requires Action/recovery → ExecutionOverride; callbacks cannot silently mutate
LR, optimizer, dataset, algorithm, reward, adapter, precision or stopping policy.

Adapt the existing console/MLflow/W&B/TensorBoard backends to EventSink later;
do not delete them or make them authoritative. PR-009a adds only core observation,
resume and sink contracts and the tiny LocalRuntime payload-construction adapter.
Collectors, exporters and NativeWorker integration are deferred.
