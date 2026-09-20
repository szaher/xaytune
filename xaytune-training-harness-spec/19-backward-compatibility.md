# Backward Compatibility and Migration

## 1. Existing API commitment

Keep current public APIs during migration:

```python
xaytune.finetune(...)
xaytune.pretrain(...)
xaytune.align(...)
xaytune.evaluate(...)
```

Keep current CLI commands.

## 2. Migration stages

### Stage A

New control-plane code exists but old APIs use old internals.

### Stage B

One legacy path (SFT) is internally routed through new `TrainingSpec → NativeCompiler → LocalRuntime`.

### Stage C

Other recipes migrate.

### Stage D

Legacy callback/checkpoint compatibility bridges translate to new events/contracts.

### Stage E

Old internal paths may be deprecated only after parity tests.

## 3. Pipeline

Existing deterministic pipeline remains.

Do not redefine it as Experiment.

Difference:

```text
Pipeline:
known steps before execution

Experiment:
adaptive graph determined during execution
```

Possible composition:

```text
ExperimentNode
  executes a deterministic Pipeline
```

but this is not required for MVP.

## 4. Callback compatibility

Current callback events:

```text
train_start
train_end
epoch_start
epoch_end
step_start
step_end
eval_start
eval_end
checkpoint_saved
error
```

Bridge them to structured events.

Legacy callback registration should continue working during transition.

## 5. Checkpoint migration

Existing checkpoint formats should be:

- supported by a LegacyCheckpointCodec, or
- rejected with an explicit migration message

Never silently treat an incompatible checkpoint as valid.

## 6. Config migration

Legacy config remains parseable.

New experiment YAML is separate.

Do not overload one schema with both every legacy field and new control-plane fields.

Provide:

```bash
xaytune migrate-config old.yaml --output experiment.yaml
```

later.

## 7. Versioning

New experiment schema starts:

```text
xaytune.ai/v1alpha1
```

Control-plane public APIs remain experimental until first compatibility commitment.

## 8. Deprecation

A deprecation requires:

- replacement documented
- compatibility layer available
- warning for at least one minor release
- migration guide
