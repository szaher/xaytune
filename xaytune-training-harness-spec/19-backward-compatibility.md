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

## 1a. Live Python objects: the in-process exception

Today's API accepts live objects:

```python
xaytune.finetune(model=live_model, tokenizer=live_tokenizer, callbacks=[cb])
```

These **cannot cross the compile/execute boundary**. `TrainingExecutionSpec` is
JSON, crosses a process boundary, and carries no open Python objects, no
closures and no live tokenizers (see `05-training-spec-and-compilation.md` §4).
A `torch.nn.Module` in memory is not expressible as an `ArtifactInput`.

So there are two paths, and which one a caller gets is determined by what they
pass:

| | Serializable refs | Live Python objects |
|---|---|---|
| Path | Portable control plane | In-process compatibility |
| `portable` | `true` | `false` |
| `durable_controller` | `true` | `false` |
| `remote_execution` | `true` | `false` |
| Restart-survivable | yes | no |

The in-process path keeps today's behaviour exactly: the training runs in the
calling process against the objects handed to it, and no control-plane
guarantees apply. It is not deprecated and not second-class — interactive and
notebook use is a real requirement, and forcing a model to be round-tripped
through disk to fine-tune it would be a regression.

What matters is that the distinction is **explicit and detected at submission**,
not discovered when a controller restart finds an experiment it cannot rebuild.
An experiment constructed with live objects reports `portable=false` and refuses
`durable_controller=true` rather than accepting it and failing later.

This exception is lifted only when a serialization or plugin contract exists for
the object in question — for example a callback registered through the plugin
registry rather than passed as an instance. Until then, live objects mean
in-process.

## 2. Migration stages

### Stage A

New control-plane code exists but old APIs use old internals.

### Stage B

One legacy path (SFT) is internally routed through new `CandidateSpec → NativeCompiler → TrainingExecutionSpec → LocalRuntime`.

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
