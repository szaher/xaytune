# Public API and CLI

## 1. Remote-friendly first

Primary semantics:

```python
handle = experiment.submit()
```

not:

```python
experiment.run()  # still supported as convenience
```

## 2. Python API

```python
import xaytune

experiment = xaytune.Experiment(
    name="support-qwen",
    objective=xaytune.Objective.maximize(
        "task_success",
        target=0.85,
    ),
    training=xaytune.SFT(
        model=xaytune.ModelRef(
            uri="Qwen/Qwen3-8B",
            revision="...",
        ),
        dataset=xaytune.DatasetRef(
            uri="s3://datasets/support-v4",
            revision="sha256:...",
            split="train",
        ),
        adapter=xaytune.LoRA(
            rank=16,
            alpha=32,
        ),
        learning_rate=2e-5,
        micro_batch_size=4,
        gradient_accumulation=8,
        precision="bf16",
        epochs=3,
    ),
    evaluation=xaytune.EvaluationSpec(
        evaluators=[
            xaytune.TaskEvaluator("support-task-v2"),
        ],
    ),
    budget=xaytune.Budget(
        max_runs=8,
        max_gpu_hours=20,
    ),
    resilience=xaytune.ResiliencePolicy(
        cuda_oom="adaptive",
        worker_failure="runtime-recover",
    ),
    planner=xaytune.RuleBasedPlanner(),
    runtime=xaytune.TrainingHubRuntime(
        runtime_profile="torch-distributed",
    ),
)

handle = experiment.submit()

print(handle.experiment_id)

result = handle.wait()
```

### What is actually persisted

`RuleBasedPlanner(...)` and `TrainingHubRuntime(...)` above are **ergonomic
sugar over specs**, not objects the experiment stores. Per ADR-016 the durable
record holds:

```python
PlannerSpec(kind="rule-based", version="1.2.0", config={...})
RuntimeSpec(kind="training-hub", version="0.4.1", config={...},
            credentials_ref=SecretRef("TRAINING_HUB_TOKEN"))
ControllerHostSpec(kind="embedded", config={...})
```

which the plugin registry (ADR-008) resolves back into implementations. This is
what makes a controller restartable: after a crash it rebuilds the planner and
runtime from the record, with nothing left over from the process that created
the experiment.

A constructor that closes over a live object, a file handle or a lambda cannot
produce a spec, and **is rejected at `submit()`** rather than at restart. An
experiment that runs for six hours and then cannot be recovered is worse than
one that refuses to start. Callers who need to pass live Python objects use the
in-process compatibility path — see `19-backward-compatibility.md` §3.

## 3. Attach

```python
handle = xaytune.attach("exp_01...")
print(handle.status())
```

## 4. Synchronous convenience

```python
result = experiment.run()
```

Equivalent to submit + wait under the selected controller host.

## 5. YAML

See `schemas/experiment.example.yaml`.

## 6. CLI

### Experiment

```bash
xaytune experiment submit experiment.yaml

xaytune experiment status exp_123

xaytune experiment watch exp_123

xaytune experiment inspect exp_123

xaytune experiment graph exp_123

xaytune experiment pause exp_123

xaytune experiment resume exp_123

xaytune experiment cancel exp_123
```

### Nodes/runs

```bash
xaytune node inspect node_123
xaytune node compare node_a node_b

xaytune run inspect run_123
xaytune run attempts run_123
xaytune run logs run_123
xaytune run incidents run_123
```

### Checkpoints

```bash
xaytune checkpoint list run_123
xaytune checkpoint inspect ckpt_123
xaytune checkpoint validate ckpt_123
```

### Policy/approval

```bash
xaytune action list exp_123 --status approval-pending

xaytune action approve act_123
xaytune action reject act_123 --reason "Do not change dataset."
```

### Reproduction

```bash
xaytune node freeze node_123 --output frozen.yaml
xaytune node reproduce node_123
```

## 7. Error UX

Errors must name:

- failed layer
- requested capability
- selected backend
- alternatives when known
- remediation

Example:

```text
Cannot execute GRPO experiment with runtime profile "torch-ddp".

Reason:
  trainer compiler requires stateful rollout support,
  but the selected runtime does not advertise that capability.

Compatible runtimes discovered:
  - ray-grpo
  - training-hub/verl-grpo
```

## 8. Legacy API

Keep:

```python
xaytune.finetune(...)
xaytune.pretrain(...)
xaytune.align(...)
xaytune.evaluate(...)
```

These become wrappers around the new architecture where feasible.
