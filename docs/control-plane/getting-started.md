# Getting started with the control plane

This page takes one fine-tuning candidate from a description to a trained
model, through the control plane: compiled, submitted, observed, cancelled, and
adopted by a second process. Every step has a runnable script in
[`examples/control_plane/`](https://github.com/szaher/xaytune/tree/main/examples/control_plane).

!!! note "Pre-release"
    The control plane is on `main`, not in the `0.6.0` package on PyPI. Install
    it from a clone as below. For the `finetune()` / CLI library that `0.6.0`
    contains, see the [legacy trainer API](../getting-started.md).

## Install

```bash
git clone https://github.com/szaher/xaytune && cd xaytune
uv sync --locked                 # or: pip install -e .
uv sync --locked --extra trl     # to use the TRL trainer as well
```

`--locked` installs exactly what CI tests, from `uv.lock`. The TRL trainer
supports one minor release each of `trl` (1.13) and `transformers` (5.17), and
refuses to train on any other. See [Getting Started](../getting-started.md#supported-trl-and-transformers-releases)
for why.

`import xaytune` loads neither torch nor transformers. The controller runs
without them; training happens in worker processes, which load them.

## 1. Describe a candidate, and compile it

A `CandidateSpec` is one hypothesis: this model, this data, trained this way.
It declares **every value that changes what the model learns**, because a
value left to a trainer default would be part of the run without being part
of the candidate's identity.

```python
from xaytune.core import (
    CandidateSpec, DataSpec, DatasetRef, LRScheduleSpec, ModelRef, ModelSpec,
    OptimizationSpec, OptimizerSpec, PrecisionSpec, TrainingKind, TrainingSpec,
)

candidate = CandidateSpec(
    model=ModelSpec(model=ModelRef(uri="/models/base")),          # a local directory
    data=DataSpec(
        dataset=DatasetRef(uri="/data/train.jsonl"),                # JSONL, a "text" field
        format="text", max_seq_length=512, packing=False,
    ),
    training=TrainingSpec(
        kind=TrainingKind.SFT,
        optimization=OptimizationSpec(
            optimizer=OptimizerSpec(name="adamw", weight_decay=0.0),
            lr_schedule=LRScheduleSpec(name="constant"),
            learning_rate=2e-5, micro_batch_size=4, gradient_accumulation=1,
            epochs=1, max_grad_norm=1.0,
        ),
        precision=PrecisionSpec(dtype="fp32"),
    ),
)
print(candidate.candidate_fingerprint())   # sha256:...
```

A compiler turns it into a plan, and runs nothing:

```python
from xaytune.compilation import CompilationContext
from xaytune.compilation.native import NativeCompiler

spec = NativeCompiler().compile(
    candidate, CompilationContext(run_id="run_1", seed=7, output_uri="/out/run_1")
)
```

A candidate the compiler cannot run *exactly as declared* raises
`UnsupportedCandidateError` listing every reason, not just the first. A hub
name such as `Qwen/Qwen3-0.6B` is refused, for example: without a pinned
revision it names whatever the hub serves on the day the worker starts.

Run it: `python examples/control_plane/01_compile_a_candidate.py`. It needs no
model and no GPU.

## 2. Submit it, and wait

An `ExperimentSpec` adds what the candidate does not say: the run's seed,
which compiler, which runtime, and where models go.

```python
import asyncio
from xaytune.core import Objective, ObjectiveMetric
from xaytune.experiment import (
    CompilerSpec, EmbeddedControllerHost, ExperimentSpec, RuntimeSpec,
)

spec = ExperimentSpec(
    name="first-sft",
    objective=Objective(primary=ObjectiveMetric(name="loss", direction="minimize")),
    candidate=candidate,
    seed=7,
    compiler=CompilerSpec(name="native"),            # or "trl"
    runtime=RuntimeSpec(kind="local", config={"root": "/abs/workdir/runtime"}),
    artifact_root="/abs/workdir/artifacts",
)

async def main() -> None:
    host = EmbeddedControllerHost("/abs/workdir/state.db")
    try:
        handle = await host.submit(spec)
        result = await handle.wait()
        print(result.status, result.next_stage)       # ACTIVE, "evaluation"
        for node in result.nodes:
            for run in node.runs:
                print(run.status, [a.uri for a in run.artifacts])
    finally:
        await host.close()

asyncio.run(main())
```

`submit()` returns once the runtime has accepted the workload and that
acceptance is recorded. `wait()` returns when the controller has **nothing
left it can run**. That is not a verdict on the experiment. After training
alone, the run is `SUCCEEDED` with its model recorded as an artifact, while
the candidate and the experiment stay `ACTIVE`. `next_stage` says
`"evaluation"` is what would come next. See [concepts](concepts.md#what-wait-means).

Run it: `python examples/control_plane/02_train.py --model /abs/model --dataset /abs/train.jsonl`.
It prints each control-plane event as it is committed.

## 3. Cancel it

```python
await handle.cancel(reason="wrong learning rate")
result = await handle.wait()                           # status CANCELLED
```

Cancellation is recorded intent first, carried out as a cancel operation
against the runtime. The experiment reads `CANCELLED` only once no workload it
owns is still running.

Run it: `python examples/control_plane/03_cancel.py --model ... --dataset ...`.

## 4. End the process, and attach from another

The workload belongs to the runtime, not to the process that submitted it.

```python
# process 1
handle = await host.submit(spec)
await host.close()                                     # training keeps going

# process 2, later
host = EmbeddedControllerHost("/abs/workdir/state.db")
handle = await host.attach(experiment_id)
result = await handle.wait()
```

`attach()` reads the record, finds the attempt the runtime still holds, and
observes it from the durable telemetry cursor. It never submits it a second
time. An attempt whose submission was recorded but never reached the runtime
is issued under its original identity. A question the record cannot answer
safely raises `ReconciliationEscalatedError` rather than guessing.

Run it: `python examples/control_plane/04_restart_and_attach.py start ...`,
then `... attach <experiment-id>`.

## Evaluating the trained model

`ExperimentSpec.evaluation` takes an `EvaluationSpec` naming one evaluator.
When it is set, the host evaluates the trained model after training, through
the same journal and telemetry, records the result, and moves the candidate to
`DECIDING` with `next_stage="decision"`.

**No evaluator is built in yet.** Evaluators that wrap Xaytune's metrics and
lm-eval are the next step. Until then, evaluation needs an `Evaluator` you
register with `EmbeddedControllerHost(..., evaluators={...})`. See
[concepts](concepts.md#evaluation).

## What a candidate may not do yet

Both compilers run **full-parameter SFT on one worker** from local files.
They refuse, with reasons: adapters (LoRA/QLoRA), checkpoint intent,
algorithm variants, rewards, hub model names, and any training-relevant
value left undeclared. The TRL compiler also refuses data formats other than
`text`, and packing. The legacy trainer API still does all of these, outside
the control plane.
