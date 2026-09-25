# Getting started with the control plane

This page takes one fine-tuning candidate from a description to a trained
model, through the control plane: compiled, submitted, observed, cancelled,
adopted by a second process, evaluated, and decided. Every step has a runnable script in
[`examples/control_plane/`](https://github.com/szaher/xaytune/tree/main/examples/control_plane).

!!! note "Pre-release"
    The control plane is on `main`, not in the `0.6.0` package on PyPI. Install
    it from a clone as below. For the `finetune()` / CLI library that `0.6.0`
    contains, see the [legacy trainer API](../getting-started.md).

## Install

```bash
git clone https://github.com/szaher/xaytune && cd xaytune
uv sync --locked                 # or: pip install -e .
uv sync --locked --extra trl     # the TRL trainer as well; or: pip install -e ".[trl]"
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

## 5. Evaluate the trained model

`ExperimentSpec.evaluation` takes an `EvaluationSpec` naming one evaluator.
When it is set, the host evaluates the trained model after training, through
the same journal, runtime and telemetry, records the result, and moves the
candidate to `DECIDING`, where it is decided (section 6).

The built-in `native` evaluator measures next-token loss, perplexity and token
accuracy on a local JSONL file of held-out text:

```python
from xaytune.core.domain.evaluation import EvaluationSpec, EvaluatorSpec
from xaytune.evaluation.native import local_dataset

evaluation = EvaluationSpec(
    evaluator=EvaluatorSpec(
        name="native",
        config={
            "format": "text",                 # each record's "text" field
            "max_seq_length": 512,            # truncation, in tokens
            "batch_size": 8,
            "metrics": ["loss", "perplexity", "token_accuracy"],
            "precision": "fp32",
        },
    ),
    dataset=local_dataset("/abs/data/held-out.jsonl"),   # pinned by content digest
)
result = await (await host.submit(spec.model_copy(update={"evaluation": evaluation}))).wait()
print(result.next_stage)       # "decision": this spec's objective has no target yet
for metric in result.nodes[0].evaluations[0].result.metrics:
    print(metric.name, metric.value, metric.seed)
```

Everything that changes a number is declared, and nothing is defaulted.
`local_dataset()` records the file's digest when you build the spec; the
worker checks the file still holds those bytes, and the evaluation fails if
it does not. An evaluation the evaluator cannot run exactly as declared (an
unpinned dataset, a format it does not read, a precision it does not use) is
refused by `submit()` with every reason, before anything trains.

The metrics are next-token and token-weighted: the logits at position *i* are
scored against the token at *i + 1*, averaged over tokens rather than batches.
The evaluator is **seeded**, not deterministic: the run's seed is applied and
recorded, and the report beside the result names the device and library
versions it ran with.

Run it: `python examples/control_plane/05_train_and_evaluate.py --model ... --dataset ... --held-out ...`.

## 6. Decide

The experiment's `Objective` says what counts as good enough. With a
`target`, the evaluated candidate is decided as soon as its results are in:

```python
from xaytune.core import Objective, ObjectiveMetric, MetricConstraint

spec = spec.model_copy(update={
    "objective": Objective(
        primary=ObjectiveMetric(name="loss", direction="minimize"),
        target=2.5,                                   # good enough: loss <= 2.5
        constraints=(MetricConstraint(name="token_accuracy", operator=">=", value=0.2),),
    ),
    "evaluation": evaluation,
})
result = await (await host.submit(spec)).wait()
print(result.status, result.next_stage)               # SUCCEEDED, None  (or FAILED, None)

decision = host.repository.aggregates.decisions_for_node(str(result.nodes[0].node_id))[0]
print(decision.outcome, decision.reason)
```

A target met completes the candidate, and the experiment `SUCCEEDED`, with
`best_node_id` naming the candidate. A target missed rejects the candidate,
and the experiment `FAILED`. A violated constraint rejects the candidate but
leaves the experiment `ACTIVE`, since another candidate could still succeed;
`next_stage` is then `"planning"`. The decision is recorded with its evidence
in the same commit that applies it.

Without a target, or with a metric the objective names but the evaluation did
not report, nothing is decided. The candidate stays `DECIDING`, the reason is
recorded as a `DecisionDeferred` event, and `next_stage` is `"decision"`. See
[concepts](concepts.md#decisions).

Run it: `python examples/control_plane/05_train_and_evaluate.py ... --target 2.5`.

## What is not supported yet

Both compilers run **full-parameter SFT on one worker** from local files.
They refuse, with reasons: adapters (LoRA/QLoRA), checkpoint intent,
algorithm variants, rewards, hub model names, and any training-relevant
value left undeclared. The TRL compiler also refuses data formats other than
`text`, and packing. The legacy trainer API still does all of these, outside
the control plane.

The `native` evaluator, likewise, reads only local plain text and evaluates in
`fp32`. It refuses slices, a dataset revision or split, and dataset
fingerprints it cannot verify.
