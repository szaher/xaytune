"""Describe a candidate, see it refused, fix it, and compile it into a plan.

Runs anywhere: nothing here loads a model, reads a file, or starts a process.
A compiler compiles; it never executes. The plan it returns is data -- the
thing Xaytune records before anything runs, and rebuilds after a restart.

    python examples/control_plane/01_compile_a_candidate.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from xaytune.compilation import CompilationContext, UnsupportedCandidateError
from xaytune.compilation.native import NativeCompiler
from xaytune.core import (
    CandidateSpec,
    DatasetRef,
    DataSpec,
    LRScheduleSpec,
    ModelRef,
    ModelSpec,
    OptimizationSpec,
    OptimizerSpec,
    PrecisionSpec,
    ResolvedExecutionPlan,
    RuntimeOperationTarget,
    TrainingKind,
    TrainingSpec,
)


def candidate(model_uri: str, dataset_uri: str) -> CandidateSpec:
    """One SFT hypothesis, with every value that changes what the model learns declared."""
    return CandidateSpec(
        model=ModelSpec(model=ModelRef(uri=model_uri)),
        data=DataSpec(
            dataset=DatasetRef(uri=dataset_uri),
            format="text",
            max_seq_length=512,
            packing=False,
        ),
        training=TrainingSpec(
            kind=TrainingKind.SFT,
            optimization=OptimizationSpec(
                optimizer=OptimizerSpec(name="adamw", weight_decay=0.0),
                lr_schedule=LRScheduleSpec(name="constant"),
                learning_rate=2e-5,
                micro_batch_size=4,
                gradient_accumulation=1,
                epochs=1,
                max_grad_norm=1.0,
            ),
            precision=PrecisionSpec(dtype="fp32"),
        ),
    )


def main() -> None:
    # Never created: compilation only names these paths for a worker to read.
    workdir = Path(tempfile.gettempdir()).resolve() / "xaytune-example"
    compiler = NativeCompiler()
    context = CompilationContext(
        run_id="run_example", seed=7, output_uri=str(workdir / "artifacts" / "run_example")
    )

    # A hub name is not a model: without a pinned revision it names whatever
    # the hub serves on the day the worker starts. The compiler says so, with
    # every reason at once rather than one per attempt.
    unpinned = candidate("Qwen/Qwen3-0.6B", "data/train.jsonl")
    try:
        compiler.compile(unpinned, context)
    except UnsupportedCandidateError as refusal:
        print(f"refused by {refusal.compiler}:")
        for reason in refusal.reasons:
            print(f"  - {reason}")

    # Absolute local paths. Compilation does not read them -- the worker does.
    pinned = candidate(str(workdir / "models" / "base"), str(workdir / "data" / "train.jsonl"))
    print(f"\ncandidate fingerprint: {pinned.candidate_fingerprint()}")

    spec = compiler.compile(pinned, context)
    plan = ResolvedExecutionPlan(
        spec=spec,
        runtime="local",
        target=RuntimeOperationTarget(kind="training-attempt", id="attempt_example"),
    )
    print(f"compiled by {spec.compiler.name} {spec.compiler.version}")
    print(f"worker entrypoint: {spec.entrypoint.module}:{spec.entrypoint.function}")
    print(f"submit request digest: {plan.request_digest('submit')}")

    # The plan is serializable, and reading it back gives the same request --
    # which is what lets a restarted controller prove it is re-issuing the
    # submission it recorded, not a new one.
    restored = ResolvedExecutionPlan.model_validate_json(plan.model_dump_json())
    assert restored.request_digest("submit") == plan.request_digest("submit")
    print("\nworker config:")
    print(json.dumps(spec.model_dump(mode="json")["config"], indent=2))


if __name__ == "__main__":
    main()
