"""The experiment the runnable examples submit: one SFT candidate, run locally.

Shared by ``02_train.py``, ``03_cancel.py`` and ``04_restart_and_attach.py`` so
they differ only in what they do with it. ``01_compile_a_candidate.py`` builds
the same candidate inline, field by field.

Every value that changes what the model learns is declared. The compilers
refuse a candidate that leaves one to a trainer default, because the default
would then be part of the run but not of the candidate's fingerprint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xaytune.core import (
    CandidateSpec,
    DatasetRef,
    DataSpec,
    LRScheduleSpec,
    ModelRef,
    ModelSpec,
    Objective,
    ObjectiveMetric,
    OptimizationSpec,
    OptimizerSpec,
    PrecisionSpec,
    TrainingKind,
    TrainingSpec,
)
from xaytune.experiment import CompilerSpec, ExperimentSpec, RuntimeSpec


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="local Hugging Face model directory")
    parser.add_argument("--dataset", required=True, help="local JSONL with a 'text' field")
    parser.add_argument("--compiler", choices=("native", "trl"), default="native")
    parser.add_argument(
        "--workdir",
        default="xaytune-workdir",
        help="holds the control-plane database, the runtime registry and the models",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=512)


def state_path(args: argparse.Namespace) -> Path:
    """The control-plane database: the record every handle reads."""
    return Path(args.workdir).resolve() / "state.db"


def experiment(args: argparse.Namespace) -> ExperimentSpec:
    # Absolute paths throughout: the worker is another process, with its own
    # working directory, and resolves them there.
    workdir = Path(args.workdir).resolve()
    candidate = CandidateSpec(
        model=ModelSpec(model=ModelRef(uri=str(Path(args.model).resolve()))),
        data=DataSpec(
            dataset=DatasetRef(uri=str(Path(args.dataset).resolve())),
            format="text",
            max_seq_length=args.max_seq_length,
            packing=False,
        ),
        training=TrainingSpec(
            kind=TrainingKind.SFT,
            optimization=OptimizationSpec(
                optimizer=OptimizerSpec(name="adamw", weight_decay=0.0),
                lr_schedule=LRScheduleSpec(name="constant"),
                learning_rate=args.learning_rate,
                micro_batch_size=2,
                gradient_accumulation=1,
                epochs=1,
                max_steps=args.max_steps,
                max_grad_norm=1.0,
            ),
            precision=PrecisionSpec(dtype="fp32"),
        ),
    )
    return ExperimentSpec(
        name="control-plane-sft",
        objective=Objective(primary=ObjectiveMetric(name="loss", direction="minimize")),
        candidate=candidate,
        # The run's seed, not the candidate's: two seeds are two samples of
        # one hypothesis, under one fingerprint.
        seed=args.seed,
        compiler=CompilerSpec(name=args.compiler),
        runtime=RuntimeSpec(kind="local", config={"root": str(workdir / "runtime")}),
        artifact_root=str(workdir / "artifacts"),
    )
