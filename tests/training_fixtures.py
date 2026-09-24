"""Shared fixtures for the end-to-end tests: compilers, and the controller above them.

Every compiler is held to one contract, so every end-to-end test runs on the
same fixtures: the same tiny offline model, the same candidate
shape, the same runtime and the same way of reading telemetry back. A fixture
that differed per compiler would let a difference in the fixture pass for a
difference in the compiler.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from xaytune.core.domain.candidate import (
    CandidateSpec,
    DataSpec,
    LRScheduleSpec,
    ModelSpec,
    OptimizationSpec,
    OptimizerSpec,
    PrecisionSpec,
    TrainingKind,
    TrainingSpec,
)
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.execution import ResolvedExecutionPlan, TrainingExecutionSpec
from xaytune.core.ids import OperationId
from xaytune.core.refs import DatasetRef, ModelRef
from xaytune.runtimes.local import LocalRuntime

TERMINAL = frozenset({"succeeded", "failed", "cancelled", "unknown"})


def tiny_model(directory: Path) -> Path:
    """A real Hugging Face artifact, built here rather than downloaded.

    Roughly four thousand parameters, saved with ``save_pretrained``. It is not
    a stand-in: ``load_model`` resolves it through the ordinary
    ``AutoModelForCausalLM.from_pretrained`` path a production model takes, so
    what this proves is that a serialized ``CandidateSpec`` can name an
    artifact a separate process resolves normally -- not that a test-only
    branch of the loader works.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "### Instruction:": 3,
        "### Response:": 4,
        "hello": 5,
        "world": 6,
        "train": 7,
        ".": 8,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        # The model config below names <eos> (id 1) as BOS too. A tokenizer
        # that disagreed would be an incoherent artifact, and transformers'
        # Trainer "aligns" such a pair by rewriting the model config.
        bos_token="<eos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_layer=1,
            n_head=1,
            n_embd=16,
            n_positions=32,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )

    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory


def tiny_dataset(path: Path, data_format: str = "alpaca") -> Path:
    """Four samples, on disk, as local JSONL in *data_format*.

    ``"text"`` is the one format both compilers read with the same meaning --
    the whole sequence is the training target -- so it is what a test running
    one candidate through two trainers uses.
    """
    if data_format == "alpaca":
        samples = [
            {"instruction": "train", "input": "", "output": "hello world ."},
            {"instruction": "train", "input": "hello", "output": "world ."},
            {"instruction": "train", "input": "world", "output": "hello ."},
            {"instruction": "train", "input": "", "output": "train ."},
        ]
    elif data_format == "text":
        samples = [
            {"text": "hello world ."},
            {"text": "world hello ."},
            {"text": "train hello world ."},
            {"text": "train ."},
        ]
    else:
        raise ValueError(f"no tiny dataset in format {data_format!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")
    return path


def assert_fixture_is_a_real_artifact(model_dir: Path) -> None:
    """Fail here rather than inside the worker if the fixture is not valid.

    Without this, "the worker failed" and "the test fixture was never a
    loadable model" look identical from the outside, and the second one sends
    you reading worker code for a bug that is in the test.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def sft_candidate(model_dir: Path, dataset: Path, data_format: str = "alpaca") -> CandidateSpec:
    """The smallest candidate that is still a real SFT hypothesis.

    Absolute paths: the worker has its own working directory, and a relative
    one would quietly prove that cwd was inherited rather than that the plan
    carried everything the worker needed (ADR-016).
    """
    return CandidateSpec(
        model=ModelSpec(model=ModelRef(uri=str(model_dir.resolve()))),
        data=DataSpec(
            dataset=DatasetRef(uri=str(dataset.resolve())),
            format=data_format,
            # The tiny model has n_positions=32; longer would overrun it.
            max_seq_length=32,
            packing=False,
        ),
        training=TrainingSpec(
            kind=TrainingKind.SFT,
            optimization=OptimizationSpec(
                optimizer=OptimizerSpec(name="adamw", weight_decay=0.0),
                lr_schedule=LRScheduleSpec(name="constant"),
                learning_rate=1e-3,
                micro_batch_size=2,
                gradient_accumulation=1,
                epochs=1,
                max_steps=2,
                max_grad_norm=1.0,
            ),
            # CPU in CI; half-precision autocast is not what is being tested.
            precision=PrecisionSpec(dtype="fp32"),
        ),
    )


def compilation_context(tmp_path: Path, *, output_uri: str | None = None):
    from xaytune.compilation import CompilationContext

    return CompilationContext(
        run_id="run_e2e",
        seed=7,
        output_uri=output_uri or str(tmp_path / "artifacts"),
        checkpoint_store_uri=str(tmp_path / "checkpoints"),
    )


def offline_plan(spec: TrainingExecutionSpec, target_id: str) -> ResolvedExecutionPlan:
    """*spec* resolved for the local runtime, forbidden from reaching the network.

    Hostile to accidental network access: if a run passes under this, the
    worker crossed the process boundary with everything it needed.
    """
    offline = spec.model_copy(
        update={
            "environment": {
                **dict(spec.environment),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        }
    )
    return ResolvedExecutionPlan(
        spec=offline,
        runtime="local",
        target=RuntimeOperationTarget(kind="training-attempt", id=target_id),
    )


async def settle(runtime: LocalRuntime, ref: object) -> object:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 120.0
    status = await runtime.get_status(ref)  # type: ignore[arg-type]
    while status.state not in TERMINAL and loop.time() < deadline:
        await asyncio.sleep(0.05)
        status = await runtime.get_status(ref)  # type: ignore[arg-type]
    return status


def run_to_completion(runtime_dir: Path, plan: ResolvedExecutionPlan) -> tuple[object, list]:
    """Submit, wait, and read the telemetry back through the same reference.

    The reference is the one submission returned. A workload is keyed by the
    operation that started it, not by the attempt it serves, so rebuilding a
    reference from the target would name a workload that does not exist.
    """

    async def go(runtime: LocalRuntime) -> tuple[object, list]:
        ref = await runtime.submit_or_get(OperationId.generate(), plan)
        status = await settle(runtime, ref)
        return status, [event async for event in runtime.watch(ref)]

    runtime = LocalRuntime(runtime_dir)
    try:
        return asyncio.run(go(runtime))
    finally:
        runtime.close()


def emitted_types(events: list) -> list[str]:
    return [event.payload.data.type for event in events]


def incident_reasons(events: list) -> list[str]:
    return [
        event.payload.data.reason
        for event in events
        if event.payload.data.type == "IncidentObserved"
    ]


def imported_names(module_path: Path) -> set[str]:
    """Every name *module_path* imports, read from its syntax tree."""
    import ast

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ImportFrom, ast.Import)):
            imported.update(alias.name for alias in node.names)
    return imported
