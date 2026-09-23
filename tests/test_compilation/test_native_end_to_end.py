"""The first path that runs all the way through, for real.

```text
CandidateSpec(SFT)
      ↓ NativeCompiler
TrainingExecutionSpec   ── JSON round-trip ──>
      ↓ resolver
ResolvedExecutionPlan(runtime="local")
      ↓ LocalRuntime.submit_or_get
launcher (the telemetry supervisor)
      ↓
NativeWorker
      ↓
Trainer.train()
```

One test for the whole seam rather than one per mapping helper. Every piece of
this has been contract-tested against a fake on one side or the other; what has
never been shown is that a candidate compiled by a real compiler runs to
completion on a real runtime and reports what happened. A mapping helper can be
correct in isolation and still produce a spec that nothing can execute.

Offline and tiny on purpose: no network, no Hugging Face download, no GPU. A
test needing any of those is one nobody runs, and an end-to-end test nobody
runs is worse than none, because it reports the seam as covered.
"""

from __future__ import annotations

import asyncio
import json
import math
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

_TERMINAL = frozenset({"succeeded", "failed", "cancelled", "unknown"})


def _tiny_model(directory: Path) -> Path:
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
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
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


def _tiny_dataset(path: Path) -> Path:
    """Four alpaca samples, on disk, in the format the local loader reads."""
    samples = [
        {"instruction": "train", "input": "", "output": "hello world ."},
        {"instruction": "train", "input": "hello", "output": "world ."},
        {"instruction": "train", "input": "world", "output": "hello ."},
        {"instruction": "train", "input": "", "output": "train ."},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")
    return path


def _assert_fixture_is_a_real_artifact(model_dir: Path) -> None:
    """Fail here rather than inside the worker if the fixture is not valid.

    Without this, "NativeWorker failed" and "the test fixture was never a
    loadable model" look identical from the outside, and the second one sends
    you reading worker code for a bug that is in the test.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def _sft_candidate(model_dir: Path, dataset: Path) -> CandidateSpec:
    """The smallest candidate that is still a real SFT hypothesis.

    Absolute paths: the worker has its own working directory, and a relative
    one would quietly prove that cwd was inherited rather than that the plan
    carried everything the worker needed (ADR-016).
    """
    return CandidateSpec(
        model=ModelSpec(model=ModelRef(uri=str(model_dir.resolve()))),
        data=DataSpec(
            dataset=DatasetRef(uri=str(dataset.resolve())),
            format="alpaca",
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


async def _settle(runtime: LocalRuntime, ref: object) -> object:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 120.0
    status = await runtime.get_status(ref)  # type: ignore[arg-type]
    while status.state not in _TERMINAL and loop.time() < deadline:
        await asyncio.sleep(0.05)
        status = await runtime.get_status(ref)  # type: ignore[arg-type]
    return status


def test_an_sft_candidate_compiles_runs_and_reports(tmp_path) -> None:
    """The milestone: a hypothesis becomes a finished run without a fake anywhere."""
    from xaytune.compilation.native import NativeCompiler

    model_dir = _tiny_model(tmp_path / "tiny-model")
    dataset = _tiny_dataset(tmp_path / "data" / "train.jsonl")
    _assert_fixture_is_a_real_artifact(model_dir)

    candidate = _sft_candidate(model_dir, dataset)
    compiler = NativeCompiler()

    assert compiler.supports(candidate), "the first compiler must support plain SFT"

    spec = compiler.compile(candidate, _context(tmp_path))

    # It has to survive the boundary it was built to cross.
    restored = TrainingExecutionSpec.model_validate(json.loads(spec.model_dump_json()))
    assert restored == spec

    assert spec.compiler.descriptor == compiler.descriptor, "ADR-008: the plan names its producer"
    assert spec.candidate_fingerprint == candidate.candidate_fingerprint()
    assert "native" in spec.entrypoint.module

    # Hostile to accidental network access: if this passes, the worker crossed
    # the process boundary with everything it needed.
    offline = restored.model_copy(
        update={
            "environment": {
                **dict(restored.environment),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        }
    )
    plan = ResolvedExecutionPlan(
        spec=offline,
        runtime="local",
        target=RuntimeOperationTarget(kind="training-attempt", id="ra_e2e"),
    )

    runtime = LocalRuntime(tmp_path / "runtime")
    try:
        status, events = asyncio.run(_run(runtime, plan))
    finally:
        runtime.close()

    assert status.state == "succeeded", status.detail

    emitted = [event.payload.data.type for event in events]
    assert "TrainingStarted" in emitted, "the worker must say it began"
    assert "TrainingMetricObserved" in emitted, "a real train() reports at least one metric"
    assert "TrainingCompleted" in emitted, "and must say it finished"

    # Real optimizer steps with finite metrics. Deliberately not "loss went
    # down": with a four-thousand-parameter random model that is evidence, not
    # a contract, and a test built on it would be flaky by design.
    metrics = [e.payload.data for e in events if e.payload.data.type == "TrainingMetricObserved"]
    assert [m.optimizer_step for m in metrics] == [1, 2], "max_steps=2 means two steps"
    assert all(m.loss is not None and math.isfinite(m.loss) for m in metrics)

    # No raw state_dict checkpoint the candidate never asked for.
    assert not list((tmp_path / "artifacts").glob("checkpoint-*"))

    # ADR-014 §1a: the supervisor owns sequencing, and it is gapless.
    positions = [(event.stream_generation, event.sequence) for event in events]
    assert positions == sorted(positions)
    assert [s for _, s in positions] == list(range(len(positions)))

    # The plan declared a model output, so the run has to produce one -- and
    # "produce" means an artifact something else can load, not a raw state
    # dict inside a checkpoint directory. This passed without these checks
    # while the declared output did not exist.
    assert "ArtifactProduced" in emitted, "the declared model output was never produced"
    assert emitted.index("ArtifactProduced") > emitted.index("TrainingCompleted")

    produced = next(e for e in events if e.payload.data.type == "ArtifactProduced")
    artifact = produced.payload.data.artifact_ref
    declared = next(o for o in spec.outputs if o.kind == "model")
    assert artifact.kind == "model"
    assert artifact.uri == declared.uri

    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoModelForCausalLM.from_pretrained(artifact.uri, local_files_only=True)
    AutoTokenizer.from_pretrained(artifact.uri, local_files_only=True)


async def _run(runtime: LocalRuntime, plan: ResolvedExecutionPlan) -> tuple[object, list]:
    """Submit, wait, and read the telemetry back through the same reference.

    The reference is the one submission returned. A workload is keyed by the
    operation that started it, not by the attempt it serves, so rebuilding a
    reference from the target would name a workload that does not exist.
    """
    ref = await runtime.submit_or_get(OperationId.generate(), plan)
    status = await _settle(runtime, ref)
    return status, [event async for event in runtime.watch(ref)]


def _context(tmp_path):
    from xaytune.compilation import CompilationContext

    return CompilationContext(
        run_id="run_e2e",
        seed=7,
        output_uri=str(tmp_path / "artifacts"),
        checkpoint_store_uri=str(tmp_path / "checkpoints"),
    )


def test_the_worker_never_sees_the_envelope() -> None:
    """ADR-014 §1a: exactly one telemetry supervisor assigns sequence.

    The worker emits observation *bodies*; the launcher wraps them. If the
    worker could build a ``RuntimeEventEnvelope`` it would be choosing a
    sequence, and two writers to one stream is the condition that makes a gap
    indistinguishable from a reorder. Checked by import, not by reading the
    source, because an import is what would actually let it happen.
    """
    import ast

    worker = Path(__file__).resolve().parents[2] / "xaytune" / "workers" / "native.py"
    tree = ast.parse(worker.read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    forbidden = {"RuntimeEventEnvelope", "StreamCursor", "LocalRuntime"}
    assert not (imported & forbidden), f"the worker imports {imported & forbidden}"


def test_a_training_failure_is_reported_at_both_levels(tmp_path) -> None:
    """Training-level and process-level evidence, and neither is a duplicate.

    A dataset that does not exist compiles perfectly -- the compiler inspects
    nothing, by design -- so the failure has to surface at execution. It must
    be attributed there: the worker says training failed, and the launcher
    separately says the process exited non-zero.
    """
    from xaytune.compilation.native import NativeCompiler

    model_dir = _tiny_model(tmp_path / "tiny-model")
    missing = tmp_path / "data" / "never-written.jsonl"
    candidate = _sft_candidate(model_dir, missing)

    spec = NativeCompiler().compile(candidate, _context(tmp_path))
    plan = ResolvedExecutionPlan(
        spec=spec.model_copy(
            update={
                "environment": {
                    **dict(spec.environment),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
            }
        ),
        runtime="local",
        target=RuntimeOperationTarget(kind="training-attempt", id="ra_fails"),
    )

    runtime = LocalRuntime(tmp_path / "runtime")
    try:
        status, events = asyncio.run(_run(runtime, plan))
    finally:
        runtime.close()

    emitted = [event.payload.data.type for event in events]
    assert status.state == "failed"
    assert "TrainingFailed" in emitted, "the worker must say training failed"
    assert "TrainingCompleted" not in emitted
    assert "ArtifactProduced" not in emitted

    reasons = [
        event.payload.data.reason
        for event in events
        if event.payload.data.type == "IncidentObserved"
    ]
    assert "nonzero-exit" in reasons, "and the launcher must say the process failed"


def test_training_that_completes_but_cannot_publish_is_a_failed_run(tmp_path) -> None:
    """Training finished; the declared output does not exist. Both are facts.

    ```text
    TrainingCompleted
    IncidentObserved(artifact-publication-failed)
    no ArtifactProduced
    non-zero exit  ->  status failed
    ```

    Triggered by an output location that is a regular file, which is the case
    that first fooled the worker: ``save_pretrained`` logs an error for it and
    returns without raising, and the worker announced a model that was never
    written, under a run reporting success.
    """
    from xaytune.compilation import CompilationContext
    from xaytune.compilation.native import NativeCompiler

    occupied = tmp_path / "not-a-directory"
    occupied.write_text("occupied")
    context = CompilationContext(
        run_id="run_pub",
        seed=7,
        output_uri=str(occupied),
        checkpoint_store_uri=str(tmp_path / "checkpoints"),
    )
    candidate = _sft_candidate(
        _tiny_model(tmp_path / "tiny-model"), _tiny_dataset(tmp_path / "data" / "t.jsonl")
    )
    spec = NativeCompiler().compile(candidate, context)
    plan = ResolvedExecutionPlan(
        spec=spec.model_copy(
            update={
                "environment": {
                    **dict(spec.environment),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
            }
        ),
        runtime="local",
        target=RuntimeOperationTarget(kind="training-attempt", id="ra_publish"),
    )

    runtime = LocalRuntime(tmp_path / "runtime")
    try:
        status, events = asyncio.run(_run(runtime, plan))
    finally:
        runtime.close()

    emitted = [event.payload.data.type for event in events]
    reasons = [
        event.payload.data.reason
        for event in events
        if event.payload.data.type == "IncidentObserved"
    ]

    assert "TrainingCompleted" in emitted, "training did finish"
    assert "TrainingFailed" not in emitted, "and saying otherwise would be false"
    assert "artifact-publication-failed" in reasons
    assert "ArtifactProduced" not in emitted, "a model that was never written"
    assert status.state == "failed"
    assert occupied.read_text() == "occupied"
