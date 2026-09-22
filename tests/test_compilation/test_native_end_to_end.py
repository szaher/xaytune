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

from xaytune.core.domain.candidate import (
    CandidateSpec,
    DataSpec,
    ModelSpec,
    OptimizationSpec,
    TrainingKind,
    TrainingSpec,
)
from xaytune.core.domain.operation import RuntimeOperationTarget
from xaytune.core.execution import ResolvedExecutionPlan, TrainingExecutionSpec
from xaytune.core.ids import OperationId
from xaytune.core.refs import DatasetRef, ModelRef
from xaytune.runtimes.local import LocalRuntime

_TERMINAL = frozenset({"succeeded", "failed", "cancelled", "unknown"})


def _sft_candidate() -> CandidateSpec:
    """The smallest candidate that is still a real SFT hypothesis."""
    return CandidateSpec(
        model=ModelSpec(model=ModelRef(uri="xaytune-test-tiny")),
        data=DataSpec(dataset=DatasetRef(uri="xaytune-test-tiny-dataset")),
        training=TrainingSpec(
            kind=TrainingKind.SFT,
            optimization=OptimizationSpec(learning_rate=1e-3, max_steps=2),
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

    candidate = _sft_candidate()
    compiler = NativeCompiler()

    assert compiler.supports(candidate), "the first compiler must support plain SFT"

    spec = compiler.compile(candidate, _context(tmp_path))

    # It has to survive the boundary it was built to cross.
    restored = TrainingExecutionSpec.model_validate(json.loads(spec.model_dump_json()))
    assert restored == spec

    assert spec.compiler.descriptor == compiler.descriptor, "ADR-008: the plan names its producer"
    assert spec.candidate_fingerprint == candidate.candidate_fingerprint()
    assert "native" in spec.entrypoint.module

    plan = ResolvedExecutionPlan(
        spec=restored,
        runtime="local",
        target=RuntimeOperationTarget(kind="training-attempt", id="ra_e2e"),
    )

    runtime = LocalRuntime(tmp_path / "runtime")
    try:
        status = asyncio.run(_run(runtime, plan))
        events = asyncio.run(_events(runtime, plan))
    finally:
        runtime.close()

    assert status.state == "succeeded", status.detail

    emitted = [event.payload.data.type for event in events]
    assert "TrainingStarted" in emitted, "the worker must say it began"
    assert "TrainingMetricObserved" in emitted, "a real train() reports at least one metric"
    assert "TrainingCompleted" in emitted, "and must say it finished"

    # ADR-014 §1a: the supervisor owns sequencing, and it is gapless.
    positions = [(event.stream_generation, event.sequence) for event in events]
    assert positions == sorted(positions)
    assert [s for _, s in positions] == list(range(len(positions)))


async def _run(runtime: LocalRuntime, plan: ResolvedExecutionPlan) -> object:
    ref = await runtime.submit_or_get(OperationId.generate(), plan)
    return await _settle(runtime, ref)


async def _events(runtime: LocalRuntime, plan: ResolvedExecutionPlan) -> list:
    from xaytune.core.refs import RuntimeRef

    ref = RuntimeRef(backend="local", external_id=plan.target.id)
    return [event async for event in runtime.watch(ref)]


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
    from pathlib import Path

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
