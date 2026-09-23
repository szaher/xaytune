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

import json
import math
from pathlib import Path

from xaytune.core.execution import TrainingExecutionSpec

from .conftest import (
    assert_fixture_is_a_real_artifact,
    compilation_context,
    emitted_types,
    imported_names,
    incident_reasons,
    offline_plan,
    run_to_completion,
    sft_candidate,
    tiny_dataset,
    tiny_model,
)


def test_an_sft_candidate_compiles_runs_and_reports(tmp_path) -> None:
    """The milestone: a hypothesis becomes a finished run without a fake anywhere."""
    from xaytune.compilation.native import NativeCompiler

    model_dir = tiny_model(tmp_path / "tiny-model")
    dataset = tiny_dataset(tmp_path / "data" / "train.jsonl")
    assert_fixture_is_a_real_artifact(model_dir)

    candidate = sft_candidate(model_dir, dataset)
    compiler = NativeCompiler()

    assert compiler.supports(candidate), "the first compiler must support plain SFT"

    spec = compiler.compile(candidate, compilation_context(tmp_path))

    # It has to survive the boundary it was built to cross.
    restored = TrainingExecutionSpec.model_validate(json.loads(spec.model_dump_json()))
    assert restored == spec

    assert spec.compiler.descriptor == compiler.descriptor, "ADR-008: the plan names its producer"
    assert spec.candidate_fingerprint == candidate.candidate_fingerprint()
    assert "native" in spec.entrypoint.module

    status, events = run_to_completion(tmp_path / "runtime", offline_plan(restored, "ra_e2e"))

    assert status.state == "succeeded", status.detail

    emitted = emitted_types(events)
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


def test_the_worker_never_sees_the_envelope() -> None:
    """ADR-014 §1a: exactly one telemetry supervisor assigns sequence.

    The worker emits observation *bodies*; the launcher wraps them. If the
    worker could build a ``RuntimeEventEnvelope`` it would be choosing a
    sequence, and two writers to one stream is the condition that makes a gap
    indistinguishable from a reorder. Checked by import, not by reading the
    source, because an import is what would actually let it happen.
    """
    worker = Path(__file__).resolve().parents[2] / "xaytune" / "workers" / "native.py"
    imported = imported_names(worker)

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

    model_dir = tiny_model(tmp_path / "tiny-model")
    missing = tmp_path / "data" / "never-written.jsonl"
    candidate = sft_candidate(model_dir, missing)

    spec = NativeCompiler().compile(candidate, compilation_context(tmp_path))
    status, events = run_to_completion(tmp_path / "runtime", offline_plan(spec, "ra_fails"))

    emitted = emitted_types(events)
    assert status.state == "failed"
    assert "TrainingFailed" in emitted, "the worker must say training failed"
    assert "TrainingCompleted" not in emitted
    assert "ArtifactProduced" not in emitted
    assert "nonzero-exit" in incident_reasons(events), (
        "and the launcher must say the process failed"
    )


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
    from xaytune.compilation.native import NativeCompiler

    occupied = tmp_path / "not-a-directory"
    occupied.write_text("occupied")
    candidate = sft_candidate(
        tiny_model(tmp_path / "tiny-model"), tiny_dataset(tmp_path / "data" / "t.jsonl")
    )
    spec = NativeCompiler().compile(
        candidate, compilation_context(tmp_path, output_uri=str(occupied))
    )
    status, events = run_to_completion(tmp_path / "runtime", offline_plan(spec, "ra_publish"))

    emitted = emitted_types(events)

    assert "TrainingCompleted" in emitted, "training did finish"
    assert "TrainingFailed" not in emitted, "and saying otherwise would be false"
    assert "artifact-publication-failed" in incident_reasons(events)
    assert "ArtifactProduced" not in emitted, "a model that was never written"
    assert status.state == "failed"
    assert occupied.read_text() == "occupied"
