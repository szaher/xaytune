"""The second trainer, run all the way through, behind the same contract.

```text
CandidateSpec(SFT, text)
      ↓ TRLCompiler
TrainingExecutionSpec   ── JSON round-trip ──>
      ↓ resolver
ResolvedExecutionPlan(runtime="local")
      ↓ LocalRuntime.submit_or_get
launcher (the telemetry supervisor)
      ↓
TRLWorker
      ↓
SFTTrainer.train()
```

PR-010 proved one trainer runs behind the compile/execute boundary. That is not
yet evidence the boundary is trainer-neutral: a contract with one
implementation tends to describe that implementation. This is the second one,
built on a trainer Xaytune does not own and whose defaults it does not choose,
held to the same observable contract -- the same runtime, the same telemetry
vocabulary, the same artifact rules, and the same failure attribution.

Same fixtures as the native test, deliberately (see ``conftest.py``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

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

pytestmark = pytest.mark.trl


@pytest.fixture(autouse=True)
def _trl_installed() -> None:
    """Per test, not at import: a module-level skip happens during collection,
    before ``XAYTUNE_REQUIRE_TRL`` can turn it into a failure."""
    pytest.importorskip("trl")


def test_an_sft_candidate_compiles_runs_and_reports_through_trl(tmp_path) -> None:
    """The same milestone as PR-010, on a trainer Xaytune does not own."""
    from xaytune.compilation.trl import TRLCompiler

    model_dir = tiny_model(tmp_path / "tiny-model")
    dataset = tiny_dataset(tmp_path / "data" / "train.jsonl", "text")
    assert_fixture_is_a_real_artifact(model_dir)

    candidate = sft_candidate(model_dir, dataset, "text")
    compiler = TRLCompiler()

    assert compiler.supports(candidate), compiler.supports(candidate).reasons

    spec = compiler.compile(candidate, compilation_context(tmp_path))

    restored = TrainingExecutionSpec.model_validate(json.loads(spec.model_dump_json()))
    assert restored == spec

    assert spec.compiler.descriptor == compiler.descriptor, "ADR-008: the plan names its producer"
    assert spec.compiler.name == "trl"
    assert spec.candidate_fingerprint == candidate.candidate_fingerprint()
    assert spec.entrypoint.module == "xaytune.workers.trl"

    status, events = run_to_completion(tmp_path / "runtime", offline_plan(restored, "ra_trl"))

    assert status.state == "succeeded", status.detail

    emitted = emitted_types(events)
    assert "TrainingStarted" in emitted
    assert "TrainingCompleted" in emitted

    metrics = [e.payload.data for e in events if e.payload.data.type == "TrainingMetricObserved"]
    assert [m.optimizer_step for m in metrics] == [1, 2], "max_steps=2 means two steps"
    assert all(m.loss is not None and math.isfinite(m.loss) for m in metrics)
    assert all(m.learning_rate is not None and math.isfinite(m.learning_rate) for m in metrics), (
        "a constant schedule still has a learning rate to report"
    )

    # TRL saves every 500 steps by default; the candidate asked for none.
    assert not list((tmp_path / "artifacts").glob("checkpoint-*"))

    positions = [(event.stream_generation, event.sequence) for event in events]
    assert positions == sorted(positions)
    assert [s for _, s in positions] == list(range(len(positions)))

    assert emitted.count("ArtifactProduced") == 1
    assert emitted.index("ArtifactProduced") > emitted.index("TrainingCompleted")

    produced = next(e for e in events if e.payload.data.type == "ArtifactProduced")
    artifact = produced.payload.data.artifact_ref
    declared = next(o for o in spec.outputs if o.kind == "model")
    assert artifact.kind == "model"
    assert artifact.uri == declared.uri

    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoModelForCausalLM.from_pretrained(artifact.uri, local_files_only=True)
    AutoTokenizer.from_pretrained(artifact.uri, local_files_only=True)


def test_the_plan_carries_a_wire_schema_not_sft_config(tmp_path) -> None:
    """``SFTConfig`` is TRL's API, and it changes between TRL releases.

    A plan carrying it would make every recorded plan depend on the TRL
    version that wrote it. The plan carries Xaytune's own versioned schema,
    and the worker is the only place that translates it into ``SFTConfig``.
    """
    from xaytune.compilation.trl import TRLCompiler
    from xaytune.workers.trl_schema import TRLSftConfig

    candidate = sft_candidate(
        tiny_model(tmp_path / "m"), tiny_dataset(tmp_path / "d.jsonl", "text"), "text"
    )
    spec = TRLCompiler().compile(candidate, compilation_context(tmp_path))

    wire = TRLSftConfig.model_validate(dict(spec.config))
    assert wire.api_version == "xaytune.trl-sft/v1alpha1"
    assert wire.model_dump(mode="json") == dict(spec.config)


def test_the_trl_worker_never_sees_the_envelope() -> None:
    """ADR-014 §1a holds for every worker, not only the one Xaytune wrote first."""
    worker = Path(__file__).resolve().parents[2] / "xaytune" / "workers" / "trl.py"
    imported = imported_names(worker)

    forbidden = {"RuntimeEventEnvelope", "StreamCursor", "LocalRuntime"}
    assert not (imported & forbidden), f"the worker imports {imported & forbidden}"


def test_a_trl_training_failure_is_reported_at_both_levels(tmp_path) -> None:
    """A dataset that does not exist compiles, then fails, attributed twice."""
    from xaytune.compilation.trl import TRLCompiler

    candidate = sft_candidate(
        tiny_model(tmp_path / "tiny-model"), tmp_path / "data" / "never-written.jsonl", "text"
    )
    spec = TRLCompiler().compile(candidate, compilation_context(tmp_path))
    status, events = run_to_completion(tmp_path / "runtime", offline_plan(spec, "ra_trl_fails"))

    emitted = emitted_types(events)
    assert status.state == "failed"
    assert "TrainingFailed" in emitted
    assert "TrainingCompleted" not in emitted
    assert "ArtifactProduced" not in emitted
    assert "nonzero-exit" in incident_reasons(events)


def test_trl_training_that_completes_but_cannot_publish_is_a_failed_run(tmp_path) -> None:
    """Training finished; the declared output does not exist. Both are facts."""
    from xaytune.compilation.trl import TRLCompiler

    occupied = tmp_path / "not-a-directory"
    occupied.write_text("occupied")
    candidate = sft_candidate(
        tiny_model(tmp_path / "tiny-model"),
        tiny_dataset(tmp_path / "data" / "t.jsonl", "text"),
        "text",
    )
    spec = TRLCompiler().compile(candidate, compilation_context(tmp_path, output_uri=str(occupied)))
    status, events = run_to_completion(tmp_path / "runtime", offline_plan(spec, "ra_trl_pub"))

    emitted = emitted_types(events)
    assert "TrainingCompleted" in emitted
    assert "TrainingFailed" not in emitted
    assert "artifact-publication-failed" in incident_reasons(events)
    assert "ArtifactProduced" not in emitted
    assert status.state == "failed"
    assert occupied.read_text() == "occupied"
