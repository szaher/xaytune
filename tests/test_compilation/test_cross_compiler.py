"""One candidate, two trainers, one contract.

```text
                       ┌─ NativeCompiler ─ NativeWorker ─ Trainer.train()
CandidateSpec(SFT) ────┤
                       └─ TRLCompiler ──── TRLWorker ──── SFTTrainer.train()
```

The claim PR-011 exists to test: the compile/execute boundary is
trainer-neutral. If it is, a controller cannot tell from the control plane
which trainer ran -- only from the execution identity, which is where ADR-011
§5 says the difference belongs.

**Asserted to be the same:** the candidate and its fingerprint, the telemetry
lifecycle (event types in order, and incident reasons), the optimizer steps
metrics arrive at, the artifact contract, and the terminal classification --
for success, for a training failure, and for a publication failure.

**Asserted to differ:** the compiler, the worker, and the request digest --
execution identity. Runs through the two are not interchangeable replicates.

**Deliberately not compared:** weights and losses. The trainers need not
produce the same trajectory, and a test demanding it would be testing
numerics nobody promised.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests.training_fixtures import (
    compilation_context,
    offline_plan,
    run_to_completion,
    sft_candidate,
    tiny_dataset,
    tiny_model,
)
from xaytune.core.execution import TrainingExecutionSpec

pytestmark = pytest.mark.trl


@pytest.fixture(autouse=True)
def _trl_installed() -> None:
    pytest.importorskip("trl")


def _compilers() -> dict:
    from xaytune.compilation.native import NativeCompiler
    from xaytune.compilation.trl import TRLCompiler

    return {"native": NativeCompiler(), "trl": TRLCompiler()}


def _lifecycle(events: list) -> list[str]:
    """What a controller reads: event types in order, incidents by reason.

    Metric *values* are not part of it; metric *positions* are, because a
    missing or extra step is a lifecycle difference.
    """
    shape = []
    for event in events:
        data = event.payload.data
        if data.type == "IncidentObserved":
            shape.append(f"IncidentObserved({data.reason})")
        else:
            shape.append(data.type)
    return shape


def _run_both(tmp_path: Path, *, dataset: Path, output: Path | None = None) -> dict:
    model_dir = tiny_model(tmp_path / "model")
    candidate = sft_candidate(model_dir, dataset, "text")

    results = {}
    for name, compiler in _compilers().items():
        assert compiler.supports(candidate), (name, compiler.supports(candidate).reasons)
        root = tmp_path / name
        spec = compiler.compile(
            candidate,
            compilation_context(root, output_uri=str(output) if output else None),
        )
        restored = TrainingExecutionSpec.model_validate(json.loads(spec.model_dump_json()))
        assert restored == spec, f"{name}: the plan must survive serialization"
        plan = offline_plan(restored, f"ra_{name}")
        status, events = run_to_completion(root / "runtime", plan)
        results[name] = {"spec": spec, "plan": plan, "status": status, "events": events}
    results["candidate"] = candidate
    return results


def test_one_candidate_succeeds_the_same_way_on_both_trainers(tmp_path) -> None:
    results = _run_both(tmp_path, dataset=tiny_dataset(tmp_path / "d.jsonl", "text"))
    native, trl, candidate = results["native"], results["trl"], results["candidate"]

    # The same scientific proposition...
    assert native["spec"].candidate_fingerprint == candidate.candidate_fingerprint()
    assert trl["spec"].candidate_fingerprint == candidate.candidate_fingerprint()

    # ...observed identically by the control plane.
    assert native["status"].state == trl["status"].state == "succeeded"
    assert (
        _lifecycle(native["events"])
        == _lifecycle(trl["events"])
        == [
            "WorkerReady",
            "TrainingStarted",
            "TrainingMetricObserved",
            "TrainingMetricObserved",
            "TrainingCompleted",
            "ArtifactProduced",
        ]
    )

    for name in ("native", "trl"):
        events = results[name]["events"]
        metrics = [
            e.payload.data for e in events if e.payload.data.type == "TrainingMetricObserved"
        ]
        assert [m.optimizer_step for m in metrics] == [1, 2], name
        assert all(m.loss is not None and math.isfinite(m.loss) for m in metrics), name

        started = next(e.payload.data for e in events if e.payload.data.type == "TrainingStarted")
        completed = next(
            e.payload.data for e in events if e.payload.data.type == "TrainingCompleted"
        )
        assert (started.optimizer_step, completed.optimizer_step) == (0, 2), name

        (artifact,) = [
            e.payload.data.artifact_ref for e in events if e.payload.data.type == "ArtifactProduced"
        ]
        declared = next(o for o in results[name]["spec"].outputs if o.kind == "model")
        assert (artifact.kind, artifact.uri) == ("model", declared.uri), name

        from transformers import AutoModelForCausalLM, AutoTokenizer

        AutoModelForCausalLM.from_pretrained(artifact.uri, local_files_only=True)
        AutoTokenizer.from_pretrained(artifact.uri, local_files_only=True)

        positions = [(e.stream_generation, e.sequence) for e in events]
        assert [s for _, s in positions] == list(range(len(positions))), name

    # Different execution: the difference lives in execution identity, and
    # nowhere a controller would mistake it for a different candidate.
    assert native["spec"].compiler.name == "native"
    assert trl["spec"].compiler.name == "trl"
    assert native["spec"].entrypoint != trl["spec"].entrypoint
    assert native["plan"].request_digest("submit") != trl["plan"].request_digest("submit")


def test_a_training_failure_is_attributed_the_same_way_on_both(tmp_path) -> None:
    results = _run_both(tmp_path, dataset=tmp_path / "never-written.jsonl")

    assert results["native"]["status"].state == results["trl"]["status"].state == "failed"
    assert (
        _lifecycle(results["native"]["events"])
        == _lifecycle(results["trl"]["events"])
        == [
            "WorkerReady",
            "TrainingFailed",
            "IncidentObserved(nonzero-exit)",
        ]
    )


def test_a_publication_failure_is_attributed_the_same_way_on_both(tmp_path) -> None:
    occupied = tmp_path / "not-a-directory"
    occupied.write_text("occupied")

    results = _run_both(
        tmp_path, dataset=tiny_dataset(tmp_path / "d.jsonl", "text"), output=occupied
    )

    assert results["native"]["status"].state == results["trl"]["status"].state == "failed"
    assert (
        _lifecycle(results["native"]["events"])
        == _lifecycle(results["trl"]["events"])
        == [
            "WorkerReady",
            "TrainingStarted",
            "TrainingMetricObserved",
            "TrainingMetricObserved",
            "TrainingCompleted",
            "IncidentObserved(artifact-publication-failed)",
            "IncidentObserved(nonzero-exit)",
        ]
    )
    assert occupied.read_text() == "occupied"
