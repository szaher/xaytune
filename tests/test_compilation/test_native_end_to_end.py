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

from tests.training_fixtures import (
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
from xaytune.core.execution import TrainingExecutionSpec


def test_native_worker_restores_managed_full_exact_checkpoint_after_resize(tmp_path) -> None:
    """Real LocalRuntime workers resume at the next sample with a smaller microbatch."""
    from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec
    from xaytune.compilation.attempt_resolution import resolve_training_attempt
    from xaytune.compilation.native import NativeCompiler
    from xaytune.core.domain.candidate import CheckpointIntent, LRScheduleSpec
    from xaytune.core.domain.run import ExecutionOverride, RunAttempt
    from xaytune.core.ids import ActionId, RunAttemptId, RunId
    from xaytune.core.immutable import FrozenDict

    model_dir = tiny_model(tmp_path / "tiny-model")
    dataset = tiny_dataset(tmp_path / "data" / "train.jsonl")
    candidate = sft_candidate(model_dir, dataset)
    training = candidate.training.model_copy(
        update={
            "optimization": candidate.training.optimization.model_copy(
                update={"lr_schedule": LRScheduleSpec(name="linear")}
            )
        }
    )
    candidate = candidate.model_copy(
        update={
            "training": training.model_copy(
                update={"checkpoint": CheckpointIntent(every_optimizer_steps=1)}
            )
        }
    )
    spec = NativeCompiler().compile(candidate, compilation_context(tmp_path))
    first_id = RunAttemptId.generate()
    first = offline_plan(spec, str(first_id))
    status, events = run_to_completion(tmp_path / "runtime", first)
    assert status.state == "succeeded", status.detail
    commits = [
        event.payload.data for event in events if event.payload.data.type == "CheckpointCommitted"
    ]
    assert [commit.optimizer_step for commit in commits] == [1, 2]
    assert commits[0].resume_guarantee.data.value == "exact"
    assert commits[0].data_cursor.next_sample_offset == 2

    manager = CheckpointManager(
        SerializedStateCodec(), LocalCheckpointStore(tmp_path / "checkpoints")
    )
    reference = commits[0].checkpoint_ref
    localized = asyncio.run(manager.store.get(reference))
    assert localized.manifest.optimizer_step == 1
    action = ActionId.generate()
    successor = RunAttempt(
        id=RunAttemptId.generate(),
        run_id=RunId.generate(),
        attempt_number=2,
        checkpoint_ref=reference,
        execution_overrides=(
            ExecutionOverride(
                id="micro-resize",
                kind="micro_batch_resize",
                reason="governed OOM resize",
                values=FrozenDict({"from": 2, "to": 1}),
                preserves=("effective_batch_size",),
                action_id=action,
            ),
            ExecutionOverride(
                id="accum-adjust",
                kind="gradient_accumulation_adjustment",
                reason="preserve effective batch",
                values=FrozenDict({"from": 1, "to": 2}),
                preserves=("effective_batch_size",),
                action_id=action,
            ),
            ExecutionOverride(
                id="restore",
                kind="checkpoint_restore",
                reason="FULL+EXACT restore",
                values=FrozenDict({"checkpoint_id": str(reference.id)}),
                action_id=action,
            ),
        ),
    )
    resolved = resolve_training_attempt(first.spec, successor, "local")
    from xaytune.runtimes.worker import ObservationWriter
    from xaytune.trainer.scheduler import create_scheduler
    from xaytune.workers.native_checkpoint import NativeCheckpointAdapter

    adapter = NativeCheckpointAdapter(
        resolved,
        dataset,
        seed=7,
        dataset_size=4,
        micro_batch_size=1,
        gradient_accumulation=2,
        writer=ObservationWriter(tmp_path / "unused-observations.jsonl"),
    )
    apply = adapter.bind_restore(reference)
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = create_scheduler(optimizer, "linear", 2, 0)
    restored_state = apply(model, optimizer, scheduler, None)
    assert restored_state.global_step == 1
    saved_model = torch.load(
        localized.directory / "model.pt", weights_only=True, map_location="cpu"
    )
    assert all(torch.equal(value, saved_model[name]) for name, value in model.state_dict().items())
    assert {int(state["step"]) for state in optimizer.state_dict()["state"].values()} == {1}
    import random

    import numpy as np

    state_manifest = localized.manifest.state_manifest
    assert state_manifest.rng is not None
    assert json.loads(json.dumps(random.getstate())) == json.loads(
        (localized.directory / state_manifest.rng.python.uri).read_text()
    )
    saved_numpy = json.loads((localized.directory / state_manifest.rng.numpy.uri).read_text())
    assert np.random.get_state()[1].tolist() == saved_numpy[1]
    assert torch.equal(
        torch.get_rng_state(),
        torch.load(
            localized.directory / state_manifest.rng.torch_cpu.uri,
            weights_only=True,
            map_location="cpu",
        ),
    )
    resumed = resolved.model_copy(
        update={"spec": resolved.spec.model_copy(update={"environment": first.spec.environment})}
    )
    status, events = run_to_completion(tmp_path / "resumed-runtime", resumed)
    assert status.state == "succeeded", status.detail
    metrics = [
        event.payload.data.optimizer_step
        for event in events
        if event.payload.data.type == "TrainingMetricObserved"
    ]
    assert metrics == [2]
    assert resumed.spec.config["optimization"]["micro_batch_size"] == 1
    assert resumed.spec.config["optimization"]["gradient_accumulation"] == 2


def test_native_managed_checkpoint_refuses_partial_optimizer_window(tmp_path) -> None:
    from xaytune.compilation.native import NativeCompiler
    from xaytune.core.domain.candidate import CheckpointIntent
    from xaytune.core.ids import RunAttemptId

    candidate = sft_candidate(tiny_model(tmp_path / "model"), tiny_dataset(tmp_path / "data.jsonl"))
    training = candidate.training.model_copy(
        update={
            "optimization": candidate.training.optimization.model_copy(
                update={"micro_batch_size": 3, "gradient_accumulation": 1}
            ),
            "checkpoint": CheckpointIntent(every_optimizer_steps=1),
        }
    )
    candidate = candidate.model_copy(update={"training": training})
    spec = NativeCompiler().compile(candidate, compilation_context(tmp_path))
    status, events = run_to_completion(
        tmp_path / "runtime", offline_plan(spec, str(RunAttemptId.generate()))
    )
    assert status.state == "failed"
    assert "CheckpointCommitted" not in emitted_types(events)


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
