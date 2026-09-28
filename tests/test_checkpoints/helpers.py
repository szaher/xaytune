from __future__ import annotations

import hashlib
from pathlib import Path

from xaytune.checkpoints import CheckpointState
from xaytune.core.checkpoint import (
    CheckpointCompatibilityKey,
    CheckpointContext,
    RestoreContext,
)
from xaytune.core.clock import utc_now
from xaytune.core.ids import ArtifactId, CheckpointId, RunAttemptId
from xaytune.core.refs import ArtifactRef
from xaytune.core.resume import (
    CheckpointStateManifest,
    DataCursor,
    ResumeGuarantee,
    RNGState,
    SamplerState,
)


def make_bundle(
    directory: Path,
    *,
    attempt_id: RunAttemptId | None = None,
    candidate: str = "candidate-a",
    execution: str = "execution-a",
) -> tuple[CheckpointState, CheckpointContext, RestoreContext]:
    directory.mkdir()
    attempt_id = attempt_id or RunAttemptId.generate()

    def component(name: str) -> ArtifactRef:
        contents = ('{"captured": "' + name + '"}').encode()
        (directory / (name + ".json")).write_bytes(contents)
        return ArtifactRef(
            id=ArtifactId.generate(),
            kind="checkpoint_state",
            uri=name + ".json",
            digest="sha256:" + hashlib.sha256(contents).hexdigest(),
            producer_attempt_id=attempt_id,
        )

    state = CheckpointStateManifest(
        model=component("model"),
        optimizer=component("optimizer"),
        scheduler=component("scheduler"),
        scaler=component("scaler"),
        rng=RNGState(
            python=component("python-rng"),
            numpy=component("numpy-rng"),
            torch_cpu=component("torch-rng"),
        ),
        micro_step=0,
        applied_intervention_application_ids=(),
    )
    cursor = DataCursor(
        dataset_fingerprint="dataset-a",
        ordering_fingerprint="ordering-a",
        epoch=2,
        next_sample_offset=400,
        examples_seen=400,
        tokens_seen=2048,
        sampler_state=SamplerState(provider="indexed", version="1", state_ref=component("sampler")),
    )
    compatibility = CheckpointCompatibilityKey(
        state_format="test-json/v1",
        model_fingerprint="model-a",
        optimizer_layout="adam/v1",
        scheduler_layout="linear/v1",
        distributed_strategy="single-process",
        sharding_scheme="none",
        topology_fingerprint="single-cpu-worker",
        framework_versions={"test-framework": "1"},
    )
    context = CheckpointContext(
        checkpoint_id=CheckpointId.generate(),
        producer_attempt_id=attempt_id,
        candidate_fingerprint=candidate,
        execution_fingerprint=execution,
        compatibility=compatibility,
        created_at=utc_now(),
    )
    guarantee = ResumeGuarantee(state="full", data="exact", boundary="optimizer-step")
    return (
        CheckpointState(directory, 100, cursor, guarantee, state),
        context,
        RestoreContext(
            candidate_fingerprint=candidate,
            compatibility=compatibility,
            dataset_fingerprint=cursor.dataset_fingerprint,
            ordering_fingerprint=cursor.ordering_fingerprint,
            required_guarantee=guarantee,
        ),
    )
