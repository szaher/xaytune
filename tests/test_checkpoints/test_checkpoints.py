from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest
from pydantic import ValidationError

from xaytune.checkpoints import (
    CheckpointCompatibilityError,
    CheckpointCorruptionError,
    CheckpointManager,
    LocalCheckpointStore,
    SerializedStateCodec,
)
from xaytune.checkpoints import store as store_module
from xaytune.core.checkpoint import CheckpointFile, CheckpointManifest
from xaytune.core.errors import IdempotencyConflictError
from xaytune.core.ids import RunAttemptId
from xaytune.core.resume import ResumeGuarantee

from .helpers import make_bundle


def run(coro):
    return asyncio.run(coro)


def test_roundtrip_and_restart_preserve_bytes_state_and_provenance(tmp_path):
    state, context, restore = make_bundle(tmp_path / "source")
    store = LocalCheckpointStore(tmp_path / "store")
    manager = CheckpointManager(SerializedStateCodec(), store)
    reference = run(manager.save(state, context))

    reopened = LocalCheckpointStore(tmp_path / "store")
    restored = run(CheckpointManager(SerializedStateCodec(), reopened).restore(reference, restore))

    assert run(reopened.list()) == (reference,)
    assert restored.manifest.context == context
    assert restored.manifest.data_cursor == state.data_cursor
    assert restored.manifest.state_manifest == state.state_manifest
    assert restored.manifest.resume_guarantee == state.resume_guarantee
    assert CheckpointManifest.model_validate_json(restored.manifest.model_dump_json()) == (
        restored.manifest
    )
    for file in restored.manifest.files:
        assert (restored.source / file.path).read_bytes() == (state.source / file.path).read_bytes()


def test_identical_save_returns_the_original_checkpoint_and_changed_state_conflicts(tmp_path):
    state, context, _ = make_bundle(tmp_path / "source")
    manager = CheckpointManager(SerializedStateCodec(), LocalCheckpointStore(tmp_path / "store"))
    reference = run(manager.save(state, context))
    assert run(manager.save(state, context)) == reference
    assert tuple(manager.store._staging.iterdir()) == ()
    with pytest.raises(IdempotencyConflictError):
        run(manager.save(replace(state, optimizer_step=101), context))
    assert run(manager.store.list()) == (reference,)


def test_staging_is_not_listed_and_cannot_be_restored(tmp_path):
    state, context, _ = make_bundle(tmp_path / "source")
    codec = SerializedStateCodec()
    manifest = run(codec.encode(state, tmp_path / "encoded", context))
    store = LocalCheckpointStore(tmp_path / "store")
    staging = run(store.put_staging(tmp_path / "encoded", manifest))
    assert run(store.list()) == ()
    with pytest.raises(CheckpointCorruptionError):
        run(store.get(manifest.reference(staging.directory.as_uri())))
    reference = run(store.commit(staging))
    assert run(store.commit(staging)) == reference  # rename consumed the staging path


@pytest.mark.parametrize("boundary", ["file-sync", "directory-sync", "rename", "acknowledgement"])
def test_publication_failures_and_retry_are_restart_safe(tmp_path, monkeypatch, boundary):
    state, context, _ = make_bundle(tmp_path / "source")
    store = LocalCheckpointStore(tmp_path / "store")
    manager = CheckpointManager(SerializedStateCodec(), store)
    with monkeypatch.context() as patch:

        def fail(*args):
            raise OSError("injected crash")

        if boundary == "file-sync":
            patch.setattr(os, "fsync", fail)
        elif boundary == "rename":
            patch.setattr(os, "rename", fail)
        else:
            original = store_module.sync_directory

            def fail_directory(directory):
                if (boundary == "acknowledgement" and directory == store._committed) or (
                    boundary == "directory-sync" and directory.name.startswith("bundle-")
                ):
                    fail()
                original(directory)

            patch.setattr(store_module, "sync_directory", fail_directory)
        with pytest.raises(OSError, match="injected crash"):
            run(manager.save(state, context))
    reopened = LocalCheckpointStore(tmp_path / "store")
    assert len(run(reopened.list())) == (1 if boundary == "acknowledgement" else 0)
    reference = run(CheckpointManager(SerializedStateCodec(), reopened).save(state, context))
    assert run(reopened.list()) == (reference,)


@pytest.mark.parametrize("damage", ["missing", "bytes", "extra", "manifest", "symlink"])
def test_corrupt_published_checkpoints_are_refused(tmp_path, damage):
    state, context, restore = make_bundle(tmp_path / "source")
    store = LocalCheckpointStore(tmp_path / "store")
    manager = CheckpointManager(SerializedStateCodec(), store)
    ref = run(manager.save(state, context))
    directory = store._committed / str(ref.id)
    path = directory / "model.json"
    if damage == "missing":
        path.unlink()
    elif damage == "bytes":
        path.write_bytes(b"modified")
    elif damage == "extra":
        (directory / "unlisted.bin").write_bytes(b"extra")
    elif damage == "symlink":
        path.unlink()
        path.symlink_to(state.source / "model.json")
    else:
        manifest_path = directory / "manifest.json"
        data = json.loads(manifest_path.read_text())
        data["optimizer_step"] = 101
        manifest_path.write_text(json.dumps(data))
    with pytest.raises(CheckpointCorruptionError):
        run(manager.restore(ref, restore))
    with pytest.raises(CheckpointCorruptionError):
        run(store.list())


@pytest.mark.parametrize("operation", ["get", "list"])
def test_committed_manifest_dot_path_is_corruption(tmp_path, operation):
    state, context, _ = make_bundle(tmp_path / "source")
    store = LocalCheckpointStore(tmp_path / "store")
    reference = run(CheckpointManager(SerializedStateCodec(), store).save(state, context))
    manifest_path = store._committed / str(reference.id) / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["files"][0]["path"] = "."
    manifest_path.write_text(json.dumps(data))

    with pytest.raises(CheckpointCorruptionError, match="manifest is missing or invalid"):
        run(store.get(reference) if operation == "get" else store.list())


@pytest.mark.parametrize(
    "field,value",
    [
        ("uri", "../escape"),
        ("uri", "/outside"),
        ("uri", "file:///outside"),
        ("uri", "absent.json"),
        ("digest", None),
        ("digest", "sha256:" + "0" * 64),
        ("producer_attempt_id", RunAttemptId.generate()),
        ("producer_attempt_id", None),
    ],
)
def test_codec_requires_captured_components_inside_bundle_with_provenance(tmp_path, field, value):
    state, context, _ = make_bundle(tmp_path / "source")
    changed = state.state_manifest.model.model_copy(update={field: value})
    state = replace(
        state, state_manifest=state.state_manifest.model_copy(update={"model": changed})
    )
    with pytest.raises(CheckpointCorruptionError):
        run(SerializedStateCodec().encode(state, tmp_path / "encoded", context))


@pytest.mark.parametrize(
    "field", ["optimizer", "scheduler", "scaler", "rng", "applied_intervention_application_ids"]
)
def test_full_exact_claim_requires_all_declared_state(tmp_path, field):
    state, context, _ = make_bundle(tmp_path / "source")
    state = replace(state, state_manifest=state.state_manifest.model_copy(update={field: None}))
    with pytest.raises(ValidationError, match="FULL"):
        run(SerializedStateCodec().encode(state, tmp_path / "encoded", context))


def test_mid_accumulation_cannot_claim_exact_and_model_only_is_explicit(tmp_path):
    state, context, restore = make_bundle(tmp_path / "source")
    codec = SerializedStateCodec()
    mid = replace(state, state_manifest=state.state_manifest.model_copy(update={"micro_step": 2}))
    with pytest.raises(ValidationError, match="boundary"):
        run(codec.encode(mid, tmp_path / "invalid", context))
    state = replace(
        state,
        data_cursor=None,
        resume_guarantee=ResumeGuarantee(
            state="model-only", data="none", boundary="optimizer-step"
        ),
        state_manifest=state.state_manifest.model_copy(
            update={
                "optimizer": None,
                "scheduler": None,
                "scaler": None,
                "rng": None,
                "applied_intervention_application_ids": None,
            }
        ),
    )
    manager = CheckpointManager(codec, LocalCheckpointStore(tmp_path / "store"))
    ref = run(manager.save(state, context))
    with pytest.raises(CheckpointCompatibilityError, match="guarantee"):
        run(manager.restore(ref, restore))
    weak = restore.model_copy(update={"required_guarantee": state.resume_guarantee})
    assert run(manager.restore(ref, weak)).manifest.resume_guarantee == state.resume_guarantee


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_fingerprint", "other"),
        ("dataset_fingerprint", "other"),
        ("ordering_fingerprint", "other"),
        ("dataset_fingerprint", None),
        ("ordering_fingerprint", None),
    ],
)
def test_restore_refuses_incompatible_science_and_data_before_decode(
    tmp_path, monkeypatch, field, value
):
    state, context, restore = make_bundle(tmp_path / "source")
    codec = SerializedStateCodec()
    manager = CheckpointManager(codec, LocalCheckpointStore(tmp_path / "store"))
    ref = run(manager.save(state, context))

    def decode(*args):
        pytest.fail("decode must not be called for incompatible state")

    monkeypatch.setattr(codec, "decode", decode)
    with pytest.raises(CheckpointCompatibilityError):
        run(manager.restore(ref, restore.model_copy(update={field: value})))


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_fingerprint", "other"),
        ("optimizer_layout", "other"),
        ("scheduler_layout", "other"),
        ("framework_versions", {"test-framework": "2"}),
        ("topology_fingerprint", "two-workers"),
        ("adapter_fingerprint", "lora"),
        ("tokenizer_fingerprint", "other"),
        ("sharding_scheme", "other"),
        ("distributed_strategy", "other"),
        ("state_format", "other"),
    ],
)
def test_compatibility_is_explicit_exact_match(tmp_path, field, value):
    state, context, restore = make_bundle(tmp_path / "source")
    manager = CheckpointManager(SerializedStateCodec(), LocalCheckpointStore(tmp_path / "store"))
    ref = run(manager.save(state, context))
    changed = restore.compatibility.model_copy(update={field: value})
    with pytest.raises(CheckpointCompatibilityError):
        run(manager.restore(ref, restore.model_copy(update={"compatibility": changed})))


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "./",
        "./.",
        "../model",
        "/model",
        "./model",
        "a//b",
        "a/../b",
        "a\\b",
        "file:model",
        "manifest.json",
        "manifest.json/x",
    ],
)
def test_file_paths_are_canonical_and_contained(path):
    with pytest.raises(ValidationError):
        CheckpointFile(path=path, size_bytes=1, digest="sha256:" + "1" * 64)


def test_committed_reference_cannot_be_forged_or_redirected(tmp_path):
    state, context, _ = make_bundle(tmp_path / "source")
    store = LocalCheckpointStore(tmp_path / "store")
    ref = run(CheckpointManager(SerializedStateCodec(), store).save(state, context))
    for field, value in (
        ("uri", "file:///outside"),
        ("digest", None),
        ("compatibility_key", "other"),
        ("global_step", 999),
    ):
        with pytest.raises(CheckpointCorruptionError):
            run(store.get(ref.model_copy(update={field: value})))


def test_unknown_manifest_version_is_refused(tmp_path):
    state, context, _ = make_bundle(tmp_path / "source")
    store = LocalCheckpointStore(tmp_path / "store")
    ref = run(CheckpointManager(SerializedStateCodec(), store).save(state, context))
    path = store._committed / str(ref.id) / "manifest.json"
    data = json.loads(path.read_text())
    data["schema_version"] = "xaytune.checkpoint/v999"
    path.write_text(json.dumps(data))
    with pytest.raises(CheckpointCorruptionError):
        run(store.get(ref))


@pytest.mark.parametrize("conflict", [False, True])
def test_two_processes_publish_one_bundle_or_refuse_conflicting_content(tmp_path, conflict):
    state, context, _ = make_bundle(tmp_path / "source")
    store = LocalCheckpointStore(tmp_path / "store")
    codec = SerializedStateCodec()
    first = run(codec.encode(state, tmp_path / "encoded-a", context))
    second = run(
        codec.encode(
            replace(state, optimizer_step=101) if conflict else state,
            tmp_path / "encoded-b",
            context,
        )
    )
    staging = [
        run(store.put_staging(tmp_path / name, manifest))
        for name, manifest in (("encoded-a", first), ("encoded-b", second))
    ]
    code = """
import asyncio, sys
from pathlib import Path
from xaytune.checkpoints import LocalCheckpointStore, StagingRef
from xaytune.core.ids import CheckpointId
from xaytune.core.errors import IdempotencyConflictError
try:
    ref = asyncio.run(LocalCheckpointStore(Path(sys.argv[1])).commit(StagingRef(
        CheckpointId.validate(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])))
    print(ref.digest)
except IdempotencyConflictError:
    print('CONFLICT')
"""
    children = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                code,
                str(store.root),
                str(item.checkpoint_id),
                str(item.directory),
                item.manifest_digest,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for item in staging
    ]
    outputs = []
    for child in children:
        output, error = child.communicate(timeout=20)
        assert child.returncode == 0, error
        outputs.append(output.strip())
    assert outputs.count("CONFLICT") == int(conflict)
    assert len(run(store.list())) == 1
    if not conflict:
        assert outputs[0] == outputs[1] == first.manifest_digest


@pytest.mark.parametrize(
    "component",
    ["optimizer", "scheduler", "scaler", "python-rng", "numpy-rng", "torch-rng", "sampler"],
)
def test_every_required_component_is_verified_against_captured_bytes(tmp_path, component):
    state, context, _ = make_bundle(tmp_path / "source")
    (state.source / (component + ".json")).unlink()
    with pytest.raises(CheckpointCorruptionError, match="missing"):
        run(SerializedStateCodec().encode(state, tmp_path / "encoded", context))


def test_unknown_state_schema_and_codec_version_are_refused(tmp_path):
    state, context, restore = make_bundle(tmp_path / "source")
    state = replace(
        state, state_manifest=state.state_manifest.model_copy(update={"schema_version": "unknown"})
    )
    with pytest.raises(CheckpointCompatibilityError, match="schema"):
        run(SerializedStateCodec().encode(state, tmp_path / "unknown", context))
    state = replace(
        state,
        state_manifest=state.state_manifest.model_copy(
            update={"schema_version": "xaytune.checkpoint-state/v1alpha1"}
        ),
    )
    codec = SerializedStateCodec()
    store = LocalCheckpointStore(tmp_path / "store")
    ref = run(CheckpointManager(codec, store).save(state, context))
    changed_codec = SerializedStateCodec()
    changed_codec.descriptor = codec.descriptor.model_copy(update={"plugin_version": "2"})
    with pytest.raises(CheckpointCompatibilityError, match="layout"):
        run(CheckpointManager(changed_codec, store).restore(ref, restore))


def test_codec_cannot_change_the_requested_producer_or_candidate(tmp_path):
    state, context, _ = make_bundle(tmp_path / "source")

    class ForgedCodec(SerializedStateCodec):
        async def encode(self, state, destination, context):
            return await super().encode(
                state,
                destination,
                context.model_copy(update={"candidate_fingerprint": "another-candidate"}),
            )

    store = LocalCheckpointStore(tmp_path / "store")
    with pytest.raises(CheckpointCorruptionError, match="provenance"):
        run(CheckpointManager(ForgedCodec(), store).save(state, context))
    assert run(store.list()) == ()


def test_unknown_plugin_api_is_refused_at_manager_boundary(tmp_path):
    from xaytune.core.errors import IncompatiblePluginError

    codec = SerializedStateCodec()
    codec.descriptor = codec.descriptor.model_copy(update={"api_version": "unknown/v99"})
    with pytest.raises(IncompatiblePluginError):
        CheckpointManager(codec, LocalCheckpointStore(tmp_path / "store"))


@pytest.mark.parametrize("field", ["optimizer_layout", "scheduler_layout"])
def test_captured_state_cannot_use_unknown_layout_compatibility(tmp_path, field):
    state, context, _ = make_bundle(tmp_path / "source")
    context = context.model_copy(
        update={"compatibility": context.compatibility.model_copy(update={field: None})}
    )
    with pytest.raises(CheckpointCompatibilityError, match="unknown"):
        run(SerializedStateCodec().encode(state, tmp_path / "encoded", context))


def test_worker_rng_references_are_verified_and_preserved(tmp_path):
    from xaytune.core.ids import ArtifactId
    from xaytune.core.resume import WorkerRNGState

    state, context, restore = make_bundle(tmp_path / "source")
    accelerator = state.state_manifest.rng.torch_cpu.model_copy(
        update={"id": ArtifactId.generate()}
    )
    loader = state.data_cursor.sampler_state.state_ref.model_copy(
        update={"id": ArtifactId.generate()}
    )
    workers = (
        WorkerRNGState(logical_worker_id="worker-0", accelerator=accelerator, dataloader=(loader,)),
    )
    state = replace(
        state,
        state_manifest=state.state_manifest.model_copy(
            update={"rng": state.state_manifest.rng.model_copy(update={"workers": workers})}
        ),
    )
    manager = CheckpointManager(SerializedStateCodec(), LocalCheckpointStore(tmp_path / "store"))
    ref = run(manager.save(state, context))
    assert run(manager.restore(ref, restore)).manifest.state_manifest.rng.workers == workers
    bad = workers[0].model_copy(
        update={"accelerator": accelerator.model_copy(update={"digest": "sha256:" + "0" * 64})}
    )
    state = replace(
        state,
        state_manifest=state.state_manifest.model_copy(
            update={"rng": state.state_manifest.rng.model_copy(update={"workers": (bad,)})}
        ),
    )
    with pytest.raises(CheckpointCorruptionError):
        run(SerializedStateCodec().encode(state, tmp_path / "invalid", context))


def test_symlink_source_and_duplicate_manifest_keys_are_refused(tmp_path):
    state, context, _ = make_bundle(tmp_path / "source")
    link = tmp_path / "link"
    link.symlink_to(state.source, target_is_directory=True)
    with pytest.raises(CheckpointCorruptionError):
        run(
            SerializedStateCodec().encode(
                replace(state, source=link), tmp_path / "invalid", context
            )
        )
    store = LocalCheckpointStore(tmp_path / "store")
    ref = run(CheckpointManager(SerializedStateCodec(), store).save(state, context))
    path = store._committed / str(ref.id) / "manifest.json"
    text = path.read_text()
    path.write_text('{"codec": "forged", ' + text[1:])
    with pytest.raises(CheckpointCorruptionError):
        run(store.get(ref))
