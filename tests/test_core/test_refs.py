"""Value objects: immutability, closed vocabularies, serialization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from xaytune.core.ids import ArtifactId, EvaluationId, RunAttemptId
from xaytune.core.refs import (
    Actor,
    ArtifactRef,
    CheckpointRef,
    ControllerHostRef,
    DatasetRef,
    ModelRef,
    ResourceUsage,
    RuntimeRef,
)


class TestActor:
    @pytest.mark.parametrize(
        "actor_type", ["human", "rule", "llm_agent", "search_provider", "system"]
    )
    def test_accepts_every_declared_actor_type(self, actor_type):
        assert Actor(type=actor_type, id="x").type == actor_type

    def test_rejects_an_unknown_actor_type(self):
        with pytest.raises(ValidationError):
            Actor(type="daemon", id="x")

    def test_is_frozen(self):
        actor = Actor(type="human", id="alex")
        with pytest.raises(ValidationError):
            actor.id = "someone-else"


class TestDatasetRef:
    def test_fingerprints_are_part_of_identity(self):
        """Same URI, different template, different scientific input (ADR-006)."""
        base = DatasetRef(uri="support-v4", revision="2026-01-01")
        retemplated = base.model_copy(update={"template_fingerprint": "sha256:tpl"})
        assert base != retemplated

    def test_round_trips(self):
        ref = DatasetRef(
            uri="s3://bucket/data",
            revision="v4",
            split="train",
            content_digest="sha256:aaa",
            tokenizer_fingerprint="sha256:tok",
        )
        assert DatasetRef.model_validate_json(ref.model_dump_json()) == ref

    def test_compares_by_value(self):
        """Identity is the field contents, not the object.

        Note these are not hashable: the ``metadata`` dict field makes frozen
        Pydantic models unhashable, so they cannot go in a set as-is.
        """
        assert DatasetRef(uri="s3://bucket/data") == DatasetRef(uri="s3://bucket/data")
        assert DatasetRef(uri="a") != DatasetRef(uri="b")


class TestArtifactRef:
    @pytest.mark.parametrize(
        "kind",
        [
            "model",
            "adapter",
            "checkpoint",
            "tokenizer",
            "metrics",
            "evaluation_report",
            "dataset_snapshot",
            "logs",
            "execution_manifest",
            "provenance_bundle",
        ],
    )
    def test_accepts_every_declared_kind(self, kind):
        ref = ArtifactRef(id=ArtifactId.generate(), kind=kind, uri="s3://x")
        assert ref.kind == kind

    def test_rejects_an_unknown_kind(self):
        with pytest.raises(ValidationError):
            ArtifactRef(id=ArtifactId.generate(), kind="spreadsheet", uri="s3://x")

    def test_records_its_producer(self):
        attempt_id = RunAttemptId.generate()
        ref = ArtifactRef(
            id=ArtifactId.generate(),
            kind="adapter",
            uri="s3://x",
            producer_attempt_id=attempt_id,
        )
        assert ref.producer_attempt_id == attempt_id
        assert ref.producer_evaluation_id is None

    def test_rejects_a_wrong_producer_id_type(self):
        with pytest.raises(ValidationError):
            ArtifactRef(
                id=ArtifactId.generate(),
                kind="adapter",
                uri="s3://x",
                producer_attempt_id=EvaluationId.generate(),
            )


class TestRuntimeAndHostRefs:
    def test_runtime_ref_is_backend_neutral(self):
        """The core must not assume Kubernetes identifiers."""
        ref = RuntimeRef(backend="local", external_id="pid-1234")
        assert ref.namespace is None
        assert RuntimeRef.model_validate_json(ref.model_dump_json()) == ref

    @pytest.mark.parametrize("kind", ["embedded", "local_daemon", "remote"])
    def test_controller_host_kinds(self, kind):
        assert ControllerHostRef(kind=kind).kind == kind

    def test_rejects_an_unknown_controller_host_kind(self):
        with pytest.raises(ValidationError):
            ControllerHostRef(kind="serverless")


class TestMiscRefs:
    def test_model_ref_round_trips(self):
        ref = ModelRef(uri="Qwen/Qwen3-8B", revision="main", digest="sha256:m")
        assert ModelRef.model_validate_json(ref.model_dump_json()) == ref

    def test_resource_usage_defaults_to_all_unknown(self):
        usage = ResourceUsage()
        assert usage.gpu_hours is None
        assert usage.cost is None

    def test_checkpoint_ref_round_trips(self):
        from xaytune.core.ids import CheckpointId

        ref = CheckpointRef(id=CheckpointId.generate(), uri="s3://ckpt", global_step=500)
        assert CheckpointRef.model_validate_json(ref.model_dump_json()) == ref

    def test_extra_fields_are_rejected(self):
        with pytest.raises(ValidationError):
            ModelRef(uri="x", unexpected="y")
