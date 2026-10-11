"""A small pinned support suite: three tasks, a policy attachment, held-out answers."""

from __future__ import annotations

import hashlib
from typing import Any

from xaytune.benchmarks import MappingBenchmarkSource
from xaytune.core.domain.benchmark import BenchmarkSuiteManifest, TaskSpec

POLICY = "Refunds are issued within five days of approval."
POLICY_URI = "https://bench.example.com/support/policy-v3.txt"
SNAPSHOT = b"\x00kb-index-v3\x00"
SNAPSHOT_URI = "s3://bench-snapshots/support/kb-v3.bin"


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def answer(task_id: str) -> str:
    return f"Held-out answer for {task_id}."


def answer_uri(task_id: str) -> str:
    return f"https://bench.example.com/answers/{task_id.replace('/', '-')}.txt"


def ref(uri: str, data: bytes, media_type: str = "text/plain") -> dict[str, Any]:
    return {"kind": "ref", "uri": uri, "digest": digest(data), "media_type": media_type}


def task_record(task_id: str, **changes: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "xaytune.task/v1alpha1",
        "id": task_id,
        "revision": "2026.10.1",
        "inputs": {
            "variables": {"customer": "Ann Lee", "ticket": f"Question in {task_id}"},
            "attachments": [{"name": "policy", "content": ref(POLICY_URI, POLICY.encode())}],
        },
        "output": {"kind": "text-answer", "media_type": "text/plain"},
        "reference": ref(answer_uri(task_id), answer(task_id).encode()),
    }
    record.update(changes)
    return record


def task(task_id: str, **changes: Any) -> TaskSpec:
    return TaskSpec.model_validate(task_record(task_id, **changes))


TASK_IDS = ("support/refund-001", "support/refund-002", "support/shipping-001")


def manifest_record(tasks: tuple[TaskSpec, ...] | None = None, **changes: Any) -> dict[str, Any]:
    members = tasks if tasks is not None else tuple(task(i) for i in TASK_IDS)
    record: dict[str, Any] = {
        "schema_version": "xaytune.benchmark-suite/v1alpha1",
        "id": "support/core",
        "revision": "v3",
        "provider": {"name": "acme-bench", "version": "1.4.0"},
        "members": [
            {"task_id": t.id, "revision": t.revision, "task_fingerprint": t.task_fingerprint()}
            for t in members
        ],
        "reproducibility": "static",
        "environment_snapshot": None,
        "reset": "per-task",
    }
    record.update(changes)
    return record


def manifest(**changes: Any) -> BenchmarkSuiteManifest:
    return BenchmarkSuiteManifest.model_validate(manifest_record(**changes))


def blobs() -> dict[str, bytes]:
    held = {POLICY_URI: POLICY.encode(), SNAPSHOT_URI: SNAPSHOT}
    held.update({answer_uri(i): answer(i).encode() for i in TASK_IDS})
    return held


def source(
    manifests: tuple[BenchmarkSuiteManifest, ...] | None = None,
    tasks: tuple[TaskSpec, ...] | None = None,
    held: dict[str, bytes] | None = None,
    aliases: dict[tuple[str, str], str] | None = None,
) -> MappingBenchmarkSource:
    return MappingBenchmarkSource(
        manifests=manifests if manifests is not None else (manifest(),),
        tasks=tasks if tasks is not None else tuple(task(i) for i in TASK_IDS),
        blobs=held if held is not None else blobs(),
        aliases=aliases if aliases is not None else {("support/core", "stable"): "v3"},
    )
