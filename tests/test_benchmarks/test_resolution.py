"""Resolving a suite: aliases pinned, pins immovable, every task and byte verified (ADR-020 B4)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_benchmarks.fixtures import (
    POLICY,
    POLICY_URI,
    SNAPSHOT,
    SNAPSHOT_URI,
    TASK_IDS,
    answer,
    answer_uri,
    blobs,
    manifest,
    ref,
    source,
    task,
    task_record,
)
from xaytune.benchmarks import (
    BenchmarkSource,
    UnverifiedBenchmarkError,
    resolve_benchmark_suite,
    resolve_reference,
    resolve_task_inputs,
    verify_suite_content,
)
from xaytune.core.domain.benchmark import BenchmarkSuiteRequest, ResolvedBenchmarkSuite, TaskSpec

STABLE = BenchmarkSuiteRequest(suite_id="support/core", revision="stable")
PINNED = BenchmarkSuiteRequest(suite_id="support/core", revision="v3")


def test_an_alias_resolves_to_a_pin_and_both_are_kept() -> None:
    resolved = resolve_benchmark_suite(STABLE, source())
    assert resolved.request == STABLE  # what was asked for
    assert resolved.ref.revision == "v3"  # what it was
    assert resolved.ref.suite_fingerprint == manifest().suite_fingerprint()
    assert [t.id for t in resolved.tasks] == list(TASK_IDS)
    assert isinstance(source(), BenchmarkSource)


def test_a_pinned_ref_resolves_to_exactly_itself_after_the_alias_moves() -> None:
    first = resolve_benchmark_suite(STABLE, source())
    newer = manifest(revision="v4", tasks=(task(TASK_IDS[0]),))
    moved = source(manifests=(manifest(), newer), aliases={("support/core", "stable"): "v4"})
    assert resolve_benchmark_suite(STABLE, moved).ref.revision == "v4"  # the alias moved
    again = resolve_benchmark_suite(first.ref, moved)  # the pin did not
    assert again.ref == first.ref and again.tasks == first.tasks


def test_a_pinned_suite_whose_manifest_changed_is_refused() -> None:
    pinned = resolve_benchmark_suite(STABLE, source()).ref
    rewritten = manifest(tasks=tuple(task(i) for i in reversed(TASK_IDS)))  # same id and revision
    with pytest.raises(UnverifiedBenchmarkError, match="the pinned suite has changed"):
        resolve_benchmark_suite(pinned, source(manifests=(rewritten,)))


def test_a_suite_the_source_does_not_hold_is_refused() -> None:
    with pytest.raises(UnverifiedBenchmarkError, match="does not hold the requested suite"):
        resolve_benchmark_suite(
            BenchmarkSuiteRequest(suite_id="support/core", revision="v9"), source()
        )


def _tasks_with(position: int, replacement: TaskSpec | None) -> tuple[TaskSpec, ...]:
    tasks = [task(i) for i in TASK_IDS]
    if replacement is None:
        del tasks[position]
    else:
        tasks[position] = replacement
    return tuple(tasks)


@pytest.mark.parametrize(
    ("tasks", "held", "reason"),
    [
        (
            _tasks_with(1, None),
            None,
            "support/refund-002: the source does not hold the pinned revision",
        ),
        (
            _tasks_with(
                1,
                task(
                    TASK_IDS[1],
                    inputs={"variables": {"customer": "x", "ticket": "y"}, "attachments": []},
                ),
            ),
            None,
            "support/refund-002: the task does not have its pinned fingerprint",
        ),
        (
            None,
            {k: v for k, v in blobs().items() if k != POLICY_URI},
            "attachments.policy: the source does not hold it",
        ),
        (
            None,
            {**blobs(), POLICY_URI: b"tampered"},
            "attachments.policy: the bytes do not have the pinned digest",
        ),
        (
            None,
            {**blobs(), answer_uri(TASK_IDS[2]): b"leaked?"},
            "support/shipping-001 reference: the bytes do not have",
        ),
    ],
)
def test_every_member_and_byte_is_verified(
    tasks: tuple[TaskSpec, ...] | None, held: dict[str, bytes] | None, reason: str
) -> None:
    with pytest.raises(UnverifiedBenchmarkError, match=reason) as refused:
        resolve_benchmark_suite(PINNED, source(tasks=tasks, held=held))
    for uri in (POLICY_URI, answer_uri(TASK_IDS[2])):
        assert uri not in str(refused.value)


def test_every_reason_is_reported_at_once() -> None:
    held = {k: v for k, v in blobs().items() if k != POLICY_URI}
    with pytest.raises(UnverifiedBenchmarkError) as refused:
        resolve_benchmark_suite(PINNED, source(tasks=_tasks_with(2, None), held=held))
    # Two tasks lose their policy attachment; the third is missing outright.
    assert len(refused.value.reasons) == 3


@pytest.mark.parametrize(
    ("content", "held", "reason"),
    [
        (ref(POLICY_URI, POLICY.encode(), "application/pdf"), None, "media type is not text/"),
        (ref(POLICY_URI, b"\xff\xfe"), {POLICY_URI: b"\xff\xfe"}, "not valid UTF-8"),
    ],
)
def test_text_content_must_be_text(
    content: dict[str, Any], held: dict[str, bytes] | None, reason: str
) -> None:
    record = task_record(TASK_IDS[0])
    record["inputs"]["attachments"][0]["content"] = content
    odd = TaskSpec.model_validate(record)
    suite = manifest(tasks=(odd,))
    with pytest.raises(UnverifiedBenchmarkError, match=reason):
        resolve_benchmark_suite(
            PINNED, source(manifests=(suite,), tasks=(odd,), held={**blobs(), **(held or {})})
        )


def test_a_snapshot_suite_verifies_its_snapshot() -> None:
    snapshot = ref(SNAPSHOT_URI, SNAPSHOT, "application/octet-stream")
    suite = manifest(reproducibility="snapshot", environment_snapshot=snapshot)
    resolved = resolve_benchmark_suite(PINNED, source(manifests=(suite,)))
    assert resolved.manifest.reproducibility == "snapshot"
    held = {k: v for k, v in blobs().items() if k != SNAPSHOT_URI}
    with pytest.raises(UnverifiedBenchmarkError, match="environment_snapshot: the source does not"):
        resolve_benchmark_suite(PINNED, source(manifests=(suite,), held=held))


# ---- structure on load, bytes against a source -------------------------------------------


def _loaded_snapshot_suite() -> ResolvedBenchmarkSuite:
    """A resolved snapshot suite, stored and parsed back: structure only, no bytes."""
    snapshot = ref(SNAPSHOT_URI, SNAPSHOT, "application/octet-stream")
    suite = manifest(reproducibility="snapshot", environment_snapshot=snapshot)
    resolved = resolve_benchmark_suite(PINNED, source(manifests=(suite,)))
    return ResolvedBenchmarkSuite.model_validate_json(resolved.model_dump_json())


def test_a_loaded_suite_parses_structurally_and_its_bytes_verify_against_a_source() -> None:
    loaded = _loaded_snapshot_suite()
    assert verify_suite_content(loaded, source()) is None


def test_parsing_proves_structure_not_bytes() -> None:
    """A loaded record still parses when its content is gone or changed; the byte check refuses."""
    loaded = _loaded_snapshot_suite()
    held = blobs()
    held[POLICY_URI] = b"tampered policy"
    held[answer_uri(TASK_IDS[1])] = b"another answer"
    del held[SNAPSHOT_URI]
    with pytest.raises(UnverifiedBenchmarkError) as refused:
        verify_suite_content(loaded, source(held=held))
    assert refused.value.reasons == (
        f"{TASK_IDS[0]} attachments.policy: the bytes do not have the pinned digest",
        f"{TASK_IDS[1]} attachments.policy: the bytes do not have the pinned digest",
        f"{TASK_IDS[1]} reference: the bytes do not have the pinned digest",
        f"{TASK_IDS[2]} attachments.policy: the bytes do not have the pinned digest",
        "environment_snapshot: the source does not hold it",
    )
    assert "https://" not in str(refused.value) and "s3://" not in str(refused.value)


# ---- agent-visible inputs vs held-out references -------------------------------------------------


def test_task_inputs_are_verified_text_and_never_the_reference() -> None:
    spec = task(TASK_IDS[0])
    inputs = resolve_task_inputs(spec, source())
    assert dict(inputs) == {"policy": POLICY}
    assert answer(TASK_IDS[0]) not in repr(inputs)
    with pytest.raises(UnverifiedBenchmarkError, match="attachments.policy"):
        resolve_task_inputs(spec, source(held={POLICY_URI: b"tampered"}))


def test_the_reference_is_resolved_only_on_request_for_evaluation() -> None:
    spec = task(TASK_IDS[0])
    assert resolve_reference(spec, source()) == answer(TASK_IDS[0])
    assert resolve_reference(task(TASK_IDS[0], reference=None), source()) is None
    with pytest.raises(UnverifiedBenchmarkError, match="reference: the bytes do not have"):
        resolve_reference(spec, source(held={answer_uri(TASK_IDS[0]): b"wrong"}))


def test_inline_attachments_need_no_source() -> None:
    record = task_record(TASK_IDS[0], reference=None)
    record["inputs"]["attachments"][0]["content"] = {"kind": "inline", "text": POLICY}
    assert dict(resolve_task_inputs(TaskSpec.model_validate(record), source(held={}))) == {
        "policy": POLICY
    }
