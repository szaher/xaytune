"""Pinned task and suite contracts, their identities, sampling and cohorts (ADR-020 B1-B8)."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError

from tests.test_benchmarks.fixtures import (
    POLICY,
    POLICY_URI,
    SNAPSHOT,
    SNAPSHOT_URI,
    TASK_IDS,
    answer,
    answer_uri,
    manifest,
    manifest_record,
    ref,
    source,
    task,
    task_record,
)
from tests.test_core.test_harness import PINNED_V1, _harness
from xaytune.benchmarks import resolve_benchmark_suite
from xaytune.core.domain.benchmark import (
    AllTasks,
    BenchmarkSuiteManifest,
    BenchmarkSuiteRef,
    BenchmarkSuiteRequest,
    FirstTasks,
    ResolvedBenchmarkSuite,
    SeededSample,
    TaskAssignment,
    TaskSpec,
    UnverifiedAssignmentError,
    agent_visible,
    assign_tasks,
    assignment_mismatches,
    benchmark_cohort,
    record_assignment,
    suite_identity_v1,
    task_identity_v1,
    verify_assignment,
)
from xaytune.core.domain.harness import AgentHarnessSpec

REQUEST = BenchmarkSuiteRequest(suite_id="support/core", revision="stable")


def _resolved() -> ResolvedBenchmarkSuite:
    return resolve_benchmark_suite(REQUEST, source())


# ---- task identity ------------------------------------------------------------------------


def test_a_task_is_identified_by_its_content_not_its_order_metadata_or_location() -> None:
    base = task(TASK_IDS[0])
    reordered = TaskSpec.model_validate(
        json.loads(
            json.dumps(task_record(TASK_IDS[0])),
            object_pairs_hook=lambda pairs: dict(reversed(pairs)),
        )
    )
    annotated = task(TASK_IDS[0], metadata={"owner": "qa"})
    moved = task_record(TASK_IDS[0])
    moved["inputs"]["attachments"][0]["content"] = ref(
        "gs://mirror/policy.txt", POLICY.encode(), "text/markdown"
    )
    inline = task_record(TASK_IDS[0])
    inline["inputs"]["attachments"][0]["content"] = {"kind": "inline", "text": POLICY}
    for same in (
        reordered,
        annotated,
        TaskSpec.model_validate(moved),
        TaskSpec.model_validate(inline),
    ):
        assert same.task_fingerprint() == base.task_fingerprint()


def _edited(path: str, value: Any) -> TaskSpec:
    record = task_record(TASK_IDS[0])
    *parents, last = path.split(".")
    node: Any = record
    for key in parents:
        node = node[int(key)] if isinstance(node, list) else node[key]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
    return TaskSpec.model_validate(record)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("id", "support/refund-009"),
        ("revision", "2026.10.2"),
        ("inputs.variables.ticket", "A different question"),
        ("inputs.variables", {"customer": "Ann Lee"}),
        ("inputs.attachments.0.name", "rules"),
        ("inputs.attachments.0.content", {"kind": "inline", "text": "Refunds: ten days."}),
        ("inputs.attachments", []),
        ("output.media_type", "text/markdown"),
        ("reference", ref(answer_uri("x"), b"another answer")),
        ("reference", None),
    ],
)
def test_every_input_byte_and_pin_is_task_identity(path: str, value: Any) -> None:
    assert _edited(path, value).task_fingerprint() != task(TASK_IDS[0]).task_fingerprint()


def test_task_identity_is_a_frozen_v1_projection() -> None:
    identity = task_identity_v1(task(TASK_IDS[0]))
    assert identity == {
        "type": "task",
        "version": 1,
        "id": "support/refund-001",
        "revision": "2026.10.1",
        "variables": {"customer": "Ann Lee", "ticket": "Question in support/refund-001"},
        "attachments": [{"name": "policy", "content": ref(POLICY_URI, POLICY.encode())["digest"]}],
        "output": {"kind": "text-answer", "media_type": "text/plain"},
        "reference": ref(answer_uri(TASK_IDS[0]), answer(TASK_IDS[0]).encode())["digest"],
    }
    assert task(TASK_IDS[0]).task_fingerprint() == PINNED_TASK_V1


PINNED_TASK_V1 = "sha256:10fa3bb3104b4fa6864dc1b64e81d007b51d910d2b268584b523eb4e1f12bc64"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"revision": "latest"}, "moving alias"),
        ({"revision": "main"}, "moving alias"),
        ({"revision": "v2-latest"}, "moving alias"),
        ({"id": "Support/Refund"}, "lowercase"),
        ({"id": "support//refund"}, "lowercase"),
        ({"reference": {"kind": "inline", "text": "the answer"}}, "reference"),
        ({"schema_version": "xaytune.task/v2"}, "schema_version"),
        ({"output": {"kind": "json", "media_type": "text/plain"}}, "output"),
        ({"inputs": {"variables": {"n": 3}, "attachments": []}}, "strings"),
        (
            {
                "inputs": {
                    "variables": {},
                    "attachments": [
                        {"name": "b", "content": {"kind": "inline", "text": "x"}},
                        {"name": "a", "content": {"kind": "inline", "text": "y"}},
                    ],
                }
            },
            "sorted",
        ),
        ({"evaluator": "exact-match"}, "evaluator"),
    ],
)
def test_what_cannot_be_pinned_is_refused(changes: dict[str, Any], reason: str) -> None:
    with pytest.raises(ValidationError, match=reason):
        TaskSpec.model_validate(task_record(TASK_IDS[0], **changes))


def test_every_task_field_is_required() -> None:
    for field in ("schema_version", "id", "revision", "inputs", "output", "reference"):
        record = task_record(TASK_IDS[0])
        del record[field]
        with pytest.raises(ValidationError, match=field):
            TaskSpec.model_validate(record)


def test_the_held_out_reference_is_never_agent_visible() -> None:
    spec = task(TASK_IDS[0])
    assert spec.reference is not None
    visible = agent_visible(spec).model_dump_json()
    assert spec.reference.digest not in visible
    assert spec.reference.uri not in visible
    assert "reference" not in visible


# ---- suite identity ---------------------------------------------------------------------------


def test_member_order_is_suite_identity() -> None:
    reversed_tasks = tuple(task(i) for i in reversed(TASK_IDS))
    assert manifest(tasks=reversed_tasks).suite_fingerprint() != manifest().suite_fingerprint()


@pytest.mark.parametrize(
    "changes",
    [
        {"revision": "v4"},
        {"provider": {"name": "acme-bench", "version": "1.5.0"}},
        {"tasks": tuple(task(i) for i in TASK_IDS[:2])},
        {"tasks": (task(TASK_IDS[0], revision="2026.10.2"), task(TASK_IDS[1]), task(TASK_IDS[2]))},
        {
            "reproducibility": "snapshot",
            "environment_snapshot": ref(SNAPSHOT_URI, SNAPSHOT, "application/octet-stream"),
        },
    ],
)
def test_a_different_suite_is_a_different_fingerprint(changes: dict[str, Any]) -> None:
    assert manifest(**changes).suite_fingerprint() != manifest().suite_fingerprint()


def test_suite_metadata_is_not_identity_and_the_projection_is_frozen() -> None:
    assert manifest(metadata={"note": "x"}).suite_fingerprint() == manifest().suite_fingerprint()
    identity = suite_identity_v1(manifest())
    assert set(identity) == {
        "type", "version", "id", "revision", "provider", "members", "reproducibility",
        "environment_snapshot", "reset",
    }  # fmt: skip
    assert identity["type"] == "benchmark-suite" and identity["version"] == 1
    assert manifest().suite_fingerprint() == PINNED_SUITE_V1


PINNED_SUITE_V1 = "sha256:4536c723153e49c5663b177c80a8e7484ca8d1e45dcc1ed00600bf9675062a4d"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"revision": "latest"}, "moving alias"),
        ({"provider": {"name": "acme-bench", "version": "latest"}}, "moving alias"),
        ({"members": []}, "members"),
        ({"reproducibility": "dynamic"}, "reproducibility"),
        ({"reset": "per-suite"}, "reset"),
        ({"reproducibility": "snapshot"}, "snapshot suite pins"),
        ({"environment_snapshot": ref(SNAPSHOT_URI, SNAPSHOT)}, "snapshot suite pins"),
    ],
)
def test_what_a_suite_cannot_claim_is_refused(changes: dict[str, Any], reason: str) -> None:
    with pytest.raises(ValidationError, match=reason):
        BenchmarkSuiteManifest.model_validate(manifest_record(**changes))


def test_a_suite_lists_each_task_once() -> None:
    record = manifest_record()
    record["members"].append(record["members"][0])
    with pytest.raises(ValidationError, match="each task once"):
        BenchmarkSuiteManifest.model_validate(record)


def test_a_pinned_ref_cannot_name_an_alias() -> None:
    with pytest.raises(ValidationError, match="moving alias"):
        BenchmarkSuiteRef(
            suite_id="support/core",
            revision="stable-latest",
            suite_fingerprint="sha256:" + "0" * 64,
        )
    assert BenchmarkSuiteRequest(suite_id="support/core", revision="stable").revision == "stable"


# ---- a resolved suite verifies itself ----------------------------------------------------


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (lambda r: r["ref"].update(suite_fingerprint="sha256:" + "1" * 64), "not the manifest's"),
        (lambda r: r["ref"].update(revision="v4"), "disagree on suite id or revision"),
        (lambda r: r["request"].update(suite_id="support/other"), "different suite"),
        (lambda r: r["tasks"].reverse(), "not the member at that position"),
        (lambda r: r["tasks"].pop(), "exactly the manifest's members"),
        (
            lambda r: r["tasks"][0]["inputs"]["variables"].update(ticket="edited"),
            "pinned fingerprint",
        ),
    ],
)
def test_a_resolved_suite_edited_after_resolution_does_not_parse(edit: Any, reason: str) -> None:
    record = json.loads(_resolved().model_dump_json())
    edit(record)
    with pytest.raises(ValidationError, match=reason):
        ResolvedBenchmarkSuite.model_validate(record)


def test_a_resolved_suite_round_trips() -> None:
    resolved = _resolved()
    again = ResolvedBenchmarkSuite.model_validate_json(resolved.model_dump_json())
    assert again == resolved
    assert again.task(TASK_IDS[1]).id == TASK_IDS[1]
    with pytest.raises(KeyError):
        again.task("support/nope")


# ---- sampling and cohorts -----------------------------------------------------------------------


def test_sampling_is_deterministic_and_keeps_manifest_order() -> None:
    suite = _resolved()
    assert assign_tasks(suite, AllTasks(), seed=1) == TASK_IDS
    assert assign_tasks(suite, FirstTasks(count=2), seed=1) == TASK_IDS[:2]
    sample = assign_tasks(suite, SeededSample(count=2), seed=0)
    assert len(sample) == 2 and sample == tuple(i for i in TASK_IDS if i in sample)
    assert assign_tasks(_resolved(), SeededSample(count=2), seed=0) == sample
    assert sample == PINNED_SAMPLE_SEED_0 != TASK_IDS[:2]
    seeds = {assign_tasks(suite, SeededSample(count=1), seed=s) for s in range(20)}
    assert len(seeds) > 1  # the seed matters


PINNED_SAMPLE_SEED_0 = ("support/refund-001", "support/shipping-001")
"""``assign_tasks(fixture suite, SeededSample(2), seed=0)``: frozen forever (ADR-020 B5)."""


def test_sampling_more_than_the_suite_holds_is_refused() -> None:
    with pytest.raises(ValueError, match="asks for 4 tasks; the suite has 3"):
        assign_tasks(_resolved(), FirstTasks(count=4), seed=0)
    with pytest.raises(ValueError, match="integer seed"):
        assign_tasks(_resolved(), SeededSample(count=1), seed=True)


def test_reset_semantics_are_recorded_and_required() -> None:
    assert suite_identity_v1(manifest())["reset"] == "per-task"
    record = manifest_record()
    del record["reset"]
    with pytest.raises(ValidationError, match="reset"):
        BenchmarkSuiteManifest.model_validate(record)


def test_a_cohort_is_suite_and_policy_never_the_seed() -> None:
    ref_ = _resolved().ref
    base = benchmark_cohort(ref_, SeededSample(count=2))
    assert benchmark_cohort(ref_, SeededSample(count=2)) == base
    assert benchmark_cohort(ref_, SeededSample(count=1)) != base
    assert benchmark_cohort(ref_, AllTasks()) != base
    other = ref_.model_copy(update={"suite_fingerprint": "sha256:" + "2" * 64})
    assert benchmark_cohort(other, SeededSample(count=2)) != base
    assert benchmark_cohort(ref_, AllTasks()) == PINNED_COHORT_ALL


PINNED_COHORT_ALL = "sha256:dad6af3b624accb049ddd12b7c45b91f8e37b6837981026c00130552481eb392"


# ---- benchmarks are not harness identity --------------------------------------------------


def test_a_benchmark_never_touches_harness_identity() -> None:
    assert _harness().harness_fingerprint() == PINNED_V1
    fields = set(AgentHarnessSpec.model_fields)
    assert not fields & {"benchmark", "suite", "tasks", "task", "sampling", "environment"}
    _resolved()  # resolving, sampling and cohorting a suite leaves the harness as it was
    assert _harness().harness_fingerprint() == PINNED_V1


def test_the_contracts_are_pure_core() -> None:
    code = (
        "import sys, xaytune.core.domain.benchmark\n"
        "bad = [m for m in ('xaytune.benchmarks', 'xaytune.harness', 'xaytune.agent', "
        "'xaytune.storage', 'xaytune.runtimes', 'torch', 'transformers') if m in sys.modules]\n"
        "print(','.join(bad))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == ""


# ---- matched assignments (ADR-020 B7) -------------------------------------------------------


def test_runs_under_the_same_replicate_seed_are_matched_whatever_the_candidate() -> None:
    """The assignment depends on suite, policy and seed only, so paired runs match."""
    policy = SeededSample(count=2)
    first = record_assignment(_resolved(), policy, seed=3)
    second = record_assignment(resolve_benchmark_suite(REQUEST, source()), policy, seed=3)
    assert first == second
    assert assignment_mismatches(first, second) == ()
    assert first.cohort == benchmark_cohort(_resolved().ref, policy)


def test_unmatched_assignments_are_reported_with_every_reason() -> None:
    suite = _resolved()
    base = record_assignment(suite, SeededSample(count=2), seed=0)
    other_seed = record_assignment(suite, SeededSample(count=2), seed=1)
    assert base.task_ids != other_seed.task_ids  # same cohort, different tasks
    assert assignment_mismatches(base, other_seed) == (
        "different replicate seeds: the runs are not paired",
        "different task assignments",
    )
    other_policy = record_assignment(suite, AllTasks(), seed=0)
    assert "different benchmark cohorts" in assignment_mismatches(base, other_policy)[0]


def test_an_assignment_records_a_real_cohort_and_at_least_one_task() -> None:
    with pytest.raises(ValidationError, match="digest"):
        TaskAssignment(cohort="cohort-1", seed=0, task_ids=("support/refund-001",))
    with pytest.raises(ValidationError, match="task_ids"):
        TaskAssignment(cohort="sha256:" + "0" * 64, seed=0, task_ids=())


# ---- verifying a recorded assignment (ADR-020 B7) -----------------------------------------

SEEDED = SeededSample(count=2)
NOT_THE_ASSIGNMENT = "the tasks are not the ones this suite, policy and seed assign"


@pytest.mark.parametrize(
    ("policy", "seed"),
    [(AllTasks(), 0), (FirstTasks(count=2), 5), (SEEDED, 0), (SEEDED, 1), (SEEDED, 41)],
)
def test_a_recorded_assignment_verifies_against_its_suite(policy: Any, seed: int) -> None:
    recorded = record_assignment(_resolved(), policy, seed)
    parsed = TaskAssignment.model_validate_json(recorded.model_dump_json())
    assert verify_assignment(_resolved(), policy, parsed) == recorded


def _claim(task_ids: tuple[str, ...], seed: int = 0, policy: Any = SEEDED) -> TaskAssignment:
    """A record built by hand, as one read from storage or another process would be."""
    cohort = benchmark_cohort(_resolved().ref, policy)
    return TaskAssignment.model_construct(cohort=cohort, seed=seed, task_ids=task_ids)


@pytest.mark.parametrize(
    ("claim", "reasons"),
    [
        pytest.param(
            _claim(("support/refund-001", "support/unknown-009")),
            (
                "the assignment names tasks that are not members of the suite",
                NOT_THE_ASSIGNMENT,
            ),
            id="incorrect-member",
        ),
        pytest.param(
            _claim(("support/refund-001", "support/refund-001")),
            ("the assignment lists a task more than once", NOT_THE_ASSIGNMENT),
            id="duplicate-ids",
        ),
        pytest.param(
            _claim(("support/shipping-001", "support/refund-001")),
            ("the tasks are not in manifest order", NOT_THE_ASSIGNMENT),
            id="reordered",
        ),
        pytest.param(
            TaskAssignment(
                cohort=benchmark_cohort(_resolved().ref, AllTasks()),
                seed=0,
                task_ids=PINNED_SAMPLE_SEED_0,
            ),
            ("the cohort is not this suite's and sampling policy's",),
            id="incorrect-cohort",
        ),
        pytest.param(
            _claim(PINNED_SAMPLE_SEED_0, seed=1),
            (NOT_THE_ASSIGNMENT,),
            id="mismatched-seed",
        ),
        pytest.param(
            _claim(TASK_IDS[:1]),
            (NOT_THE_ASSIGNMENT,),
            id="too-few-tasks",
        ),
    ],
)
def test_an_assignment_that_the_suite_does_not_assign_is_refused(
    claim: TaskAssignment, reasons: tuple[str, ...]
) -> None:
    with pytest.raises(UnverifiedAssignmentError) as refused:
        verify_assignment(_resolved(), SEEDED, claim)
    assert refused.value.reasons == reasons


def test_an_assignment_for_another_suite_is_refused() -> None:
    """A real assignment of one suite does not verify against a reordered one."""
    reordered_tasks = tuple(task(i) for i in reversed(TASK_IDS))
    other = resolve_benchmark_suite(REQUEST, source(manifests=(manifest(tasks=reordered_tasks),)))
    recorded = record_assignment(other, SEEDED, 0)
    with pytest.raises(UnverifiedAssignmentError) as refused:
        verify_assignment(_resolved(), SEEDED, recorded)
    assert refused.value.reasons[0] == "the cohort is not this suite's and sampling policy's"


def test_a_policy_the_suite_cannot_satisfy_is_refused() -> None:
    policy = FirstTasks(count=len(TASK_IDS) + 1)
    claim = _claim(TASK_IDS, policy=policy)
    with pytest.raises(UnverifiedAssignmentError) as refused:
        verify_assignment(_resolved(), policy, claim)
    assert refused.value.reasons == (
        "the sampling policy does not apply to this suite: "
        "the sampling policy asks for 4 tasks; the suite has 3",
    )


def test_fabricated_records_can_agree_but_do_not_verify() -> None:
    """The review's case: agreement between two claims is not proof of either."""
    fabricated = TaskAssignment(
        cohort=benchmark_cohort(_resolved().ref, SEEDED),
        seed=0,
        task_ids=("support/unrelated-001", "support/unrelated-002"),
    )
    twin = TaskAssignment.model_validate(fabricated.model_dump())
    assert assignment_mismatches(fabricated, twin) == ()
    for claim in (fabricated, twin):
        with pytest.raises(UnverifiedAssignmentError, match="not members"):
            verify_assignment(_resolved(), SEEDED, claim)


def test_a_parsed_assignment_lists_each_task_once() -> None:
    with pytest.raises(ValidationError, match="each task once"):
        TaskAssignment(
            cohort="sha256:" + "0" * 64,
            seed=0,
            task_ids=("support/refund-001", "support/refund-001"),
        )
