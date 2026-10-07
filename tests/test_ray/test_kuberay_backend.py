"""KubeRayJobsBackend: the same runtimes, submitted as KubeRay RayJobs -- once, whatever happens.

``FakeKubernetes`` is the API server, and its operator runs each RayJob's Ray
job on ``ProcessJobs``: real supervisors and real workers, as in the
``RayJobsRuntime`` matrix. The runtimes are the merged ones, unchanged; only
the submission backend differs.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.test_ray.kube_support import FakeKubernetes, status_error
from tests.test_ray.ray_support import ProcessJobs
from tests.test_ray.test_ray_jobs_runtime import _FAILS, _TRAINS, _plan, _settle, _types
from xaytune.core.errors import IdempotencyConflictError
from xaytune.core.execution import ResourceRequirements
from xaytune.core.ids import OperationId
from xaytune.ray import (
    KubeRayConfig,
    KubeRayJobsBackend,
    RayJobsConfig,
    RayJobsRuntime,
    RaySubmissionBackend,
    RayTrainConfig,
    RayTrainRuntime,
    RayUnavailableError,
    ray_jobs_runtime,
    ray_train_runtime,
)
from xaytune.ray.runtime.jobs import ACCEPTED, REJECTED
from xaytune.ray.submission import RayJobsBackend, submission_backend
from xaytune.ray.submission.kuberay import (
    STOP_REQUESTED,
    SUBMISSION_DIGEST,
    SUBMISSION_ID,
    job_status,
    rayjob_name,
)
from xaytune.runtimes import UnsupportedPlanError
from xaytune.runtimes.local.paths import WorkloadPaths, read_json

NAMESPACE = "ml"
_EXISTING = {"kind": "existing", "selector": {"ray.io/cluster": "trainers"}}
_TEMPLATE = {
    "rayVersion": "2.59.0",
    "headGroupSpec": {
        "rayStartParams": {},
        "template": {
            "spec": {"containers": [{"name": "ray-head", "image": "registry/xaytune-ray:1"}]}
        },
    },
    "workerGroupSpecs": [
        {
            "groupName": "gpu",
            "replicas": 2,
            "rayStartParams": {},
            "template": {
                "spec": {"containers": [{"name": "ray-worker", "image": "registry/xaytune-ray:1"}]}
            },
        }
    ],
}
_EPHEMERAL = {"kind": "ephemeral", "template": _TEMPLATE}


def _kuberay(cluster: dict[str, Any] = _EXISTING, **rayjob: Any) -> dict[str, Any]:
    if cluster["kind"] == "ephemeral":
        rayjob.setdefault("deletion_policy", "delete-cluster")
    return {
        "kind": "kuberay",
        "namespace": NAMESPACE,
        "context": "kind-xaytune",
        "cluster": cluster,
        "rayjob": rayjob,
    }


def _backend(kube: FakeKubernetes, **config: Any) -> KubeRayJobsBackend:
    return KubeRayJobsBackend(KubeRayConfig.model_validate(_kuberay(**config)), api=kube)


def _jobs_runtime(root: Path, kube: FakeKubernetes, **config: Any) -> RayJobsRuntime:
    settings = {"submission": _kuberay(**config), "runtime_env": {}, "shared_state_root": str(root)}
    return RayJobsRuntime(
        RayJobsConfig.model_validate(settings), submission=_backend(kube, **config)
    )


def _submit(backend: KubeRayJobsBackend, submission_id: str = "op_1", **overrides: Any) -> None:
    arguments: dict[str, Any] = {
        "metadata": {"xaytune.request_digest": "sha256:a"},
        "resources": {},
        "runtime_env": {},
    }
    arguments.update(overrides)
    entrypoint = arguments.pop("entrypoint", "python -m xaytune.ray.runtime.supervisor /s/op_1")
    backend.submit(submission_id, entrypoint, **arguments)


@pytest.fixture
def jobs() -> Iterator[ProcessJobs]:
    fake = ProcessJobs()
    yield fake
    fake.close()


@pytest.fixture
def kube(jobs: ProcessJobs) -> FakeKubernetes:
    return FakeKubernetes(jobs)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "shared"


# ---- the backend on its own ----------------------------------------------------------------


def test_it_is_a_submission_backend_like_the_jobs_api_one() -> None:
    assert isinstance(_backend(FakeKubernetes()), RaySubmissionBackend)
    assert isinstance(RayJobsBackend("http://h:8265"), RaySubmissionBackend)


def test_the_rayjob_name_is_a_pure_function_of_the_id_and_valid_for_kuberay() -> None:
    names = {rayjob_name(f"op_01J{index:07d}ABCDEFGHJKMNPQRSTV") for index in range(500)}
    assert len(names) == 500
    for name in names:
        assert len(name) <= 47  # KubeRay's RayJob limit
        assert re.fullmatch(r"[a-z]([-a-z0-9]*[a-z0-9])?", name)  # a DNS-1035 label
    assert rayjob_name("op_X") == rayjob_name("op_X")


def test_the_same_submission_creates_one_rayjob_and_a_retry_finds_it() -> None:
    kube = FakeKubernetes()
    backend = _backend(kube)
    _submit(backend)
    _submit(backend)  # a retry: "already exists" is this very RayJob
    assert len(kube.objects) == 1 and kube.creates == 2
    _submit(_backend(kube))  # and so is a reconstructed backend's
    assert len(kube.objects) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"entrypoint": "python -m elsewhere /s/op_1"},
        {"metadata": {"xaytune.request_digest": "sha256:b"}},
        {"resources": {"entrypoint_num_cpus": 2}},
        {"runtime_env": {"pip": ["torch"]}},
    ],
    ids=["entrypoint", "metadata", "resources", "runtime-env"],
)
def test_another_submission_under_the_same_id_is_a_conflict(change: dict[str, Any]) -> None:
    kube = FakeKubernetes()
    _submit(_backend(kube))
    with pytest.raises(IdempotencyConflictError):
        _submit(_backend(kube), **change)
    assert len(kube.objects) == 1


def test_the_same_id_placed_differently_is_a_conflict_not_a_second_rayjob() -> None:
    kube = FakeKubernetes()
    _submit(_backend(kube))
    elsewhere = {"kind": "existing", "selector": {"ray.io/cluster": "other"}}
    with pytest.raises(IdempotencyConflictError):
        _submit(_backend(kube, cluster=elsewhere))
    with pytest.raises(IdempotencyConflictError):
        _submit(_backend(kube, cluster=_EPHEMERAL))
    assert len(kube.objects) == 1


def test_a_rayjob_under_the_name_that_is_not_this_ids_is_never_adopted() -> None:
    kube = FakeKubernetes()
    backend = _backend(kube)
    _submit(backend)
    (stored,) = kube.objects.values()
    stored["metadata"]["annotations"][SUBMISSION_ID] = "op_someone_else"
    with pytest.raises(IdempotencyConflictError):
        backend.info("op_1")
    with pytest.raises(IdempotencyConflictError):
        _submit(backend)
    with pytest.raises(IdempotencyConflictError):
        backend.stop("op_1")
    assert not kube.patches


def test_a_completed_rayjob_stays_completed_across_reconstruction_and_stop() -> None:
    kube = FakeKubernetes()
    _submit(_backend(kube))
    kube.set_status(
        NAMESPACE, rayjob_name("op_1"), jobStatus="SUCCEEDED", jobDeploymentStatus="Complete"
    )
    reconstructed = _backend(kube)
    job = reconstructed.info("op_1")
    assert job is not None and job.status == "SUCCEEDED"
    assert job.metadata == {"xaytune.request_digest": "sha256:a"}
    reconstructed.stop("op_1")  # finished: nothing to stop, and its record stays
    assert not kube.patches
    job = _backend(kube).info("op_1")
    assert job is not None and job.status == "SUCCEEDED"


def test_kubernetes_saying_not_found_is_the_only_absence() -> None:
    kube = FakeKubernetes()
    backend = _backend(kube)
    assert backend.info("op_never") is None


@pytest.mark.parametrize("failure", ["unreachable", "forbidden", "no-crds", "server-error"])
def test_anything_but_not_found_is_unavailable_never_absent(failure: str) -> None:
    kube = FakeKubernetes()
    backend = _backend(kube)
    _submit(backend)
    if failure == "server-error":

        def broken(*_: Any, **__: Any) -> Any:
            raise status_error(500, "InternalError")

        kube.get_namespaced_custom_object = broken  # type: ignore[method-assign]
        kube.delete_namespaced_custom_object = broken  # type: ignore[method-assign]
        kube.create_namespaced_custom_object = broken  # type: ignore[method-assign]
    elif failure == "unreachable":
        kube.unreachable = True
    elif failure == "forbidden":
        kube.forbidden = True
    else:
        kube.crds_installed = False  # a 404 for the resource type, not for the RayJob
    with pytest.raises(RayUnavailableError):
        backend.info("op_1")
    with pytest.raises(RayUnavailableError):
        backend.info("op_never")
    with pytest.raises(RayUnavailableError):
        backend.stop("op_1")
    with pytest.raises(RayUnavailableError):
        _submit(backend, "op_2")


def test_an_unavailable_error_never_carries_the_servers_body() -> None:
    kube = FakeKubernetes()

    def leaky(*_: Any, **__: Any) -> Any:
        error = status_error(422, "Invalid")
        error.body = json.dumps({"reason": "Invalid", "message": "token=s3cr3t rejected"})
        raise error

    kube.create_namespaced_custom_object = leaky  # type: ignore[method-assign]
    with pytest.raises(RayUnavailableError) as raised:
        _submit(_backend(kube))
    assert "s3cr3t" not in str(raised.value) and "Invalid" in str(raised.value)


def test_stopping_an_ephemeral_rayjob_suspends_it_once_and_keeps_it_as_the_record() -> None:
    kube = FakeKubernetes()
    backend = _backend(kube, cluster=_EPHEMERAL)
    _submit(backend)
    backend.stop("op_1")
    backend.stop("op_1")  # idempotent: already suspended by Xaytune
    backend.stop("op_never")
    ((name, patch, content_type),) = kube.patches
    assert name == rayjob_name("op_1")
    assert patch["spec"] == {"suspend": True}
    assert STOP_REQUESTED in patch["metadata"]["annotations"]
    assert content_type == "application/merge-patch+json"
    (rayjob,) = kube.objects.values()  # kept: nothing is ever deleted
    assert rayjob["status"] == {"jobDeploymentStatus": "Suspended"}
    job = _backend(kube, cluster=_EPHEMERAL).info("op_1")  # after a restart too
    assert job is not None and job.status == "STOPPED"


def test_stopping_a_rayjob_on_an_existing_cluster_changes_nothing_it_cannot_record() -> None:
    """KubeRay ignores ``suspend`` with a clusterSelector; deleting would erase the status."""
    kube = FakeKubernetes()
    backend = _backend(kube)
    _submit(backend)
    backend.stop("op_1")
    backend.stop("op_1")
    assert not kube.patches and len(kube.objects) == 1
    job = backend.info("op_1")
    assert job is not None and job.status == "PENDING"  # never a STOPPED it did not observe


def test_a_suspension_xaytune_did_not_ask_for_is_waiting_not_stopped() -> None:
    kube = FakeKubernetes()
    backend = _backend(kube, cluster=_EPHEMERAL)
    _submit(backend)
    kube.set_status(NAMESPACE, rayjob_name("op_1"), jobDeploymentStatus="Suspended")
    job = backend.info("op_1")
    assert job is not None and job.status == "PENDING"


def test_on_an_existing_cluster_the_rayjob_selects_it_and_owns_none() -> None:
    kube = FakeKubernetes()
    _submit(_backend(kube), resources={"entrypoint_num_cpus": 1, "entrypoint_num_gpus": 1})
    (rayjob,) = kube.objects.values()
    spec = rayjob["spec"]
    assert spec["clusterSelector"] == {"ray.io/cluster": "trainers"}
    assert "rayClusterSpec" not in spec and "shutdownAfterJobFinishes" not in spec
    assert spec["jobId"] == "op_1" and spec["backoffLimit"] == 0
    assert spec["entrypointNumCpus"] == 1 and spec["entrypointNumGpus"] == 1
    assert rayjob["metadata"]["namespace"] == NAMESPACE
    assert rayjob["metadata"]["annotations"][SUBMISSION_ID] == "op_1"
    assert rayjob["metadata"]["annotations"][SUBMISSION_DIGEST].startswith("sha256:")


def test_an_ephemeral_cluster_is_the_rayjobs_own_and_goes_while_the_rayjob_stays() -> None:
    kube = FakeKubernetes()
    _submit(_backend(kube, cluster=_EPHEMERAL, ttl_seconds_after_finished=30))
    (rayjob,) = kube.sent
    spec = rayjob["spec"]
    assert spec["rayClusterSpec"] == _TEMPLATE
    assert "clusterSelector" not in spec
    assert spec["shutdownAfterJobFinishes"] is True  # the cluster goes after the job
    assert spec["ttlSecondsAfterFinished"] == 30
    assert "deletionStrategy" not in spec  # never DeleteSelf: the RayJob stays


def test_configured_labels_and_annotations_reach_the_rayjob() -> None:
    kube = FakeKubernetes()
    labels = {"team": "post-training", "example.com/cost-center": "ml-42"}
    _submit(_backend(kube, labels=labels, annotations={"example.com/owner": "alice"}))
    (rayjob,) = kube.objects.values()
    assert rayjob["metadata"]["labels"] == {**labels, "app.kubernetes.io/managed-by": "xaytune"}
    assert rayjob["metadata"]["annotations"]["example.com/owner"] == "alice"


def test_the_runtime_env_is_the_rayjobs_and_a_resource_it_cannot_express_is_refused() -> None:
    from xaytune.ray import RaySubmissionRefusedError

    kube = FakeKubernetes()
    _submit(_backend(kube), runtime_env={"pip": ["torch"], "env_vars": {"A": "1"}})
    (rayjob,) = kube.objects.values()
    assert json.loads(rayjob["spec"]["runtimeEnvYAML"]) == {
        "env_vars": {"A": "1"},
        "pip": ["torch"],
    }
    with pytest.raises(RaySubmissionRefusedError, match="entrypoint_memory"):
        _submit(_backend(kube), "op_2", resources={"entrypoint_memory": 2**30})
    assert len(kube.objects) == 1 and kube.creates == 1


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ({}, "PENDING"),
        ({"jobDeploymentStatus": "Initializing"}, "PENDING"),
        ({"jobDeploymentStatus": "Suspended"}, "PENDING"),  # not Xaytune's suspension
        ({"jobDeploymentStatus": "Waiting"}, "PENDING"),
        ({"jobStatus": "PENDING", "jobDeploymentStatus": "Running"}, "PENDING"),
        ({"jobStatus": "RUNNING", "jobDeploymentStatus": "Running"}, "RUNNING"),
        ({"jobStatus": "SUCCEEDED", "jobDeploymentStatus": "Complete"}, "SUCCEEDED"),
        ({"jobStatus": "FAILED", "jobDeploymentStatus": "Complete"}, "FAILED"),
        ({"jobStatus": "STOPPED", "jobDeploymentStatus": "Complete"}, "STOPPED"),
        ({"jobDeploymentStatus": "Failed", "reason": "SubmissionFailed"}, "FAILED"),
        ({"jobDeploymentStatus": "ValidationFailed"}, "FAILED"),
        ({"jobStatus": "RUNNING", "jobDeploymentStatus": "Failed"}, "FAILED"),
    ],
)
def test_a_rayjobs_status_is_rays_unless_its_deployment_ended_first(
    status: dict[str, str], expected: str
) -> None:
    assert job_status(status) == expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ({"jobDeploymentStatus": "Suspended"}, "STOPPED"),
        ({"jobDeploymentStatus": "Suspending"}, "PENDING"),  # not yet: stopping is async
        ({"jobStatus": "RUNNING", "jobDeploymentStatus": "Suspending"}, "PENDING"),
        ({"jobStatus": "SUCCEEDED", "jobDeploymentStatus": "Complete"}, "SUCCEEDED"),
        ({"jobDeploymentStatus": "Initializing"}, "PENDING"),
    ],
)
def test_suspended_is_stopped_only_once_xaytune_asked_and_kuberay_suspended_it(
    status: dict[str, str], expected: str
) -> None:
    assert job_status(status, stop_requested=True) == expected


# ---- the live RayJob is verified against its bound identity on every read --------------------


def _mutate(stored: dict[str, Any], path: str, value: Any) -> None:
    *parents, leaf = [segment.replace("~", "/") for segment in path.split("/")]
    node = stored
    for key in parents:
        node = node[int(key)] if isinstance(node, list) else node[key]
    node[leaf] = value


_DRIFT = [
    ("existing", "spec/clusterSelector", {"ray.io/cluster": "elsewhere"}),
    ("existing", "spec/entrypoint", "python -m something_else"),
    ("existing", "spec/runtimeEnvYAML", '{"pip": ["evil"]}'),
    ("existing", "spec/entrypointNumCpus", 8),
    ("existing", "spec/metadata", {"xaytune.request_digest": "sha256:other"}),
    ("existing", "spec/activeDeadlineSeconds", 60),
    ("existing", "spec/submissionMode", "HTTPMode"),
    ("existing", "spec/ttlSecondsAfterFinished", 30),
    ("existing", "metadata/labels/kueue.x-k8s.io~queue-name", "gpu-queue"),
    ("existing", "metadata/annotations/team", "someone-else"),
    ("ephemeral", "spec/rayClusterSpec/headGroupSpec/rayStartParams", {"num-cpus": "64"}),
    ("ephemeral", "spec/rayClusterSpec/workerGroupSpecs/0/replicas", 20),
    ("ephemeral", "spec/rayClusterSpec/workerGroupSpecs/0/numOfHosts", 4),
    ("ephemeral", "spec/shutdownAfterJobFinishes", False),
    ("ephemeral", "spec/clusterSelector", {"ray.io/cluster": "trainers"}),
]


@pytest.mark.parametrize(
    ("mode", "path", "value"), _DRIFT, ids=[f"{m}:{p.split('/')[-1]}" for m, p, _ in _DRIFT]
)
def test_a_rayjob_changed_after_it_was_bound_is_refused_never_adopted(
    mode: str, path: str, value: Any
) -> None:
    kube = FakeKubernetes()
    config: dict[str, Any] = {"cluster": _EPHEMERAL} if mode == "ephemeral" else {}
    _submit(_backend(kube, **config), resources={"entrypoint_num_cpus": 1})
    (stored,) = kube.objects.values()
    _mutate(stored, path, value)
    reconstructed = _backend(kube, **config)  # a restarted controller's
    with pytest.raises(IdempotencyConflictError):
        reconstructed.info("op_1")
    with pytest.raises(IdempotencyConflictError):
        _submit(reconstructed, resources={"entrypoint_num_cpus": 1})  # a retry
    with pytest.raises(IdempotencyConflictError):
        reconstructed.stop("op_1")
    assert not kube.patches and kube.creates == 2 and len(kube.objects) == 1


_CONCURRENT_DRIFT = [
    ("spec/clusterSelector", {"ray.io/cluster": "elsewhere"}),
    ("spec/entrypoint", "python -m something_else"),
    ("spec/rayClusterSpec/headGroupSpec/rayStartParams", {"num-cpus": "64"}),
    ("metadata/labels/team", "someone-else"),
]


@pytest.mark.parametrize(
    ("path", "value"), _CONCURRENT_DRIFT, ids=[p.split("/")[-1] for p, _ in _CONCURRENT_DRIFT]
)
def test_a_rayjob_changed_between_verifying_and_suspending_it_is_never_suspended(
    path: str, value: Any
) -> None:
    """GET, verify -- someone edits the RayJob -- PATCH: the stale cancellation is refused."""
    kube = FakeKubernetes()
    backend = _backend(kube, cluster=_EPHEMERAL)
    _submit(backend)
    kube.concurrently(lambda stored: _mutate(stored, path, value))
    with pytest.raises(IdempotencyConflictError):
        backend.stop("op_1")
    (stored,) = kube.objects.values()
    assert not stored["spec"].get("suspend")
    assert STOP_REQUESTED not in stored["metadata"]["annotations"]
    assert not kube.patches


def test_a_benign_write_between_verifying_and_suspending_is_reverified_and_suspended() -> None:
    """KubeRay updated the status in between: the same RayJob, so it is suspended -- on the
    version that was verified again, never on the stale one."""
    kube = FakeKubernetes()
    backend = _backend(kube, cluster=_EPHEMERAL)
    _submit(backend)
    (stored,) = kube.objects.values()
    stale = stored["metadata"]["resourceVersion"]

    def kuberay_reconciles(rayjob: dict[str, Any]) -> None:
        rayjob["status"] = {"jobDeploymentStatus": "Initializing"}
        rayjob["metadata"]["managedFields"] = [{"manager": "kuberay-operator"}]
        rayjob["metadata"]["generation"] = 2

    kube.concurrently(kuberay_reconciles)
    backend.stop("op_1")
    ((_, patch, _),) = kube.patches  # the refused attempt applied nothing
    assert patch["metadata"]["resourceVersion"] != stale
    assert stored["spec"]["suspend"] is True
    assert STOP_REQUESTED in stored["metadata"]["annotations"]
    job = backend.info("op_1")
    assert job is not None and job.status == "STOPPED"


def test_a_rayjob_that_keeps_changing_is_not_suspended_on_an_unverified_version() -> None:
    kube = FakeKubernetes()
    backend = _backend(kube, cluster=_EPHEMERAL)
    _submit(backend)

    def keeps_reconciling(rayjob: dict[str, Any]) -> None:
        rayjob["status"] = {"jobDeploymentStatus": "Initializing"}
        kube.concurrently(keeps_reconciling)

    kube.concurrently(keeps_reconciling)
    with pytest.raises(RayUnavailableError, match="kept changing"):
        backend.stop("op_1")
    (stored,) = kube.objects.values()
    assert not stored["spec"].get("suspend") and not kube.patches


def test_what_may_change_after_binding_does_not_change_the_identity() -> None:
    """Status, server metadata, server defaults, and the suspension Xaytune asks for."""
    kube = FakeKubernetes()
    backend = _backend(kube, cluster=_EPHEMERAL)
    _submit(backend)
    (stored,) = kube.objects.values()
    # Stored with the defaults the server added, which the identity tolerates.
    assert stored["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["numOfHosts"] == 1
    stored["metadata"].update(
        resourceVersion="99",
        generation=7,
        managedFields=[{"manager": "kuberay-operator"}],
        finalizers=[],
    )
    stored["status"] = {"jobStatus": "RUNNING", "jobDeploymentStatus": "Running"}
    job = _backend(kube, cluster=_EPHEMERAL).info("op_1")
    assert job is not None and job.status == "RUNNING"
    stored["status"] = {}
    backend.stop("op_1")  # spec.suspend and the stop marker
    job = _backend(kube, cluster=_EPHEMERAL).info("op_1")
    assert job is not None and job.status == "STOPPED"


def test_a_restarted_runtime_refuses_to_adopt_a_rayjob_moved_to_another_cluster(
    root: Path,
) -> None:
    held = ProcessJobs(hold=True)
    kube = FakeKubernetes(held)
    operation = OperationId.generate()

    async def scenario() -> None:
        await _jobs_runtime(root, kube).submit_or_get(operation, _plan(_TRAINS))
        (stored,) = kube.objects.values()
        stored["spec"]["clusterSelector"] = {"ray.io/cluster": "elsewhere"}
        stored["spec"]["entrypoint"] = "python -c 'print(\"not the plan\")'"
        restarted = _jobs_runtime(root, kube)
        with pytest.raises(IdempotencyConflictError):
            await restarted.lookup_operation(operation)
        with pytest.raises(IdempotencyConflictError):
            await restarted.submit_or_get(operation, _plan(_TRAINS))

    try:
        asyncio.run(scenario())
    finally:
        held.close()
    # Refused on reading it: never re-created, never adopted.
    assert kube.creates == 1 and len(kube.objects) == 1 and held.submissions == 1


# ---- configuration: placement is identity ---------------------------------------------------


def test_everything_that_places_a_job_changes_the_placement_digest() -> None:
    def digest(**config: Any) -> str:
        return _backend(FakeKubernetes(), **config).placement_digest

    base = digest()
    variants = [
        digest(cluster={"kind": "existing", "selector": {"ray.io/cluster": "other"}}),
        digest(cluster=_EPHEMERAL),
        digest(cluster={**_EPHEMERAL, "template": {**_TEMPLATE, "rayVersion": "2.59.1"}}),
        digest(labels={"team": "post-training"}),
        digest(annotations={"team": "x"}),
    ]
    assert len({base, *variants}) == 1 + len(variants)
    assert digest() == base
    other_namespace = KubeRayConfig.model_validate({**_kuberay(), "namespace": "other"})
    assert KubeRayJobsBackend(other_namespace).placement_digest != base
    other_context = KubeRayConfig.model_validate({**_kuberay(), "context": None})
    assert KubeRayJobsBackend(other_context).placement_digest != base


@pytest.mark.parametrize(
    "change",
    [
        {"cluster": {"kind": "existing", "selector": {"app": "ray"}}},
        {"cluster": {"kind": "ephemeral", "template": {"workerGroupSpecs": []}}},
        {"cluster": {"kind": "ephemeral", "template": _TEMPLATE}},  # no deletion policy
        {"rayjob": {"deletion_policy": "delete-cluster"}},  # not its cluster to delete
        {"rayjob": {"ttl_seconds_after_finished": 5}},
        {"rayjob": {"labels": {"xaytune.io/anything": "x"}}},
        {"rayjob": {"annotations": {"xaytune.io/submission-id": "op_x"}}},
        {"rayjob": {"annotations": {"xaytune.io/submission-digest": "sha256:0"}}},
        {"rayjob": {"annotations": {"xaytune.io/stop-requested": "now"}}},
        {"rayjob": {"labels": {"xaytune.request_digest": "sha256:0"}}},
        {"rayjob": {"annotations": {"xaytune.placement_digest": "sha256:0"}}},
        {"rayjob": {"annotations": {"xaytune.target_id": "ra_x"}}},
        {"rayjob": {"labels": {"kueue.x-k8s.io/queue-name": "gpu-queue"}}},
        {"rayjob": {"labels": {"kueue.x-k8s.io/priority-class": "production"}}},
        {"rayjob": {"annotations": {"kueue.x-k8s.io/queue-name": "gpu-queue"}}},
        {"rayjob": {"annotations": {"ray.io/ft-enabled": "true"}}},
        {"cluster": _EPHEMERAL, "rayjob": {"deletion_policy": "delete-self"}},
        {"cluster": _EPHEMERAL, "rayjob": {"deletion_policy": "DeleteSelf"}},
        {"cluster": _EPHEMERAL, "rayjob": {"deletion_policy": "keep-cluster"}},
        {"rayjob": {"labels": {"app.kubernetes.io/managed-by": "me"}}},
        {"rayjob": {"labels": {"queue": "not a label value"}}},
        {"rayjob": {"labels": {"queue": 3}}},
        {"rayjob": {"spec": {"suspend": True}}},  # no arbitrary RayJob fields
        {"namespace": "Not_A_Namespace"},
        {"context": "x", "token": "s3cr3t"},
        {
            "cluster": {
                "kind": "existing",
                "selector": {"ray.io/cluster": "a"},
                "rayClusterSpec": {},
            }
        },
    ],
    ids=[
        "selector-without-cluster",
        "template-without-head",
        "ephemeral-without-policy",
        "existing-with-policy",
        "ttl-without-delete",
        "reserved-label",
        "reserved-annotation",
        "reserved-submission-digest",
        "reserved-stop-marker",
        "reserved-request-digest",
        "reserved-placement-digest",
        "reserved-target",
        "kueue-queue-label",
        "kueue-priority-label",
        "kueue-annotation",
        "kuberay-control-annotation",
        "delete-self",
        "delete-self-kuberay-spelling",
        "keep-cluster",
        "managed-by",
        "invalid-label-value",
        "non-string-label",
        "arbitrary-spec",
        "namespace",
        "unknown-key",
        "unmodelled-cluster-field",
    ],
)
def test_kuberay_configuration_is_typed_and_closed(change: dict[str, Any]) -> None:
    config = {**_kuberay(), **change}
    if "context" not in change:
        config["context"] = "kind-xaytune"
    with pytest.raises(ValueError):
        KubeRayConfig.model_validate(config)


def test_the_context_must_be_stated_even_when_it_is_in_cluster() -> None:
    config = _kuberay()
    del config["context"]
    with pytest.raises(ValueError):
        KubeRayConfig.model_validate(config)


def test_a_runtime_config_names_its_submission_and_address_stays_shorthand(tmp_path: Path) -> None:
    common = {"runtime_env": {}, "shared_state_root": str(tmp_path)}
    shorthand = RayJobsConfig.model_validate({"address": "http://h:8265", **common})
    explicit = RayJobsConfig.model_validate(
        {"submission": {"kind": "ray-jobs", "address": "http://h:8265"}, **common}
    )
    assert shorthand == explicit
    assert isinstance(submission_backend(shorthand.submission), RayJobsBackend)
    kuberay = RayTrainConfig.model_validate({"submission": _kuberay(), **common})
    assert isinstance(submission_backend(kuberay.submission), KubeRayJobsBackend)
    with pytest.raises(ValueError):
        RayJobsConfig.model_validate(
            {"address": "http://h:8265", "submission": _kuberay(), **common}
        )
    # The hosts' factories build either from configuration alone, contacting nothing.
    assert isinstance(ray_jobs_runtime({"submission": _kuberay(), **common}), RayJobsRuntime)
    assert isinstance(ray_train_runtime({"submission": _kuberay(), **common}), RayTrainRuntime)


# ---- the runtimes, unchanged, on KubeRay -----------------------------------------------------


def test_ray_jobs_runtime_runs_a_workload_once_on_kuberay(
    root: Path, kube: FakeKubernetes, jobs: ProcessJobs
) -> None:
    runtime = _jobs_runtime(root, kube)
    operation = OperationId.generate()

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, _plan(_TRAINS))
        assert await runtime.submit_or_get(operation, _plan(_TRAINS)) == ref
        status = await _settle(runtime, ref)
        assert status.state == "succeeded", status
        events = [event async for event in runtime.watch(ref)]
        assert _types(events)[-1] == "TrainingCompleted"
        outcome = await runtime.lookup_operation(operation)
        assert outcome is not None and outcome.disposition == "completed"

    asyncio.run(scenario())
    assert len(kube.objects) == 1 and jobs.submissions == 1
    (rayjob,) = kube.objects.values()
    assert rayjob["metadata"]["name"] == rayjob_name(str(operation))
    accepted = read_json(root / str(operation) / ACCEPTED)
    assert accepted is not None
    assert accepted["placement_digest"] == _backend(kube).placement_digest
    assert rayjob["spec"]["metadata"]["xaytune.placement_digest"] == accepted["placement_digest"]


def test_ray_jobs_runtime_reports_a_failed_worker_on_kuberay(
    root: Path, kube: FakeKubernetes
) -> None:
    runtime = _jobs_runtime(root, kube)

    async def scenario() -> None:
        ref = await runtime.submit_or_get(OperationId.generate(), _plan(_FAILS))
        status = await _settle(runtime, ref)
        assert status.state == "failed" and status.exit_code == 3

    asyncio.run(scenario())


def test_ray_train_runtime_trains_as_a_group_on_kuberay(root: Path, kube: FakeKubernetes) -> None:
    settings = {"submission": _kuberay(), "runtime_env": {}, "shared_state_root": str(root)}
    runtime = RayTrainRuntime(RayTrainConfig.model_validate(settings), submission=_backend(kube))
    plan = _plan(_TRAINS, runtime="ray-train", resources=ResourceRequirements(workers=2))

    async def scenario() -> None:
        ref = await runtime.submit_or_get(OperationId.generate(), plan)
        status = await _settle(runtime, ref)
        assert status.state == "succeeded", status
        events = [event async for event in runtime.watch(ref)]
        assert _types(events)[0] == "WorkerReady"
        assert _types(events)[-1] == "TrainingCompleted"

    asyncio.run(scenario())
    (rayjob,) = kube.objects.values()
    assert "xaytune.ray.runtime.train_driver" in rayjob["spec"]["entrypoint"]


def test_a_restarted_controller_adopts_the_same_rayjob(
    root: Path, kube: FakeKubernetes, jobs: ProcessJobs
) -> None:
    operation = OperationId.generate()

    async def scenario() -> None:
        first = _jobs_runtime(root, kube)
        ref = await first.submit_or_get(operation, _plan(_TRAINS))
        # The controller dies. Another process, another backend, the same cluster:
        second = _jobs_runtime(root, kube)
        outcome = await second.lookup_operation(operation)
        assert outcome is not None and outcome.runtime_ref == ref
        assert await second.submit_or_get(operation, _plan(_TRAINS)) == ref
        assert (await _settle(second, ref)).state == "succeeded"

    asyncio.run(scenario())
    assert kube.creates == 1 and len(kube.objects) == 1 and jobs.submissions == 1


def test_a_create_whose_answer_was_lost_is_found_by_the_retry(
    root: Path, kube: FakeKubernetes, jobs: ProcessJobs
) -> None:
    operation = OperationId.generate()
    kube.lose_create_response = True

    async def scenario() -> None:
        runtime = _jobs_runtime(root, kube)
        ref = await runtime.submit_or_get(operation, _plan(_TRAINS))
        kube.lose_create_response = False
        assert await _jobs_runtime(root, kube).submit_or_get(operation, _plan(_TRAINS)) == ref
        assert (await _settle(runtime, ref)).state == "succeeded"

    asyncio.run(scenario())
    assert len(kube.objects) == 1 and jobs.submissions == 1


def test_another_request_or_placement_under_the_same_operation_is_a_conflict(
    root: Path, kube: FakeKubernetes
) -> None:
    operation = OperationId.generate()
    elsewhere = {"kind": "existing", "selector": {"ray.io/cluster": "other"}}

    async def scenario() -> None:
        await _jobs_runtime(root, kube).submit_or_get(operation, _plan(_TRAINS))
        with pytest.raises(IdempotencyConflictError) as request:
            await _jobs_runtime(root, kube).submit_or_get(operation, _plan(_FAILS))
        assert request.value.differing == ("request_digest",)
        with pytest.raises(IdempotencyConflictError) as placement:
            await _jobs_runtime(root, kube, cluster=elsewhere).submit_or_get(
                operation, _plan(_TRAINS)
            )
        assert placement.value.differing == ("placement",)
        with pytest.raises(IdempotencyConflictError):
            await _jobs_runtime(root, kube, labels={"team": "x"}).submit_or_get(
                operation, _plan(_TRAINS)
            )

    asyncio.run(scenario())
    assert len(kube.objects) == 1


def test_a_placement_change_is_refused_even_once_the_rayjob_is_gone(
    root: Path, kube: FakeKubernetes
) -> None:
    """``accepted.json`` remembers the placement, as it remembers the request."""
    operation = OperationId.generate()

    async def scenario() -> None:
        runtime = _jobs_runtime(root, kube)
        ref = await runtime.submit_or_get(operation, _plan(_TRAINS))
        assert (await _settle(runtime, ref)).state == "succeeded"
        kube.objects.clear()  # someone deleted the RayJob
        with pytest.raises(IdempotencyConflictError):
            await _jobs_runtime(root, kube, cluster=_EPHEMERAL).submit_or_get(
                operation, _plan(_TRAINS)
            )
        outcome = await _jobs_runtime(root, kube).lookup_operation(operation)
        assert outcome is not None and outcome.disposition == "completed"

    asyncio.run(scenario())
    assert kube.creates == 1


def test_an_unreachable_api_server_is_never_never_received(
    root: Path, kube: FakeKubernetes
) -> None:
    operation = OperationId.generate()

    async def scenario() -> None:
        runtime = _jobs_runtime(root, kube)
        kube.unreachable = True
        with pytest.raises(RayUnavailableError):
            await runtime.lookup_operation(operation)
        with pytest.raises(RayUnavailableError):
            await runtime.submit_or_get(operation, _plan(_TRAINS))
        kube.unreachable = False
        assert await runtime.lookup_operation(operation) is None  # a real 404, no evidence

    asyncio.run(scenario())


def test_a_resource_kuberay_cannot_express_is_refused_and_recorded(
    root: Path, kube: FakeKubernetes
) -> None:
    operation = OperationId.generate()
    plan = _plan(_TRAINS, resources=ResourceRequirements(memory_bytes=2**30))

    async def scenario() -> None:
        runtime = _jobs_runtime(root, kube)
        with pytest.raises(UnsupportedPlanError, match="memory"):
            await runtime.submit_or_get(operation, plan)
        with pytest.raises(UnsupportedPlanError):
            await runtime.submit_or_get(operation, plan)
        outcome = await runtime.lookup_operation(operation)
        assert outcome is not None and outcome.disposition == "rejected"

    asyncio.run(scenario())
    assert not kube.objects and kube.creates == 0
    assert read_json(root / str(operation) / REJECTED) is not None


def test_cancelling_a_running_workload_on_kuberay_is_graceful(
    root: Path, kube: FakeKubernetes
) -> None:
    sleeps = "writer.write(t.TrainingStartedPayload())\ntime.sleep(60)\n"

    async def scenario() -> None:
        runtime = _jobs_runtime(root, kube)
        ref = await runtime.submit_or_get(OperationId.generate(), _plan(sleeps))
        while (await runtime.get_status(ref)).state != "running":
            await asyncio.sleep(0.05)
        cancel = OperationId.generate()
        await runtime.cancel(ref, cancel)
        await runtime.cancel(ref, cancel)
        assert (await _settle(runtime, ref)).state == "cancelled"

    asyncio.run(scenario())
    assert not kube.patches, "a started workload is cancelled by its supervisor, its RayJob kept"


def test_cancelling_an_ephemeral_rayjob_before_it_runs_suspends_it_and_is_cancelled(
    root: Path,
) -> None:
    """The cancellation KubeRay records: suspended, its cluster gone, the RayJob kept."""
    held = ProcessJobs(hold=True)
    kube = FakeKubernetes(held)
    operation = OperationId.generate()

    async def scenario() -> None:
        runtime = _jobs_runtime(root, kube, cluster=_EPHEMERAL)
        ref = await runtime.submit_or_get(operation, _plan(_TRAINS))
        assert (await runtime.get_status(ref)).state == "pending"
        cancel = OperationId.generate()
        await runtime.cancel(ref, cancel)
        await runtime.cancel(ref, cancel)
        assert len(kube.patches) == 1 and len(kube.objects) == 1
        status = await runtime.get_status(ref)
        assert status.state == "cancelled", status
        # A restarted controller reads the same RayJob, and the same ending.
        restarted = _jobs_runtime(root, kube, cluster=_EPHEMERAL)
        assert (await restarted.get_status(ref)).state == "cancelled"
        outcome = await restarted.lookup_operation(operation)
        assert outcome is not None and outcome.disposition == "completed"
        assert outcome.status is not None and outcome.status.state == "cancelled"
        assert await restarted.submit_or_get(operation, _plan(_TRAINS)) == ref
        held.release(str(operation))  # resources free up: a stopped job does not start
        assert (await restarted.get_status(ref)).state == "cancelled"

    try:
        asyncio.run(scenario())
    finally:
        held.close()
    assert kube.creates == 1 and held.submissions == 1
    assert read_json(WorkloadPaths(root / str(operation)).started) is None


@pytest.mark.parametrize("runtime_kind", ["ray-jobs", "ray-train"])
def test_cancelling_a_rayjob_on_an_existing_cluster_before_it_runs_waits_then_never_trains(
    root: Path, runtime_kind: str
) -> None:
    """Nothing stops it there without erasing its record: it stays pending, truthfully.

    When KubeRay does start it, the supervisor (or the train driver) finds the
    durable cancellation first, and the worker never runs.
    """
    held = ProcessJobs(hold=True)
    kube = FakeKubernetes(held)
    operation = OperationId.generate()
    settings = {"submission": _kuberay(), "runtime_env": {}, "shared_state_root": str(root)}
    if runtime_kind == "ray-jobs":
        runtime: Any = _jobs_runtime(root, kube)
        plan = _plan(_TRAINS)
    else:
        runtime = RayTrainRuntime(
            RayTrainConfig.model_validate(settings), submission=_backend(kube)
        )
        plan = _plan(_TRAINS, runtime="ray-train", resources=ResourceRequirements(workers=2))

    async def scenario() -> None:
        ref = await runtime.submit_or_get(operation, plan)
        await runtime.cancel(ref, OperationId.generate())
        assert not kube.patches and len(kube.objects) == 1
        status = await runtime.get_status(ref)
        assert status.state == "pending" and "cancellation requested" in (status.detail or "")
        outcome = await runtime.lookup_operation(operation)
        assert outcome is not None and outcome.disposition == "accepted"
        held.release(str(operation))  # KubeRay starts it after all
        status = await _settle(runtime, ref)
        assert status.state == "cancelled", status
        events = [event async for event in runtime.watch(ref)]
        assert "TrainingStarted" not in _types(events)

    try:
        asyncio.run(scenario())
    finally:
        held.close()
    assert read_json(WorkloadPaths(root / str(operation)).started) is None
    assert kube.creates == 1 and held.submissions == 1


# ---- boundaries ------------------------------------------------------------------------------


def test_nothing_above_the_submission_backend_loads_kubernetes() -> None:
    probe = (
        "import sys\n"
        "import xaytune.core, xaytune.planning, xaytune.agent, xaytune.storage\n"
        "import xaytune.experiment, xaytune.daemon, xaytune.runtimes, xaytune.ray\n"
        "from xaytune.ray import KubeRayConfig, KubeRayJobsBackend\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('kubernetes', 'ray')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
    package = Path(__file__).resolve().parents[2] / "xaytune"
    importers = sorted(
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if any(
            (
                isinstance(node, ast.Import)
                and any(a.name.split(".")[0] == "kubernetes" for a in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").split(".")[0] == "kubernetes"
            )
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
    )
    assert importers == ["ray/submission/kuberay.py"]
    # Nor does anything outside the submission package import the KubeRay backend --
    # the runtimes included: they are handed a backend, or build the one configured.
    kuberay_importers = sorted(
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if any(
            isinstance(node, ast.ImportFrom)
            and (
                (node.module or "") == "xaytune.ray.submission.kuberay"
                or any(alias.name.startswith("KubeRay") for alias in node.names)
            )
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
    )
    assert kuberay_importers == ["ray/__init__.py", "ray/submission/__init__.py"]
