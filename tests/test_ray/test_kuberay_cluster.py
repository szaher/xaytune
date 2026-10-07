"""KubeRayJobsBackend against a real Kubernetes API server with the KubeRay operator installed.

What the fake cannot prove: that KubeRay's CRDs accept the RayJobs this
backend writes, that the API server answers "already exists" and "not found"
the way the backend reads them, that a suspended ephemeral RayJob loses its
cluster and keeps its record, and -- on a RayCluster this module creates from
Ray's own image -- that the statuses KubeRay reports for real Ray jobs read as
Ray's: a job that finished stays finished, one that failed is failed, and an
ephemeral cluster runs its job and goes while the RayJob stays. Xaytune's
supervisor does not run here (the image has no Xaytune, and no shared
filesystem is mounted); the matrix runs the runtimes on ``ProcessJobs``
behind a fake operator.

Needs ``XAYTUNE_KUBERAY_CONTEXT``: a kubeconfig context (``KUBECONFIG`` as
usual) whose cluster runs the KubeRay operator. Every test works in a
namespace of its own, created here and deleted here; nothing else in the
cluster is touched. Skipped without it, failed under
``XAYTUNE_REQUIRE_KUBERAY=1`` (the ``kuberay`` CI job).
"""

from __future__ import annotations

import os
import secrets
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from xaytune.core.errors import IdempotencyConflictError
from xaytune.ray import KubeRayConfig, KubeRayJobsBackend, RayUnavailableError
from xaytune.ray.submission.kuberay import GROUP, PLURAL, VERSION, rayjob_name

CONTEXT_ENV = "XAYTUNE_KUBERAY_CONTEXT"
RAY_IMAGE = os.environ.get("XAYTUNE_KUBERAY_RAY_IMAGE", "rayproject/ray:2.59.0-py312-cpu")
RAY_CLUSTER = "xaytune-ray"

pytestmark = pytest.mark.kuberay

_TEMPLATE = {
    "headGroupSpec": {
        "rayStartParams": {},
        "template": {
            "spec": {
                "containers": [
                    {
                        "name": "ray-head",
                        # Never pulled to completion: the RayJob is stopped first.
                        "image": "registry.invalid/xaytune-ray:never",
                        "resources": {"requests": {"cpu": "10m", "memory": "16Mi"}},
                    }
                ]
            }
        },
    },
    # A worker group too, so the defaults KubeRay adds to one (numOfHosts,
    # priority, scaleStrategy, ...) are read back on a real cluster.
    "workerGroupSpecs": [
        {
            "groupName": "workers",
            "replicas": 0,
            "minReplicas": 0,
            "maxReplicas": 1,
            "rayStartParams": {},
            "template": {
                "spec": {
                    "containers": [
                        {"name": "ray-worker", "image": "registry.invalid/xaytune-ray:never"}
                    ]
                }
            },
        }
    ],
}


class Cluster:
    def __init__(self, context: str, namespace: str, api: Any) -> None:
        self.context = context
        self.namespace = namespace
        self.api = api

    def backend(self, **cluster: Any) -> KubeRayJobsBackend:
        settings: dict[str, Any] = {}
        if cluster.get("kind") == "ephemeral":
            settings["deletion_policy"] = "delete-cluster"
        config = KubeRayConfig.model_validate(
            {
                "kind": "kuberay",
                "namespace": self.namespace,
                "context": self.context,
                "cluster": cluster
                or {"kind": "existing", "selector": {"ray.io/cluster": "absent"}},
                "rayjob": settings,
            }
        )
        return KubeRayJobsBackend(config)  # a real client, from the context

    def rayjobs(self) -> list[dict[str, Any]]:
        listed = self.api.list_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL)
        return list(listed["items"])

    def wait_until(self, predicate: Any, what: str, timeout: float = 90) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.5)
        raise AssertionError(f"never: {what}")


@pytest.fixture(scope="module")
def cluster() -> Iterator[Cluster]:
    context = os.environ.get(CONTEXT_ENV)
    if not context:
        pytest.skip(f"{CONTEXT_ENV} names no Kubernetes context with KubeRay")
    kubernetes = pytest.importorskip("kubernetes", reason="needs xaytune[kuberay]")
    api_client = kubernetes.config.new_client_from_config(context=context)
    core = kubernetes.client.CoreV1Api(api_client)
    namespace = f"xaytune-test-{secrets.token_hex(4)}"
    core.create_namespace({"metadata": {"name": namespace}})
    try:
        yield Cluster(context, namespace, kubernetes.client.CustomObjectsApi(api_client))
    finally:
        # Only the namespace this module created.
        core.delete_namespace(namespace)


def _submit(backend: KubeRayJobsBackend, submission_id: str, **overrides: Any) -> None:
    arguments: dict[str, Any] = {
        "metadata": {"xaytune.request_digest": "sha256:a"},
        "resources": {"entrypoint_num_cpus": 1},
        "runtime_env": {"env_vars": {"XAYTUNE": "1"}},
    }
    arguments.update(overrides)
    entrypoint = arguments.pop(
        "entrypoint", f"python -m xaytune.ray.runtime.supervisor /shared/{submission_id}"
    )
    backend.submit(submission_id, entrypoint, **arguments)


def test_kuberay_accepts_the_rayjob_once_and_a_restarted_backend_adopts_it(
    cluster: Cluster,
) -> None:
    submission_id = f"op_{secrets.token_hex(6).upper()}"
    _submit(cluster.backend(), submission_id)
    _submit(cluster.backend(), submission_id)  # a retry: AlreadyExists, the same RayJob
    mine = [
        item for item in cluster.rayjobs() if item["metadata"]["name"] == rayjob_name(submission_id)
    ]
    assert len(mine) == 1
    assert mine[0]["spec"]["jobId"] == submission_id
    assert mine[0]["spec"]["entrypointNumCpus"] == 1
    job = cluster.backend().info(submission_id)  # a reconstructed backend
    assert job is not None and job.status == "PENDING"  # no such RayCluster to run on
    assert job.metadata == {"xaytune.request_digest": "sha256:a"}


def test_another_submission_or_placement_under_the_id_is_a_conflict(cluster: Cluster) -> None:
    submission_id = f"op_{secrets.token_hex(6).upper()}"
    _submit(cluster.backend(), submission_id)
    with pytest.raises(IdempotencyConflictError):
        _submit(cluster.backend(), submission_id, entrypoint="python -m elsewhere")
    with pytest.raises(IdempotencyConflictError):
        _submit(
            cluster.backend(kind="existing", selector={"ray.io/cluster": "other"}),
            submission_id,
        )
    names = [item["metadata"]["name"] for item in cluster.rayjobs()]
    assert names.count(rayjob_name(submission_id)) == 1


def test_not_found_is_absence_and_stopping_on_an_existing_cluster_keeps_the_rayjob(
    cluster: Cluster,
) -> None:
    backend = cluster.backend()
    assert backend.info("op_NEVER_SUBMITTED") is None
    submission_id = f"op_{secrets.token_hex(6).upper()}"
    _submit(backend, submission_id)
    name = rayjob_name(submission_id)
    cluster.wait_until(
        lambda: (
            "ray.io/rayjob-finalizer"
            in (
                cluster.api.get_namespaced_custom_object(
                    GROUP, VERSION, cluster.namespace, PLURAL, name
                )["metadata"].get("finalizers")
                or []
            )
        ),
        "KubeRay took the RayJob on",
    )
    backend.stop(submission_id)
    backend.stop(submission_id)
    stored = cluster.api.get_namespaced_custom_object(
        GROUP, VERSION, cluster.namespace, PLURAL, name
    )
    assert not stored["spec"].get("suspend")  # KubeRay would ignore it here
    assert "deletionTimestamp" not in stored["metadata"]
    job = backend.info(submission_id)
    assert job is not None and job.status == "PENDING"  # truthfully: not yet started


def test_stopping_an_ephemeral_rayjob_suspends_it_its_cluster_goes_and_it_reads_stopped(
    cluster: Cluster,
) -> None:
    backend = cluster.backend(kind="ephemeral", template=_TEMPLATE)
    submission_id = f"op_{secrets.token_hex(6).upper()}"
    _submit(backend, submission_id)
    name = rayjob_name(submission_id)

    def owned_clusters() -> list[dict[str, Any]]:
        listed = cluster.api.list_namespaced_custom_object(
            GROUP, VERSION, cluster.namespace, "rayclusters"
        )
        return [
            item
            for item in listed["items"]
            if any(
                owner.get("kind") == "RayJob" and owner.get("name") == name
                for owner in item["metadata"].get("ownerReferences", [])
            )
        ]

    cluster.wait_until(lambda: len(owned_clusters()) == 1, "KubeRay created the RayJob's cluster")
    job = backend.info(submission_id)
    assert job is not None and job.status == "PENDING"
    backend.stop(submission_id)
    backend.stop(submission_id)

    def stopped() -> bool:
        job = backend.info(submission_id)
        return job is not None and job.status == "STOPPED"

    cluster.wait_until(stopped, "KubeRay suspended the RayJob")
    cluster.wait_until(lambda: not owned_clusters(), "the RayJob's own cluster was deleted")
    reconstructed = cluster.backend(kind="ephemeral", template=_TEMPLATE)
    job = reconstructed.info(submission_id)  # the RayJob stays, and so does its ending
    assert job is not None and job.status == "STOPPED"


def test_a_rayjob_edited_after_it_was_created_is_refused_and_defaults_are_not_edits(
    cluster: Cluster,
) -> None:
    for backend in (
        cluster.backend(),
        cluster.backend(kind="ephemeral", template=_TEMPLATE),
    ):
        submission_id = f"op_{secrets.token_hex(6).upper()}"
        _submit(backend, submission_id)
        name = rayjob_name(submission_id)
        cluster.wait_until(
            lambda name=name: (
                (
                    cluster.api.get_namespaced_custom_object(
                        GROUP, VERSION, cluster.namespace, PLURAL, name
                    ).get("status")
                    or {}
                ).get("jobDeploymentStatus")
                is not None
            ),
            "KubeRay reconciled the RayJob",
        )
        # As stored -- defaulted by the API server and KubeRay -- it is still the RayJob bound.
        job = backend.info(submission_id)
        assert job is not None and job.status == "PENDING"
        cluster.api.patch_namespaced_custom_object(
            GROUP,
            VERSION,
            cluster.namespace,
            PLURAL,
            name,
            {"spec": {"entrypoint": "python -c 'print(2)'"}},
            _content_type="application/merge-patch+json",
        )
        with pytest.raises(IdempotencyConflictError):
            backend.info(submission_id)
        with pytest.raises(IdempotencyConflictError):
            _submit(backend, submission_id)


class _WritesBeforeThePatch:
    """The real API, with another writer's real write landing just before the first patch."""

    def __init__(self, api: Any, write: Any) -> None:
        self._api = api
        self._write: Any = write

    def __getattr__(self, name: str) -> Any:
        return getattr(self._api, name)

    def patch_namespaced_custom_object(self, *args: Any, **kwargs: Any) -> Any:
        if self._write is not None:
            write, self._write = self._write, None
            write()
        return self._api.patch_namespaced_custom_object(*args, **kwargs)


@pytest.mark.parametrize("write", ["identity", "status"])
def test_the_api_server_refuses_a_suspension_of_a_version_that_was_not_verified(
    cluster: Cluster, write: str
) -> None:
    """GET, verify, someone writes, PATCH: Kubernetes refuses the stale patch (409), and the
    RayJob is suspended only if it is still the one bound."""
    submission_id = f"op_{secrets.token_hex(6).upper()}"
    _submit(cluster.backend(kind="ephemeral", template=_TEMPLATE), submission_id)
    name = rayjob_name(submission_id)

    def concurrent() -> None:
        if write == "identity":
            cluster.api.patch_namespaced_custom_object(
                GROUP,
                VERSION,
                cluster.namespace,
                PLURAL,
                name,
                {"metadata": {"labels": {"team": "someone-else"}}},
                _content_type="application/merge-patch+json",
            )
        else:
            cluster.api.patch_namespaced_custom_object_status(
                GROUP,
                VERSION,
                cluster.namespace,
                PLURAL,
                name,
                {"status": {"message": "a status write, not an edit"}},
                _content_type="application/merge-patch+json",
            )

    config = cluster.backend(kind="ephemeral", template=_TEMPLATE).config
    backend = KubeRayJobsBackend(
        config, api=_WritesBeforeThePatch(cluster.backend()._objects(), concurrent)
    )
    if write == "identity":
        with pytest.raises(IdempotencyConflictError):
            backend.stop(submission_id)
        stored = cluster.api.get_namespaced_custom_object(
            GROUP, VERSION, cluster.namespace, PLURAL, name
        )
        assert not stored["spec"].get("suspend")
        assert "xaytune.io/stop-requested" not in stored["metadata"].get("annotations", {})
    else:
        backend.stop(submission_id)

        def stopped() -> bool:
            job = cluster.backend(kind="ephemeral", template=_TEMPLATE).info(submission_id)
            return job is not None and job.status == "STOPPED"

        cluster.wait_until(stopped, "the re-verified RayJob was suspended")


def test_an_unreachable_api_server_is_unavailable_never_absent(
    cluster: Cluster, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kubeconfig = tmp_path / "unreachable"
    kubeconfig.write_text(
        "apiVersion: v1\nkind: Config\n"
        "clusters: [{name: nowhere, cluster: {server: 'https://127.0.0.1:1'}}]\n"
        "users: [{name: nobody, user: {token: placeholder}}]\n"
        "contexts: [{name: nowhere, context: {cluster: nowhere, user: nobody}}]\n"
        "current-context: nowhere\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KUBECONFIG", str(kubeconfig))
    config = KubeRayConfig.model_validate(
        {
            "kind": "kuberay",
            "namespace": cluster.namespace,
            "context": "nowhere",
            "cluster": {"kind": "existing", "selector": {"ray.io/cluster": "absent"}},
        }
    )
    backend = KubeRayJobsBackend(config)
    with pytest.raises(RayUnavailableError):
        backend.info("op_ANY")
    with pytest.raises(RayUnavailableError):
        backend.stop("op_ANY")
    missing = KubeRayConfig.model_validate({**config.model_dump(), "context": "no-such-context"})
    with pytest.raises(RayUnavailableError, match="no-such-context"):
        KubeRayJobsBackend(missing).info("op_ANY")


# ---- real Ray jobs ------------------------------------------------------------------------------


def _ray_pod(name: str, memory: str) -> dict[str, Any]:
    return {
        "spec": {
            "containers": [
                {
                    "name": name,
                    "image": RAY_IMAGE,
                    "imagePullPolicy": "IfNotPresent",
                    # A test cluster: Ray's memory monitor would otherwise kill
                    # jobs on a head that the dashboard keeps near its limit.
                    "env": [{"name": "RAY_memory_monitor_refresh_ms", "value": "0"}],
                    "resources": {
                        "requests": {"cpu": "250m", "memory": memory},
                        "limits": {"memory": memory},
                    },
                }
            ]
        }
    }


_RAY_HEAD = {
    "rayVersion": "2.59.0",
    "headGroupSpec": {
        # Ray sizes its object store from the node's memory, not the pod's
        # limit, and is OOM-killed at start without a bound of its own.
        "rayStartParams": {"num-cpus": "2", "object-store-memory": "300000000"},
        "template": _ray_pod("ray-head", "4Gi"),
    },
}


@pytest.fixture(scope="module")
def ray_cluster(cluster: Cluster) -> Cluster:
    """A RayCluster of this module's own, in its namespace, that RayJobs select."""
    cluster.api.create_namespaced_custom_object(
        GROUP,
        VERSION,
        cluster.namespace,
        "rayclusters",
        {
            "apiVersion": f"{GROUP}/{VERSION}",
            "kind": "RayCluster",
            "metadata": {"name": RAY_CLUSTER},
            "spec": _RAY_HEAD,
        },
    )

    def ready() -> bool:
        found = cluster.api.get_namespaced_custom_object(
            GROUP, VERSION, cluster.namespace, "rayclusters", RAY_CLUSTER
        )
        return (found.get("status") or {}).get("state") == "ready"

    cluster.wait_until(ready, f"RayCluster {RAY_CLUSTER} is ready", timeout=600)
    return cluster


def _finished(backend: KubeRayJobsBackend, submission_id: str, cluster: Cluster) -> Any:
    def ended() -> bool:
        job = backend.info(submission_id)
        return job is not None and job.status in ("SUCCEEDED", "FAILED", "STOPPED")

    cluster.wait_until(ended, f"{submission_id} ended", timeout=300)
    return backend.info(submission_id)


_ON_RAY = {"kind": "existing", "selector": {"ray.io/cluster": RAY_CLUSTER}}


def test_on_ray_a_finished_rayjob_stays_finished_across_reconstruction_and_stop(
    ray_cluster: Cluster,
) -> None:
    submission_id = f"op_{secrets.token_hex(6).upper()}"
    _submit(ray_cluster.backend(**_ON_RAY), submission_id, entrypoint="python -c 'print(1)'")
    job = _finished(ray_cluster.backend(**_ON_RAY), submission_id, ray_cluster)
    assert job.status == "SUCCEEDED", job
    reconstructed = ray_cluster.backend(**_ON_RAY)
    reconstructed.stop(submission_id)  # finished: left alone, and its record kept
    job = ray_cluster.backend(**_ON_RAY).info(submission_id)
    assert job is not None and job.status == "SUCCEEDED"
    assert job.metadata == {"xaytune.request_digest": "sha256:a"}


def test_on_ray_a_failed_job_is_failed(ray_cluster: Cluster) -> None:
    submission_id = f"op_{secrets.token_hex(6).upper()}"
    backend = ray_cluster.backend(**_ON_RAY)
    _submit(backend, submission_id, entrypoint="python -c 'import sys; sys.exit(3)'")
    job = _finished(backend, submission_id, ray_cluster)
    assert job.status == "FAILED", job


def test_on_ray_an_ephemeral_cluster_runs_the_job_and_goes_while_the_rayjob_stays(
    ray_cluster: Cluster,
) -> None:
    submission_id = f"op_{secrets.token_hex(6).upper()}"
    backend = ray_cluster.backend(kind="ephemeral", template=_RAY_HEAD)
    _submit(backend, submission_id, entrypoint="python -c 'print(1)'")
    job = _finished(backend, submission_id, ray_cluster)
    assert job.status == "SUCCEEDED"
    stored = ray_cluster.api.get_namespaced_custom_object(
        GROUP, VERSION, ray_cluster.namespace, PLURAL, rayjob_name(submission_id)
    )
    own = stored["status"]["rayClusterName"]

    def cluster_gone() -> bool:
        listed = ray_cluster.api.list_namespaced_custom_object(
            GROUP, VERSION, ray_cluster.namespace, "rayclusters"
        )
        return own not in [item["metadata"]["name"] for item in listed["items"]]

    ray_cluster.wait_until(cluster_gone, "the RayJob's own cluster was deleted", timeout=180)
    job = ray_cluster.backend(kind="ephemeral", template=_RAY_HEAD).info(submission_id)
    assert job is not None and job.status == "SUCCEEDED"


def test_the_kuberay_tests_are_required_where_kuberay_is() -> None:
    """CI's kuberay job sets both; a skip there would hide a suite that never ran."""
    if os.environ.get("XAYTUNE_REQUIRE_KUBERAY") == "1":
        assert os.environ.get(CONTEXT_ENV)
