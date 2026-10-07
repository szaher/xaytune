"""A Kubernetes API server holding KubeRay ``RayJob`` resources, with an operator behind it.

``FakeKubernetes`` answers the three ``CustomObjectsApi`` calls
``KubeRayJobsBackend`` makes, the way the API server does where the backend
depends on it: one object per name (``409 AlreadyExists``), a ``NotFound``
``Status`` naming the object, a plain-text 404 for a resource type that is
not installed, and errors that are not answers at all. Behind it, an optional
"operator" runs each RayJob's Ray job on :class:`ProcessJobs` -- real
supervisors, real workers -- and mirrors the job's status onto the RayJob.
Suspension is KubeRay 1.7's, as observed on a real cluster: a RayJob that owns
its cluster (``rayClusterSpec`` with ``shutdownAfterJobFinishes``) goes to
``Suspended``, its cluster -- and so its job -- deleted, its ``jobStatus``
cleared, the RayJob kept; a RayJob that selects an existing cluster ignores
``suspend``. A stored RayJob carries the server metadata and the defaults a
real API server and KubeRay add. It answers no deletes: the backend must
never need one.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

from tests.test_ray.ray_support import ProcessJobs

GROUP, VERSION, PLURAL = "ray.io", "v1", "rayjobs"


class ApiError(Exception):
    """What ``kubernetes.client.ApiException`` carries: a status and the raw body."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"({status})")
        self.status = status
        self.body = body


def status_error(status: int, reason: str, name: str | None = None) -> ApiError:
    details = {"name": name, "group": GROUP, "kind": PLURAL} if name else {}
    body = {"kind": "Status", "status": "Failure", "reason": reason, "details": details}
    return ApiError(status, json.dumps(body))


class FakeKubernetes:
    """A ``CustomObjectsApi`` over an in-memory API server, optionally with a KubeRay operator."""

    def __init__(self, jobs: ProcessJobs | None = None) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.jobs = jobs
        self.creates = 0
        self.sent: list[dict[str, Any]] = []
        self.patches: list[tuple[str, dict[str, Any], str | None]] = []
        self.unreachable = False
        self.forbidden = False
        self.crds_installed = True
        self.lose_create_response = False
        self._versions = 0
        self._before_next_patch: Callable[[dict[str, Any]], None] | None = None

    # -- the three calls --------------------------------------------------

    def create_namespaced_custom_object(
        self, group: str, version: str, namespace: str, plural: str, body: dict[str, Any], **_: Any
    ) -> dict[str, Any]:
        self._answerable(group, version, plural)
        self.creates += 1
        self.sent.append(copy.deepcopy(body))
        name = body["metadata"]["name"]
        if (namespace, name) in self.objects:
            raise status_error(409, "AlreadyExists", name)
        stored = copy.deepcopy(body)
        stored["metadata"].update(
            uid=f"uid-{len(self.objects)}",
            resourceVersion=self._next_version(),
            generation=1,
            creationTimestamp="2026-10-07T00:00:00Z",
            managedFields=[{"manager": "xaytune", "operation": "Update"}],
            finalizers=["ray.io/rayjob-finalizer"],
        )
        _default(stored["spec"])
        stored["status"] = {}
        self.objects[(namespace, name)] = stored
        if self.jobs is not None:
            spec = stored["spec"]
            self.jobs.submit(
                spec["jobId"],
                spec["entrypoint"],
                metadata=spec.get("metadata", {}),
                resources={},
                runtime_env=json.loads(spec.get("runtimeEnvYAML") or "{}"),
            )
        if self.lose_create_response:
            # Created -- and the answer lost on the way back.
            raise ConnectionResetError("connection reset by peer")
        return copy.deepcopy(stored)

    def get_namespaced_custom_object(
        self, group: str, version: str, namespace: str, plural: str, name: str, **_: Any
    ) -> dict[str, Any]:
        self._answerable(group, version, plural)
        stored = self.objects.get((namespace, name))
        if stored is None:
            raise status_error(404, "NotFound", name)
        self._reconcile(stored)
        return copy.deepcopy(stored)

    def patch_namespaced_custom_object(
        self,
        group: str,
        version: str,
        namespace: str,
        plural: str,
        name: str,
        body: dict[str, Any],
        _content_type: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        self._answerable(group, version, plural)
        stored = self.objects.get((namespace, name))
        if stored is None:
            raise status_error(404, "NotFound", name)
        if self._before_next_patch is not None:
            # Another writer, after the caller read the RayJob and before this patch.
            concurrent, self._before_next_patch = self._before_next_patch, None
            concurrent(stored)
            stored["metadata"]["resourceVersion"] = self._next_version()
        precondition = (body.get("metadata") or {}).get("resourceVersion")
        if precondition is not None and precondition != stored["metadata"]["resourceVersion"]:
            raise status_error(409, "Conflict", name)
        self.patches.append((name, copy.deepcopy(body), _content_type))
        _merge(stored, body)
        stored["metadata"]["resourceVersion"] = self._next_version()
        spec = stored["spec"]
        if spec.get("suspend") and "rayClusterSpec" in spec:
            # KubeRay deletes the owned cluster -- the job with it -- and keeps the RayJob.
            if self.jobs is not None:
                self.jobs.stop(spec["jobId"])
            stored["status"] = {"jobDeploymentStatus": "Suspended"}
        return copy.deepcopy(stored)

    # -- the operator and the server's state ----------------------------------

    def set_status(self, namespace: str, name: str, **status: str) -> None:
        """Report a RayJob status as KubeRay would, for a RayJob with no operator behind it."""
        stored = self.objects[(namespace, name)]
        stored["status"] = dict(status)
        stored["metadata"]["resourceVersion"] = self._next_version()

    def concurrently(self, write: Callable[[dict[str, Any]], None]) -> None:
        """Have *write* change the stored RayJob just before the next patch reaches it."""
        self._before_next_patch = write

    def _next_version(self) -> str:
        self._versions += 1
        return str(self._versions)

    def _reconcile(self, stored: dict[str, Any]) -> None:
        if self.jobs is None:
            return
        if stored["spec"].get("suspend") and "rayClusterSpec" in stored["spec"]:
            return  # Suspended: no cluster, no job to mirror
        job = self.jobs.info(stored["spec"]["jobId"])
        if job is None:
            return
        terminal = job.status in ("STOPPED", "SUCCEEDED", "FAILED")
        stored["status"] = {
            "jobId": job.submission_id,
            "jobStatus": job.status,
            "jobDeploymentStatus": "Complete"
            if terminal
            else ("Running" if job.status == "RUNNING" else "Initializing"),
        }

    def _answerable(self, group: str, version: str, plural: str) -> None:
        if self.unreachable:
            raise ConnectionRefusedError("[Errno 61] Connection refused")
        if self.forbidden:
            raise status_error(403, "Forbidden")
        if not self.crds_installed or (group, version, plural) != (GROUP, VERSION, PLURAL):
            raise ApiError(404, "404 page not found")


def _merge(target: dict[str, Any], patch: dict[str, Any]) -> None:
    """A JSON merge patch (RFC 7386), as the API server applies one."""
    for key, value in patch.items():
        if value is None:
            target.pop(key, None)
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _default(spec: dict[str, Any]) -> None:
    """What the API server and KubeRay 1.7.1 add to a stored RayJob, as observed on kind."""
    spec.setdefault("ttlSecondsAfterFinished", 0)
    cluster = spec.get("rayClusterSpec")
    if cluster is None:
        return
    cluster["headGroupSpec"].setdefault("template", {}).setdefault("metadata", {})
    for group in cluster.get("workerGroupSpecs", []):
        group.setdefault("numOfHosts", 1)
        group.setdefault("priority", 0)
        group.setdefault("scaleStrategy", {})
        template = group.setdefault("template", {})
        template.setdefault("metadata", {})
        for container in template.get("spec", {}).get("containers", []):
            container.setdefault("resources", {})
