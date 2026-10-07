"""``KubeRayJobsBackend``: submission as a KubeRay ``RayJob`` custom resource.

```text
submit(submission_id, ...)
   ↓  Kubernetes API: create RayJob <name derived from submission_id>
RayJob (ray.io/v1)
   ↓  the KubeRay operator
   ├─ clusterSelector  → an existing RayCluster              (ExistingCluster)
   └─ rayClusterSpec   → a RayCluster the RayJob owns         (EphemeralCluster)
   ↓
Ray job, jobId = submission_id
```

The same :class:`~xaytune.ray.submission.protocol.RaySubmissionBackend` as
:class:`~xaytune.ray.submission.jobs.RayJobsBackend`, so ``RayJobsRuntime``
and ``RayTrainRuntime`` run on it unchanged and never learn Kubernetes is
there.

**Exactly once, through the API server's own uniqueness.** A RayJob's name is
a pure function of the submission id (:func:`rayjob_name`), and Kubernetes
refuses to create a second object under a name. A retry, or a controller
restarted after a crash, therefore finds the RayJob it created, never makes a
second one: a create answered "already exists" is accepted only if the
existing RayJob records this submission id *and* the digest of this exact
resource -- anything else is an
:class:`~xaytune.core.errors.IdempotencyConflictError`. And every read checks
the live RayJob against that digest (:func:`rayjob_identity`): one whose
cluster, entrypoint, environment or resources were edited after it was
created is refused, never adopted.

**Placement is identity.** Everything in :class:`KubeRayConfig` -- the
namespace and context, the cluster selected or the template a cluster is made
from, the RayJob's labels, annotations and deletion policy -- is in
:attr:`KubeRayJobsBackend.placement_digest`, which the runtime records with
the job. The same operation id with a different topology is refused, not
adopted. The configuration is typed: there is no field for an arbitrary
RayJob spec, so nothing reaches the RayJob that the identity does not cover.
Secret values never appear in it -- a template refers to a ``Secret`` by name.

**"No such RayJob" only when Kubernetes says exactly that.** ``info`` returns
``None`` for a ``NotFound`` naming this RayJob, and nothing else: an
unreachable API server, a forbidden request, or a 404 for the resource type
itself (KubeRay's CRDs not installed) raise
:class:`~xaytune.ray.submission.protocol.RayUnavailableError`.

**The RayJob is the record, and is never deleted.** Nothing here deletes a
RayJob, and no configuration lets KubeRay delete one (``DeleteSelf``): its
status is the evidence a controller reads after any restart -- including how
a job ended before its supervisor could record it. An ephemeral cluster is
deleted once its job finishes; the RayJob that owned it stays.

**Stopping keeps the evidence.** ``stop`` asks KubeRay to stop the way it
records: an ephemeral RayJob is *suspended* (``spec.suspend``, with
``xaytune.io/stop-requested`` on the RayJob), KubeRay deletes its cluster and
reports ``Suspended``, which -- Xaytune having asked -- reads as ``STOPPED``.
KubeRay does not suspend a RayJob that selects an existing cluster, and
deleting the RayJob would destroy its status, so such a RayJob is left as it
is: it stays ``PENDING`` until KubeRay starts it, and then the supervisor
finds the durable cancellation request and ends without running the worker.

**No platform scheduler, yet.** Kueue takes over a RayJob's admission and its
``spec.suspend`` once the RayJob carries Kueue's labels, so Kueue's keys
(``kueue.x-k8s.io/...``) -- and KubeRay's own (``ray.io/...``) -- are refused
in ``rayjob.labels`` and ``rayjob.annotations``. Queues, flavors and
workloads are the platform's, and admission, preemption and cancellation
under Kueue are a slice of their own.

Lazily imports the ``kubernetes`` client, and only to build one
(``pip install xaytune[kuberay]``).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import Field

from xaytune.core.clock import utc_now
from xaytune.core.errors import IdempotencyConflictError
from xaytune.core.fingerprint import fingerprint
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.ray.submission.jobs import _plain
from xaytune.ray.submission.protocol import (
    RayJob,
    RayJobStatus,
    RaySubmissionRefusedError,
    RayUnavailableError,
)

__all__ = [
    "GROUP",
    "PLURAL",
    "VERSION",
    "EphemeralCluster",
    "ExistingCluster",
    "KubeRayConfig",
    "KubeRayJobsBackend",
    "RayJobSettings",
    "rayjob_identity",
    "rayjob_name",
]

GROUP = "ray.io"
VERSION = "v1"
PLURAL = "rayjobs"

SUBMISSION_ID = "xaytune.io/submission-id"
"""Annotation: the submission id the RayJob was created for, in full."""
SUBMISSION_DIGEST = "xaytune.io/submission-digest"
"""Annotation: the RayJob's identity as created (:func:`rayjob_identity`), checked on every read."""
STOP_REQUESTED = "xaytune.io/stop-requested"
"""Annotation: Xaytune suspended this RayJob to stop it; ``Suspended`` then means stopped."""
MANAGED_BY = "app.kubernetes.io/managed-by"

# Keys a caller may not put on the RayJob: Xaytune's own (its annotations, and
# the ``xaytune.*`` job metadata), KubeRay's controls, and Kueue's, which would
# hand the RayJob's admission and suspension to a scheduler this slice does not
# speak to.
_RESERVED_PREFIXES = ("xaytune.", "xaytune.io/", "ray.io/", "kueue.x-k8s.io/")
_CLUSTER_KEY = "ray.io/cluster"
_LABEL_VALUE = re.compile(r"^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$")
_TERMINAL = frozenset({"STOPPED", "SUCCEEDED", "FAILED"})
# What a RayJob may legitimately change after Xaytune created it, and so what
# its identity leaves out: everything the server and KubeRay write
# (``status``, and metadata other than name, namespace, labels and
# annotations), the suspension Xaytune itself requests, and these annotations.
_MUTABLE_ANNOTATIONS = frozenset({SUBMISSION_DIGEST, STOP_REQUESTED})
# Fields the API server (CRD defaults) and KubeRay 1.7 add to a stored RayJob,
# each left out only while it holds that default -- observed on a real
# cluster. Any other addition or change is a different RayJob.
_SERVER_DEFAULTS: dict[str, Any] = {
    "ttlSecondsAfterFinished": 0,
    "numOfHosts": 1,
    "priority": 0,
    "scaleStrategy": {},
    "metadata": {},
    "resources": {},
}
_TIMEOUT_SECONDS = 30
_STOP_ATTEMPTS = 5
# How Ray's job-submission options become RayJob fields. Memory has no field.
_RESOURCE_FIELDS = {
    "entrypoint_num_cpus": "entrypointNumCpus",
    "entrypoint_num_gpus": "entrypointNumGpus",
}


def rayjob_name(submission_id: str) -> str:
    """The RayJob for *submission_id*: always the same name, never another's.

    A DNS-1035 label within KubeRay's 47 characters, whatever the id looks
    like. The full id is recorded on the RayJob too, and checked on every
    read, so a name that somehow belonged to another id is a conflict rather
    than an adoption.
    """
    return "xay-" + hashlib.sha256(submission_id.encode("utf-8")).hexdigest()[:32]


# ---- configuration ----------------------------------------------------------------------------


class ExistingCluster(FrozenDomainModel):
    """Run on a RayCluster that already exists, selected by name: ``clusterSelector``.

    Xaytune neither creates nor deletes it.
    """

    kind: Literal["existing"] = "existing"
    selector: FrozenDict
    """The RayJob's ``clusterSelector``. KubeRay selects by ``ray.io/cluster``: required."""

    def model_post_init(self, __context: Any) -> None:
        if not str(self.selector.get(_CLUSTER_KEY, "")).strip():
            raise ValueError(f"selector must name the cluster under {_CLUSTER_KEY!r}")


class EphemeralCluster(FrozenDomainModel):
    """Run on a RayCluster the RayJob creates and owns: ``rayClusterSpec``.

    The template is KubeRay's ``RayClusterSpec``, as KubeRay defines it, and
    is part of every job's identity in full.
    """

    kind: Literal["ephemeral"] = "ephemeral"
    template: FrozenDict
    """A ``RayClusterSpec``: ``headGroupSpec`` and any ``workerGroupSpecs``."""

    def model_post_init(self, __context: Any) -> None:
        if not isinstance(self.template.get("headGroupSpec"), Mapping):
            raise ValueError("template must be a RayClusterSpec with a headGroupSpec")


class RayJobSettings(FrozenDomainModel):
    """What the RayJob itself carries besides the job: metadata for the platform, and cleanup."""

    labels: FrozenDict = Field(default_factory=FrozenDict)
    """Copied onto the RayJob. Not Xaytune's, KubeRay's or Kueue's keys."""
    annotations: FrozenDict = Field(default_factory=FrozenDict)
    """Copied onto the RayJob. Not Xaytune's, KubeRay's or Kueue's keys."""
    deletion_policy: str | None = None
    """An ephemeral cluster's fate once its job finishes: ``delete-cluster``, and only that.

    Required for an ephemeral cluster, refused for an existing one. KubeRay's
    ``shutdownAfterJobFinishes``: the cluster goes, the RayJob -- the job's
    record -- stays. ``delete-self`` is refused: it would delete the RayJob,
    and with it how the job ended. ``keep-cluster`` is refused too: KubeRay
    suspends (stops) only a RayJob whose cluster is shut down after it.
    """
    ttl_seconds_after_finished: int = Field(default=0, ge=0)
    """How long a finished job's ephemeral cluster is kept before it is deleted."""

    def model_post_init(self, __context: Any) -> None:
        reasons = []
        for name, values in (("labels", self.labels), ("annotations", self.annotations)):
            reserved = sorted(key for key in values if key.startswith(_RESERVED_PREFIXES))
            if reserved:
                reasons.append(
                    f"{name} {reserved} are reserved: Xaytune's, KubeRay's and Kueue's keys "
                    f"({', '.join(_RESERVED_PREFIXES)}) cannot be set"
                )
            if any(not isinstance(value, str) for value in values.values()):
                reasons.append(f"{name} values must be strings")
        if self.deletion_policy == "delete-self":
            reasons.append(
                "deletion_policy delete-self would delete the RayJob, which is the job's record"
            )
        elif self.deletion_policy not in (None, "delete-cluster"):
            reasons.append(
                f"deletion_policy must be delete-cluster, not {self.deletion_policy!r}: the "
                f"RayJob stays, and only a RayJob whose cluster is shut down can be stopped"
            )
        if MANAGED_BY in self.labels:
            reasons.append(f"the {MANAGED_BY} label is Xaytune's")
        invalid = sorted(
            key
            for key, value in self.labels.items()
            if isinstance(value, str) and (len(value) > 63 or not _LABEL_VALUE.match(value))
        )
        if invalid:
            reasons.append(f"label values are not valid Kubernetes label values: {invalid}")
        if self.ttl_seconds_after_finished and self.deletion_policy is None:
            reasons.append("ttl_seconds_after_finished applies only to an ephemeral cluster")
        if reasons:
            raise ValueError("; ".join(reasons))


class KubeRayConfig(FrozenDomainModel):
    """Submission as KubeRay RayJobs: where, on which cluster, carrying what.

    Every field is part of the placement digest, so the record says where a
    job ran and a changed placement is never mistaken for the same request.
    """

    kind: Literal["kuberay"] = "kuberay"
    namespace: str
    """The namespace the RayJobs are created in."""
    context: str | None
    """The kubeconfig context to use; ``None`` for the pod's own service account (in-cluster).

    Required either way, so which Kubernetes API was used is on record and
    never the current context of whichever kubeconfig is around. Neither is
    the namespace implied: every request names ``namespace``.
    """
    cluster: Annotated[ExistingCluster | EphemeralCluster, Field(discriminator="kind")]
    rayjob: RayJobSettings = Field(default_factory=RayJobSettings)

    def model_post_init(self, __context: Any) -> None:
        reasons = []
        if not re.fullmatch(r"[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?", self.namespace):
            reasons.append(f"namespace must be a Kubernetes namespace name, not {self.namespace!r}")
        ephemeral = isinstance(self.cluster, EphemeralCluster)
        if ephemeral and self.rayjob.deletion_policy is None:
            reasons.append("an ephemeral cluster needs rayjob.deletion_policy: delete-cluster")
        if not ephemeral and self.rayjob.deletion_policy is not None:
            reasons.append("rayjob.deletion_policy applies only to an ephemeral cluster")
        if reasons:
            raise ValueError("; ".join(reasons))


# ---- the backend -------------------------------------------------------------------------------


class KubeRayJobsBackend:
    """A ``RaySubmissionBackend`` that creates, reads and suspends KubeRay ``RayJob`` resources.

    Args:
        config: The namespace, the Kubernetes context, the cluster and the
            RayJob's own settings.
        api: A ``kubernetes.client.CustomObjectsApi`` (or one that answers
            the same three calls: create, get, patch). Built from ``config.context`` unless given.
    """

    def __init__(self, config: KubeRayConfig, *, api: Any = None) -> None:
        self.config = config
        self._api = api

    @property
    def placement_digest(self) -> str:
        """The digest of the whole configuration: namespace, context, cluster, RayJob settings."""
        return fingerprint(self.config.model_dump(mode="json"))

    def _objects(self) -> Any:
        if self._api is None:
            self._api = _custom_objects_api(self.config.context)
        return self._api

    def submit(
        self,
        submission_id: str,
        entrypoint: str,
        *,
        metadata: Mapping[str, str],
        resources: Mapping[str, Any],
        runtime_env: Mapping[str, Any],
    ) -> None:
        body = self.rayjob(
            submission_id,
            entrypoint,
            metadata=metadata,
            resources=resources,
            runtime_env=runtime_env,
        )
        name = body["metadata"]["name"]
        try:
            self._objects().create_namespaced_custom_object(
                GROUP,
                VERSION,
                self.config.namespace,
                PLURAL,
                body,
                _request_timeout=_TIMEOUT_SECONDS,
            )
            return
        except RayUnavailableError:
            raise
        except Exception as error:
            if not _is_already_exists(error):
                raise RayUnavailableError(
                    f"Kubernetes did not accept RayJob {name} for job {submission_id}: "
                    f"{_describe(error)}"
                ) from None
        # The name is taken: by this very submission (a retry, a restart), or
        # by something else -- which is a conflict, never an adoption.
        existing = self._get(name)
        if existing is None:
            raise RayUnavailableError(
                f"RayJob {name} was reported to exist and then not found; retry the submission"
            )
        _require_bound(existing, submission_id)
        recorded = (existing.get("metadata", {}).get("annotations") or {}).get(SUBMISSION_DIGEST)
        if recorded != body["metadata"]["annotations"][SUBMISSION_DIGEST]:
            raise IdempotencyConflictError(submission_id, ("rayjob",))

    def info(self, submission_id: str) -> RayJob | None:
        name = rayjob_name(submission_id)
        existing = self._get(name)
        if existing is None:
            return None
        _require_bound(existing, submission_id)
        spec = existing.get("spec") or {}
        status = existing.get("status") or {}
        return RayJob(
            submission_id=submission_id,
            status=job_status(status, stop_requested=_stop_requested(existing)),
            metadata={str(k): str(v) for k, v in (spec.get("metadata") or {}).items()},
            message=status.get("message") or None,
        )

    def stop(self, submission_id: str) -> None:
        """Ask KubeRay to stop the job without losing the RayJob that records it.

        An ephemeral RayJob is suspended, and marked as suspended by Xaytune:
        KubeRay deletes its cluster and reports ``Suspended``, which
        :meth:`info` then reads as ``STOPPED``. A RayJob on an existing
        cluster cannot be suspended (KubeRay ignores it there), and deleting
        it would erase its status, so it is left untouched: it stays
        ``PENDING`` until KubeRay starts it, and whoever asked to stop it
        must also tell the job itself -- as the Ray runtimes do, through the
        cancellation request their supervisor reads first. A finished
        RayJob, or one already being suspended, is left alone.

        **Only the RayJob that was verified is suspended.** The patch carries
        the ``resourceVersion`` of the object whose identity was just
        checked, so Kubernetes refuses it (``409 Conflict``) if anything
        wrote the RayJob in between. Then the RayJob is read and verified
        again: an unchanged identity (KubeRay updated its status, say) is
        suspended on that version; a changed one is an
        :class:`~xaytune.core.errors.IdempotencyConflictError`, and the stale
        cancellation is never applied.
        """
        name = rayjob_name(submission_id)
        for _ in range(_STOP_ATTEMPTS):
            existing = self._get(name)
            if existing is None:
                return
            _require_bound(existing, submission_id)
            spec = existing.get("spec") or {}
            if job_status(existing.get("status") or {}) in _TERMINAL:
                return  # finished: its RayJob is its record
            if "rayClusterSpec" not in spec:
                return  # an existing cluster: nothing stops it here that keeps its record
            if spec.get("suspend") and _stop_requested(existing):
                return
            verified = (existing.get("metadata") or {}).get("resourceVersion")
            if not verified:
                raise RayUnavailableError(
                    f"RayJob {name} has no resourceVersion; it cannot be suspended "
                    f"only as it was verified"
                )
            try:
                self._objects().patch_namespaced_custom_object(
                    GROUP,
                    VERSION,
                    self.config.namespace,
                    PLURAL,
                    name,
                    {
                        "metadata": {
                            # The precondition: this version, the one verified, or nothing.
                            "resourceVersion": verified,
                            "annotations": {STOP_REQUESTED: utc_now().isoformat()},
                        },
                        "spec": {"suspend": True},
                    },
                    _content_type="application/merge-patch+json",
                    _request_timeout=_TIMEOUT_SECONDS,
                )
                return
            except Exception as error:
                if _is_not_found(error, name):
                    return
                if not _is_conflict(error):
                    raise RayUnavailableError(
                        f"cannot ask Kubernetes to suspend RayJob {name}: {_describe(error)}"
                    ) from None
            # Written since it was verified: read it, and verify it, again.
        raise RayUnavailableError(
            f"RayJob {name} kept changing while it was being suspended; nothing was applied "
            f"on a version that was not verified, and the stop can be retried"
        )

    def rayjob(
        self,
        submission_id: str,
        entrypoint: str,
        *,
        metadata: Mapping[str, str],
        resources: Mapping[str, Any],
        runtime_env: Mapping[str, Any],
    ) -> dict[str, Any]:
        """The RayJob :meth:`submit` creates for this job -- the same one every time.

        Raises:
            RaySubmissionRefusedError: A resource the RayJob has no field for.
        """
        spec: dict[str, Any] = {
            "entrypoint": entrypoint,
            "jobId": submission_id,
            "metadata": {str(k): str(v) for k, v in metadata.items()},
            "submissionMode": "K8sJobMode",
            # Retrying is recovery, and recovery is the controller's (ADR-013).
            "backoffLimit": 0,
        }
        plain_env = _plain(runtime_env)
        if plain_env:
            # JSON is YAML, and is what Ray reads back.
            spec["runtimeEnvYAML"] = json.dumps(plain_env, sort_keys=True)
        unexpressible = sorted(set(resources) - set(_RESOURCE_FIELDS))
        if unexpressible:
            raise RaySubmissionRefusedError(
                f"a KubeRay RayJob has no field for {', '.join(unexpressible)}; "
                f"KubeRay would run the job without it"
            )
        for option, value in resources.items():
            spec[_RESOURCE_FIELDS[option]] = value
        cluster = self.config.cluster
        if isinstance(cluster, ExistingCluster):
            spec["clusterSelector"] = _plain(cluster.selector)
        else:
            spec["rayClusterSpec"] = _plain(cluster.template)
            if self.config.rayjob.deletion_policy == "delete-cluster":
                spec["shutdownAfterJobFinishes"] = True
                spec["ttlSecondsAfterFinished"] = self.config.rayjob.ttl_seconds_after_finished
        body: dict[str, Any] = {
            "apiVersion": f"{GROUP}/{VERSION}",
            "kind": "RayJob",
            "metadata": {
                "name": rayjob_name(submission_id),
                "namespace": self.config.namespace,
                "labels": {**_plain(self.config.rayjob.labels), MANAGED_BY: "xaytune"},
                "annotations": {
                    **_plain(self.config.rayjob.annotations),
                    SUBMISSION_ID: submission_id,
                },
            },
            "spec": spec,
        }
        body["metadata"]["annotations"][SUBMISSION_DIGEST] = rayjob_identity(body)
        return body

    def _get(self, name: str) -> dict[str, Any] | None:
        try:
            found = self._objects().get_namespaced_custom_object(
                GROUP,
                VERSION,
                self.config.namespace,
                PLURAL,
                name,
                _request_timeout=_TIMEOUT_SECONDS,
            )
        except RayUnavailableError:
            raise
        except Exception as error:
            if _is_not_found(error, name):
                return None
            raise RayUnavailableError(
                f"cannot ask Kubernetes about RayJob {name}: {_describe(error)}"
            ) from None
        return dict(found)


def job_status(status: Mapping[str, Any], *, stop_requested: bool = False) -> RayJobStatus:
    """A RayJob's status as Ray's job status: the job's own, unless its deployment ended first.

    ``jobStatus`` is Ray's, mirrored by KubeRay. A deployment that failed --
    the cluster never came up, submission failed, validation refused the
    RayJob, a deadline passed -- ends the job without Ray ever reporting one.
    ``Suspended`` is ``STOPPED`` only when Xaytune suspended the RayJob to
    stop it (*stop_requested*); otherwise it is waiting, like everything else
    before a job runs (admission, a cluster): ``PENDING``.
    """
    reported = str(status.get("jobStatus") or "")
    deployment = str(status.get("jobDeploymentStatus") or "")
    if reported in _TERMINAL:
        return reported  # type: ignore[return-value]
    if deployment in ("Failed", "ValidationFailed", "Complete"):
        return "FAILED"
    if deployment == "Suspended" and stop_requested:
        return "STOPPED"
    if reported == "RUNNING" and deployment not in ("Suspending", "Suspended"):
        return "RUNNING"
    return "PENDING"


def _stop_requested(rayjob: Mapping[str, Any]) -> bool:
    annotations = (rayjob.get("metadata") or {}).get("annotations") or {}
    return STOP_REQUESTED in annotations


def rayjob_identity(rayjob: Mapping[str, Any]) -> str:
    """The digest of what a RayJob *is*: everything Xaytune bound when it created it.

    The same projection of the RayJob as created and of the live object: its
    kind, name, namespace, labels and annotations, and its whole spec --
    without what may legitimately change afterwards: ``status``, server
    metadata (uid, finalizers, resourceVersion, ...), ``spec.suspend``, the
    stop marker, and the defaults the server adds (each only at its default
    value). A live RayJob whose identity differs from the one recorded on it
    was changed after Xaytune created it -- its cluster, its entrypoint, its
    environment -- and is not the job Xaytune submitted.
    """
    metadata = rayjob.get("metadata") or {}
    annotations = metadata.get("annotations") or {}
    spec = {key: value for key, value in (rayjob.get("spec") or {}).items() if key != "suspend"}
    return fingerprint(
        {
            "apiVersion": rayjob.get("apiVersion"),
            "kind": rayjob.get("kind"),
            "metadata": {
                "name": metadata.get("name"),
                "namespace": metadata.get("namespace"),
                "labels": dict(metadata.get("labels") or {}),
                "annotations": {
                    key: value
                    for key, value in annotations.items()
                    if key not in _MUTABLE_ANNOTATIONS
                },
            },
            "spec": _without_defaults(spec),
        }
    )


def _without_defaults(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_defaults(item)
            for key, item in value.items()
            if not (key in _SERVER_DEFAULTS and item == _SERVER_DEFAULTS[key])
        }
    if isinstance(value, list):
        return [_without_defaults(item) for item in value]
    return value


def _require_bound(rayjob: Mapping[str, Any], submission_id: str) -> None:
    """This RayJob is the one Xaytune created for *submission_id*, unchanged since.

    Raises:
        IdempotencyConflictError: It records another submission id, or its
            identity no longer matches the one bound when it was created.
            Never adopted, by a retry or after a restart.
    """
    annotations = (rayjob.get("metadata") or {}).get("annotations") or {}
    if annotations.get(SUBMISSION_ID) != submission_id:
        # The name is this id's, the RayJob is not.
        raise IdempotencyConflictError(submission_id, ("rayjob",))
    if annotations.get(SUBMISSION_DIGEST) != rayjob_identity(rayjob):
        # Changed after it was bound: another cluster, entrypoint, environment...
        raise IdempotencyConflictError(submission_id, ("rayjob",))


def _reason(error: BaseException) -> tuple[int | None, Mapping[str, Any]]:
    """The HTTP status and the Kubernetes ``Status`` body of an API error, if it has them."""
    status = getattr(error, "status", None)
    try:
        body = json.loads(getattr(error, "body", None) or "")
    except (TypeError, ValueError):
        body = {}
    return (status if isinstance(status, int) else None), (body if isinstance(body, dict) else {})


def _is_not_found(error: BaseException, name: str) -> bool:
    """Kubernetes said *this RayJob* does not exist -- and only that.

    A 404 for the resource type (KubeRay's CRDs missing), or from anything
    that is not the API server's ``Status``, is an inability to answer.
    """
    status, body = _reason(error)
    details = body.get("details") or {}
    return status == 404 and body.get("reason") == "NotFound" and details.get("name") == name


def _is_conflict(error: BaseException) -> bool:
    """Kubernetes refused a write made on a ``resourceVersion`` that is no longer current."""
    status, body = _reason(error)
    return status == 409 and body.get("reason") == "Conflict"


def _is_already_exists(error: BaseException) -> bool:
    status, body = _reason(error)
    return status == 409 and body.get("reason") == "AlreadyExists"


def _describe(error: BaseException) -> str:
    """The error's kind, status and reason; never its body, which may echo the request."""
    status, body = _reason(error)
    parts = [type(error).__name__]
    if status is not None:
        parts.append(str(status))
    if body.get("reason"):
        parts.append(str(body["reason"]))
    return " ".join(parts)


def _custom_objects_api(context: str | None) -> Any:
    try:
        from kubernetes import client as kube_client
        from kubernetes import config as kube_config
    except ImportError as missing:  # pragma: no cover - depends on the install
        raise RayUnavailableError(
            "submitting to KubeRay needs the Kubernetes client: install xaytune[kuberay]"
        ) from missing
    try:
        if context is None:
            configuration = kube_client.Configuration()
            kube_config.load_incluster_config(client_configuration=configuration)
            api_client = kube_client.ApiClient(configuration)
        else:
            api_client = kube_config.new_client_from_config(context=context)
    except Exception as error:
        where = "in-cluster" if context is None else f"context {context!r}"
        raise RayUnavailableError(
            f"cannot configure the Kubernetes client ({where}): {type(error).__name__}"
        ) from None
    return kube_client.CustomObjectsApi(api_client)
