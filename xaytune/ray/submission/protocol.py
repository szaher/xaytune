"""How a Ray runtime gets a job onto a cluster: the submission boundary.

```text
RayJobsRuntime ─┐                        ┌─ RayJobsBackend       (Ray Jobs API, an existing cluster)
RayTrainRuntime ┴─ RaySubmissionBackend ─┤
                                         └─ KubeRayJobsBackend   (a RayJob custom resource)
```

What runs and how it is submitted are separate choices. A runtime decides
what the job does -- one supervised Xaytune worker, a Ray Train worker
group -- and a submission backend only gets it onto a cluster under an id,
describes it, and stops it. Neither inherits from the other.

**"No such job" is an answer; "I could not ask" is not.**
:meth:`RaySubmissionBackend.info` returns ``None`` only when the cluster says
the job does not exist, because that is what lets a controller re-issue a
submission. Anything else -- the cluster unreachable, an error mid-request --
raises :class:`RayUnavailableError`, and a caller that cannot tell must not
conclude the job is absent.

**Where a job runs is part of what it is.** A backend whose configuration
decides more than which cluster answers -- the topology of a cluster it
creates, the labels a scheduler admits it by -- says so in
:attr:`RaySubmissionBackend.placement_digest`, and the job's identity
includes it: the same id placed differently is a different request.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "RayJob",
    "RayJobStatus",
    "RaySubmissionBackend",
    "RaySubmissionRefusedError",
    "RayUnavailableError",
]

RayJobStatus = Literal["PENDING", "RUNNING", "STOPPED", "SUCCEEDED", "FAILED"]


class RayUnavailableError(Exception):
    """Ray could not be asked, or did not answer. Nothing is concluded from it."""


class RaySubmissionRefusedError(Exception):
    """The backend cannot express this submission; nothing was sent, and nothing will be.

    Deterministic, unlike :class:`RayUnavailableError`: the same submission is
    refused the same way every time, so a caller records the refusal rather
    than retrying.
    """


@dataclass(frozen=True)
class RayJob:
    """What the cluster reports about one submitted job."""

    submission_id: str
    status: RayJobStatus
    metadata: Mapping[str, str] = field(default_factory=dict)
    message: str | None = None
    exit_code: int | None = None


@runtime_checkable
class RaySubmissionBackend(Protocol):
    """Submit, describe and stop jobs under ids the cluster refuses to reuse."""

    @property
    def placement_digest(self) -> str | None:
        """The digest of how this backend places jobs, or ``None`` if only the cluster varies.

        Everything in the backend's configuration that changes what a job
        runs on -- never a secret value -- and nothing about the job itself.
        A caller records it with every job and refuses the same id placed
        differently.
        """
        ...

    def submit(
        self,
        submission_id: str,
        entrypoint: str,
        *,
        metadata: Mapping[str, str],
        resources: Mapping[str, Any],
        runtime_env: Mapping[str, Any],
    ) -> None:
        """Submit a job under *submission_id*, which the cluster refuses to reuse.

        Raises:
            RaySubmissionRefusedError: The backend cannot express this job.
            RayUnavailableError: The cluster could not be asked, or did not
                accept the job; whether it holds one is :meth:`info`'s answer.
        """
        ...

    def info(self, submission_id: str) -> RayJob | None:
        """The job, or ``None`` only if the cluster says there is none.

        Raises:
            RayUnavailableError: The cluster could not be asked.
        """
        ...

    def stop(self, submission_id: str) -> None:
        """Ask the cluster to stop the job. Stopping a finished job does nothing.

        Asynchronous: the job's status, not this call, says when it stopped --
        so a stop must leave that status observable. A backend that cannot
        stop a job without erasing its record leaves the job as it is, and the
        caller's own cancellation (the request its supervisor reads first)
        ends it once it runs.
        """
        ...
