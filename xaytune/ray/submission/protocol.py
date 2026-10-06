"""How a Ray runtime gets a job onto a cluster: the submission boundary.

```text
RayJobsRuntime ─┐                        ┌─ RayJobsBackend    (Ray Jobs API, an existing cluster)
RayTrainRuntime ┴─ RaySubmissionBackend ─┤
                                         └─ KubeRay RayJob    (PR-033c)
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
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "RayJob",
    "RayJobStatus",
    "RaySubmissionBackend",
    "RayUnavailableError",
]

RayJobStatus = Literal["PENDING", "RUNNING", "STOPPED", "SUCCEEDED", "FAILED"]


class RayUnavailableError(Exception):
    """Ray could not be asked, or did not answer. Nothing is concluded from it."""


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

    def submit(
        self,
        submission_id: str,
        entrypoint: str,
        *,
        metadata: Mapping[str, str],
        resources: Mapping[str, Any],
        runtime_env: Mapping[str, Any],
    ) -> None:
        """Submit a job under *submission_id*, which the cluster refuses to reuse."""
        ...

    def info(self, submission_id: str) -> RayJob | None:
        """The job, or ``None`` only if the cluster says there is none.

        Raises:
            RayUnavailableError: The cluster could not be asked.
        """
        ...

    def stop(self, submission_id: str) -> None:
        """Ask the cluster to stop the job. Stopping a finished job does nothing.

        Asynchronous: the job's status, not this call, says when it stopped.
        """
        ...
