"""Getting a job onto a Ray cluster, apart from what the job does (PR-033a, PR-033c).

```text
SubmissionConfig ── submission_backend() ──┬─ RayJobsBackend       kind "ray-jobs"
                                           └─ KubeRayJobsBackend   kind "kuberay"
```

A runtime is configured with a :data:`SubmissionConfig` and asks
:func:`submission_backend` for the backend it names; it never learns which.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from xaytune.ray.submission.jobs import RayJobsBackend, RayJobsBackendConfig
from xaytune.ray.submission.kuberay import (
    EphemeralCluster,
    ExistingCluster,
    KubeRayConfig,
    KubeRayJobsBackend,
    RayJobSettings,
)
from xaytune.ray.submission.protocol import (
    RayJob,
    RayJobStatus,
    RaySubmissionBackend,
    RaySubmissionRefusedError,
    RayUnavailableError,
)

__all__ = [
    "EphemeralCluster",
    "ExistingCluster",
    "KubeRayConfig",
    "KubeRayJobsBackend",
    "RayJob",
    "RayJobSettings",
    "RayJobStatus",
    "RayJobsBackend",
    "RayJobsBackendConfig",
    "RaySubmissionBackend",
    "RaySubmissionRefusedError",
    "RayUnavailableError",
    "SubmissionConfig",
    "submission_backend",
]

SubmissionConfig = Annotated[RayJobsBackendConfig | KubeRayConfig, Field(discriminator="kind")]
"""How jobs reach the cluster, by ``kind``: ``ray-jobs`` or ``kuberay``."""


def submission_backend(config: RayJobsBackendConfig | KubeRayConfig) -> RaySubmissionBackend:
    """The backend *config* describes. Contacts nothing: a backend connects when first used."""
    if isinstance(config, KubeRayConfig):
        return KubeRayJobsBackend(config)
    return RayJobsBackend(config.address)
