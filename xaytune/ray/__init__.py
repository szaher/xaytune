"""Xaytune on Ray, kept apart by what each piece is (PR-033a).

```text
xaytune.ray
├── submission   how a job reaches a cluster    RayJobsBackend; KubeRay later (PR-033c)
└── runtime      what runs there                RayJobsRuntime; RayTrainRuntime (PR-033b)
```

A runtime composes a submission backend; neither inherits from the other, so
Ray Train over KubeRay is a pairing, not a class. Ray Tune is a search
provider (PR-034) and not a runtime.

Xaytune does not create clusters: every backend here submits to an existing
one, named by its address.

```python
RuntimeSpec(kind="ray-jobs", config={
    "address": "http://ray-head:8265",
    "runtime_env": {},
    "shared_state_root": "/shared/xaytune",
})
```

Importable without Ray: only :class:`~xaytune.ray.submission.RayJobsBackend`
needs it, when it first talks to a cluster (``pip install xaytune[ray]``).
"""

from __future__ import annotations

from xaytune.ray.runtime import BACKEND, RayJobsConfig, RayJobsRuntime, ray_jobs_runtime
from xaytune.ray.submission import (
    RayJob,
    RayJobsBackend,
    RayJobStatus,
    RaySubmissionBackend,
    RayUnavailableError,
)

__all__ = [
    "BACKEND",
    "RayJob",
    "RayJobStatus",
    "RayJobsBackend",
    "RayJobsConfig",
    "RayJobsRuntime",
    "RaySubmissionBackend",
    "RayUnavailableError",
    "ray_jobs_runtime",
]
