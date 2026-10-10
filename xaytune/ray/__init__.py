"""Xaytune on Ray, kept apart by what each piece is (PR-033a).

```text
xaytune.ray
├── submission   how a job reaches a cluster    RayJobsBackend, KubeRayJobsBackend
├── runtime      what runs there                RayJobsRuntime, RayTrainRuntime
└── search       what to try next               RayTuneSearchProvider (not re-exported)
```

A runtime composes a submission backend; neither inherits from the other, so
Ray Train over KubeRay is a pairing, not a class. Ray Tune is a search
provider (PR-034, :mod:`xaytune.ray.search`) and not a runtime: it touches no
cluster, and runs the same whatever executes the candidates.

Xaytune does not manage clusters. The Ray Jobs API backend submits to an
existing one, named by its address; the KubeRay backend creates ``RayJob``
resources, which either select an existing RayCluster or carry the template
of one the RayJob owns -- KubeRay's mechanisms, not Xaytune's.

```python
RuntimeSpec(kind="ray-jobs", config={
    "address": "http://ray-head:8265",   # shorthand for submission={"kind": "ray-jobs", ...}
    "runtime_env": {},
    "shared_state_root": "/shared/xaytune",
})
RuntimeSpec(kind="ray-train", config={
    "submission": {
        "kind": "kuberay",
        "namespace": "ml",
        "context": None,                 # in-cluster; or a kubeconfig context name
        "cluster": {"kind": "existing", "selector": {"ray.io/cluster": "trainers"}},
    },
    "runtime_env": {},
    "shared_state_root": "/shared/xaytune",
})
```

Importable without Ray or Kubernetes: only
:class:`~xaytune.ray.submission.RayJobsBackend` needs Ray, and
:class:`~xaytune.ray.submission.KubeRayJobsBackend` the Kubernetes client,
when each first talks to a cluster (``pip install xaytune[ray]``,
``xaytune[kuberay]``).
"""

from __future__ import annotations

from xaytune.ray.runtime import (
    BACKEND,
    RayClusterConfig,
    RayJobsConfig,
    RayJobsRuntime,
    RayTrainConfig,
    RayTrainRuntime,
    ray_jobs_runtime,
    ray_train_runtime,
)
from xaytune.ray.submission import (
    EphemeralCluster,
    ExistingCluster,
    KubeRayConfig,
    KubeRayJobsBackend,
    RayJob,
    RayJobsBackend,
    RayJobsBackendConfig,
    RayJobSettings,
    RayJobStatus,
    RaySubmissionBackend,
    RaySubmissionRefusedError,
    RayUnavailableError,
    SubmissionConfig,
    submission_backend,
)

__all__ = [
    "BACKEND",
    "EphemeralCluster",
    "ExistingCluster",
    "KubeRayConfig",
    "KubeRayJobsBackend",
    "RayClusterConfig",
    "RayJob",
    "RayJobSettings",
    "RayJobStatus",
    "RayJobsBackend",
    "RayJobsBackendConfig",
    "RayJobsConfig",
    "RayJobsRuntime",
    "RaySubmissionBackend",
    "RaySubmissionRefusedError",
    "RayTrainConfig",
    "RayTrainRuntime",
    "RayUnavailableError",
    "SubmissionConfig",
    "ray_jobs_runtime",
    "ray_train_runtime",
    "submission_backend",
]
