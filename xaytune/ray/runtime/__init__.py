"""Ray-backed ``RuntimeBackend``s: what runs on the cluster (PR-033a).

``RayJobsRuntime`` runs one supervised Xaytune worker per plan;
``RayTrainRuntime`` (PR-033b) trains as a Ray Train worker group. Both get
their jobs onto a cluster through a
:class:`~xaytune.ray.submission.RaySubmissionBackend`, by composition.
"""

from __future__ import annotations

from xaytune.ray.runtime._workloads import RayClusterConfig
from xaytune.ray.runtime.jobs import (
    BACKEND,
    RayJobsConfig,
    RayJobsRuntime,
    ray_jobs_runtime,
)
from xaytune.ray.runtime.train import RayTrainConfig, RayTrainRuntime, ray_train_runtime

__all__ = [
    "BACKEND",
    "RayClusterConfig",
    "RayJobsConfig",
    "RayJobsRuntime",
    "RayTrainConfig",
    "RayTrainRuntime",
    "ray_jobs_runtime",
    "ray_train_runtime",
]
