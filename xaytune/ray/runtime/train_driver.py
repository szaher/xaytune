"""The entrypoint of every training job ``RayTrainRuntime`` submits: a real ``TorchTrainer``.

```text
python -m xaytune.ray.runtime.train_driver <workload directory>
  └─ TorchTrainer(train_loop_per_worker=run_rank,
                  scaling_config=ScalingConfig(**launch.json["scaling"]),
                  run_config=RunConfig(name=<operation id>,
                                       storage_path=<workload directory>/ray-train,
                                       failure_config=FailureConfig(max_failures=0)))
```

The scaling is the one the runtime derived from the plan and recorded in
``launch.json`` before Ray was asked -- read here, never re-derived, so the
group that runs is the group on record.

``max_failures=0``: a failed group is reported, not silently retried. Retrying
is recovery, and recovery is the controller's (ADR-013). ``storage_path`` is
set because Ray Train's default is the home directory of whichever node runs
the driver.

Besides :mod:`xaytune.ray.submission.jobs`, the only module that imports
``ray``; nothing imports it -- Ray runs it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from xaytune.core.clock import utc_now
from xaytune.core.execution import ResolvedExecutionPlan
from xaytune.ray.runtime._workloads import LAUNCH
from xaytune.ray.runtime.train_group import RankContext, WorkerGroup, drive, free_port
from xaytune.runtimes.local.paths import WorkloadPaths, read_json, write_atomic

__all__ = ["TorchTrainerGroup", "build_trainer", "main"]

STORAGE = "ray-train"


class TorchTrainerGroup:
    """A :class:`~xaytune.ray.runtime.train_group.WorkerGroup` that is a Ray Train worker group."""

    def __init__(self, paths: WorkloadPaths, scaling: dict[str, Any]) -> None:
        self.paths = paths
        self.scaling = scaling
        self.world_size = int(scaling["num_workers"])

    def run(self, rank_fn: Any) -> None:
        build_trainer(self.paths, self.scaling, rank_fn).fit()


def build_trainer(paths: WorkloadPaths, scaling: dict[str, Any], rank_fn: Any) -> Any:
    """The ``TorchTrainer`` for this workload: its scaling, its name, its storage, no retries."""
    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    return TorchTrainer(
        train_loop_per_worker=_TrainLoop(rank_fn),
        scaling_config=ScalingConfig(**scaling),
        run_config=RunConfig(
            name=paths.directory.name,
            storage_path=str(paths.directory / STORAGE),
            failure_config=FailureConfig(max_failures=0),
        ),
    )


class _TrainLoop:
    """What every Ray Train worker runs: learn its placement, then run its rank."""

    def __init__(self, rank_fn: Any) -> None:
        self.rank_fn = rank_fn

    def __call__(self) -> None:
        import ray.train
        from ray.train.collective import broadcast_from_rank_zero
        from ray.util import get_node_ip_address

        context = ray.train.get_context()
        rank = context.get_world_rank()
        # Rank 0 chooses where the workers' own process group meets -- on its
        # node, on a port free there -- and every rank learns it from rank 0.
        meeting = broadcast_from_rank_zero(
            {"addr": get_node_ip_address(), "port": free_port()} if rank == 0 else None
        )
        self.rank_fn(
            RankContext(
                rank=rank,
                world_size=context.get_world_size(),
                local_rank=context.get_local_rank(),
                local_world_size=context.get_local_world_size(),
                node_rank=context.get_node_rank(),
                master_addr=str(meeting["addr"]),
                master_port=int(meeting["port"]),
            )
        )


def _group_for(plan: ResolvedExecutionPlan, paths: WorkloadPaths) -> WorkerGroup:
    launch = read_json(paths.directory / LAUNCH)
    if launch is None or "scaling" not in launch:
        raise RuntimeError(f"{paths.directory / LAUNCH} records no scaling for this workload")
    return TorchTrainerGroup(paths, dict(launch["scaling"]))


def run(directory: Path) -> int:
    paths = WorkloadPaths(directory)

    def cancel() -> None:
        write_atomic(paths.cancel, {"requested_at": utc_now().isoformat(), "by": "ray"})

    return drive(directory, _group_for, on_termination=cancel)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: python -m xaytune.ray.runtime.train_driver <workload-directory>")
        return 2
    return run(Path(arguments[0]))


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    raise SystemExit(main())
