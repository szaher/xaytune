"""``train_driver`` with the worker group simulated: the real driver and ranks, no Ray.

``ProcessJobs(train=True)`` runs this in place of
:mod:`xaytune.ray.runtime.train_driver`, the way a cluster image supplies the
entrypoint module. Everything but the group is the production code -- the
claim, the single sequencer, every rank's worker, cancellation, aborts and
the group's outcome. The group is ``world_size`` threads, each a rank on one
node, meeting on a port free here, as ``TorchTrainer``'s workers would.

``launch.json``'s scaling decides the group's size, exactly as for Ray.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from xaytune.core.clock import utc_now
from xaytune.core.execution import ResolvedExecutionPlan
from xaytune.ray.runtime._workloads import LAUNCH
from xaytune.ray.runtime.train_group import RankContext, drive, free_port
from xaytune.runtimes.local.paths import WorkloadPaths, read_json, write_atomic

FAIL_GROUP = "fake-group-fails"
"""A file whose presence makes the simulated group fail as a group, after its ranks ran."""


class ThreadGroup:
    def __init__(self, paths: WorkloadPaths, world_size: int) -> None:
        self.paths = paths
        self.world_size = world_size

    def run(self, rank_fn: Callable[[RankContext], None]) -> None:
        port = free_port()
        errors: list[BaseException] = []

        def rank(index: int) -> None:
            try:
                rank_fn(
                    RankContext(
                        rank=index,
                        world_size=self.world_size,
                        local_rank=index,
                        local_world_size=self.world_size,
                        node_rank=0,
                        master_addr="127.0.0.1",
                        master_port=port,
                    )
                )
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        threads = [threading.Thread(target=rank, args=(i,)) for i in range(self.world_size)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        if (self.paths.directory / FAIL_GROUP).exists():
            raise RuntimeError("the simulated worker group failed")


def _group_for(plan: ResolvedExecutionPlan, paths: WorkloadPaths) -> Any:
    launch = read_json(paths.directory / LAUNCH) or {}
    return ThreadGroup(paths, int(launch["scaling"]["num_workers"]))


def main() -> int:
    directory = Path(sys.argv[1])
    paths = WorkloadPaths(directory)

    def cancel() -> None:
        write_atomic(paths.cancel, {"requested_at": utc_now().isoformat(), "by": "ray"})

    return drive(directory, _group_for, on_termination=cancel)


if __name__ == "__main__":
    raise SystemExit(main())
