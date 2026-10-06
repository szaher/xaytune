"""The entrypoint of every Ray job ``RayJobsRuntime`` submits.

```text
ray job (submission_id = the operation id)
  └─ python -m xaytune.ray.runtime.supervisor <workload directory>
       └─ the worker, exactly as LocalRuntime would run it
```

It is the same supervisor the local launcher runs
(:func:`~xaytune.runtimes.local.launcher.supervise_workload`): the single
writer of the workload's telemetry stream (ADR-014 §1a), the parent that reaps
the worker and records how it ended. So a worker under Ray speaks the
telemetry it speaks locally, and the controller reads the same envelopes.

What differs is the claim and the cancellation. Ray's submission id is the
claim that only one job exists per operation; this process additionally
claims the workload directory, so an entrypoint Ray ran twice cannot start a
second worker. And a termination request to this process -- Ray stopping the
job -- becomes the workload's durable ``cancel.request``, delivered to the
worker's process group once, as the local launcher delivers it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from xaytune.core.clock import utc_now
from xaytune.core.execution import ResolvedExecutionPlan
from xaytune.runtimes.local.launcher import supervise_workload
from xaytune.runtimes.local.paths import WorkloadPaths, write_atomic

__all__ = ["CLAIM", "main", "run"]

CLAIM = "supervisor.claim"


def run(directory: Path) -> int:
    """Supervise the workload in *directory*; its worker's exit code, or 0 if not ours."""
    paths = WorkloadPaths(directory)
    try:
        descriptor = os.open(directory / CLAIM, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # Ray ran this entrypoint again for the same job. The first run owns
        # the workload; a second worker must never exist.
        return 0
    with os.fdopen(descriptor, "w", encoding="utf-8") as claim:
        claim.write(f"{os.getpid()} {utc_now().isoformat()}\n")

    plan = ResolvedExecutionPlan.model_validate_json(paths.plan.read_text(encoding="utf-8"))

    def cancel() -> None:
        write_atomic(paths.cancel, {"requested_at": utc_now().isoformat(), "by": "ray"})

    return supervise_workload(paths, plan, on_termination=cancel)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: python -m xaytune.ray.runtime.supervisor <workload-directory>")
        return 2
    code = run(Path(arguments[0]))
    # A worker killed by a signal reports a negative code; a process exit code
    # cannot be negative, and the finished record already holds the signal.
    return code if code >= 0 else 1


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    raise SystemExit(main())
