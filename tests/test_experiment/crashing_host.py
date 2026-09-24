"""A controller process that dies at a chosen moment, for the restart tests.

Run as ``python -m tests.test_experiment.crashing_host STATE SPEC MODE``. It
submits the experiment in SPEC (an ``ExperimentSpec`` as JSON) against STATE
and kills itself with ``SIGKILL`` -- not an exception, not a clean shutdown,
the thing a crash actually is -- at the point MODE names:

``running``         once the attempt is RUNNING: training is live and the
                    submission is confirmed.
``lost-response``   after the runtime accepted the submission, before the
                    controller recorded it: the operation stays INTENDED while
                    a workload exists.
``never-sent``      before the runtime is asked at all: INTENDED, and nothing
                    was started.
``cancel-intended`` once RUNNING, after recording a cancellation and before
                    issuing its effect: the intent is durable, the workload
                    still runs.

The workload runs in its own session, so it outlives this process, which is
the situation PR-012a exists for.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any

from xaytune.core.refs import Actor
from xaytune.experiment import EmbeddedControllerHost, ExperimentSpec


def _die() -> None:
    os.kill(os.getpid(), signal.SIGKILL)


class _DyingAt:
    """LocalRuntime, except that submission kills the process at MODE."""

    def __init__(self, runtime: Any, mode: str) -> None:
        self._runtime = runtime
        self._mode = mode

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    async def submit_or_get(self, operation_id: Any, plan: Any) -> Any:
        if self._mode == "never-sent":
            _die()
        reference = await self._runtime.submit_or_get(operation_id, plan)
        if self._mode == "lost-response":
            _die()
        return reference


async def _main(state: Path, spec: ExperimentSpec, mode: str) -> None:
    from xaytune.experiment.host import _local_runtime

    host = EmbeddedControllerHost(
        state, runtimes={"local": lambda config: _DyingAt(_local_runtime(config), mode)}
    )
    handle = await host.submit(spec)
    if mode not in ("running", "cancel-intended"):
        raise AssertionError(f"mode {mode!r} should have died during submit")
    while True:
        result = host._result(handle.experiment_id)
        (run,) = result.nodes[0].runs
        if run.attempt_status is not None and run.attempt_status.value == "running":
            if mode == "cancel-intended":
                host.repository.request_experiment_cancellation(
                    handle.experiment_id,
                    reason="cancelled, then the controller died",
                    actor=Actor(type="human", id="operator"),
                )
            _die()
        await asyncio.sleep(0.02)


if __name__ == "__main__":
    state_path, spec_path, crash_mode = sys.argv[1:4]
    asyncio.run(
        _main(
            Path(state_path),
            ExperimentSpec.model_validate_json(Path(spec_path).read_text()),
            crash_mode,
        )
    )
