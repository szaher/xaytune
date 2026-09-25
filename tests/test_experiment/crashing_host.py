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

The same four points for the **evaluation** that follows training, when the
spec asks for one (PR-013) -- training has succeeded and been recorded:

``evaluating``            once the evaluation attempt is RUNNING.
``eval-lost-response``    after the runtime accepted the evaluation, before
                          the controller recorded it.
``eval-never-sent``       before the runtime is asked for the evaluation.
``eval-cancel-intended``  once the evaluation is RUNNING, after recording a
                          cancellation and before issuing its effect.
``eval-stream-lost``      the evaluation's telemetry stream ends just after its
                          EvaluationCompleted, while the workload still runs
                          (ADR-014 §1a); dies once the controller has recorded
                          that degradation, before the workload ends.

And one for the decision that follows (PR-015):

``deciding``              the evaluation's result is recorded and the node is
                          DECIDING; dies as the decision engine is asked, so
                          no decision exists.

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

from tests.evaluation_fixtures import EVALUATORS
from xaytune.core.refs import Actor
from xaytune.core.state.status import EvaluationAttemptStatus
from xaytune.evaluation.native import NativeEvaluator
from xaytune.experiment import EmbeddedControllerHost, ExperimentSpec

_EVALUATION_MODES = (
    "evaluating",
    "eval-lost-response",
    "eval-never-sent",
    "eval-cancel-intended",
    "eval-stream-lost",
)


def _die() -> None:
    os.kill(os.getpid(), signal.SIGKILL)


class _DieWhenDeciding:
    """A decision engine that kills the controller instead of deciding."""

    name = "dies"
    version = "0"

    def decide(self, context: Any) -> Any:
        _die()


class _DyingAt:
    """LocalRuntime, except that submission kills the process at MODE."""

    def __init__(self, runtime: Any, mode: str) -> None:
        self._runtime = runtime
        self._mode = mode

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    async def submit_or_get(self, operation_id: Any, plan: Any) -> Any:
        evaluating = plan.target.kind == "evaluation-attempt"
        if self._mode == "never-sent" or (self._mode == "eval-never-sent" and evaluating):
            _die()
        reference = await self._runtime.submit_or_get(operation_id, plan)
        if self._mode == "lost-response" or (self._mode == "eval-lost-response" and evaluating):
            _die()
        return reference

    async def watch(self, reference: Any, cursor: Any = None) -> Any:
        """The stream, ended just after an EvaluationCompleted in ``eval-stream-lost``.

        The workload runs on; only its telemetry is gone -- what a dead
        supervisor looks like to the controller.
        """
        async for envelope in self._runtime.watch(reference, cursor):
            yield envelope
            if (
                self._mode == "eval-stream-lost"
                and envelope.payload.data.type == "EvaluationCompleted"
            ):
                return


async def _main(state: Path, spec: ExperimentSpec, mode: str) -> None:
    from xaytune.experiment.host import _local_runtime

    host = EmbeddedControllerHost(
        state,
        runtimes={"local": lambda config: _DyingAt(_local_runtime(config), mode)},
        evaluators={**EVALUATORS, "native": NativeEvaluator},
        decision_engine=_DieWhenDeciding() if mode == "deciding" else None,
    )
    handle = await host.submit(spec)
    if mode == "deciding":
        await handle.wait()
        raise AssertionError("the controller should have died deciding")
    if mode in _EVALUATION_MODES:
        await _die_while_evaluating(host, handle, mode)
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


async def _die_while_evaluating(host: Any, handle: Any, mode: str) -> None:
    """Wait for the evaluation to reach the point MODE names, and die there.

    The lost-response and never-sent modes die inside ``submit_or_get``,
    which the controller calls for the evaluation once training has
    succeeded; this loop only has to outlast them.
    """
    repo = host.repository
    experiment_id = str(handle.experiment_id)
    while True:
        if mode == "eval-stream-lost" and any(
            event.event_type == "TelemetryDegraded" and event.aggregate_type == "EvaluationAttempt"
            for event in repo.events.events_for_experiment(experiment_id)
        ):
            _die()
        for node in repo.aggregates.nodes_for_experiment(experiment_id):
            for run in repo.aggregates.evaluation_runs_for_node(str(node.id)):
                for attempt in repo.aggregates.evaluation_attempts_for_run(str(run.id)):
                    if attempt.status is EvaluationAttemptStatus.RUNNING and mode in (
                        "evaluating",
                        "eval-cancel-intended",
                    ):
                        if mode == "eval-cancel-intended":
                            repo.request_experiment_cancellation(
                                handle.experiment_id,
                                reason="cancelled mid-evaluation, then the controller died",
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
