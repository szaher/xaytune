"""Record real experiments into state databases, for the Studio browser tests.

    python -m tests.studio_record DIRECTORY             → prints DIRECTORY/state.db
    python -m tests.studio_record --training DIRECTORY  → prints DIRECTORY/state.db
    python -m tests.studio_record --live DIRECTORY      → prints "ready DIRECTORY/state.db",
                                                          waits for DIRECTORY/release

The default is the spec 18 adaptive scenario, run by a real controller host,
exactly as ``tests/test_api`` records it (UX-02): training with a CUDA OOM
and a recovered attempt, evaluation, a decision, a planner branch and the
child's training.

``--training`` records one real training run (Track B): a native worker
trains a tiny causal LM on CPU under the built-in local runtime, and its
telemetry -- loss and learning rate at every optimizer step -- reaches the
metric-point store through the controller. One host submits it and closes
while it trains; a second host adopts the workload and replays its stream,
so the recorded series has already survived a controller restart.

``--live`` is for watching an experiment while it runs: the spec 18 scenario
again, with each training attempt reporting its loss. The first attempt
reports four steps, then the script prints ``ready`` and holds that attempt
running until a file named ``release`` appears; only then does it hit its
CUDA OOM, so a page opened in between sees the recovery attempt start.

Nothing here writes records by hand; every host is closed before it returns.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from tests.test_api.conftest import _record

RELEASE_TIMEOUT_SECONDS = 120


def record_training(directory: Path, *, steps: int = 24) -> Path:
    """One real native training run of *steps* optimizer steps; the state database."""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    from tests.test_experiment.test_host_behaviour import _spec
    from tests.training_fixtures import tiny_dataset
    from xaytune.experiment import EmbeddedControllerHost

    dataset = directory / "data" / "train.jsonl"
    samples = tiny_dataset(dataset, "text").read_text().splitlines()
    dataset.write_text("\n".join(samples * steps) + "\n")
    spec = _spec(directory, dataset=dataset)
    optimization = spec.candidate.training.optimization.model_copy(update={"max_steps": steps})
    training = spec.candidate.training.model_copy(update={"optimization": optimization})
    spec = spec.model_copy(
        update={
            "name": "tiny-sft-metrics",
            "candidate": spec.candidate.model_copy(update={"training": training}),
        }
    )
    state = directory / "state.db"

    async def submit() -> str:
        host = EmbeddedControllerHost(state)
        try:
            handle = await host.submit(spec)
            await asyncio.sleep(1.0)
            return str(handle.experiment_id)
        finally:
            await host.close()

    async def adopt(experiment_id: str) -> None:
        host = EmbeddedControllerHost(state)
        try:
            handle = await host.attach(experiment_id)
            result = await asyncio.wait_for(handle.wait(), timeout=300)
            ((run,),) = [node.runs for node in result.nodes]
            if not result.quiescent or run.status.value != "succeeded":
                raise RuntimeError(f"the training run ended {run.status.value}")
        finally:
            await host.close()

    asyncio.run(adopt(asyncio.run(submit())))
    return state


def _live_runtime(manager: Any, directory: Path, ready: Callable[[], None]) -> Any:
    from tests.test_experiment.adaptive_fixtures import AdaptiveRuntime
    from xaytune.core.telemetry import TrainingMetricObserved

    class LiveRuntime(AdaptiveRuntime):
        """The spec 18 runtime, reporting loss; the OOM attempt waits for *release*."""

        def __init__(self) -> None:
            super().__init__(manager, directory)
            self.held: dict[str, int] = {}
            self.first_step = 1

        async def _train(self, plan: Any, target: Any) -> None:
            oom = plan.spec.config["adapter_rank"] == self.oom_rank
            await super()._train(plan, target)
            steps = range(self.first_step, self.first_step + 4)
            self.first_step = 1 if not oom else self.first_step + 4
            reported = [
                self._training(
                    target, index, TrainingMetricObserved(optimizer_step=step, loss=4.0 / step)
                )
                for index, step in enumerate(steps)
            ]
            rest = [
                self._training(target, len(reported) + index, event.payload.data)
                for index, event in enumerate(self.events[target.id])
            ]
            self.events[target.id] = (*reported, *rest)
            if oom:
                self.held[target.id] = len(reported)

        async def watch(self, reference: Any, cursor: Any) -> AsyncIterator[Any]:
            held = self.held.pop(reference.external_id, None)
            for index, event in enumerate(self.events[reference.external_id]):
                if index == held:
                    # Every point before this one is recorded: the host
                    # observed each event before asking for the next.
                    ready()
                    await _released(directory / "release")
                yield event

    return LiveRuntime()


async def _released(release: Path) -> None:
    for _ in range(RELEASE_TIMEOUT_SECONDS * 10):
        if release.exists():
            return
        await asyncio.sleep(0.1)
    raise TimeoutError(f"nothing created {release}")


def record_live(directory: Path) -> Path:
    """The spec 18 scenario with loss, held at its first attempt until *release* exists."""
    from tests.test_experiment.adaptive_fixtures import adaptive_spec
    from tests.test_experiment.test_adaptive_mvp import _host
    from xaytune.checkpoints import CheckpointManager, LocalCheckpointStore, SerializedStateCodec

    state = directory / "state.db"
    manager = CheckpointManager(SerializedStateCodec(), LocalCheckpointStore(directory / "bundles"))
    runtime = _live_runtime(manager, directory, lambda: print(f"ready {state}", flush=True))

    async def scenario() -> None:
        host = _host(directory, {"manager": manager, "runtime": runtime})
        try:
            handle = await host.submit(adaptive_spec(directory))
            result = await asyncio.wait_for(handle.wait(), timeout=RELEASE_TIMEOUT_SECONDS + 60)
            if result.status.value != "succeeded":
                raise RuntimeError(f"the live experiment ended {result.status.value}")
        finally:
            await host.close()

    asyncio.run(scenario())
    return state


def main(argv: list[str]) -> int:
    mode = argv[0] if argv[:1] in (["--training"], ["--live"]) else None
    arguments = argv[1:] if mode else argv
    if len(arguments) != 1:
        print(
            "usage: python -m tests.studio_record [--training | --live] DIRECTORY",
            file=sys.stderr,
        )
        return 2
    directory = Path(arguments[0])
    directory.mkdir(parents=True, exist_ok=True)
    if mode == "--training":
        state = record_training(directory)
    elif mode == "--live":
        state = record_live(directory)
    else:
        state = _record(directory).state
    print(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
