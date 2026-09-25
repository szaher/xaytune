"""Submit one SFT candidate, follow its events, and wait for the result.

    python examples/control_plane/02_train.py \\
        --model /abs/path/to/hf-model-dir \\
        --dataset /abs/path/to/train.jsonl \\
        --compiler native            # or: trl

The control plane is not on PyPI yet, so install from a clone of main. For
``--compiler trl``, include the TRL extra::

    uv sync --locked --extra trl     # or, with pip: pip install -e ".[trl]"

The dataset is local JSONL with a ``text`` field per line, the one format both
built-in trainers read with the same meaning. The model is a local Hugging
Face directory: a hub name is refused, because without a pinned revision it
names whatever the hub serves on the day the worker starts.

Everything is recorded under ``--workdir``: the control-plane database, the
local runtime's registry and the trained model. Run it twice and you have two
experiments in the same record.

What ``wait()`` returns is *controller quiescence*, not a verdict. With no
evaluation configured, a trained node stays ``ACTIVE`` and ``next_stage`` is
``"evaluation"``: the stage that would run next, if anything ran it.
"""

from __future__ import annotations

import argparse
import asyncio

from sft_experiment import add_arguments, experiment, state_path

from xaytune.experiment import EmbeddedControllerHost, ExperimentHandle


async def follow(handle: ExperimentHandle) -> None:
    """Print each control-plane event as it is committed."""
    async for event in handle.events():
        status = event.payload.get("status")
        suffix = f" -> {status}" if status else ""
        print(f"  [{event.sequence:>3}] {event.aggregate_type}: {event.event_type}{suffix}")


async def run(args: argparse.Namespace) -> None:
    host = EmbeddedControllerHost(state_path(args))
    try:
        handle = await host.submit(experiment(args))
        print(f"submitted {handle.experiment_id}")
        follower = asyncio.create_task(follow(handle))
        result = await handle.wait()
        # Stop following. Anything not printed yet is still in the record:
        # handle.events() replays the whole history, here or after attach().
        follower.cancel()

        print(f"\nexperiment {result.experiment_id}: {result.status.value}")
        print(f"quiescent: {result.quiescent}, next stage: {result.next_stage}")
        for node in result.nodes:
            print(f"node {node.node_id}: {node.status.value}")
            for outcome in node.runs:
                attempt = outcome.attempt_status.value if outcome.attempt_status else "-"
                print(f"  run {outcome.run_id}: {outcome.status.value} (attempt {attempt})")
                for artifact in outcome.artifacts:
                    print(f"    {artifact.kind}: {artifact.uri}")
    finally:
        await host.close()


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(run(parse()))
