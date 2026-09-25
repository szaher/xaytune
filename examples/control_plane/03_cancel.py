"""Cancel a running experiment, and see the record say when it stopped.

    python examples/control_plane/03_cancel.py --model /abs/model --dataset /abs/train.jsonl

Cancellation is intent recorded first -- an Action -- then carried out as a
cancel operation for each live attempt. The experiment stays ``ACTIVE`` while
that happens and becomes ``CANCELLED`` only once no workload it owns is still
executing, so the status never claims a stop that has not happened.
"""

from __future__ import annotations

import argparse
import asyncio

from sft_experiment import add_arguments, experiment, state_path

from xaytune.experiment import EmbeddedControllerHost, ExperimentHandle


async def until_training(handle: ExperimentHandle) -> None:
    """Return once the attempt is recorded as running."""
    async for event in handle.events():
        if event.aggregate_type == "RunAttempt" and event.payload.get("status") == "running":
            return


async def run(args: argparse.Namespace) -> None:
    host = EmbeddedControllerHost(state_path(args))
    try:
        handle = await host.submit(experiment(args))
        print(f"submitted {handle.experiment_id}")
        await until_training(handle)
        print(f"training; status {(await handle.status()).value}")

        await handle.cancel(reason="example: cancelled while training")
        print(f"cancel requested; status {(await handle.status()).value}")

        result = await handle.wait()
        print(f"settled; status {result.status.value}, next stage {result.next_stage}")
        for node in result.nodes:
            for outcome in node.runs:
                attempt = outcome.attempt_status.value if outcome.attempt_status else "-"
                print(f"  run {outcome.run_id}: {outcome.status.value} (attempt {attempt})")
    finally:
        await host.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    asyncio.run(run(parser.parse_args()))
