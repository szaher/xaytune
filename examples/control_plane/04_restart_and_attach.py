"""Submit from one process, end it, and adopt the running experiment from another.

    python examples/control_plane/04_restart_and_attach.py start \\
        --model /abs/model --dataset /abs/train.jsonl
    # prints an experiment id, then exits while training continues

    python examples/control_plane/04_restart_and_attach.py attach <experiment-id>

The workload belongs to the runtime, not to the process that submitted it.
``start`` records the submission, sees the runtime accept it, and exits. The
worker keeps training. ``attach`` reads the record, finds the attempt the
runtime still holds, and observes it from the durable telemetry cursor -- it
never submits it a second time. Run ``attach`` after training has finished and
it adopts the outcome instead.

Both commands must use the same ``--workdir``: that is where the record is.
"""

from __future__ import annotations

import argparse
import asyncio

from sft_experiment import add_arguments, experiment, state_path

from xaytune.experiment import EmbeddedControllerHost


async def start(args: argparse.Namespace) -> None:
    host = EmbeddedControllerHost(state_path(args))
    try:
        handle = await host.submit(experiment(args))
        print(f"submitted {handle.experiment_id}; status {(await handle.status()).value}")
    finally:
        # Stops observing. The workload is the runtime's and keeps running.
        await host.close()
    print("this process is ending; training continues. Attach with:")
    print(f"  python {__file__} attach {handle.experiment_id} --workdir {args.workdir}")


async def attach(args: argparse.Namespace) -> None:
    host = EmbeddedControllerHost(state_path(args))
    try:
        handle = await host.attach(args.experiment_id)
        print(f"attached to {handle.experiment_id}; status {(await handle.status()).value}")
        result = await handle.wait()
        print(f"settled; status {result.status.value}, next stage {result.next_stage}")
        for node in result.nodes:
            for outcome in node.runs:
                attempt = outcome.attempt_status.value if outcome.attempt_status else "-"
                print(f"  run {outcome.run_id}: {outcome.status.value} (attempt {attempt})")
                for artifact in outcome.artifacts:
                    print(f"    {artifact.kind}: {artifact.uri}")
    finally:
        await host.close()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    add_arguments(commands.add_parser("start", help="submit, then exit while it trains"))
    attaching = commands.add_parser("attach", help="adopt a submitted experiment")
    attaching.add_argument("experiment_id")
    attaching.add_argument("--workdir", default="xaytune-workdir")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    asyncio.run(start(args) if args.command == "start" else attach(args))
