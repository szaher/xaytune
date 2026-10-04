"""``xaytune submit|attach|status|watch|events|results|actions|cancel|approve|reject`` (PR-029).

The command line over :class:`~xaytune.daemon.LocalDaemonControllerHost`, and
nothing else: every command reads the durable record or sends a mailbox
request through that host, so the CLI has no way of its own to change the
database or call a runtime. A daemon serving the same state database carries
every request out:

```text
xaytune submit experiment.yaml --state state.db    → the experiment id; exits
xaytune watch <experiment id> --state state.db     events until it is at rest
xaytune cancel <experiment id> --state state.db    a cancel request
xaytune actions <experiment id> --state state.db   pending approvals among them
xaytune approve <action id> --reason "..." --state state.db
```

**Killing a command is safe.** A mutating command prints its request id
before it sends the request. Once sent, the request is the daemon's: the
command can be killed at any point and the work still happens. To find out
how a killed command's request ended, run it again with ``--request-id`` --
the same request, which the daemon never carries out twice.

``--state`` defaults to ``$XAYTUNE_STATE``. Heavy imports wait until a
command runs, so ``xaytune --help`` stays fast.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xaytune.daemon import LocalDaemonControllerHost
    from xaytune.experiment import ExperimentResult

__all__ = ["CONTROL_COMMANDS", "add_control_commands", "run_control_command"]

CONTROL_COMMANDS = (
    "submit",
    "attach",
    "status",
    "watch",
    "events",
    "results",
    "actions",
    "cancel",
    "approve",
    "reject",
)

_STATE_VARIABLE = "XAYTUNE_STATE"


def add_control_commands(subparsers: Any) -> None:
    """Register the daemon commands on the ``xaytune`` parser's subcommands."""

    def command(name: str, help: str) -> argparse.ArgumentParser:
        parser: argparse.ArgumentParser = subparsers.add_parser(name, help=help)
        parser.add_argument(
            "--state",
            default=os.environ.get(_STATE_VARIABLE),
            help=f"The daemon's state database (default: ${_STATE_VARIABLE})",
        )
        return parser

    def mutating(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--request-id",
            default=None,
            help="Send this request again: how to retry, or learn the outcome of, "
            "a command that was killed",
        )
        parser.add_argument(
            "--timeout",
            type=float,
            default=None,
            help="Seconds to wait for the daemon to carry the request out "
            "(default: as long as it takes)",
        )

    submit = command("submit", "Hand an experiment to the daemon; print its id")
    submit.add_argument("spec", help="ExperimentSpec as YAML or JSON")
    submit.add_argument("--experiment-id", default=None, help="Use this experiment id")
    mutating(submit)

    attach = command("attach", "Ask the daemon to adopt an experiment already recorded")
    attach.add_argument("experiment_id")
    mutating(attach)

    status = command("status", "Where an experiment stands now")
    status.add_argument("experiment_id")
    status.add_argument("--json", action="store_true", help="Print the result as JSON")

    watch = command("watch", "Follow an experiment's events until the daemon is at rest on it")
    watch.add_argument("experiment_id")
    watch.add_argument("--after", type=int, default=0, help="Only events after this sequence")
    watch.add_argument("--timeout", type=float, default=None, help="Give up after SECONDS")

    events = command("events", "An experiment's durable events")
    events.add_argument("experiment_id")
    events.add_argument("--after", type=int, default=0, help="Only events after this sequence")
    events.add_argument("--follow", action="store_true", help="Keep printing new events")
    events.add_argument("--json", action="store_true", help="One JSON object per line")

    results = command("results", "Wait until the daemon is at rest; print the result as JSON")
    results.add_argument("experiment_id")
    results.add_argument(
        "--no-wait", action="store_true", help="Print the record's result now, without waiting"
    )
    results.add_argument("--timeout", type=float, default=None, help="Give up after SECONDS")

    actions = command("actions", "An experiment's actions, as governance left them")
    actions.add_argument("experiment_id")
    actions.add_argument("--json", action="store_true", help="One JSON object per line")

    cancel = command("cancel", "Ask the daemon to cancel an experiment")
    cancel.add_argument("experiment_id")
    cancel.add_argument("--reason", default="cancelled from the xaytune CLI")
    mutating(cancel)

    for name, verb in (("approve", "Approve"), ("reject", "Reject")):
        resolve = command(name, f"{verb} an action awaiting approval, as a human")
        resolve.add_argument("action_id")
        resolve.add_argument("--reason", required=True, help="Why; recorded with the decision")
        resolve.add_argument(
            "--approver", default=None, help="Who is deciding (default: your user name)"
        )
        mutating(resolve)


def run_control_command(args: argparse.Namespace) -> int:
    """Run one daemon command; its exit status."""
    from pydantic import ValidationError

    from xaytune.core.errors import XaytuneError

    if not args.state:
        print(f"error: --state is required (or set ${_STATE_VARIABLE})", file=sys.stderr)
        return 2
    run: Callable[[LocalDaemonControllerHost, argparse.Namespace], Coroutine[Any, Any, int]] = (
        _COMMANDS[args.command]
    )
    try:
        return asyncio.run(_with_host(args, run))
    except KeyboardInterrupt:
        return 130
    except (
        XaytuneError,
        ValidationError,
        TimeoutError,
        asyncio.TimeoutError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {_describe(error)}", file=sys.stderr)
        return 1


async def _with_host(
    args: argparse.Namespace,
    run: Callable[[LocalDaemonControllerHost, argparse.Namespace], Coroutine[Any, Any, int]],
) -> int:
    from xaytune.daemon import LocalDaemonControllerHost

    host = LocalDaemonControllerHost(args.state, handoff_timeout=getattr(args, "timeout", None))
    try:
        return await run(host, args)
    finally:
        await host.close()


# ---- the commands ---------------------------------------------------------------


async def _submit(host: LocalDaemonControllerHost, args: argparse.Namespace) -> int:
    import yaml

    from xaytune.core.ids import ControllerRequestId
    from xaytune.experiment import ExperimentSpec

    with open(args.spec, encoding="utf-8") as source:
        spec = ExperimentSpec.model_validate(yaml.safe_load(source))
    request_id = _request_id(args, ControllerRequestId)
    _warn_if_no_daemon(host)
    handle = await host.submit(spec, request_id=request_id, experiment_id=args.experiment_id)
    print(handle.experiment_id)
    return 0


async def _attach(host: LocalDaemonControllerHost, args: argparse.Namespace) -> int:
    from xaytune.core.ids import ControllerRequestId

    request_id = _request_id(args, ControllerRequestId)
    _warn_if_no_daemon(host)
    handle = await host.attach(args.experiment_id, request_id=request_id)
    _print_result(host.result(handle.experiment_id))
    return 0


async def _status(host: LocalDaemonControllerHost, args: argparse.Namespace) -> int:
    result = host.result(args.experiment_id)
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _print_result(result)
    return 0


async def _watch(host: LocalDaemonControllerHost, args: argparse.Namespace) -> int:
    handle = host.handle(args.experiment_id)
    waiting = asyncio.ensure_future(handle.wait())
    cursor = args.after
    loop = asyncio.get_running_loop()
    deadline = None if args.timeout is None else loop.time() + args.timeout
    try:
        while True:
            # Looked at before printing: once the daemon is at rest, the
            # events printed next include the last one it rested on.
            rested = waiting.done()
            for event in host._events_after(handle.experiment_id, cursor):
                assert event.sequence is not None
                cursor = event.sequence
                print(_event_line(event), flush=True)
            if rested:
                break
            if deadline is not None and loop.time() >= deadline:
                raise TimeoutError(f"experiment {handle.experiment_id} is not at rest yet")
            await asyncio.wait({waiting}, timeout=0.2)
        result = waiting.result()
    finally:
        waiting.cancel()
    _print_result(result)
    return 0


async def _events(host: LocalDaemonControllerHost, args: argparse.Namespace) -> int:
    handle = host.handle(args.experiment_id)
    cursor = args.after
    while True:
        batch = host._events_after(handle.experiment_id, cursor)
        for event in batch:
            assert event.sequence is not None
            cursor = event.sequence
            print(event.model_dump_json() if args.json else _event_line(event), flush=True)
        if not batch:
            if not args.follow:
                return 0
            await asyncio.sleep(0.2)


async def _results(host: LocalDaemonControllerHost, args: argparse.Namespace) -> int:
    handle = host.handle(args.experiment_id)
    if args.no_wait:
        result = host.result(handle.experiment_id)
    else:
        result = await asyncio.wait_for(handle.wait(), args.timeout)
    print(result.model_dump_json(indent=2))
    return 0


async def _actions(host: LocalDaemonControllerHost, args: argparse.Namespace) -> int:
    for governed in await host.handle(args.experiment_id).actions():
        if args.json:
            print(governed.model_dump_json())
            continue
        action = governed.action
        verdict = governed.decision.verdict.value if governed.decision is not None else "-"
        print(
            f"{action.id}  {action.status.value:<16} {action.type:<28} "
            f"{action.target.kind}:{action.target.id}  policy={verdict}  {action.reason}"
        )
    return 0


async def _cancel(host: LocalDaemonControllerHost, args: argparse.Namespace) -> int:
    from xaytune.core.ids import ControllerRequestId

    request_id = _request_id(args, ControllerRequestId)
    _warn_if_no_daemon(host)
    await host.cancel(args.experiment_id, reason=args.reason, request_id=request_id)
    print(f"cancellation requested; the daemon issued its effects ({args.experiment_id})")
    return 0


async def _resolve(host: LocalDaemonControllerHost, args: argparse.Namespace) -> int:
    import getpass

    from xaytune.core.ids import ControllerRequestId
    from xaytune.core.refs import Actor

    request_id = _request_id(args, ControllerRequestId)
    approver = Actor(type="human", id=args.approver or getpass.getuser())
    _warn_if_no_daemon(host)
    resolve = host.approve_action if args.command == "approve" else host.reject_action
    governed = await resolve(
        args.action_id, approver=approver, reason=args.reason, request_id=request_id
    )
    print(f"{governed.action.id}  {governed.action.status.value}")
    return 0


_COMMANDS: dict[
    str, Callable[[LocalDaemonControllerHost, argparse.Namespace], Coroutine[Any, Any, int]]
] = {
    "submit": _submit,
    "attach": _attach,
    "status": _status,
    "watch": _watch,
    "events": _events,
    "results": _results,
    "actions": _actions,
    "cancel": _cancel,
    "approve": _resolve,
    "reject": _resolve,
}


# ---- output -------------------------------------------------------------------


def _request_id(args: argparse.Namespace, minted: Any) -> Any:
    """The request id to send: the caller's, to retry, or a new one -- printed before it is sent."""
    request_id = args.request_id or minted.generate()
    print(f"request {request_id}", file=sys.stderr, flush=True)
    return request_id


def _warn_if_no_daemon(host: LocalDaemonControllerHost) -> None:
    from xaytune.core.clock import utc_now

    lease = host.client.lease()
    if lease is None or not lease.is_live(utc_now()):
        print(
            "warning: no daemon holds a live lease on this database; the request is "
            "recorded and will be carried out when one starts",
            file=sys.stderr,
            flush=True,
        )


def _print_result(result: ExperimentResult) -> None:
    print(f"experiment  {result.experiment_id}")
    print(f"status      {result.status.value}")
    print(f"quiescent   {str(result.quiescent).lower()}")
    print(f"next stage  {result.next_stage or '-'}")
    for node in result.nodes:
        print(f"node        {node.node_id}  {node.status.value}")
        for run in node.runs:
            attempt = run.attempt_status.value if run.attempt_status is not None else "-"
            print(f"  run       {run.run_id}  {run.status.value}  (attempt {attempt})")
        for evaluation in node.evaluations:
            print(
                f"  eval      {evaluation.evaluation_run_id}  cycle {evaluation.evaluation_cycle}"
                f"  {evaluation.status.value}"
            )
    if result.budget is not None:
        print(f"budget      {json.dumps(result.budget.model_dump(mode='json'), sort_keys=True)}")


def _event_line(event: Any) -> str:
    return (
        f"{event.sequence:>6}  {event.occurred_at.isoformat(timespec='seconds')}  "
        f"{event.event_type:<28} {event.aggregate_type}:{event.aggregate_id}"
    )


def _describe(error: BaseException) -> str:
    return str(error) or type(error).__name__
