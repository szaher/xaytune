"""``python -m xaytune.daemon --state STATE_DB --config MODULE:FACTORY``

The daemon process's entrypoint, not the Xaytune CLI (PR-029). It runs in the
foreground and does not daemonize itself: supervise it with systemd, launchd,
a container or tmux. SIGTERM and SIGINT shut it down in a controlled way
(ADR-004 §7); workloads keep running.

Exit status: 0 after a controlled shutdown; 2 for a configuration error; 3 when
another daemon already holds the state database; 4 on a platform without
``fcntl`` locking.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from collections.abc import Sequence

from xaytune.daemon.config import DaemonConfigurationError, load_config
from xaytune.daemon.lock import (
    DaemonAlreadyRunningError,
    UnsupportedPlatformError,
    require_locking,
)
from xaytune.daemon.server import LocalDaemonControllerServer

EXIT_CONFIGURATION = 2
EXIT_ALREADY_RUNNING = 3
EXIT_UNSUPPORTED_PLATFORM = 4


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m xaytune.daemon",
        description="Run a local Xaytune controller over one state database.",
    )
    parser.add_argument("--state", required=True, help="the control-plane SQLite database")
    parser.add_argument(
        "--config",
        required=True,
        help="'package.module:factory' returning a DaemonConfig",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="seconds between looks at the request mailbox (default 0.5)",
    )
    parser.add_argument("--instance-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    try:
        # First: an event loop without signal handlers would otherwise fail
        # with an unrelated error before the lock is ever tried.
        require_locking()
    except UnsupportedPlatformError as exc:
        print(f"xaytune daemon: {exc}", file=sys.stderr)
        return EXIT_UNSUPPORTED_PLATFORM
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        config = load_config(args.config)
        daemon = LocalDaemonControllerServer(
            args.state,
            config,
            poll_interval=args.poll_interval,
            instance_id=args.instance_id,
        )
    except (DaemonConfigurationError, ValueError) as exc:
        print(f"xaytune daemon: {exc}", file=sys.stderr)
        return EXIT_CONFIGURATION
    try:
        asyncio.run(_serve(daemon))
    except DaemonAlreadyRunningError as exc:
        print(f"xaytune daemon: {exc}", file=sys.stderr)
        return EXIT_ALREADY_RUNNING
    except UnsupportedPlatformError as exc:
        print(f"xaytune daemon: {exc}", file=sys.stderr)
        return EXIT_UNSUPPORTED_PLATFORM
    return 0


async def _serve(daemon: LocalDaemonControllerServer) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    await daemon.serve(stop)


if __name__ == "__main__":
    sys.exit(main())
