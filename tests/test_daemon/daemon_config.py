"""The configuration the daemon subprocess tests run with (``--config``).

``tests.test_daemon.daemon_config:create_config``. Every implementation is
named, as the daemon requires. ``XAYTUNE_TEST_FAULT`` injects a crash:

``kill-before-admission``  the process dies resolving the runtime, after
                           the request was read and before anything was
                           admitted: the request stays PENDING.
``kill-before-submit``     dies after admission, before the runtime got the
``kill-after-submit``      submission -- or after it did, before the daemon
                           recorded it: see :mod:`.file_runtime`.
"""

from __future__ import annotations

import os
import signal
from collections.abc import Mapping
from typing import Any

from tests.test_daemon.file_runtime import FileRuntime
from xaytune.compilation.native import NativeCompiler
from xaytune.daemon import DaemonConfig
from xaytune.decision import ThresholdDecisionEngine
from xaytune.planning import PLANNERS
from xaytune.policy import DenyAllPolicy


def create_config() -> DaemonConfig:
    fault = os.environ.get("XAYTUNE_TEST_FAULT")

    def file_runtime(config: Mapping[str, Any]) -> Any:
        if fault == "kill-before-admission":
            os.kill(os.getpid(), signal.SIGKILL)
        return FileRuntime(config["root"], fault=fault)

    return DaemonConfig(
        compilers={"native": NativeCompiler},
        runtimes={"file": file_runtime},
        evaluators={},
        planners=PLANNERS,
        decision_engine=ThresholdDecisionEngine(),
        policy=DenyAllPolicy(),
        checkpoint_manager=None,
        recovery_request_for_incident=None,
    )


def not_a_config() -> object:
    return {"compilers": {}}
