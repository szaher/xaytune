"""The configuration the daemon subprocess tests run with (``--config``).

``tests.test_daemon.daemon_config:create_config``. Every implementation is
named, as the daemon requires. ``XAYTUNE_TEST_FAULT`` injects a crash:

``kill-before-admission``  the process dies resolving the runtime, after
                           the request was read and before anything was
                           admitted: the request stays PENDING.
``kill-before-submit``     dies after admission, before the runtime got the
``kill-after-submit``      submission -- or after it did, before the daemon
                           recorded it: see :mod:`.file_runtime`.

``XAYTUNE_TEST_LEASE_TTL`` sets the lease TTL in seconds, so a test that
kills a daemon waits seconds, not the default 30, for its lease to expire.
``XAYTUNE_TEST_POLICY=review`` makes every applicable proposal await a
human's approval.
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
from xaytune.experiment import PolicyVerdict
from xaytune.planning import PLANNERS
from xaytune.policy import DenyAllPolicy, RulePolicyEngine


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
        policy=(
            RulePolicyEngine(default=PolicyVerdict.REQUIRE_APPROVAL)
            if os.environ.get("XAYTUNE_TEST_POLICY") == "review"
            else DenyAllPolicy()
        ),
        checkpoint_manager=None,
        recovery_request_for_incident=None,
        lease_ttl_seconds=float(os.environ.get("XAYTUNE_TEST_LEASE_TTL", "30")),
    )


def not_a_config() -> object:
    return {"compilers": {}}
