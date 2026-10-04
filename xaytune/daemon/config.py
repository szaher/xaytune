"""What the daemon runs with: every implementation, chosen explicitly (ADR-004 §6).

The daemon never falls back to the embedded host's defaults. A project names
its compilers, runtimes, evaluators, planners, decision engine, policy and
recovery configuration in a factory, and points the daemon at it::

    # myproject/xaytune_config.py
    def create_config() -> DaemonConfig:
        return DaemonConfig(
            compilers={"native": NativeCompiler},
            runtimes={"local": lambda config: LocalRuntime(config["root"])},
            evaluators={"native": NativeEvaluator},
            planners=PLANNERS,
            decision_engine=AdaptiveThresholdDecisionEngine(),
            policy=DenyAllPolicy(),
            checkpoint_manager=None,
            recovery_request_for_incident=None,
        )

    python -m xaytune.daemon --state state.db --config myproject.xaytune_config:create_config

Choosing a built-in is fine; choosing it silently is not. The decision engine
in particular is not yet recorded with an experiment (``DecisionEngineSpec``
persistence is separate work), so keeping it the same across daemon restarts
against one database is the operator's responsibility.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from xaytune.checkpoints import CheckpointManager
from xaytune.compilation import TrainerCompiler
from xaytune.core.domain.incident import Incident
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.core.domain.specs import PlannerSpec
from xaytune.core.errors import XaytuneError
from xaytune.decision import DecisionEngine
from xaytune.evaluation import Evaluator
from xaytune.planning import Planner
from xaytune.policy import PolicyEngine
from xaytune.runtimes import RuntimeBackend

__all__ = ["DaemonConfig", "DaemonConfigurationError", "load_config"]


class DaemonConfigurationError(XaytuneError):
    """The ``--config`` reference does not resolve to a :class:`DaemonConfig`."""


@dataclass(frozen=True, kw_only=True)
class DaemonConfig:
    """The implementations a daemon's controller uses. Every implementation is required.

    The implementation fields are those of
    :class:`~xaytune.experiment.EmbeddedControllerHost`, which the daemon
    delegates its control to; ``None`` is accepted only where the embedded
    host gives it a meaning -- no checkpoint manager, no first recovery
    request -- and must still be written down.

    ``lease_ttl_seconds`` is the one timing setting, and has a default: how
    long the daemon's controller lease lasts unrenewed (ADR-004 §8). It is
    renewed every third of that, and after a crash the next daemon waits for
    it to expire -- so it bounds both how fast a dead daemon is replaced and
    how long the event loop may stall before the daemon loses its lease.
    """

    compilers: Mapping[str, Callable[[], TrainerCompiler]]
    runtimes: Mapping[str, Callable[[Mapping[str, Any]], RuntimeBackend]]
    evaluators: Mapping[str, Callable[[], Evaluator]]
    planners: Mapping[str, Callable[[PlannerSpec], Planner]]
    decision_engine: DecisionEngine
    policy: PolicyEngine
    checkpoint_manager: CheckpointManager | None
    recovery_request_for_incident: Callable[[Incident], RecoveryRequest | None] | None
    lease_ttl_seconds: float = 30.0

    def __post_init__(self) -> None:
        ttl = self.lease_ttl_seconds
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)):
            raise DaemonConfigurationError(f"lease_ttl_seconds must be a number, not {ttl!r}")
        if not math.isfinite(ttl) or ttl <= 0:
            raise DaemonConfigurationError(
                f"lease_ttl_seconds must be finite and positive, not {ttl!r}"
            )


def load_config(reference: str) -> DaemonConfig:
    """Import ``module:factory`` and call the factory for the daemon's configuration.

    Raises:
        DaemonConfigurationError: If the reference is malformed, does not
            import, names nothing callable, or the factory returns anything
            but a :class:`DaemonConfig`.
    """
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise DaemonConfigurationError(
            f"--config must look like 'package.module:factory', not {reference!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise DaemonConfigurationError(f"cannot import {module_name!r}: {exc}") from exc
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise DaemonConfigurationError(f"{reference!r} names nothing callable")
    config = factory()
    if not isinstance(config, DaemonConfig):
        raise DaemonConfigurationError(
            f"{reference!r} returned {type(config).__name__}, not a DaemonConfig"
        )
    return config
