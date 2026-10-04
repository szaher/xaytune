"""A deterministic agent model that answers from a script, for tests and offline runs.

:class:`ScriptedAgentModel` makes no network call and needs no credentials:
each ``generate`` takes the next entry of its script. That is what lets a
planner built on :class:`~xaytune.agent.model.AgentModel` be tested in CI
exactly, including the answers a real model gets wrong -- a scripted answer is
**not** checked here, so a malformed one reaches
:func:`~xaytune.agent.model.invoke_agent_model` and is refused there, as a real
model's would be.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

from xaytune._version import __version__
from xaytune.agent.model import (
    AgentModelDescriptor,
    AgentModelIdentity,
    AgentModelInvocationError,
    AgentModelRequest,
    AgentModelResponse,
)
from xaytune.core.capabilities import PLUGIN_API_VERSIONS, PluginDescriptor
from xaytune.core.immutable import FrozenDict

__all__ = ["SCRIPTED_MODEL", "ScriptedAgentModel", "ScriptEntry"]

SCRIPTED_MODEL = AgentModelIdentity(provider="xaytune", name="scripted")

ScriptEntry = Mapping[str, Any] | AgentModelResponse | BaseException
"""Content to answer with, a whole response to return as is, or an exception to raise."""


class ScriptedAgentModel:
    """Answers each request with the next entry of its script, and remembers what it was asked.

    A mapping is answered as content, with ``finish_reason="stop"`` and usage
    unknown -- a script has no tokens to count. A response is returned
    unchanged; an exception is raised. A request past the end of the script
    raises :class:`~xaytune.agent.model.AgentModelInvocationError`.
    """

    def __init__(
        self, script: Iterable[ScriptEntry], *, model: AgentModelIdentity = SCRIPTED_MODEL
    ) -> None:
        self.descriptor = AgentModelDescriptor(
            plugin=PluginDescriptor(
                api_version=PLUGIN_API_VERSIONS[0],
                name="scripted-agent-model",
                plugin_version=__version__,
                provider="xaytune",
                xaytune_version=__version__,
            ),
            model=model,
        )
        self._script: deque[ScriptEntry] = deque(script)
        self._requests: list[AgentModelRequest] = []

    @property
    def requests(self) -> tuple[AgentModelRequest, ...]:
        """Every request asked so far, in order -- including ones answered with an exception."""
        return tuple(self._requests)

    @property
    def remaining(self) -> int:
        """How many script entries are left."""
        return len(self._script)

    async def generate(self, request: AgentModelRequest) -> AgentModelResponse:
        self._requests.append(request)
        if not self._script:
            raise AgentModelInvocationError(
                f"the script ran out after {len(self._requests) - 1} answers"
            )
        entry = self._script.popleft()
        if isinstance(entry, BaseException):
            raise entry
        if isinstance(entry, AgentModelResponse):
            return entry
        model = self.descriptor.model
        return AgentModelResponse(
            content=FrozenDict(entry),
            model=model.name,
            model_revision=model.revision,
            finish_reason="stop",
        )
