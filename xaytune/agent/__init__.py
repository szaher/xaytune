"""Agent models: the boundary where xaytune asks a model for structured output (PR-030).

```text
PlanningContext ─▶ LLMPlanner (PR-031) ─▶ AgentModelRequest ─▶ AgentModel ─▶ AgentModelResponse
                                                               (this package)
```

This package knows how to ask a model for an answer that conforms to a
schema, and how to refuse one that does not. It knows nothing of experiments,
candidates, actions, policy, storage or runtimes, and imports only
:mod:`xaytune.core`. Provider adapters implement :class:`AgentModel`;
:class:`ScriptedAgentModel` is the deterministic one for tests.
"""

from __future__ import annotations

from xaytune.agent.model import (
    AGENT_MODEL_REQUEST_IDENTITY_VERSION,
    AgentMessage,
    AgentModel,
    AgentModelDescriptor,
    AgentModelError,
    AgentModelIdentity,
    AgentModelInvocationError,
    AgentModelOutputError,
    AgentModelRequest,
    AgentModelResponse,
    AgentModelUsage,
    agent_model_request_identity_v1,
    invoke_agent_model,
)
from xaytune.agent.schema import (
    UnsupportedResponseSchemaError,
    check_response_schema,
    schema_violations,
)
from xaytune.agent.scripted import SCRIPTED_MODEL, ScriptedAgentModel, ScriptEntry

__all__ = [
    "AGENT_MODEL_REQUEST_IDENTITY_VERSION",
    "SCRIPTED_MODEL",
    "AgentMessage",
    "AgentModel",
    "AgentModelDescriptor",
    "AgentModelError",
    "AgentModelIdentity",
    "AgentModelInvocationError",
    "AgentModelOutputError",
    "AgentModelRequest",
    "AgentModelResponse",
    "AgentModelUsage",
    "ScriptEntry",
    "ScriptedAgentModel",
    "UnsupportedResponseSchemaError",
    "agent_model_request_identity_v1",
    "check_response_schema",
    "invoke_agent_model",
    "schema_violations",
]
