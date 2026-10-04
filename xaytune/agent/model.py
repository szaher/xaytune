"""The agent-model boundary: how xaytune asks a model for structured output (PR-030).

```text
AgentModelRequest ──invoke_agent_model()──▶ AgentModel.generate() ──▶ AgentModelResponse
                     (descriptor checked,                           (content checked against
                      exactly one call)                              the request's schema)
```

**An agent model knows how to ask a model for structured output, and nothing
else.** It does not know what an experiment, candidate, action, policy,
repository, runtime or controller is; turning a planning context into a
request, and an answer into a proposal, is a planner's work (PR-031). It holds
no execution authority: no runtime, cluster client, filesystem, shell or
repository is reachable from here, and nothing an answer says is executed.

Rules this contract fixes, so adapters cannot each decide them:

- **Structured output only.** A request always carries a response schema
  (:mod:`xaytune.agent.schema`); free text never comes back as an answer.
- **Fail closed.** :func:`invoke_agent_model` checks every answer against its
  request's schema and refuses the whole answer on any violation
  (:class:`AgentModelOutputError`). Nothing is partially accepted.
- **Provider-neutral.** Requests and responses are plain serializable records;
  a provider SDK's objects never cross this boundary, in either direction.
- **No hidden reasoning.** A caller that wants a rationale asks for one as a
  field of its schema, and gets exactly that field. Nothing here requests,
  returns or records chain-of-thought.
- **Identity.** :func:`agent_model_request_identity_v1` names one logical
  request: the model identity, system prompt, messages, response schema and
  generation parameters. Not usage, latency, provider request ids or the
  answer -- so a recorded invocation (PR-032) can say which request it answered.
- **One call, one invocation.** An adapter may retry transport failures inside
  ``generate``; the caller still sees one call, for one request identity.
- **No secrets.** Credentials are adapter configuration. They never enter a
  request, a response or a descriptor, which is all that is ever serialized.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import Field, StrictInt, field_validator, model_validator

from xaytune.agent.schema import check_response_schema, schema_violations
from xaytune.core.capabilities import PluginDescriptor, require_supported_plugin
from xaytune.core.fingerprint import fingerprint
from xaytune.core.immutable import FrozenDict, FrozenDomainModel, thaw
from xaytune.core.observability import Finite

__all__ = [
    "AGENT_MODEL_REQUEST_IDENTITY_VERSION",
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
    "agent_model_request_identity_v1",
    "invoke_agent_model",
]

AGENT_MODEL_REQUEST_IDENTITY_VERSION = 1

_Name = Annotated[str, Field(min_length=1)]
_Count = Annotated[StrictInt, Field(ge=0)]


class AgentModelError(Exception):
    """An agent model gave no usable answer."""


class AgentModelInvocationError(AgentModelError):
    """The model could not be asked, or did not answer: transport, provider or adapter failure.

    Its message is safe to log and record. One raised by
    :func:`invoke_agent_model` for an adapter's own exception carries only
    that exception's type and the request fingerprint: SDK and adapter text
    can hold URLs, headers, credentials or provider payloads. The exception
    itself is discarded at the boundary -- neither ``__cause__`` nor
    ``__context__`` holds it, so no traceback can print it. An adapter with
    a diagnostic it has deliberately sanitized raises this error itself.
    """


class AgentModelOutputError(AgentModelError):
    """The model answered, and the answer is not what was asked for. Nothing of it is used."""

    def __init__(self, request_fingerprint: str, reasons: tuple[str, ...]) -> None:
        self.request_fingerprint = request_fingerprint
        self.reasons = reasons
        super().__init__(
            f"agent model output refused for request {request_fingerprint}: " + "; ".join(reasons)
        )


class AgentModelIdentity(FrozenDomainModel):
    """Which model answers: who serves it, its name, and a pinned revision if there is one.

    ``provider`` is the model's provider (``anthropic``, ``openai``, ``vllm``),
    not the adapter's -- that is the descriptor's plugin. ``revision`` is
    ``None`` when the provider exposes no pinned revision; it is never guessed.
    """

    provider: _Name
    name: _Name
    revision: _Name | None = None


class AgentModelDescriptor(FrozenDomainModel):
    """An agent model's identity: the adapter that calls it, and the model it calls.

    The two are separate because they change separately: the same adapter
    release calls many models, and one model is reachable through several
    adapters. Adapter configuration -- endpoints, credentials -- is not here.
    """

    plugin: PluginDescriptor
    model: AgentModelIdentity


class AgentMessage(FrozenDomainModel):
    """One turn of the conversation a request carries."""

    role: Literal["user", "assistant"]
    content: _Name


class AgentModelRequest(FrozenDomainModel):
    """One logical request for a structured answer: durable, serializable model input.

    ``response_schema`` is mandatory and must be in the supported subset
    (:mod:`xaytune.agent.schema`); a request asking for anything else cannot
    be built. ``messages`` end with the user's turn. ``temperature`` and
    ``max_output_tokens`` left as ``None`` mean the adapter's defaults, and
    are part of the request's identity as ``None``.
    """

    system: str
    messages: tuple[AgentMessage, ...] = Field(min_length=1)
    response_schema: FrozenDict
    temperature: Annotated[Finite, Field(ge=0)] | None = None
    max_output_tokens: Annotated[StrictInt, Field(ge=1)] | None = None

    @field_validator("response_schema")
    @classmethod
    def _supported_schema(cls, schema: FrozenDict) -> FrozenDict:
        check_response_schema(schema)
        return schema

    @model_validator(mode="after")
    def _ends_with_the_user(self) -> AgentModelRequest:
        if self.messages[-1].role != "user":
            raise ValueError("a request's messages end with the user's turn")
        return self

    def fingerprint(self, model: AgentModelIdentity) -> str:
        """This request's identity when sent to *model*: :func:`agent_model_request_identity_v1`."""
        return fingerprint(agent_model_request_identity_v1(model, self))


def agent_model_request_identity_v1(
    model: AgentModelIdentity, request: AgentModelRequest
) -> Mapping[str, Any]:
    """What makes two model requests the same logical request, version 1.

    An explicit projection rather than a dump, so a field added later is a
    deliberate change of identity; a test pins both models' fields against it.
    It names:

    - the model: provider, name and revision;
    - the system prompt, and each message's role and content, in order;
    - the response schema;
    - the generation parameters: temperature and maximum output tokens.

    Not the adapter or its version: they decide how a request is carried, not
    what it asks. Not usage, latency, provider request ids or the answer:
    those describe one attempt at the request, not the request.
    """
    return {
        "version": AGENT_MODEL_REQUEST_IDENTITY_VERSION,
        "model": {"provider": model.provider, "name": model.name, "revision": model.revision},
        "system": request.system,
        "messages": [
            {"role": message.role, "content": message.content} for message in request.messages
        ],
        "response_schema": thaw(request.response_schema),
        "generation": {
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
        },
    }


class AgentModelUsage(FrozenDomainModel):
    """Tokens one answer used, normalized across providers.

    A count the provider did not report is ``None``: unknown, never estimated.
    """

    input_tokens: _Count | None = None
    output_tokens: _Count | None = None


class AgentModelResponse(FrozenDomainModel):
    """A model's structured answer, and what the provider said about producing it.

    ``model`` and ``model_revision`` are what the provider reports serving.
    The name may be more specific than the one asked for (an alias resolved);
    a reported revision must equal a pinned one (:func:`invoke_agent_model`),
    and ``None`` means the provider did not say. ``content`` is
    the answer; everything else describes this attempt and is not part of the
    request's identity.
    """

    content: FrozenDict
    model: _Name
    model_revision: _Name | None = None
    finish_reason: _Name | None = None
    usage: AgentModelUsage = Field(default_factory=AgentModelUsage)
    provider_request_id: _Name | None = None
    latency_seconds: Annotated[Finite, Field(ge=0)] | None = None


@runtime_checkable
class AgentModel(Protocol):
    """Asks one model for structured answers. Callers go through :func:`invoke_agent_model`."""

    descriptor: AgentModelDescriptor

    async def generate(self, request: AgentModelRequest) -> AgentModelResponse:
        """Ask the model *request*, once.

        One call is one logical invocation: an adapter may retry transport
        failures inside it, but returns one answer or raises. Failing to get
        an answer raises :class:`AgentModelInvocationError`. Checking the
        answer against the schema is the boundary's job, so an adapter cannot
        forget it; an adapter may check earlier too.
        """
        ...


async def invoke_agent_model(model: AgentModel, request: AgentModelRequest) -> AgentModelResponse:
    """Ask *model* *request* once, and return the answer only if it is what was asked for.

    Raises:
        IncompatiblePluginError: The adapter speaks a plugin API this build
            does not implement.
        AgentModelInvocationError: The model gave no answer. For an adapter's
            own exception, only its type crosses the boundary: not its text,
            and not the exception (no cause, no context).
        AgentModelOutputError: The answer is not an :class:`AgentModelResponse`,
            it reports a revision other than the pinned one, or its content
            violates the request's schema anywhere.
    """
    require_supported_plugin(model.descriptor.plugin)
    request_fingerprint = request.fingerprint(model.descriptor.model)
    failure: str | None = None
    try:
        response = await model.generate(request)
    except AgentModelError:
        raise
    except Exception as error:
        failure = type(error).__name__
    if failure is not None:
        # Raised outside the except block, so the adapter's exception is
        # neither the cause nor the context: ``from None`` would only hide
        # the context from tracebacks, and still keep the object reachable.
        raise AgentModelInvocationError(
            f"agent model {model.descriptor.model.name!r} failed on request "
            f"{request_fingerprint}: {failure}"
        )
    if not isinstance(response, AgentModelResponse):
        raise AgentModelOutputError(
            request_fingerprint,
            (f"the adapter returned {type(response).__name__}, not an AgentModelResponse",),
        )
    pinned = model.descriptor.model.revision
    if pinned is not None and response.model_revision not in (None, pinned):
        raise AgentModelOutputError(
            request_fingerprint,
            (f"the model reports revision {response.model_revision!r}, not the pinned {pinned!r}",),
        )
    violations = schema_violations(request.response_schema, response.content)
    if violations:
        raise AgentModelOutputError(request_fingerprint, violations)
    return response
