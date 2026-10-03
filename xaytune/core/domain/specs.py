"""What the durable experiment stores in place of implementations (ADR-016).

A controller that is restarted has the record and nothing else: no Python
objects from the process that created the experiment, no closures, no open
handles. So the record names *what* ran it -- a compiler, a runtime -- as data
the plugin registry (ADR-008) resolves back into an implementation, with the
version resolved at submission so provenance can say which implementation, not
merely which name.

These are ADR-016's ``RuntimeSpec`` and ``PlannerSpec``, and a
``CompilerSpec`` alongside them: the compiler is chosen per experiment in this
phase, before capability resolution chooses one per candidate.
``ControllerHostSpec`` arrives with a second host; until then the experiment's
``controller_host`` reference says which host owns it.
"""

from __future__ import annotations

from pydantic import Field

from xaytune.core.execution import SecretRef
from xaytune.core.immutable import FrozenDict, FrozenDomainModel

__all__ = ["CompilerSpec", "PlannerSpec", "RuntimeSpec"]


class CompilerSpec(FrozenDomainModel):
    """Which ``TrainerCompiler`` compiles this experiment's candidates.

    ``version`` is ``None`` as a request and set once bound: the host resolves
    the name through the registry and records the descriptor's
    ``plugin_version``, so the record says which compiler ran rather than which
    name was asked for.
    """

    name: str = Field(min_length=1)
    version: str | None = None


class RuntimeSpec(FrozenDomainModel):
    """Which ``RuntimeBackend`` executes this experiment's attempts, and how.

    ``config`` is canonical JSON only, deep-frozen, because it is persisted and
    read back by a process that did not create it. Credentials are referenced,
    never stored: a token in ``config`` would be written to the database and
    copied into every bug report that includes an experiment record.
    """

    kind: str = Field(min_length=1)
    version: str | None = None
    config: FrozenDict = Field(default_factory=FrozenDict)
    credentials_ref: SecretRef | None = None


class PlannerSpec(FrozenDomainModel):
    """Which ``Planner`` proposes this experiment's next work, configured how (ADR-016 §1).

    ``kind`` names the planner in the host's registry; ``version`` is ``None``
    as a request and set once bound to the resolved descriptor's
    ``plugin_version``, so the record says which planner, not which name.
    ``config`` is canonical JSON only: the planner validates it into its own
    typed configuration when it is bound, and refuses what it does not
    understand, so a malformed planner is refused at submission rather than
    when the experiment first needs a plan.
    """

    kind: str = Field(min_length=1)
    version: str | None = None
    config: FrozenDict = Field(default_factory=FrozenDict)
