"""The compile half of the compile/execute boundary.

A :class:`TrainerCompiler` turns *what we are testing* into *how to run it*:

```text
CandidateSpec  ──compile──>  TrainingExecutionSpec
```

**A compiler compiles. It never executes.** That is the whole reason this seam
exists: a component that both decided how to run a workload and ran it would
have no point at which the decision could be recorded, reviewed, resolved
against a runtime's capabilities, or replayed after a restart. The boundary is
what makes the plan a durable artifact instead of a side effect.

Nothing here runs training, and nothing here talks to a runtime.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import Field

from xaytune.core.capabilities import CapabilityDocument, PluginDescriptor
from xaytune.core.domain.candidate import CandidateSpec
from xaytune.core.execution import TrainingExecutionSpec
from xaytune.core.immutable import FrozenDict, FrozenDomainModel

__all__ = ["CompilationContext", "SupportResult", "TrainerCompiler"]


class SupportResult(FrozenDomainModel):
    """Whether a compiler can handle a candidate, and if not, why.

    A bare ``False`` is not actionable: a planner that learns only "no" cannot
    tell a missing algorithm from an unsupported adapter, and cannot propose
    anything better. So refusal carries reasons.
    """

    supported: bool
    reasons: tuple[str, ...] = Field(default_factory=tuple)

    def __bool__(self) -> bool:
        return self.supported


class CompilationContext(FrozenDomainModel):
    """Everything compilation may depend on besides the candidate.

    Explicit and serializable, because compilation must be **deterministic**:
    the same candidate in the same context compiles to the same spec, or the
    ``request_digest`` derived from it means nothing and a retry looks like a
    different request.

    A compiler that read the clock, the environment or a global would break
    that silently, so anything it legitimately needs arrives here instead.

    ``run_id`` and ``seed`` are present because the *plan* needs them even
    though the *candidate* does not -- seed belongs to the realization
    (ADR-011), and the worker still has to be told what to seed with.
    """

    experiment_id: str | None = None
    node_id: str | None = None
    run_id: str | None = None

    seed: int | None = None
    replicate: int | None = None

    output_uri: str | None = None
    checkpoint_store_uri: str | None = None

    settings: FrozenDict = Field(default_factory=FrozenDict)


@runtime_checkable
class TrainerCompiler(Protocol):
    """Turns a candidate into a runtime-neutral execution spec.

    Takes the **whole** :class:`CandidateSpec`, not just its ``training``
    component. An SFT compiler could work from optimizer hyperparameters
    alone, but a GRPO or agent compiler cannot: it needs the reward, the
    environment and any pre-registered schedule to emit something runnable.
    Passing only the training program would force every RL compiler to reach
    around the interface for the rest.

    ``supports()`` takes the candidate for the same reason -- whether a
    compiler can handle a workload depends on its reward and environment, not
    only its algorithm.
    """

    descriptor: PluginDescriptor

    def capabilities(self) -> CapabilityDocument:
        """What this compiler can emit."""
        ...

    def supports(self, candidate: CandidateSpec) -> SupportResult:
        """Whether this compiler can handle *candidate*, with reasons if not."""
        ...

    def compile(
        self, candidate: CandidateSpec, context: CompilationContext
    ) -> TrainingExecutionSpec:
        """Return how to run *candidate*.

        Deterministic for the same inputs, and **must not submit anything**.
        A compiler that started a workload would produce an external effect
        with no operation record behind it -- the state ADR-013's journal
        exists to make impossible.
        """
        ...
